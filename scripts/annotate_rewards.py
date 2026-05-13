"""Reward annotation for a LeRobot dataset.

Decodes one camera's video per episode, runs a black-box ``compute_rewards``
on each video plus its language instruction, and stores three per-episode
reward variant arrays as sidecars under the dataset's local cache:

  $HF_LEROBOT_HOME/<repo_id>/meta/rewards/<reward_name>_reward/
    episode_000000.npy   # float32 (T,) – raw per-step reward in [0, 1]
    ...
    config.json

  $HF_LEROBOT_HOME/<repo_id>/meta/rewards/<reward_name>_max_reward/
    episode_000000.npy   # float32 (T,) – max reward of episode broadcast to all steps

  $HF_LEROBOT_HOME/<repo_id>/meta/rewards/<reward_name>_advantage/
    episode_000000.npy   # float32 (T,) – MC advantage A_t = G_t - V(s_t)
    ...
    config.json          # V(s_t) baseline is the mean MC return at step t across the dataset

The sidecar layout mirrors LeRobot's per-episode parquet structure so multiple
reward annotations can coexist and individual episodes can be recomputed.

Usage:
  uv run scripts/annotate_rewards.py \
      --repo-id memmelma/block_stack \
      --reward-name default \
      --stub
"""

import asyncio
import dataclasses
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
import tqdm
import tyro
from lerobot.common.constants import HF_LEROBOT_HOME

try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reward_vlm"))
    from rvlm.classes.rvlm import RVLM
    _rvlm: RVLM | None = None

    def _get_rvlm() -> RVLM:
        global _rvlm
        if _rvlm is None:
            _rvlm = RVLM()
        return _rvlm
except Exception as e:
    print(f"Error importing RVLM: {e}")

async def _compute_rewards_async(
    video: np.ndarray,
    instruction: str,
    semaphore: asyncio.Semaphore,
) -> np.ndarray:
    """Run RVLM reward computation for a single episode, respecting a concurrency semaphore."""
    async with semaphore:
        progress: list[float | None] = await _get_rvlm().compute_progress_async(
            video, instruction
        )
    arr = np.array([v if v is not None else 0.0 for v in progress], dtype=np.float32)
    return arr


async def _batch_compute_rewards(
    episodes: list[tuple[np.ndarray, str]],
    concurrency: int,
) -> list[np.ndarray]:
    """Compute rewards for all episodes in parallel, bounded by `concurrency`."""
    semaphore = asyncio.Semaphore(concurrency)
    results: list[np.ndarray | None] = [None] * len(episodes)

    with tqdm.tqdm(total=len(episodes), desc="Annotating (RVLM)") as pbar:
        async def _tracked(i: int, video: np.ndarray, instruction: str) -> None:
            results[i] = await _compute_rewards_async(video, instruction, semaphore)
            pbar.update(1)

        await asyncio.gather(*[
            _tracked(i, video, instruction)
            for i, (video, instruction) in enumerate(episodes)
        ])

    return results  # type: ignore[return-value]


def compute_rewards(video: np.ndarray, instruction: str) -> np.ndarray:
    """Annotate a single episode with a per-frame reward via RVLM.

    Args:
        video: (T, H, W, 3) uint8 RGB array, decoded from the scene camera.
        instruction: Language instruction for the episode.

    Returns:
        (T,) float32 array with values in [0, 1].
    """
    return asyncio.run(_compute_rewards_async(video, instruction, asyncio.Semaphore(1)))


def _stub_rewards(video: np.ndarray, instruction: str) -> np.ndarray:
    del instruction
    t = video.shape[0]
    return np.linspace(0.0, 1.0, t, dtype=np.float32)


def _subsample_frames(video: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (subsampled_video, sample_indices) with exactly n frames including first and last."""
    t = video.shape[0]
    if n >= t:
        return video, np.arange(t)
    indices = np.round(np.linspace(0, t - 1, n)).astype(int)
    return video[indices], indices


def _interpolate_rewards(rewards: np.ndarray, sample_indices: np.ndarray, full_length: int) -> np.ndarray:
    """Linearly interpolate rewards computed at sample_indices back to full_length."""
    full_indices = np.arange(full_length)
    return np.interp(full_indices, sample_indices, rewards).astype(np.float32)


def _mc_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Compute discounted Monte Carlo returns G_t = sum_{k>=0} gamma^k * r_{t+k}."""
    T = len(rewards)
    returns = np.empty(T, dtype=np.float32)
    G = 0.0
    for t in reversed(range(T)):
        G = float(rewards[t]) + gamma * G
        returns[t] = G
    return returns


def _compute_and_save_advantages(
    reward_dir: Path,
    advantage_dir: Path,
    gamma: float,
) -> None:
    """Load all reward files, compute MC returns, estimate V(s_t) as the per-step
    mean MC return across the dataset, then save A_t = G_t - V(s_t) for every episode.

    All existing advantage files are overwritten because the baseline is global and
    changes whenever new episodes are added.
    """
    reward_files = sorted(reward_dir.glob("episode_*.npy"))
    if not reward_files:
        return

    all_returns: list[np.ndarray] = []
    for f in reward_files:
        rewards = np.load(f)
        all_returns.append(_mc_returns(rewards, gamma))

    # Build per-step baseline V(s_t) = mean G_t at step t across all episodes.
    max_len = max(len(r) for r in all_returns)
    sum_g = np.zeros(max_len, dtype=np.float64)
    cnt_g = np.zeros(max_len, dtype=np.int64)
    for returns in all_returns:
        T = len(returns)
        sum_g[:T] += returns
        cnt_g[:T] += 1
    baseline = np.where(cnt_g > 0, sum_g / np.maximum(cnt_g, 1), 0.0).astype(np.float32)

    advantage_dir.mkdir(parents=True, exist_ok=True)
    for f, returns in zip(reward_files, all_returns):
        T = len(returns)
        advantage = (returns - baseline[:T]).astype(np.float32)
        np.save(advantage_dir / f.name, advantage)


@dataclasses.dataclass(frozen=True)
class Args:
    repo_id: str
    """LeRobot repo id (e.g. memmelma/block_stack). Must exist under $HF_LEROBOT_HOME."""
    reward_name: str
    """Name of this annotation. Stored under meta/rewards/<reward_name>/."""
    camera: str = "head"
    """Camera key whose video is fed to compute_rewards (default: head)."""
    episodes: tuple[int, ...] | None = None
    """Optional explicit episode indices to (re)annotate. Default: all episodes."""
    overwrite: bool = False
    """Re-annotate episodes that already have a saved .npy."""
    stub: bool = False
    """Use a deterministic linspace(0, 1, T) instead of calling compute_rewards."""
    subsample_frames: int = 8
    """Subsample each episode to this many frames (including first and last) before calling
    compute_rewards. Rewards are then linearly interpolated back to the original length T.
    Set to 0 to disable subsampling and pass the full video."""
    concurrency: int = 4
    """Number of parallel RVLM calls when annotating with the reward model (ignored with --stub)."""
    language_instruction: str | None = None
    """Override the language instruction for all episodes. If not set, the per-episode task
    string stored in episodes.jsonl is used."""
    gamma: float = 0.99
    """Discount factor for Monte Carlo return computation (used for advantage variant)."""
    push_to_hub: bool = False
    """After writing locally, upload meta/rewards/<reward_name>_{reward,max_reward,advantage}/ to the Hub."""
    private: bool = True
    """If creating/touching the Hub repo, mark it private."""


def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _decode_video(path: Path) -> np.ndarray:
    """Decode an mp4 to (T, H, W, 3) uint8 RGB. Prefers decord, falls back to torchvision."""
    try:
        import decord

        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(str(path))
        frames = vr.get_batch(list(range(len(vr)))).asnumpy()
        return frames.astype(np.uint8, copy=False)
    except Exception:  # pragma: no cover - fallback path
        import torchvision.io as tvio

        frames, _, _ = tvio.read_video(str(path), pts_unit="sec", output_format="THWC")
        return frames.numpy().astype(np.uint8, copy=False)


def main(args: Args) -> None:
    dataset_root = HF_LEROBOT_HOME / args.repo_id
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_root}. Download or convert it before annotating."
        )

    meta_dir = dataset_root / "meta"
    info = json.loads((meta_dir / "info.json").read_text())
    episodes_meta = _load_jsonl(meta_dir / "episodes.jsonl")
    tasks_meta = _load_jsonl(meta_dir / "tasks.jsonl")
    chunks_size = int(info.get("chunks_size", 1000))
    video_template = info["video_path"]  # videos/chunk-{...}/{video_key}/episode_{...}.mp4
    video_key = f"observation.images.{args.camera}"

    reward_dir = dataset_root / "meta" / "rewards" / f"{args.reward_name}_reward"
    max_reward_dir = dataset_root / "meta" / "rewards" / f"{args.reward_name}_max_reward"
    advantage_dir = dataset_root / "meta" / "rewards" / f"{args.reward_name}_advantage"
    reward_dir.mkdir(parents=True, exist_ok=True)
    max_reward_dir.mkdir(parents=True, exist_ok=True)

    target_episodes = (
        sorted(set(args.episodes)) if args.episodes is not None else [e["episode_index"] for e in episodes_meta]
    )
    ep_lookup = {e["episode_index"]: e for e in episodes_meta}

    written: list[int] = []

    # --- collect episodes that need annotation ---
    pending_indices: list[int] = []
    pending_videos: list[np.ndarray] = []
    pending_instructions: list[str] = []
    pending_num_frames: list[int] = []
    pending_sample_indices: list[np.ndarray] = []

    for ep_idx in tqdm.tqdm(target_episodes, desc="Loading videos"):
        if ep_idx not in ep_lookup:
            raise KeyError(f"Episode {ep_idx} not present in {meta_dir / 'episodes.jsonl'}")
        ep_meta = ep_lookup[ep_idx]
        num_frames = int(ep_meta["length"])
        ep_tasks = ep_meta.get("tasks") or [""]
        stored_instruction = " ".join(t for t in ep_tasks if t)
        instruction = args.language_instruction if args.language_instruction is not None else stored_instruction

        out_path = reward_dir / f"episode_{ep_idx:06d}.npy"
        if out_path.exists() and not args.overwrite:
            continue

        chunk = ep_idx // chunks_size
        video_path = dataset_root / video_template.format(
            episode_chunk=chunk, video_key=video_key, episode_index=ep_idx
        )
        if not video_path.exists():
            raise FileNotFoundError(f"Missing video for episode {ep_idx}: {video_path}")

        video = _decode_video(video_path)
        if video.shape[0] != num_frames:
            if video.shape[0] > num_frames:
                video = video[:num_frames]
            else:
                pad = np.repeat(video[-1:], num_frames - video.shape[0], axis=0)
                video = np.concatenate([video, pad], axis=0)

        if args.subsample_frames > 0:
            video_sub, sample_idx = _subsample_frames(video, args.subsample_frames)
        else:
            video_sub, sample_idx = video, np.arange(num_frames)

        pending_indices.append(ep_idx)
        pending_videos.append(video_sub)
        pending_instructions.append(instruction)
        pending_num_frames.append(num_frames)
        pending_sample_indices.append(sample_idx)

    # --- compute rewards (stub: serial; rvlm: async batch) ---
    if args.stub:
        rewards_list = [
            _stub_rewards(v, ins)
            for v, ins in tqdm.tqdm(
                zip(pending_videos, pending_instructions),
                total=len(pending_indices),
                desc=f"Annotating ({args.reward_name}, stub)",
            )
        ]
    else:
        rewards_list = asyncio.run(
            _batch_compute_rewards(
                list(zip(pending_videos, pending_instructions)),
                concurrency=args.concurrency,
            )
        )

    # --- validate, interpolate, and save ---
    for ep_idx, rewards, num_frames, sample_idx in tqdm.tqdm(
        zip(pending_indices, rewards_list, pending_num_frames, pending_sample_indices),
        total=len(pending_indices),
        desc="Saving rewards",
    ):
        rewards = np.asarray(rewards, dtype=np.float32)
        if rewards.shape[0] != len(sample_idx):
            raise ValueError(
                f"compute_rewards returned shape {rewards.shape}; expected ({len(sample_idx)},) for episode {ep_idx}"
            )
        if len(sample_idx) < num_frames:
            rewards = _interpolate_rewards(rewards, sample_idx, num_frames)
        if rewards.shape != (num_frames,):
            raise ValueError(
                f"interpolated rewards shape {rewards.shape}; expected ({num_frames},) for episode {ep_idx}"
            )
        ep_file = f"episode_{ep_idx:06d}.npy"
        np.save(reward_dir / ep_file, rewards)
        max_val = float(rewards.max())
        np.save(max_reward_dir / ep_file, np.full(rewards.shape, max_val, dtype=np.float32))
        written.append(ep_idx)

    config = {
        "reward_name": args.reward_name,
        "camera": args.camera,
        "repo_id": args.repo_id,
        "num_episodes_annotated": len(written),
        "total_episodes": len(episodes_meta),
        "stub": args.stub,
        "gamma": args.gamma,
        "created": _dt.datetime.utcnow().isoformat() + "Z",
    }
    for d in (reward_dir, max_reward_dir):
        (d / "config.json").write_text(json.dumps(config, indent=2))

    print(f"Wrote {len(written)} episode(s) → {reward_dir.name}, {max_reward_dir.name}")

    print("Computing advantages (recomputes all episodes to update global baseline)…")
    _compute_and_save_advantages(reward_dir, advantage_dir, gamma=args.gamma)
    (advantage_dir / "config.json").write_text(json.dumps({**config, "variant": "advantage"}, indent=2))
    print(f"Wrote advantages → {advantage_dir.name}")

    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )
        for local_dir in (reward_dir, max_reward_dir, advantage_dir):
            path_in_repo = f"meta/rewards/{local_dir.name}"
            api.upload_folder(
                repo_id=args.repo_id,
                repo_type="dataset",
                folder_path=str(local_dir),
                path_in_repo=path_in_repo,
            )
            print(f"Pushed {local_dir} -> {args.repo_id}:{path_in_repo}")


if __name__ == "__main__":
    main(tyro.cli(Args))
