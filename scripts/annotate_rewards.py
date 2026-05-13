"""Reward annotation for a LeRobot dataset.

Decodes one camera's video per episode, runs a black-box ``compute_rewards``
on each video plus its language instruction, and stores per-episode reward
arrays as a sidecar under the dataset's local cache:

  $HF_LEROBOT_HOME/<repo_id>/meta/rewards/<reward_name>/
    episode_000000.npy   # float32 (T,), values in [0, 1]
    ...
    config.json

The sidecar layout mirrors LeRobot's per-episode parquet structure so multiple
reward annotations can coexist and individual episodes can be recomputed.

Usage:
  uv run scripts/annotate_rewards.py \
      --repo-id memmelma/block_stack \
      --reward-name default \
      --stub
"""

import dataclasses
import datetime as _dt
import json
from pathlib import Path

import numpy as np
import tqdm
import tyro
from lerobot.common.constants import HF_LEROBOT_HOME


def compute_rewards(video: np.ndarray, instruction: str) -> np.ndarray:
    """Annotate a single episode with a per-frame reward.

    Args:
        video: (T, H, W, 3) uint8 RGB array, decoded from the scene camera.
        instruction: Language instruction for the episode.

    Returns:
        (T,) float32 array with values in [0, 1].
    """
    raise NotImplementedError(
        "Plug in your reward model here, or run with `--stub` to test the pipeline."
    )


def _stub_rewards(video: np.ndarray, instruction: str) -> np.ndarray:
    del instruction
    t = video.shape[0]
    return np.linspace(0.0, 1.0, t, dtype=np.float32)


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
    push_to_hub: bool = False
    """After writing locally, upload meta/rewards/<reward_name>/ to the Hub dataset repo."""
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
    task_by_name = {row["task"]: row["task_index"] for row in tasks_meta}
    chunks_size = int(info.get("chunks_size", 1000))
    video_template = info["video_path"]  # videos/chunk-{...}/{video_key}/episode_{...}.mp4
    video_key = f"observation.images.{args.camera}"

    out_dir = dataset_root / "meta" / "rewards" / args.reward_name
    out_dir.mkdir(parents=True, exist_ok=True)

    target_episodes = (
        sorted(set(args.episodes)) if args.episodes is not None else [e["episode_index"] for e in episodes_meta]
    )
    ep_lookup = {e["episode_index"]: e for e in episodes_meta}

    annotator = _stub_rewards if args.stub else compute_rewards
    written: list[int] = []

    for ep_idx in tqdm.tqdm(target_episodes, desc=f"Annotating ({args.reward_name})"):
        if ep_idx not in ep_lookup:
            raise KeyError(f"Episode {ep_idx} not present in {meta_dir / 'episodes.jsonl'}")
        ep_meta = ep_lookup[ep_idx]
        num_frames = int(ep_meta["length"])
        tasks = ep_meta.get("tasks") or [""]
        instruction = " ".join(t for t in tasks if t)
        if instruction and tasks[0] not in task_by_name:
            # Soft check: warn but don't crash; task may be free-form.
            pass

        out_path = out_dir / f"episode_{ep_idx:06d}.npy"
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
            # Some pipelines pad the last frame; trim or extend by replication.
            if video.shape[0] > num_frames:
                video = video[:num_frames]
            else:
                pad = np.repeat(video[-1:], num_frames - video.shape[0], axis=0)
                video = np.concatenate([video, pad], axis=0)

        rewards = np.asarray(annotator(video, instruction), dtype=np.float32)
        if rewards.shape != (num_frames,):
            raise ValueError(
                f"compute_rewards returned shape {rewards.shape}; expected ({num_frames},) for episode {ep_idx}"
            )
        rewards = np.clip(rewards, 0.0, 1.0)
        np.save(out_path, rewards)
        written.append(ep_idx)

    config_path = out_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "reward_name": args.reward_name,
                "camera": args.camera,
                "repo_id": args.repo_id,
                "num_episodes_annotated": len(written),
                "total_episodes": len(episodes_meta),
                "stub": args.stub,
                "created": _dt.datetime.utcnow().isoformat() + "Z",
            },
            indent=2,
        )
    )
    print(f"Wrote rewards for {len(written)} episode(s) to {out_dir}")

    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )
        path_in_repo = f"meta/rewards/{args.reward_name}"
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=str(out_dir),
            path_in_repo=path_in_repo,
        )
        print(f"Pushed {out_dir} -> {args.repo_id}:{path_in_repo}")


if __name__ == "__main__":
    main(tyro.cli(Args))
