"""Reward annotation for LeRobot datasets.

Decodes one camera's video per episode, runs a reward model on each video plus
its language instruction, and stores per-episode sidecar arrays under the
dataset's local cache:

  $HF_LEROBOT_HOME/<repo_id>/meta/rewards/
    <prefix>_reward/episode_000000.npy        # float32 (T,) raw per-step reward
    <prefix>_returns/episode_000000.npy       # MC return G_t  (always recomputed)
    <prefix>_advantage/episode_000000.npy     # G_t - global_mean (cross-repo, always recomputed)
    <prefix>_delta/episode_000000.npy         # r[t+1]-r[t]  (--compute-delta only)
    <prefix>_delta_returns/episode_000000.npy
    <prefix>_delta_advantage/episode_000000.npy
    <prefix>_normalized_reward/...            # topreward only: [0,1] min-max normalized
    <prefix>_normalized_returns/...
    <prefix>_normalized_advantage/...
    <prefix>_reward/config.json

<prefix> defaults to <norm_instruction>_<reward_model>  (e.g. put_the_block_rvlm).
Override with --reward-name for backwards-compat with existing training configs.

Only <prefix>_reward/ is gated by --overwrite.  All derived variants (returns,
advantages, deltas, normalization) are always recomputed because they depend on
cross-repo global statistics.

Supported reward models:
  rvlm, topreward, gvl, vlac, robodopamine, roboreward, rbm, rewind, rbm_libero
  stub  (deterministic linspace, for testing)

Usage:
  uv run scripts/annotate_rewards.py \\
      --repo-id memmelma/block_stack \\
      --reference-instruction "pick up the block" \\
      --reward-model rvlm

  # Multiple repos: advantages are computed globally across all of them
  uv run scripts/annotate_rewards.py \\
      --repo-ids memmelma/demo_a memmelma/demo_b \\
      --reference-instruction "pick up the block" \\
      --reward-model topreward \\
      --compute-delta
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import tqdm
import tyro
from lerobot.common.constants import HF_LEROBOT_HOME

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

# RVLM lives in a sibling repo (reward_vlm/) — keep the existing sys.path hack.
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reward_vlm"))
    from rvlm.classes.rvlm import RVLM as _RVLM_cls
    _rvlm_instance: "_RVLM_cls | None" = None

    def _get_rvlm() -> "_RVLM_cls":
        global _rvlm_instance
        if _rvlm_instance is None:
            _rvlm_instance = _RVLM_cls()
        return _rvlm_instance
except Exception as _e:  # noqa: BLE001
    _RVLM_cls = None  # type: ignore[assignment,misc]
    def _get_rvlm():  # type: ignore[misc]
        raise ImportError(f"RVLM not available: {_e}") from _e

# robometer is an installed package; imported lazily inside create_model().

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = [
    "rvlm", "topreward", "gvl", "vlac", "robodopamine",
    "roboreward", "rbm", "rewind", "rbm_libero", "stub",
]

_N_FRAMES_PER_CALL = 8  # context-window size for topreward / rbm per-timestep inference

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_instruction_for_key(s: str) -> str:
    """Lowercase, whitespace → underscores.  Used to build the sidecar prefix."""
    return "_".join(s.lower().split())


def parse_model_config(config_strs: tuple[str, ...]) -> dict:
    """Parse 'key=value' strings into a dict with basic type inference."""
    config: dict = {}
    for s in config_strs:
        key, val = s.split("=", 1)
        if val.lower() in ("true", "false"):
            config[key] = val.lower() == "true"
        else:
            try:
                config[key] = int(val)
            except ValueError:
                try:
                    config[key] = float(val)
                except ValueError:
                    config[key] = val
    return config


def create_model(reward_model: str, model_path: str | None, max_frames: int, **model_config):
    """Instantiate a reward model.  Mirrors reference annotate_hdf5_rewards_example.py."""
    if reward_model == "stub":
        return None
    if reward_model == "rvlm":
        return _get_rvlm()
    if reward_model == "gvl":
        from robometer.evals.baselines.gvl import GVL
        return GVL(max_frames=max_frames, **model_config)
    if reward_model == "vlac":
        from robometer.evals.baselines.vlac import VLAC
        if not model_path:
            raise ValueError("--model-path is required for vlac")
        return VLAC(model_path=model_path, **model_config)
    if reward_model == "robodopamine":
        from robometer.evals.baselines.robodopamine import RoboDopamine
        if not model_path:
            raise ValueError("--model-path is required for robodopamine")
        return RoboDopamine(model_path=model_path, **model_config)
    if reward_model == "roboreward":
        from robometer.evals.baselines.roboreward import RoboReward
        return RoboReward(model_path=model_path or "teetone/RoboReward-4B", **model_config)
    if reward_model in ("rbm", "rewind"):
        from robometer.evals.baselines.rbm_model import RBMModel
        if not model_path:
            raise ValueError("--model-path is required for rbm/rewind")
        return RBMModel(checkpoint_path=model_path)
    if reward_model == "rbm_libero":
        from robometer.evals.baselines.rbm_model import RBMModel
        return RBMModel(checkpoint_path=model_path or "jesbu1/robometer-4b-fft-libero")
    if reward_model == "topreward":
        from robometer.evals.baselines.topreward_official import TOPReward
        return TOPReward(**model_config)
    raise ValueError(f"Unknown reward_model: {reward_model!r}. Supported: {SUPPORTED_MODELS}")


# ---------------------------------------------------------------------------
# Per-episode inference
# ---------------------------------------------------------------------------

def compute_and_interpolate(
    model, frames: np.ndarray, task: str, subsample_n: int, reward_model: str
) -> np.ndarray:
    """Subsample frames, compute progress, interpolate back to full length.

    For topreward / rbm*: each query timestep t_i gets an 8-frame linspace
    prefix frames[0:t_i+1] so the model sees cumulative progress context.
    For roboreward: each k-th subsampled step gets prefix subsampled[0:k+1].
    For all others (rvlm sync, gvl, vlac, robodopamine): one batched call.
    """
    total = len(frames)

    if reward_model in ("rbm", "rewind", "rbm_libero", "topreward"):
        _, query_indices = linspace_subsample_frames(frames, subsample_n)
        per_query = []
        for t_i in query_indices:
            prefix_8, _ = linspace_subsample_frames(frames[: t_i + 1], _N_FRAMES_PER_CALL)
            result = model.compute_progress(prefix_8, task_description=task)
            per_query.append(float(result[-1]))
        progress = np.array(per_query, dtype=np.float32)
        indices = query_indices

    elif reward_model == "roboreward":
        subsampled, indices = linspace_subsample_frames(frames, subsample_n)
        per_step = []
        for k in range(len(subsampled)):
            result = model.compute_progress(subsampled[: k + 1], task_description=task)
            per_step.append(float(result[-1]))
        progress = np.array(per_step, dtype=np.float32)

    else:
        # rvlm (sync path), gvl, vlac, robodopamine: one batched call
        subsampled, indices = linspace_subsample_frames(frames, subsample_n)
        progress = np.array(
            model.compute_progress(subsampled, task_description=task), dtype=np.float32
        )
        if progress.ndim > 1:
            progress = progress.mean(axis=-1)

    if len(indices) < total:
        progress = np.interp(np.arange(total), indices, progress).astype(np.float32)
    return progress


def _stub_rewards(frames: np.ndarray, _instruction: str) -> np.ndarray:
    return np.linspace(0.0, 1.0, frames.shape[0], dtype=np.float32)


def _resolve_subsample_n(subsample_frames: int, subsample_factor: int, traj_len: int) -> int:
    if subsample_factor > 0:
        n = max(2, traj_len // subsample_factor)
        if subsample_frames > 0:
            n = max(n, subsample_frames)
        return n
    return max(2, subsample_frames) if subsample_frames > 0 else traj_len


def linspace_subsample_frames(
    frames: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(subsampled_frames, indices)`` with ``n`` frames evenly spaced in time.

    Indices span ``[0, len(frames)-1]`` (inclusive endpoints).  Matches the convention
    used by ``robometer.data.datasets.helpers.linspace_subsample_frames`` so we do not
    need to import robometer for RVLM-only annotation.
    """
    t = len(frames)
    if t == 0:
        return frames, np.array([], dtype=int)
    n = max(1, int(n))
    if n >= t:
        indices = np.arange(t, dtype=int)
    else:
        indices = np.round(np.linspace(0, t - 1, n)).astype(int)
        indices[0] = 0
        indices[-1] = t - 1
    return frames[indices], indices


# ---------------------------------------------------------------------------
# Video I/O helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _decode_video(path: Path) -> np.ndarray:
    """Decode an mp4 to (T, H, W, 3) uint8 RGB.  Prefers decord, falls back to torchvision."""
    try:
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(str(path))
        frames = vr.get_batch(list(range(len(vr)))).asnumpy()
        return frames.astype(np.uint8, copy=False)
    except Exception:  # pragma: no cover
        import torchvision.io as tvio
        frames, _, _ = tvio.read_video(str(path), pts_unit="sec", output_format="THWC")
        return frames.numpy().astype(np.uint8, copy=False)


# ---------------------------------------------------------------------------
# Per-repo annotation: sync path
# ---------------------------------------------------------------------------

def _annotate_repo_sync(
    dataset_root: Path,
    model,
    args: "Args",
    prefix: str,
    reference_instruction: str,
) -> list[int]:
    """Annotate all episodes in one repo sequentially.  Returns list of written ep indices."""
    meta_dir = dataset_root / "meta"
    info = json.loads((meta_dir / "info.json").read_text())
    episodes_meta = _load_jsonl(meta_dir / "episodes.jsonl")
    chunks_size = int(info.get("chunks_size", 1000))
    video_template = info["video_path"]
    video_key = f"observation.images.{args.camera}"

    reward_dir = dataset_root / "meta" / "rewards" / f"{prefix}_reward"
    reward_dir.mkdir(parents=True, exist_ok=True)

    target_episodes = (
        sorted(set(args.episodes)) if args.episodes is not None
        else [e["episode_index"] for e in episodes_meta]
    )
    ep_lookup = {e["episode_index"]: e for e in episodes_meta}
    written: list[int] = []

    for ep_idx in tqdm.tqdm(target_episodes, desc=f"Annotating {dataset_root.name}"):
        out_path = reward_dir / f"episode_{ep_idx:06d}.npy"
        if out_path.exists() and not args.overwrite:
            continue

        ep_meta = ep_lookup[ep_idx]
        num_frames = int(ep_meta["length"])
        stored_instr = " ".join(t for t in (ep_meta.get("tasks") or [""]) if t)
        instruction = reference_instruction if reference_instruction else stored_instr

        chunk = ep_idx // chunks_size
        video_path = dataset_root / video_template.format(
            episode_chunk=chunk, video_key=video_key, episode_index=ep_idx
        )
        if not video_path.exists():
            raise FileNotFoundError(f"Missing video for episode {ep_idx}: {video_path}")

        frames = _decode_video(video_path)
        if frames.shape[0] != num_frames:
            if frames.shape[0] > num_frames:
                frames = frames[:num_frames]
            else:
                pad = np.repeat(frames[-1:], num_frames - frames.shape[0], axis=0)
                frames = np.concatenate([frames, pad], axis=0)

        if args.stub:
            rewards = _stub_rewards(frames, instruction)
        else:
            n_sub = _resolve_subsample_n(args.subsample_frames, args.subsample_factor, num_frames)
            rewards = compute_and_interpolate(
                model, frames, instruction, n_sub, args.reward_model
            )

        np.save(out_path, rewards.astype(np.float32))
        written.append(ep_idx)

    return written


# ---------------------------------------------------------------------------
# Per-repo annotation: async path (rvlm only)
# ---------------------------------------------------------------------------

async def _batch_compute_rvlm_async(
    episodes: list[tuple[np.ndarray, str]],
    concurrency: int,
) -> list[np.ndarray]:
    semaphore = asyncio.Semaphore(concurrency)
    results: list[np.ndarray | None] = [None] * len(episodes)

    with tqdm.tqdm(total=len(episodes), desc="Annotating (RVLM async)") as pbar:
        async def _tracked(i: int, video: np.ndarray, instruction: str) -> None:
            async with semaphore:
                progress = await _get_rvlm().compute_progress_async(video, instruction)
            results[i] = np.array(
                [v if v is not None else 0.0 for v in progress], dtype=np.float32
            )
            pbar.update(1)

        await asyncio.gather(*[
            _tracked(i, video, instr)
            for i, (video, instr) in enumerate(episodes)
        ])

    return results  # type: ignore[return-value]


def _annotate_repo_async(
    dataset_root: Path,
    args: "Args",
    prefix: str,
    reference_instruction: str,
) -> list[int]:
    """RVLM-only async annotation path.  Returns list of written ep indices."""
    meta_dir = dataset_root / "meta"
    info = json.loads((meta_dir / "info.json").read_text())
    episodes_meta = _load_jsonl(meta_dir / "episodes.jsonl")
    chunks_size = int(info.get("chunks_size", 1000))
    video_template = info["video_path"]
    video_key = f"observation.images.{args.camera}"

    reward_dir = dataset_root / "meta" / "rewards" / f"{prefix}_reward"
    reward_dir.mkdir(parents=True, exist_ok=True)

    target_episodes = (
        sorted(set(args.episodes)) if args.episodes is not None
        else [e["episode_index"] for e in episodes_meta]
    )
    ep_lookup = {e["episode_index"]: e for e in episodes_meta}

    pending_indices: list[int] = []
    pending_videos: list[np.ndarray] = []
    pending_instructions: list[str] = []
    pending_num_frames: list[int] = []
    pending_sample_indices: list[np.ndarray] = []

    for ep_idx in tqdm.tqdm(target_episodes, desc=f"Loading videos {dataset_root.name}"):
        out_path = reward_dir / f"episode_{ep_idx:06d}.npy"
        if out_path.exists() and not args.overwrite:
            continue

        ep_meta = ep_lookup[ep_idx]
        num_frames = int(ep_meta["length"])
        stored_instr = " ".join(t for t in (ep_meta.get("tasks") or [""]) if t)
        instruction = reference_instruction if reference_instruction else stored_instr

        chunk = ep_idx // chunks_size
        video_path = dataset_root / video_template.format(
            episode_chunk=chunk, video_key=video_key, episode_index=ep_idx
        )
        if not video_path.exists():
            raise FileNotFoundError(f"Missing video for episode {ep_idx}: {video_path}")

        frames = _decode_video(video_path)
        if frames.shape[0] != num_frames:
            if frames.shape[0] > num_frames:
                frames = frames[:num_frames]
            else:
                frames = np.concatenate(
                    [frames, np.repeat(frames[-1:], num_frames - frames.shape[0], axis=0)], axis=0
                )

        n_sub = _resolve_subsample_n(args.subsample_frames, args.subsample_factor, num_frames)
        video_sub, sample_idx = linspace_subsample_frames(frames, n_sub)

        pending_indices.append(ep_idx)
        pending_videos.append(video_sub)
        pending_instructions.append(instruction)
        pending_num_frames.append(num_frames)
        pending_sample_indices.append(sample_idx)

    if not pending_indices:
        return []

    rewards_list = asyncio.run(
        _batch_compute_rvlm_async(
            list(zip(pending_videos, pending_instructions)),
            concurrency=args.concurrency,
        )
    )

    written: list[int] = []
    for ep_idx, rewards, num_frames, sample_idx in zip(
        pending_indices, rewards_list, pending_num_frames, pending_sample_indices
    ):
        rewards = np.asarray(rewards, dtype=np.float32)
        if len(sample_idx) < num_frames:
            rewards = np.interp(np.arange(num_frames), sample_idx, rewards).astype(np.float32)
        np.save(reward_dir / f"episode_{ep_idx:06d}.npy", rewards)
        written.append(ep_idx)

    return written


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def compute_and_write_returns(rewards_dir: Path, returns_dir: Path, gamma: float) -> None:
    """Read <rewards_dir>/episode_*.npy, write MC returns to <returns_dir>/episode_*.npy."""
    reward_files = sorted(rewards_dir.glob("episode_*.npy"))
    if not reward_files:
        return
    returns_dir.mkdir(parents=True, exist_ok=True)
    for f in tqdm.tqdm(reward_files, desc=f"Returns [{rewards_dir.parent.name}]"):
        r = np.load(f).astype(np.float32)
        G = np.empty_like(r)
        G[-1] = r[-1]
        for t in reversed(range(len(r) - 1)):
            G[t] = r[t] + gamma * G[t + 1]
        np.save(returns_dir / f.name, G)


# ---------------------------------------------------------------------------
# Delta rewards
# ---------------------------------------------------------------------------

def compute_and_write_delta_rewards(rewards_dir: Path, delta_dir: Path) -> None:
    """delta[t] = r[t+1] - r[t], delta[-1] = 0.  Reads from rewards_dir, writes to delta_dir."""
    reward_files = sorted(rewards_dir.glob("episode_*.npy"))
    if not reward_files:
        return
    delta_dir.mkdir(parents=True, exist_ok=True)
    for f in tqdm.tqdm(reward_files, desc=f"Delta rewards [{rewards_dir.parent.name}]"):
        r = np.load(f).astype(np.float32)
        delta = np.zeros_like(r)
        delta[:-1] = r[1:] - r[:-1]
        np.save(delta_dir / f.name, delta)


# ---------------------------------------------------------------------------
# Advantages (cross-repo scalar-mean baseline)
# ---------------------------------------------------------------------------

def compute_and_write_advantages(
    returns_dirs: list[Path],
    advantage_dirs: list[Path],
) -> None:
    """Scalar-mean cross-repo advantage: adv = G_t - global_mean(all G_t).

    Reads returns from every returns_dir, computes a single scalar baseline
    across all of them, then writes per-episode advantage files to each
    corresponding advantage_dir.
    """
    all_returns: list[np.ndarray] = []
    for d in returns_dirs:
        for f in sorted(d.glob("episode_*.npy")):
            all_returns.append(np.load(f))

    if not all_returns:
        print("  No returns found, skipping advantage computation.")
        return

    global_mean = float(np.concatenate(all_returns).mean())
    n_repos = len(returns_dirs)
    print(f"  Global mean return across {n_repos} repo(s): {global_mean:.4f}")

    for src, dst in zip(returns_dirs, advantage_dirs):
        dst.mkdir(parents=True, exist_ok=True)
        for f in tqdm.tqdm(
            sorted(src.glob("episode_*.npy")),
            desc=f"Advantages [{src.parent.parent.name}]",
        ):
            G = np.load(f).astype(np.float32)
            np.save(dst / f.name, (G - global_mean).astype(np.float32))


# ---------------------------------------------------------------------------
# Normalized rewards (topreward only)
# ---------------------------------------------------------------------------

def compute_and_write_normalized_rewards(
    reward_dirs: list[Path],
    norm_reward_dirs: list[Path],
) -> None:
    """Min-max normalize raw rewards to [0, 1] using cross-repo global min/max."""
    all_rewards: list[np.ndarray] = []
    for d in reward_dirs:
        for f in sorted(d.glob("episode_*.npy")):
            all_rewards.append(np.load(f))

    if not all_rewards:
        print("  No rewards found, skipping normalization.")
        return

    cat = np.concatenate(all_rewards)
    r_min, r_max = float(cat.min()), float(cat.max())
    r_range = r_max - r_min
    print(f"  Global reward min={r_min:.4f}  max={r_max:.4f}  range={r_range:.4f}")

    for src, dst in zip(reward_dirs, norm_reward_dirs):
        dst.mkdir(parents=True, exist_ok=True)
        for f in tqdm.tqdm(
            sorted(src.glob("episode_*.npy")),
            desc=f"Normalize [{src.parent.parent.name}]",
        ):
            r = np.load(f).astype(np.float32)
            r_norm = ((r - r_min) / r_range).astype(np.float32) if r_range > 0 else np.zeros_like(r)
            np.save(dst / f.name, r_norm)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Args:
    # --- required: exactly one of repo_id / repo_ids ---
    repo_id: str | None = None
    """Single LeRobot repo id.  Mutually exclusive with --repo-ids."""
    repo_ids: tuple[str, ...] | None = None
    """One or more LeRobot repo ids.  Advantages and normalization baselines are
    computed globally across all repos.  Mutually exclusive with --repo-id."""

    # --- reward model ---
    reward_model: Literal[
        "rvlm", "topreward", "gvl", "vlac", "robodopamine",
        "roboreward", "rbm", "rewind", "rbm_libero", "stub",
    ] = "rvlm"
    """Reward model to use."""

    reference_instruction: str | None = None
    """Override the language instruction for all episodes.  When set, also used
    as the basis for the default sidecar prefix (<norm_instruction>_<model>).
    If not set, the per-episode task string in episodes.jsonl is used and
    --reward-name must be provided to define the sidecar prefix."""

    reward_name: str | None = None
    """Override the sidecar prefix (default: <norm_instruction>_<reward_model>).
    Useful for backwards-compat with existing training configs."""

    # --- model knobs ---
    model_path: str | None = None
    """Checkpoint / HuggingFace path (required for vlac, robodopamine, rbm, rewind)."""
    model_config: tuple[str, ...] = ()
    """Extra model constructor kwargs as key=value pairs (e.g. model_name=gemini-3-flash)."""
    max_frames: int = 16
    """Max frames passed to GVL."""

    # --- episode selection ---
    camera: str = "head"
    """Camera key whose video is decoded (e.g. head, wrist)."""
    episodes: tuple[int, ...] | None = None
    """Optional explicit episode indices to (re)annotate.  Default: all episodes."""

    # --- subsampling ---
    subsample_frames: int = 8
    """Fixed number of frames to subsample per episode before reward inference.
    Set to 0 to disable.  Rewards are interpolated back to full length."""
    subsample_factor: int = 0
    """Downsample factor N: subsample_n = len(traj) // N.  Overrides subsample_frames
    when > 0."""

    # --- behaviour flags ---
    overwrite: bool = False
    """Re-run reward model inference even if <prefix>_reward/ files already exist.
    Without this flag, only returns/advantages/deltas are recomputed."""
    stub: bool = False
    """Use a deterministic linspace(0, 1, T) stub instead of calling the reward model."""

    # --- derived-variant control ---
    skip_returns: bool = False
    """Skip MC return computation after reward annotation."""
    skip_advantages: bool = False
    """Skip advantage computation (cross-repo global-mean subtraction)."""
    compute_delta: bool = True
    """Also compute potential-based delta rewards r[t+1]-r[t] and their returns/advantages."""
    gamma: float = 0.99
    """Discount factor for MC return computation."""

    # --- async ---
    concurrency: int = 16
    """Async concurrency limit for RVLM calls."""

    # --- hub ---
    push_to_hub: bool = False
    """After writing locally, upload all <prefix>_* variant dirs to the Hub."""
    private: bool = True
    """Mark the Hub repo as private when creating/touching it."""


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(args: Args) -> None:  # noqa: C901
    # ---- validate repo args ----
    if args.repo_id is None and args.repo_ids is None:
        raise SystemExit("Provide exactly one of --repo-id or --repo-ids.")
    if args.repo_id is not None and args.repo_ids is not None:
        raise SystemExit("--repo-id and --repo-ids are mutually exclusive.")

    repo_ids: list[str] = list(args.repo_ids) if args.repo_ids is not None else [args.repo_id]  # type: ignore[list-item]

    # ---- resolve sidecar prefix ----
    if args.reward_name is not None:
        prefix = args.reward_name
    elif args.reference_instruction is not None:
        prefix = f"{normalize_instruction_for_key(args.reference_instruction)}_{args.reward_model}"
    else:
        raise SystemExit(
            "Either --reference-instruction or --reward-name must be provided to determine "
            "the sidecar prefix."
        )

    reference_instruction = args.reference_instruction or ""

    # ---- resolve dataset roots ----
    dataset_roots: list[Path] = []
    for rid in repo_ids:
        root = HF_LEROBOT_HOME / rid
        if not root.exists():
            raise FileNotFoundError(
                f"Dataset not found at {root}. Download or convert it before annotating."
            )
        dataset_roots.append(root)

    model_cfg = parse_model_config(args.model_config)
    annotate_fn = _annotate_repo_async if (args.reward_model == "rvlm" and not args.stub) else _annotate_repo_sync

    print(f"\n{'='*60}")
    print(f"Reward model : {args.reward_model}")
    print(f"Repos        : {repo_ids}")
    print(f"Instruction  : {reference_instruction!r}")
    print(f"Prefix       : {prefix}")
    print(f"{'='*60}")

    # =======================================================================
    # Phase 1: per-repo — reward annotation, MC returns, delta rewards.
    # =======================================================================
    model = None  # lazy init

    for root in dataset_roots:
        print(f"\n--- Repo: {root} ---")

        reward_dir = root / "meta" / "rewards" / f"{prefix}_reward"
        all_exist = reward_dir.exists() and any(reward_dir.glob("episode_*.npy"))

        if all_exist and not args.overwrite:
            print(f"  Rewards already exist at '{reward_dir.name}', skipping inference (use --overwrite to force).")
        else:
            if model is None and not args.stub:
                model = create_model(
                    args.reward_model,
                    model_path=args.model_path,
                    max_frames=args.max_frames,
                    **model_cfg,
                )

            if annotate_fn is _annotate_repo_async:
                written = _annotate_repo_async(root, args, prefix, reference_instruction)
            else:
                written = _annotate_repo_sync(root, model, args, prefix, reference_instruction)
            print(f"  Wrote {len(written)} episode(s) → {prefix}_reward/")

        if not args.skip_returns:
            returns_dir = root / "meta" / "rewards" / f"{prefix}_returns"
            print(f"  Computing MC returns (gamma={args.gamma}) → {returns_dir.name}/")
            compute_and_write_returns(reward_dir, returns_dir, gamma=args.gamma)

        if args.compute_delta:
            delta_dir = root / "meta" / "rewards" / f"{prefix}_delta"
            print(f"  Computing delta rewards → {delta_dir.name}/")
            compute_and_write_delta_rewards(reward_dir, delta_dir)

            if not args.skip_returns:
                delta_returns_dir = root / "meta" / "rewards" / f"{prefix}_delta_returns"
                print(f"  Computing MC returns for delta → {delta_returns_dir.name}/")
                compute_and_write_returns(delta_dir, delta_returns_dir, gamma=args.gamma)

        # Write config.json into the reward dir
        config_payload = {
            "prefix": prefix,
            "reward_model": args.reward_model,
            "reference_instruction": reference_instruction,
            "camera": args.camera,
            "gamma": args.gamma,
            "stub": args.stub,
            "created": _dt.datetime.utcnow().isoformat() + "Z",
        }
        if reward_dir.exists():
            (reward_dir / "config.json").write_text(json.dumps(config_payload, indent=2))

    # =======================================================================
    # Phase 2: cross-repo — advantages and topreward normalization.
    # =======================================================================
    if not args.skip_advantages:
        print(f"\n{'='*60}")
        print(f"Computing advantages (cross-repo scalar-mean baseline) → {prefix}_advantage/")
        print(f"{'='*60}")
        returns_dirs = [r / "meta" / "rewards" / f"{prefix}_returns" for r in dataset_roots]
        advantage_dirs = [r / "meta" / "rewards" / f"{prefix}_advantage" for r in dataset_roots]
        compute_and_write_advantages(returns_dirs, advantage_dirs)

        if args.compute_delta:
            print(f"\nComputing advantages for delta rewards → {prefix}_delta_advantage/")
            delta_returns_dirs = [r / "meta" / "rewards" / f"{prefix}_delta_returns" for r in dataset_roots]
            delta_adv_dirs = [r / "meta" / "rewards" / f"{prefix}_delta_advantage" for r in dataset_roots]
            compute_and_write_advantages(delta_returns_dirs, delta_adv_dirs)

    if args.reward_model == "topreward":
        norm_prefix = f"{prefix}_normalized"
        print(f"\nNormalizing topreward rewards [0, 1] → {norm_prefix}_reward/")
        reward_dirs = [r / "meta" / "rewards" / f"{prefix}_reward" for r in dataset_roots]
        norm_reward_dirs = [r / "meta" / "rewards" / f"{norm_prefix}_reward" for r in dataset_roots]
        compute_and_write_normalized_rewards(reward_dirs, norm_reward_dirs)

        if not args.skip_returns:
            print(f"  Computing MC returns for normalized rewards → {norm_prefix}_returns/")
            for root in dataset_roots:
                compute_and_write_returns(
                    root / "meta" / "rewards" / f"{norm_prefix}_reward",
                    root / "meta" / "rewards" / f"{norm_prefix}_returns",
                    gamma=args.gamma,
                )

        if not args.skip_advantages:
            print(f"  Computing advantages for normalized rewards → {norm_prefix}_advantage/")
            norm_returns_dirs = [r / "meta" / "rewards" / f"{norm_prefix}_returns" for r in dataset_roots]
            norm_adv_dirs = [r / "meta" / "rewards" / f"{norm_prefix}_advantage" for r in dataset_roots]
            compute_and_write_advantages(norm_returns_dirs, norm_adv_dirs)

    # =======================================================================
    # push_to_hub: upload every <prefix>_* variant dir for each repo.
    # =======================================================================
    if args.push_to_hub:
        from huggingface_hub import HfApi
        api = HfApi()
        for repo_id, root in zip(repo_ids, dataset_roots):
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=args.private, exist_ok=True)
            rewards_root = root / "meta" / "rewards"
            variant_dirs = sorted(rewards_root.glob(f"{prefix}_*"))
            for local_dir in variant_dirs:
                if not local_dir.is_dir():
                    continue
                path_in_repo = f"meta/rewards/{local_dir.name}"
                api.upload_folder(
                    repo_id=repo_id,
                    repo_type="dataset",
                    folder_path=str(local_dir),
                    path_in_repo=path_in_repo,
                )
                print(f"Pushed {local_dir.name} → {repo_id}:{path_in_repo}")

    print("\nDone.")


if __name__ == "__main__":
    main(tyro.cli(Args))
