"""Join multiple raiden processed datasets into a single LeRobot v2.0 dataset.

Reads a YAML manifest mapping each raiden processed directory to an
``optimality`` label (``optimal`` / ``suboptimal`` / ``failure``), iterates
every episode across all listed datasets, and writes a single joint
LeRobot dataset.  Optimality (and the source dataset / source episode id)
is preserved per-episode in ``meta/optimality.jsonl`` so downstream reward
annotation, visualization and training can carry the label end-to-end.

Manifest format (``--manifest manifest.yaml``):

    datasets:
      - path: /path/to/raiden/data/processed/swb_optimal
        optimality: optimal
      - path: /path/to/raiden/data/processed/swb_failure
        optimality: failure
      # optional: name override used for the source field
      - path: /path/to/raiden/data/processed/swb_suboptimal
        optimality: suboptimal
        name: swb_suboptimal_v2

Notes:
- LeRobot's ``episode_index`` is a single global counter across all input
  datasets; we rely on ``meta/optimality.jsonl`` as the source of truth
  for "which raiden dataset did this episode come from".
- All input datasets must share FPS and source resolution; the script
  asserts this and refuses to mix.

Usage:
    uv run scripts/convert_raiden_to_lerobot_joint.py \\
        --manifest examples/datasets_manifest_swb.yaml \\
        --repo-id memmelma/swb_joint
"""

from __future__ import annotations

import dataclasses
import json
import pickle
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import tqdm
import tyro
import yaml
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

CAMERA_NAME_MAP = {
    "scene_camera": "head",
    "left_wrist_camera": "left_wrist",
    "right_wrist_camera": "right_wrist",
}
CAMERAS = ["head", "left_wrist", "right_wrist"]
_OUT_TO_SRC = {v: k for k, v in CAMERA_NAME_MAP.items()}
FPS = 30
JPEG_QUALITY = 95
DOWNSAMPLE_FACTOR = 2
MOTORS = [
    "left_joint_0", "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4", "left_joint_5",
    "left_gripper",
    "right_joint_0", "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4", "right_joint_5",
    "right_gripper",
]


def _noop_row_keep_mask(joints, actions, eps_joint, eps_gripper):
    d = np.abs(actions.astype(np.float64) - joints.astype(np.float64))
    dj_l = np.max(d[:, 0:6], axis=1)
    dg_l = d[:, 6]
    dj_r = np.max(d[:, 7:13], axis=1)
    dg_r = d[:, 13]
    is_noop = (
        (dj_l <= eps_joint) & (dg_l <= eps_gripper)
        & (dj_r <= eps_joint) & (dg_r <= eps_gripper)
    )
    return ~is_noop


def _load_episode_lowdim(ep_dir):
    lowdim_dir = ep_dir / "lowdim"
    pkl_files = sorted(lowdim_dir.glob("??????????.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No lowdim pkl files in {lowdim_dir}")
    with open(pkl_files[0], "rb") as f:
        first_frame = pickle.load(f)
    first_joints = np.asarray(first_frame["joints"], dtype=np.float32)
    if first_joints.shape != (14,):
        raise ValueError(f"Expected bimanual joints shape (14,); got {first_joints.shape} at {pkl_files[0]}")
    n = len(pkl_files)
    joints = np.zeros((n, 14), dtype=np.float32)
    task = None
    for i, p in enumerate(pkl_files):
        with open(p, "rb") as f:
            frame = pickle.load(f)
        j = np.asarray(frame["joints"], dtype=np.float32)
        if j.shape != (14,):
            raise ValueError(f"Expected joints shape (14,); got {j.shape} at {p}")
        joints[i] = j
        if task is None:
            prompt = frame.get("language_prompt", "")
            if isinstance(prompt, np.ndarray):
                prompt = prompt.item()
            task = str(prompt)
    return joints, (task or ""), n

VALID_OPTIMALITIES = ("optimal", "suboptimal", "failure")


@dataclasses.dataclass(frozen=True)
class Args:
    manifest: Path
    """YAML manifest mapping raiden processed dirs to optimality labels."""
    repo_id: str
    """Target LeRobot repo id (e.g. ``memmelma/swb_joint``)."""
    push_to_hub: bool = False
    """Push joint dataset to HF Hub after conversion."""
    private: bool = True
    """If pushing, mark hub repo as private."""
    num_demos_per_dataset: int | None = None
    """Cap number of episodes converted from each source dataset (default: all)."""
    filter_noop: bool = True
    """Drop static state→action transitions (same as single-dataset converter)."""
    noop_eps_joint: float = 1e-3
    """No-op joint delta threshold (radians)."""
    noop_eps_gripper: float = 1e-3
    """No-op gripper delta threshold."""
    noop_min_frames: int = 10
    """Skip episode if fewer frames remain after no-op filtering."""
    num_image_workers: int = 8
    """Number of threads for concurrent PNG decode + resize per episode."""
    lerobot_home: str | None = None
    """Optional override for output root (defaults to $HF_LEROBOT_HOME)."""


@dataclasses.dataclass(frozen=True)
class _SourceEpisode:
    optimality: str
    source_name: str
    source_ep_id: str
    ep_dir: Path


def _load_manifest(path: Path) -> list[_SourceEpisode]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    entries = cfg.get("datasets") if isinstance(cfg, dict) else None
    if not entries:
        raise ValueError(f"Manifest {path} has no 'datasets:' list")

    sources: list[_SourceEpisode] = []
    seen_paths: set[Path] = set()
    for entry in entries:
        p = Path(entry["path"]).expanduser().resolve()
        opt = entry["optimality"]
        name = entry.get("name") or p.name
        if opt not in VALID_OPTIMALITIES:
            raise ValueError(
                f"optimality={opt!r} must be one of {VALID_OPTIMALITIES} (path={p})"
            )
        if not p.exists():
            raise FileNotFoundError(f"Manifest path does not exist: {p}")
        if p in seen_paths:
            raise ValueError(f"Duplicate dataset path in manifest: {p}")
        seen_paths.add(p)

        ep_dirs = sorted(
            d for d in p.iterdir()
            if d.is_dir() and d.name.isdigit() and (d / "metadata.json").exists()
        )
        if not ep_dirs:
            raise SystemExit(f"No episode dirs found under {p}")
        for ep_dir in ep_dirs:
            sources.append(_SourceEpisode(
                optimality=opt,
                source_name=name,
                source_ep_id=ep_dir.name,
                ep_dir=ep_dir,
            ))
    return sources


def _validate_consistency(sources: list[_SourceEpisode]) -> None:
    """Assert all source episodes share FPS/resolution we hard-code into LeRobot features."""
    seen_fps: set[int] = set()
    seen_res: set[tuple[int, int]] = set()
    for s in sources:
        with open(s.ep_dir / "metadata.json") as f:
            md = json.load(f)
        seen_fps.add(int(md.get("framerate", FPS)))
        res = md.get("resolution")
        if res is not None:
            seen_res.add((int(res[0]), int(res[1])))
    if len(seen_fps) > 1:
        raise ValueError(f"Mixed FPS across datasets: {seen_fps}; refusing to join.")
    if seen_fps and FPS not in seen_fps:
        raise ValueError(
            f"Manifest FPS {seen_fps} != hard-coded FPS={FPS}; adjust scripts/convert_raiden_to_lerobot.py."
        )
    if len(seen_res) > 1:
        raise ValueError(f"Mixed resolutions across datasets: {seen_res}; refusing to join.")


def _process_one(
    dataset: LeRobotDataset,
    src: _SourceEpisode,
    args: Args,
) -> str | None:
    """Add a single source episode to ``dataset``.  Returns task string or ``None`` if skipped."""
    ep_dir = src.ep_dir
    joints, task, num_frames_raw = _load_episode_lowdim(ep_dir)

    if not task:
        with open(ep_dir / "metadata.json") as f:
            md = json.load(f)
        prompts = md.get("language", {}).get("prompt", [""])
        task = prompts[0] if prompts else ""

    actions = joints[1:]
    joints = joints[:-1]
    num_frames = num_frames_raw - 1

    image_paths: dict[str, list[Path]] = {}
    for out_cam in CAMERAS:
        src_cam = _OUT_TO_SRC[out_cam]
        img_dir = ep_dir / "rgb" / src_cam
        image_paths[out_cam] = sorted(img_dir.glob("??????????.png"))[:num_frames]

    for out_cam in CAMERAS:
        if not image_paths[out_cam]:
            raise FileNotFoundError(
                f"No PNG frames for camera '{out_cam}' under "
                f"{ep_dir / 'rgb' / _OUT_TO_SRC[out_cam]}. All cameras are required."
            )
    n_trim = min(num_frames, *(len(image_paths[c]) for c in CAMERAS))
    if n_trim < 1:
        print(f"skip {src.source_name}/{src.source_ep_id}: no frames")
        return None

    joints = joints[:n_trim]
    actions = actions[:n_trim]
    num_frames = n_trim
    for out_cam in CAMERAS:
        image_paths[out_cam] = image_paths[out_cam][:num_frames]

    if args.filter_noop:
        keep = _noop_row_keep_mask(joints, actions, args.noop_eps_joint, args.noop_eps_gripper)
        if not np.any(keep):
            print(f"skip {src.source_name}/{src.source_ep_id}: all frames are no-op")
            return None
        idx = np.flatnonzero(keep)
        joints = joints[idx]
        actions = actions[idx]
        for out_cam in CAMERAS:
            image_paths[out_cam] = [image_paths[out_cam][i] for i in idx]
        num_frames = int(joints.shape[0])
        if num_frames < args.noop_min_frames:
            print(
                f"skip {src.source_name}/{src.source_ep_id}: only {num_frames} frames after "
                f"noop filter (min {args.noop_min_frames})"
            )
            return None

    # Pre-load and decode all frames concurrently (I/O + decode bound).
    # Each task returns (frame_index, cam_name, rgb_array).
    def _decode_frame(i_cam):
        i, cam = i_cam
        path = image_paths[cam][i]
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read {path}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if DOWNSAMPLE_FACTOR > 1:
            h, w = img.shape[:2]
            img = cv2.resize(
                img, (w // DOWNSAMPLE_FACTOR, h // DOWNSAMPLE_FACTOR),
                interpolation=cv2.INTER_AREA,
            )
        return i, cam, img

    tasks = [(i, cam) for i in range(num_frames) for cam in CAMERAS]
    frames_buf: dict[tuple[int, str], np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=args.num_image_workers) as pool:
        for i, cam, img in tqdm.tqdm(
            pool.map(_decode_frame, tasks),
            total=len(tasks),
            desc=f"  {src.source_name}/{src.source_ep_id}",
            leave=False,
        ):
            frames_buf[(i, cam)] = img

    for i in range(num_frames):
        frame_dict: dict = {
            "observation.state": joints[i],
            "action": actions[i],
            "task": task,
        }
        for cam in CAMERAS:
            frame_dict[f"observation.images.{cam}"] = frames_buf[(i, cam)]
        dataset.add_frame(frame_dict)
    dataset.save_episode()
    return task


def main(args: Args) -> None:
    sources_all = _load_manifest(args.manifest)
    if args.num_demos_per_dataset is not None:
        # Cap per source dataset while preserving order.
        capped: list[_SourceEpisode] = []
        per_name: dict[str, int] = {}
        for s in sources_all:
            per_name[s.source_name] = per_name.get(s.source_name, 0) + 1
            if per_name[s.source_name] <= args.num_demos_per_dataset:
                capped.append(s)
        sources_all = capped

    _validate_consistency(sources_all)

    n_by_opt = {opt: sum(1 for s in sources_all if s.optimality == opt) for opt in VALID_OPTIMALITIES}
    n_by_src = {
        s: sum(1 for x in sources_all if x.source_name == s)
        for s in sorted({x.source_name for x in sources_all})
    }
    print(f"Manifest: {args.manifest}")
    print(f"  total source episodes : {len(sources_all)}")
    print(f"  by optimality         : {n_by_opt}")
    print(f"  by source dataset     : {n_by_src}")

    output_root = (
        Path(args.lerobot_home) / args.repo_id
        if args.lerobot_home is not None
        else HF_LEROBOT_HOME / args.repo_id
    )
    if output_root.exists():
        shutil.rmtree(output_root)

    features = {
        "observation.state": {"dtype": "float32", "shape": (len(MOTORS),), "names": [MOTORS]},
        "action": {"dtype": "float32", "shape": (len(MOTORS),), "names": [MOTORS]},
    }
    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (3, 720 // DOWNSAMPLE_FACTOR, 1280 // DOWNSAMPLE_FACTOR),
            "names": ["channels", "height", "width"],
        }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=FPS,
        robot_type="yam_bimanual",
        features=features,
        use_videos=True,
        tolerance_s=0.0001,
        image_writer_processes=8,
        image_writer_threads=8,
        root=(Path(args.lerobot_home) / args.repo_id) if args.lerobot_home is not None else None,
    )

    optimality_records: list[dict] = []
    tasks_set: set[str] = set()
    tasks_by_source: dict[str, set[str]] = {}
    skipped = 0
    joint_ep_idx = 0
    for src in tqdm.tqdm(sources_all, desc="Joining datasets"):
        task = _process_one(dataset, src, args)
        if task is None:
            skipped += 1
            continue
        optimality_records.append({
            "episode_index": joint_ep_idx,
            "optimality": src.optimality,
            "source": src.source_name,
            "source_episode": src.source_ep_id,
        })
        joint_ep_idx += 1
        if task:
            tasks_set.add(task)
            tasks_by_source.setdefault(src.source_name, set()).add(task)

    # Surface tasks-per-source so the user notices if datasets have differing prompts.
    multi_task_sources = {k: sorted(v) for k, v in tasks_by_source.items() if len(v) > 1}
    if multi_task_sources:
        print(f"WARNING: multiple task strings within source(s): {multi_task_sources}")
    if len({tuple(sorted(v)) for v in tasks_by_source.values()}) > 1:
        print(
            "WARNING: source datasets emit different task strings; using "
            "override_prompt_from_reward=True at training time is recommended."
        )

    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    opt_path = meta_dir / "optimality.jsonl"
    with open(opt_path, "w") as f:
        for rec in optimality_records:
            json.dump(rec, f)
            f.write("\n")
    print(f"Wrote {len(optimality_records)} entries to {opt_path}")

    tasks_jsonl_path = meta_dir / "tasks.jsonl"
    with open(tasks_jsonl_path, "w") as f:
        for i, task in enumerate(sorted(tasks_set)):
            json.dump({"task_index": i, "task": task}, f)
            f.write("\n")
    print(f"Saved {len(tasks_set)} unique task strings to {tasks_jsonl_path}")

    if skipped:
        print(f"Skipped {skipped} source episode(s) (no-op / empty).")

    print(f"\nDataset saved to {output_root}")
    print(f"  Episodes : {dataset.num_episodes}")
    print(f"  Frames   : {dataset.num_frames}")

    if args.push_to_hub:
        dataset.push_to_hub(private=args.private)
        print(f"Pushed to HuggingFace Hub: {args.repo_id} (private={args.private})")


if __name__ == "__main__":
    main(tyro.cli(Args))
