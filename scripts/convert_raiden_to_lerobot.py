"""
Convert raiden bimanual YAM data to LeRobot dataset v2.0 format for openpi
fine-tuning. Expects 14-D joint vectors per frame (no single-arm / 7-D path).

Reads `rd convert` output directly (data/processed/<task>/<episode>/):
  rgb/scene_camera/0000000000.png       (1280x720 RGB, ZED native)
  rgb/left_wrist_camera/0000000000.png   (required)
  rgb/right_wrist_camera/0000000000.png  (required)
  lowdim/0000000000.pkl                 (joints: 14D bimanual)
  metadata.json                         (num_frames, language.prompt, ...)

Output LeRobot format (always 14-D):
  observation.state: 14D [left_joint(6), left_grip(1), right_joint(6), right_grip(1)]
  action: 14D joints[t+1] in same layout
  observation.images.{head, left_wrist, right_wrist}: H.264 video

By default, **no-op filtering** drops timesteps where the absolute joint and
gripper deltas from state to action are all below small L∞ thresholds (see
``Args.filter_noop``). Episodes with too few frames after filtering are skipped.

Usage:
  uv run raiden/scripts/convert_raiden_to_lerobot.py \
      --processed-dir /home/yam/raiden/data/processed/BimanualCubePickandPlace \
      --repo-id local/raiden_bimanual_cube_pickplace
"""

import dataclasses
import json
import os
import pickle
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import tqdm
import tyro
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


# raiden camera names → pi0.5_yam expected names
CAMERA_NAME_MAP = {
    "scene_camera": "head",
    "left_wrist_camera": "left_wrist",
    "right_wrist_camera": "right_wrist",
}
CAMERAS = ["head", "left_wrist", "right_wrist"]
_OUT_TO_SRC = {v: k for k, v in CAMERA_NAME_MAP.items()}

FPS = 30
JPEG_QUALITY = 95
DOWNSAMPLE_FACTOR = 2  # store at 640x360 for reward annotation; halves storage while keeping enough detail for VLMs

# 14D: [left_joint(6), left_grip(1), right_joint(6), right_grip(1)].
# Matches the bimanual MOTORS list so the existing pi05_yam_* configs and
# OpenPiBridge work unchanged.
MOTORS = [
    "left_joint_0",
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_gripper",
    "right_joint_0",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_gripper",
]


@dataclasses.dataclass(frozen=True)
class Args:
    processed_dir: Path
    """Path to raiden's processed task dir (e.g. data/processed/<task>/)."""
    repo_id: str
    """LeRobot repo ID (e.g. local/raiden_bimanual_cube_pickplace)."""
    push_to_hub: bool = False
    """Push the dataset to HuggingFace Hub after conversion."""
    private: bool = True
    """Make the Hub repo private (default: True)."""
    num_demos: int | None = None
    """Number of episodes to convert (default: all)."""
    filter_noop: bool = True
    """Drop timesteps where pose barely changes (state → next-state)."""
    noop_eps_joint: float = 1e-3
    """Max absolute delta per joint (radians) for a timestep to count as no-op."""
    noop_eps_gripper: float = 1e-3
    """Max absolute delta per gripper dim for a timestep to count as no-op."""
    noop_min_frames: int = 10
    """When ``filter_noop`` is on, skip episode if fewer than this many frames remain."""


def _noop_row_keep_mask(
    joints: np.ndarray,
    actions: np.ndarray,
    eps_joint: float,
    eps_gripper: float,
) -> np.ndarray:
    """Boolean mask (n,) True = keep row; False = drop (static transition)."""
    d = np.abs(actions.astype(np.float64) - joints.astype(np.float64))
    dj_l = np.max(d[:, 0:6], axis=1)
    dg_l = d[:, 6]
    dj_r = np.max(d[:, 7:13], axis=1)
    dg_r = d[:, 13]
    is_noop = (
        (dj_l <= eps_joint)
        & (dg_l <= eps_gripper)
        & (dj_r <= eps_joint)
        & (dg_r <= eps_gripper)
    )
    return ~is_noop


def _load_episode_lowdim(ep_dir: Path) -> tuple[np.ndarray, str, int]:
    """Load every per-frame lowdim pkl. Expects bimanual (14,) joints per frame."""
    lowdim_dir = ep_dir / "lowdim"
    pkl_files = sorted(lowdim_dir.glob("??????????.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No lowdim pkl files in {lowdim_dir}")

    with open(pkl_files[0], "rb") as f:
        first_frame = pickle.load(f)
    first_joints = np.asarray(first_frame["joints"], dtype=np.float32)
    if first_joints.shape != (14,):
        raise ValueError(
            f"Expected bimanual joints shape (14,); got {first_joints.shape} at {pkl_files[0]}"
        )

    n = len(pkl_files)
    joints = np.zeros((n, 14), dtype=np.float32)
    task: str | None = None

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


def _infer_rgb_hw(image_paths: dict[str, list[Path]]) -> tuple[int, int]:
    """Use the first decoded frame for H×W so placeholders match real cameras."""
    for out_cam in CAMERAS:
        paths = image_paths[out_cam]
        if not paths:
            continue
        img = cv2.imread(str(paths[0]))
        if img is not None:
            return int(img.shape[0]), int(img.shape[1])
    return 720 // DOWNSAMPLE_FACTOR, 1280 // DOWNSAMPLE_FACTOR


def _png_to_jpg(src_dir: Path, dst_dir: Path, num_frames: int) -> None:
    """Decode PNGs and re-encode as JPEGs (LeRobot/ffmpeg pipeline expects .jpg)."""
    dst_dir.mkdir(parents=True, exist_ok=True)

    def _convert_one(i: int) -> None:
        src = src_dir / f"{i:010d}.png"
        dst = dst_dir / f"frame_{i:06d}.jpg"
        img = cv2.imread(str(src))
        if img is None:
            raise FileNotFoundError(f"Could not read {src}")
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_convert_one, range(num_frames)))


def _process_episode(
    dataset: LeRobotDataset,
    ep_dir: Path,
    output_dir: Path,
    args: Args,
) -> str | None:
    """Process one episode: load frames and save to LeRobot dataset.

    Returns the task string (possibly empty) for aggregating dataset tasks, or
    ``None`` if the episode is skipped (e.g. too short after no-op filtering).
    """
    joints, task, num_frames_raw = _load_episode_lowdim(ep_dir)
    print(f"convert {ep_dir.name}: bimanual lowdim, {num_frames_raw} raw frames")

    if not task:
        with open(ep_dir / "metadata.json") as f:
            md = json.load(f)
        prompts = md.get("language", {}).get("prompt", [""])
        task = prompts[0] if prompts else ""

    # action[t] = joints[t+1], drop last frame
    actions = joints[1:]
    joints = joints[:-1]
    num_frames = num_frames_raw - 1

    image_paths: dict[str, list[Path]] = {}
    for out_cam in CAMERAS:
        src_cam = _OUT_TO_SRC[out_cam]
        img_dir = ep_dir / "rgb" / src_cam
        image_paths[out_cam] = sorted(img_dir.glob("??????????.png"))[:num_frames]

    for out_cam in CAMERAS:
        n = len(image_paths[out_cam])
        if n == 0:
            raise FileNotFoundError(
                f"No ??????????.png frames for camera '{out_cam}' under {ep_dir / 'rgb' / _OUT_TO_SRC[out_cam]}. "
                "All cameras are required."
            )
    n_trim = min(num_frames, *(len(image_paths[c]) for c in CAMERAS))
    if n_trim < 1:
        raise ValueError(f"No frames to export for episode {ep_dir}")

    joints = joints[:n_trim]
    actions = actions[:n_trim]
    num_frames = n_trim
    for out_cam in CAMERAS:
        image_paths[out_cam] = image_paths[out_cam][:num_frames]

    if args.filter_noop:
        keep = _noop_row_keep_mask(
            joints, actions, args.noop_eps_joint, args.noop_eps_gripper
        )
        if not np.any(keep):
            print(
                f"skip {ep_dir.name}: no-op filter removed all {num_frames} frames "
                f"(eps_joint={args.noop_eps_joint}, eps_gripper={args.noop_eps_gripper})"
            )
            return None
        n_kept = int(np.count_nonzero(keep))
        print(
            f"convert {ep_dir.name}: noop filter {num_frames} -> {n_kept} frames "
            f"(eps_joint={args.noop_eps_joint}, eps_gripper={args.noop_eps_gripper})"
        )
        idx = np.flatnonzero(keep)
        joints = joints[idx]
        actions = actions[idx]
        for out_cam in CAMERAS:
            paths = image_paths[out_cam]
            image_paths[out_cam] = [paths[i] for i in idx if i < len(paths)]
        num_frames = int(joints.shape[0])

        if num_frames < args.noop_min_frames:
            print(
                f"skip {ep_dir.name}: only {num_frames} frames after noop filter "
                f"(min {args.noop_min_frames})"
            )
            return None

    rh, rw = _infer_rgb_hw(image_paths)

    for i in tqdm.tqdm(range(num_frames), desc=f"  {ep_dir.name}", leave=False):
        frame_dict = {
            "observation.state": joints[i],
            "action": actions[i],
            "task": task,
        }

        for out_cam in CAMERAS:
            paths = image_paths[out_cam]
            img_bgr = cv2.imread(str(paths[i]))
            if img_bgr is None:
                raise FileNotFoundError(f"Could not read {paths[i]}")
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            if DOWNSAMPLE_FACTOR > 1:
                h, w = img.shape[:2]
                img = cv2.resize(img, (w // DOWNSAMPLE_FACTOR, h // DOWNSAMPLE_FACTOR),
                                 interpolation=cv2.INTER_AREA)
            frame_dict[f"observation.images.{out_cam}"] = img

        dataset.add_frame(frame_dict)

    dataset.save_episode()
    return task


def main(args: Args):
    processed_dir = args.processed_dir
    repo_id = args.repo_id

    ep_dirs = sorted(
        d for d in processed_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and (d / "metadata.json").exists()
    )
    if not ep_dirs:
        raise SystemExit(f"No episode dirs found under {processed_dir}")
    if args.num_demos is not None:
        ep_dirs = ep_dirs[: args.num_demos]
    print(f"Found {len(ep_dirs)} episodes in {processed_dir} (converting {len(ep_dirs)})")

    output_dir = HF_LEROBOT_HOME / repo_id
    if output_dir.exists():
        shutil.rmtree(output_dir)

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": [MOTORS],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": [MOTORS],
        },
    }
    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (3, 720 // DOWNSAMPLE_FACTOR, 1280 // DOWNSAMPLE_FACTOR),
            "names": ["channels", "height", "width"],
        }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        robot_type="yam_bimanual",
        features=features,
        use_videos=True,
        tolerance_s=0.0001,
        image_writer_processes=2,
        image_writer_threads=2,
    )

    # Collect unique tasks while processing
    tasks_set = set()
    skipped = 0
    for ep_dir in tqdm.tqdm(ep_dirs, desc="Converting episodes"):
        task = _process_episode(dataset, ep_dir, output_dir, args)
        if task is None:
            skipped += 1
        elif task:
            tasks_set.add(task)
    if skipped:
        print(f"Skipped {skipped} episode(s) (no-op / min length).")

    # Save task metadata (required by LeRobot)
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    tasks_jsonl_path = meta_dir / "tasks.jsonl"
    with open(tasks_jsonl_path, "w") as f:
        for i, task in enumerate(sorted(tasks_set)):
            json.dump({"task_index": i, "task": task}, f)
            f.write("\n")
    print(f"Saved {len(tasks_set)} tasks to {tasks_jsonl_path}")

    tmp_frames_root = output_dir / "_tmp_frames"
    if tmp_frames_root.exists():
        shutil.rmtree(tmp_frames_root)

    print(f"Dataset saved to {output_dir}")
    print(f"Total episodes: {dataset.num_episodes}, Total frames: {dataset.num_frames}")

    if args.push_to_hub:
        dataset.push_to_hub(private=args.private)
        print(f"Pushed to HuggingFace Hub: {repo_id} (private={args.private})")


if __name__ == "__main__":
    main(tyro.cli(Args))
