"""Reward annotation for a single joint LeRobot dataset.

This is the joint-dataset counterpart of ``scripts/annotate_rewards.py``.
It operates on a single repo (typically produced by
``convert_raiden_to_lerobot_joint.py``), reads the per-episode success label
from the episode metadata sidecar, and writes per-episode sidecars
under ``meta/rewards/<prefix>_*``.

Supported reward sources
------------------------
- ``rvlm``                  : async RVLM, identical to single-dataset script
- ``topreward``             : per-timestep linspace(0:t, min(8,t+1))
- ``rbm`` / ``rbm_libero``  : per-timestep linspace(0:t, min(8,t+1))
- ``success``               : ``r[t] = 0.0`` for failure episodes else ``1.0``
- ``stub``                  : deterministic linspace(0, 1, T)

Outputs (under ``$HF_LEROBOT_HOME/<repo_id>/meta/rewards/``):

    <prefix>_reward/             (only rewritten with --overwrite)
    <prefix>_returns/            (always recomputed)
    <prefix>_advantage/          (always recomputed; episode-mean baseline subtracted)
    <prefix>_awr_weights/        (always recomputed: exp(clip(adv/beta, ...)))
    <prefix>_delta/              (always recomputed)
    <prefix>_delta_returns/      (always recomputed)
    <prefix>_delta_advantage/    (always recomputed)
    <prefix>_delta_awr_weights/  (always recomputed)
    <prefix>_reward/config.json  (includes reference_instruction, gamma, beta)

The AWR weight sidecars are convenience artifacts for visualization /
debugging.  Training computes AWR weights on the fly so ``beta`` can be
changed without re-annotating.

Usage:
    uv run scripts/annotate_rewards_joint.py \\
        --repo-id memmelma/swb_joint \\
        --reward-model rvlm \\
        --reference-instruction "move the star wars book ..."
"""

from __future__ import annotations

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

# Ensure sibling scripts are importable when invoked from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse self-contained helpers from the existing per-repo annotator.
from annotate_rewards import (  # noqa: E402
    _annotate_repo_async,
    _annotate_repo_sync,
    _load_jsonl,
    compute_and_write_advantages,
    compute_and_write_delta_rewards,
    compute_and_write_returns,
    create_model,
    normalize_instruction_for_key,
    parse_model_config,
)

SUPPORTED = ("rvlm", "topreward", "rbm", "rbm_libero", "success", "stub")


def _gamma_tag(gamma: float) -> str:
    """Format gamma as a compact directory-safe tag, e.g. ``g099`` for 0.99.

    Only gamma affects advantage/returns values; beta is applied on the fly at
    training time via ``DataConfig.reward_beta`` and does not need to be encoded
    in the sidecar name.
    """
    return "g" + f"{gamma:.2f}".replace(".", "")


# ---------------------------------------------------------------------------
# Success-based reward
# ---------------------------------------------------------------------------

def _write_success_rewards(
    dataset_root: Path,
    prefix: str,
    overwrite: bool,
) -> int:
    """Write per-episode reward arrays where failure→0, else→1 (constant per ep)."""
    meta_dir = dataset_root / "meta"
    opt_path = meta_dir / "optimality.jsonl"
    if not opt_path.exists():
        raise FileNotFoundError(
            f"--reward-model success requires {opt_path}. Re-run convert_raiden_to_lerobot_joint.py."
        )
    optimality_lookup = {r["episode_index"]: r["optimality"] for r in _load_jsonl(opt_path)}

    episodes_meta = _load_jsonl(meta_dir / "episodes.jsonl")
    reward_dir = meta_dir / "rewards" / f"{prefix}_reward"
    reward_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for ep in tqdm.tqdm(episodes_meta, desc="Success rewards"):
        ep_idx = int(ep["episode_index"])
        out_path = reward_dir / f"episode_{ep_idx:06d}.npy"
        if out_path.exists() and not overwrite:
            continue
        opt = optimality_lookup.get(ep_idx)
        if opt is None:
            raise KeyError(f"episode_index={ep_idx} missing from optimality.jsonl")
        value = 0.0 if opt == "failure" else 1.0
        rewards = np.full(int(ep["length"]), value, dtype=np.float32)
        np.save(out_path, rewards)
        written += 1
    return written



# ---------------------------------------------------------------------------
# AWR weights (saved as convenience sidecar; training recomputes on the fly)
# ---------------------------------------------------------------------------

def compute_and_write_awr_weights(
    advantage_dir: Path,
    weights_dir: Path,
    beta: float,
    clip: float = 100.0,
) -> None:
    """Write exp(clip(adv / beta, -clip, clip)) per episode."""
    files = sorted(advantage_dir.glob("episode_*.npy"))
    if not files:
        return
    weights_dir.mkdir(parents=True, exist_ok=True)
    for f in tqdm.tqdm(files, desc=f"AWR weights [{advantage_dir.name}]"):
        adv = np.load(f).astype(np.float32)
        w = np.exp(np.clip(adv / beta, -clip, clip)).astype(np.float32)
        np.save(weights_dir / f.name, w)


# ---------------------------------------------------------------------------
# Episode-mean baseline (intra-dataset; we only have one joint repo here)
# ---------------------------------------------------------------------------

def _compute_intra_advantages(returns_dir: Path, advantage_dir: Path) -> None:
    """Subtract the global mean return across all episodes in the single repo."""
    files = sorted(returns_dir.glob("episode_*.npy"))
    if not files:
        return
    all_returns = [np.load(f) for f in files]
    global_mean = float(np.concatenate(all_returns).mean())
    print(f"  global mean return: {global_mean:.4f} over {len(files)} episodes")
    advantage_dir.mkdir(parents=True, exist_ok=True)
    for f, r in zip(files, all_returns):
        np.save(advantage_dir / f.name, (r.astype(np.float32) - global_mean).astype(np.float32))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Args:
    repo_id: str
    """LeRobot repo id of the joint dataset to annotate."""

    reward_model: Literal[
        "rvlm", "topreward", "rbm", "rbm_libero", "success", "stub"
    ] = "rvlm"
    """Reward source.  ``success`` writes 0/1 from meta/optimality.jsonl (0 for failure, 1 for suboptimal/optimal)."""

    reference_instruction: str | None = None
    """Language instruction for the reward model.  Saved in config.json so
    training can override prompts via override_prompt_from_reward.  Required
    for model-based reward sources (rvlm, topreward, rbm[_libero])."""
    reward_name: str | None = None
    """Override sidecar prefix (default: <norm_instruction>_<reward_model>).
    Required when --reward-model=success (no instruction needed)."""

    model_path: str | None = None
    """HF / local checkpoint for rbm / rbm_libero."""
    model_config: tuple[str, ...] = ()
    """Extra key=value kwargs forwarded to the model constructor."""
    max_frames: int = 16
    """Max frames forwarded to model constructors that accept it."""

    camera: str = "head"
    """Camera name (under observation.images.<camera>) whose video is decoded."""
    episodes: tuple[int, ...] | None = None
    """Optional subset of episode indices to annotate."""

    subsample_frames: int = 8
    """Min frames for rvlm subsampling; set 0 to disable the min."""
    subsample_factor: int = 0
    """If > 0, rvlm subsample_n = traj_len // factor (clamped to >= 8)."""

    overwrite: bool = False
    """Re-run reward inference even if reward sidecars exist.  Returns /
    advantages / deltas / AWR weights are ALWAYS recomputed every run."""
    stub: bool = False
    """Replace reward inference with a deterministic linspace(0, 1, T)."""

    gamma: float = 0.99
    """Discount factor for MC returns."""
    beta: float = 2.0
    """AWR temperature: w = exp(adv / beta)."""

    compute_delta: bool = True
    """Also compute delta rewards (r[t+1]-r[t]) + their returns/advantages/weights."""

    concurrency: int = 16
    """Async concurrency for RVLM."""

    lerobot_home: str | None = None
    """Optional override for dataset root (defaults to $HF_LEROBOT_HOME)."""

    push_to_hub: bool = False
    """Upload all <prefix>_* directories to the Hub after writing."""
    private: bool = True
    """If pushing, mark hub repo as private."""


def _resolve_prefix(args: Args) -> str:
    if args.reward_name is not None:
        return args.reward_name
    if args.reward_model == "success":
        return "success"
    if args.reference_instruction is not None:
        return f"{normalize_instruction_for_key(args.reference_instruction)}_{args.reward_model}"
    raise SystemExit(
        "Either --reference-instruction or --reward-name must be provided "
        f"(or use --reward-model success).  Got reward_model={args.reward_model!r}."
    )


def main(args: Args) -> None:
    prefix = _resolve_prefix(args)
    reference_instruction = args.reference_instruction or ""
    base = Path(args.lerobot_home) if args.lerobot_home is not None else Path(HF_LEROBOT_HOME)
    root = base / args.repo_id
    if not root.exists():
        raise FileNotFoundError(f"Joint dataset not found at {root}.")

    if not (root / "meta" / "optimality.jsonl").exists():
        print(
            f"WARNING: {root / 'meta' / 'optimality.jsonl'} not found.  "
            "This script targets the joint dataset format produced by "
            "convert_raiden_to_lerobot_joint.py."
        )

    print(f"\n{'=' * 60}")
    print(f"Repo         : {args.repo_id}")
    print(f"Reward model : {args.reward_model}")
    print(f"Instruction  : {reference_instruction!r}")
    print(f"Prefix       : {prefix}")
    print(f"Gamma        : {args.gamma}    Beta: {args.beta}")
    print(f"{'=' * 60}")

    reward_dir = root / "meta" / "rewards" / f"{prefix}_reward"
    rewards_exist = reward_dir.exists() and any(reward_dir.glob("episode_*.npy"))

    # ---------------- Phase 1: reward inference ----------------
    if rewards_exist and not args.overwrite:
        print(f"\nRewards exist at {reward_dir.name}/ (use --overwrite to force).")
    elif args.reward_model == "success":
        n = _write_success_rewards(root, prefix, overwrite=args.overwrite)
        print(f"  wrote {n} success reward file(s)")
    else:
        # Patch annotate_rewards._N_FRAMES_PER_CALL behaviour for topreward/rbm:
        # the upstream helper uses a fixed 8-frame linspace prefix; we want
        # min(8, t+1) so very short prefixes still work.  Monkey-patch the
        # constant via a thin wrapper rather than copying ~80 lines.
        import annotate_rewards as _ar

        # Wrap compute_and_interpolate to use min(8, t+1) for topreward/rbm.
        _orig_compute = _ar.compute_and_interpolate

        def _patched_compute(model, frames, task, subsample_n, reward_model):
            if reward_model in ("rbm", "rewind", "rbm_libero", "topreward"):
                total = len(frames)
                _, query_indices = _ar.linspace_subsample_frames(frames, subsample_n)
                per_query = []
                for t_i in query_indices:
                    n_ctx = max(1, min(8, int(t_i) + 1))
                    prefix_ctx, _ = _ar.linspace_subsample_frames(frames[: int(t_i) + 1], n_ctx)
                    result = model.compute_progress(prefix_ctx, task_description=task)
                    per_query.append(float(result[-1]))
                progress = np.array(per_query, dtype=np.float32)
                if len(query_indices) < total:
                    progress = np.interp(np.arange(total), query_indices, progress).astype(np.float32)
                return progress
            return _orig_compute(model, frames, task, subsample_n, reward_model)

        _ar.compute_and_interpolate = _patched_compute

        # Force rvlm subsample_n >= 8.
        _orig_resolve = _ar._resolve_subsample_n

        def _patched_resolve(subsample_frames, subsample_factor, traj_len):
            n = _orig_resolve(subsample_frames, subsample_factor, traj_len)
            return max(8, n) if traj_len >= 8 else max(1, min(n, traj_len))

        _ar._resolve_subsample_n = _patched_resolve

        model_cfg = parse_model_config(args.model_config)
        model = None
        if not args.stub:
            model = create_model(
                args.reward_model,
                model_path=args.model_path,
                max_frames=args.max_frames,
                **model_cfg,
            )

        annotate_fn = (
            _annotate_repo_async
            if (args.reward_model == "rvlm" and not args.stub)
            else _annotate_repo_sync
        )
        if annotate_fn is _annotate_repo_async:
            written = _annotate_repo_async(root, args, prefix, reference_instruction)
        else:
            written = _annotate_repo_sync(root, model, args, prefix, reference_instruction)
        print(f"  wrote {len(written)} reward file(s)")

    # ---------------- Phase 2: derived signals (ALWAYS recomputed) ----------------
    # Gamma is encoded in all derived directory names so that different gamma
    # values produce distinct sidecar trees while sharing a single (expensive)
    # reward directory.  Beta is NOT encoded here because advantage values are
    # independent of beta; beta is applied on the fly at training time via
    # DataConfig.reward_beta.
    gtag = _gamma_tag(args.gamma)
    print(f"\nRecomputing derived signals (gamma_tag={gtag})...")
    returns_dir = root / "meta" / "rewards" / f"{prefix}_{gtag}_returns"
    advantage_dir = root / "meta" / "rewards" / f"{prefix}_{gtag}_advantage"
    awr_dir = root / "meta" / "rewards" / f"{prefix}_{gtag}_awr_weights"
    print(f"  returns    -> {returns_dir.name}/")
    compute_and_write_returns(reward_dir, returns_dir, gamma=args.gamma)
    print(f"  advantage  -> {advantage_dir.name}/")
    _compute_intra_advantages(returns_dir, advantage_dir)
    print(f"  awr_w      -> {awr_dir.name}/")
    compute_and_write_awr_weights(advantage_dir, awr_dir, beta=args.beta)

    if args.compute_delta:
        delta_dir = root / "meta" / "rewards" / f"{prefix}_{gtag}_delta"
        delta_returns_dir = root / "meta" / "rewards" / f"{prefix}_{gtag}_delta_returns"
        delta_adv_dir = root / "meta" / "rewards" / f"{prefix}_{gtag}_delta_advantage"
        delta_awr_dir = root / "meta" / "rewards" / f"{prefix}_{gtag}_delta_awr_weights"
        print(f"  delta      -> {delta_dir.name}/")
        compute_and_write_delta_rewards(reward_dir, delta_dir)
        print(f"  d_returns  -> {delta_returns_dir.name}/")
        compute_and_write_returns(delta_dir, delta_returns_dir, gamma=args.gamma)
        print(f"  d_adv      -> {delta_adv_dir.name}/")
        _compute_intra_advantages(delta_returns_dir, delta_adv_dir)
        print(f"  d_awr_w    -> {delta_awr_dir.name}/")
        compute_and_write_awr_weights(delta_adv_dir, delta_awr_dir, beta=args.beta)

    # ---------------- Phase 3: config.json ----------------
    config_payload = {
        "prefix": prefix,
        "reward_model": args.reward_model,
        "reference_instruction": reference_instruction,
        "camera": args.camera,
        "gamma": args.gamma,
        "beta": args.beta,
        "subsample_frames": args.subsample_frames,
        "subsample_factor": args.subsample_factor,
        "joint_dataset": True,
        "stub": args.stub,
        "created": _dt.datetime.utcnow().isoformat() + "Z",
    }
    (reward_dir / "config.json").write_text(json.dumps(config_payload, indent=2))
    print(f"\nWrote config to {reward_dir / 'config.json'}")

    # ---------------- Phase 4: push ----------------
    if args.push_to_hub:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
        rewards_root = root / "meta" / "rewards"
        for d in sorted(rewards_root.glob(f"{prefix}_*")):
            if not d.is_dir():
                continue
            api.upload_folder(
                repo_id=args.repo_id,
                repo_type="dataset",
                folder_path=str(d),
                path_in_repo=f"meta/rewards/{d.name}",
            )
            print(f"  pushed {d.name}")

    print("\nDone.")


if __name__ == "__main__":
    main(tyro.cli(Args))
