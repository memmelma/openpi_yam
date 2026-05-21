"""Sidecar reward loading for weighted / filtered behavior cloning.

Reward arrays produced by ``scripts/annotate_rewards{,_joint}.py`` live under
``<lerobot_home or $HF_LEROBOT_HOME>/<repo_id>/meta/rewards/<reward_name>/episode_*.npy``.
This module reads them at training time and exposes:

- a per-sample scalar ``weight`` via a :class:`~openpi.transforms.DataTransformFn`
  (transform selected by ``weight_scheme``; see :class:`RewardLookup`);
- an optional :class:`FilteredLeRobotDataset` that drops (episode, frame) pairs
  whose weight falls below a global quantile / absolute cutoff, so a training
  batch never contains a zero-weight sample;
- an optional :class:`OverrideTaskPrompt` transform that replaces every
  sample's ``prompt`` with the reward annotation's reference instruction
  (read from ``<reward_name>/config.json``).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from pathlib import Path

import numpy as np
import torch.utils.data
from lerobot.common.constants import HF_LEROBOT_HOME

import openpi.transforms as _transforms

logger = logging.getLogger(__name__)

_VALID_SCHEMES = {"default", "awr", "chunk", "awr_chunk"}

# ---------------------------------------------------------------------------
# RewardLookup: per-episode reward / advantage loader with weight transform
# ---------------------------------------------------------------------------


class RewardLookup:
    """Lazy per-episode reward loader.

    Reads ``meta/rewards/<reward_name>/episode_*.npy`` and returns a scalar
    per-sample weight according to ``weight_scheme``:

    - ``"default"``   – stored value used directly as weight (no transform).
    - ``"awr"``       – AWR: ``exp(value / beta)``.  ``reward_name`` should
                        point at an advantage directory (e.g. ``..._delta_advantage``).
    - ``"chunk"``     – on-the-fly advantage ``V(s_{t+N}) - V(s_t)`` where
                        N = action_horizon.  ``reward_name`` must point at a
                        *value* directory.
    - ``"awr_chunk"`` – chunk advantage then ``exp(adv / beta)``.
    """

    def __init__(
        self,
        repo_id: str,
        reward_name: str,
        action_horizon: int,
        *,
        lerobot_home: str | Path | None = None,
        beta: float = 2.0,
        weight_scheme: str = "awr",
        weight_clip: float = 100.0,
    ) -> None:
        if weight_scheme not in _VALID_SCHEMES:
            raise ValueError(f"weight_scheme must be one of {_VALID_SCHEMES}, got {weight_scheme!r}")
        base = Path(lerobot_home) if lerobot_home is not None else Path(HF_LEROBOT_HOME)
        self._dir = base / repo_id / "meta" / "rewards" / reward_name
        if not self._dir.exists():
            raise FileNotFoundError(
                f"Reward directory {self._dir} not found. "
                f"Run scripts/annotate_rewards_joint.py first."
            )
        self._action_horizon = int(action_horizon)
        self._beta = float(beta)
        self._scheme = weight_scheme
        self._clip = float(weight_clip)
        self._cache: dict[int, np.ndarray] = {}

        # The reference instruction (used by OverrideTaskPrompt) is written by
        # the annotation script next to the <prefix>_reward/ dir, not next to
        # advantage/awr dirs.  Try both: same dir and its sibling whose name
        # ends in ``_reward``.
        self._reference_instruction: str | None = None
        for cfg_path in (self._dir / "config.json", self._reward_sibling_config()):
            if cfg_path is not None and cfg_path.exists():
                try:
                    cfg = json.loads(cfg_path.read_text())
                    self._reference_instruction = cfg.get("reference_instruction") or None
                    break
                except Exception as e:  # noqa: BLE001
                    logger.warning("Could not parse %s: %s", cfg_path, e)

    def _reward_sibling_config(self) -> Path | None:
        """Find the ``<prefix>_reward/config.json`` next to ``<prefix>_*`` dirs.

        Handles the optional ``_g{gamma}_b{beta}`` tag inserted by
        ``annotate_rewards_joint.py``, e.g.
        ``<prefix>_g099_b20_delta_advantage`` → ``<prefix>_reward/config.json``.
        """
        name = self._dir.name
        if name.endswith("_reward"):
            return None  # already the canonical config location
        # Strip common suffixes (advantage, returns, delta, awr_weights, etc).
        base = name
        for suf in (
            "_delta_awr_weights", "_delta_advantage", "_delta_returns", "_delta",
            "_awr_weights", "_advantage", "_returns",
        ):
            if base.endswith(suf):
                base = base[: -len(suf)]
                break
        # Strip optional _g<digits> gamma tag added by annotate_rewards_joint.py.
        base = re.sub(r"_g\d+$", "", base)
        if base == name:
            return None  # no known suffix was stripped
        return self._dir.parent / f"{base}_reward" / "config.json"

    @property
    def reference_instruction(self) -> str | None:
        return self._reference_instruction

    @property
    def reward_dir(self) -> Path:
        return self._dir

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def weight_scheme(self) -> str:
        return self._scheme

    def _episode(self, episode_index: int) -> np.ndarray:
        arr = self._cache.get(episode_index)
        if arr is None:
            path = self._dir / f"episode_{episode_index:06d}.npy"
            if not path.exists():
                raise FileNotFoundError(f"Missing reward file: {path}")
            arr = np.load(path).astype(np.float32)
            self._cache[episode_index] = arr
        return arr

    def _chunk_mean(self, episode_index: int, frame_index: int) -> np.float32:
        arr = self._episode(int(episode_index))
        t = int(frame_index)
        end = min(t + self._action_horizon, arr.shape[0])
        if end <= t:
            return np.float32(arr[-1])
        chunk = arr[t:end]
        if chunk.shape[0] < self._action_horizon:
            pad = np.full(self._action_horizon - chunk.shape[0], arr[-1], dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)
        return np.float32(chunk.mean())

    def _chunk_advantage(self, episode_index: int, frame_index: int) -> np.float32:
        """Compute V(s_{t+N}) - V(s_t) where N = action_horizon.

        At episode boundaries t+N is clamped to the last frame index.
        Stored data must represent value function estimates V.
        """
        arr = self._episode(int(episode_index))
        t = int(frame_index)
        t_next = min(t + self._action_horizon, arr.shape[0] - 1)
        return np.float32(arr[t_next] - arr[t])

    def _transform_weight(self, raw: np.ndarray | float) -> np.ndarray | np.float32:
        if self._scheme in ("awr", "awr_chunk"):
            return np.exp(np.clip(np.asarray(raw) / self._beta, -self._clip, self._clip))
        return np.asarray(raw, dtype=np.float32)

    def weight_for(self, episode_index: int, frame_index: int) -> np.float32:
        if self._scheme in ("chunk", "awr_chunk"):
            raw = self._chunk_advantage(episode_index, frame_index)
        else:
            raw = self._chunk_mean(episode_index, frame_index)
        return np.float32(self._transform_weight(raw))

    def all_per_frame_weights(self) -> dict[int, np.ndarray]:
        """Return ``{episode_index: weight_per_frame}`` matching ``weight_for``.

        Used by :class:`FilteredLeRobotDataset` so the prefilter thresholds and
        the per-sample weights are computed identically.
        """
        out: dict[int, np.ndarray] = {}
        for path in sorted(self._dir.glob("episode_*.npy")):
            ep_idx = int(path.stem.split("_")[-1])
            arr = self._episode(ep_idx)
            n = arr.shape[0]
            if n == 0:
                out[ep_idx] = np.zeros(0, dtype=np.float32)
                continue
            H = self._action_horizon
            if self._scheme in ("chunk", "awr_chunk"):
                # V(s_{min(t+H, T-1)}) - V(s_t) for every t in [0..n-1]
                t_next = np.minimum(np.arange(n) + H, n - 1)
                raw = (arr[t_next] - arr).astype(np.float32)
            else:
                # Action-chunk mean for every t with replicate-pad at the tail.
                padded = np.concatenate([arr, np.full(max(0, H - 1), arr[-1], dtype=np.float32)])
                # csum trick: mean = (csum[t+H] - csum[t]) / H
                csum = np.concatenate([[0.0], np.cumsum(padded.astype(np.float64))])
                raw = ((csum[H:n + H] - csum[:n]) / H).astype(np.float32)
            out[ep_idx] = np.asarray(self._transform_weight(raw), dtype=np.float32)
        return out


@dataclasses.dataclass(frozen=True)
class AddRewardWeight(_transforms.DataTransformFn):
    """Inject a per-sample scalar ``weight`` from a :class:`RewardLookup`."""

    lookup: RewardLookup

    def __call__(self, data):
        ep = int(np.asarray(data["episode_index"]).item())
        frame = int(np.asarray(data["frame_index"]).item())
        return {**data, "weight": self.lookup.weight_for(ep, frame)}


@dataclasses.dataclass(frozen=True)
class OverrideTaskPrompt(_transforms.DataTransformFn):
    """Replace ``data['prompt']`` with the reward annotation's reference instruction.

    Distinguished from :class:`openpi.transforms.InjectDefaultPrompt` which
    only injects when no prompt is present; here we always overwrite.
    """

    prompt: str

    def __call__(self, data):
        return {**data, "prompt": np.asarray(self.prompt)}


# ---------------------------------------------------------------------------
# FilteredLeRobotDataset: drops (episode, frame) pairs by weight threshold
# ---------------------------------------------------------------------------


class FilteredLeRobotDataset(torch.utils.data.Dataset):
    """Wrap a LeRobotDataset to expose only frames whose weight passes the
    quantile / cutoff threshold.

    The underlying LeRobotDataset is indexed by a single global frame index
    (``episode_data_index['from'][ep] + frame``).  We materialise the list of
    valid global indices up-front from the lookup's per-frame weights.
    """

    def __init__(
        self,
        dataset,
        lookup: RewardLookup,
        *,
        weight_quantile: float | None = None,
        weight_cutoff: float | None = None,
    ) -> None:
        self._dataset = dataset
        self._lookup = lookup

        per_ep_weights = lookup.all_per_frame_weights()
        if not per_ep_weights:
            raise RuntimeError(f"No reward files found in {lookup.reward_dir}")

        ep_from = self._episode_starts(dataset)

        # Build the threshold using ALL per-frame weights so the quantile is global.
        all_w = np.concatenate(list(per_ep_weights.values())) if per_ep_weights else np.zeros(0)
        thresholds = []
        if weight_quantile is not None and all_w.size:
            thresholds.append(float(np.quantile(all_w, float(weight_quantile))))
        if weight_cutoff is not None:
            thresholds.append(float(weight_cutoff))
        self._threshold: float = max(thresholds) if thresholds else float("-inf")

        valid_global: list[int] = []
        kept = total = 0
        for ep_idx, w in per_ep_weights.items():
            if ep_idx not in ep_from:
                logger.warning(
                    "Reward file for episode %d has no matching episode start; skipping.", ep_idx
                )
                continue
            n = w.shape[0]
            mask = w >= self._threshold if np.isfinite(self._threshold) else np.ones(n, dtype=bool)
            local_idx = np.flatnonzero(mask)
            valid_global.extend((ep_from[ep_idx] + int(t)) for t in local_idx)
            kept += int(local_idx.size)
            total += int(n)

        self._valid_indices = np.asarray(valid_global, dtype=np.int64)
        if self._valid_indices.size == 0:
            raise RuntimeError(
                f"FilteredLeRobotDataset is empty after applying weight_quantile="
                f"{weight_quantile}, weight_cutoff={weight_cutoff} "
                f"(threshold={self._threshold}). Loosen the filter."
            )
        logger.info(
            "FilteredLeRobotDataset: kept %d / %d frames (%.1f%%), threshold=%.4g",
            kept, total, 100 * kept / max(total, 1), self._threshold,
        )

    @staticmethod
    def _episode_starts(dataset) -> dict[int, int]:
        """Return ``{episode_index: first_global_frame_index}``."""
        # LeRobotDataset exposes ``episode_data_index`` mapping episode -> [from, to).
        edi = getattr(dataset, "episode_data_index", None)
        if edi is None:
            # Try via dataset.meta for older lerobot APIs.
            edi = getattr(getattr(dataset, "meta", None), "episode_data_index", None)
        if edi is None:
            raise AttributeError(
                "Underlying dataset has no episode_data_index; cannot map (ep, frame) "
                "to global index. Is this a LeRobotDataset?"
            )
        # edi['from'] may be a torch tensor or numpy array.
        from_arr = edi["from"]
        try:
            from_arr = from_arr.cpu().numpy()  # type: ignore[union-attr]
        except AttributeError:
            from_arr = np.asarray(from_arr)
        return {int(i): int(v) for i, v in enumerate(from_arr)}

    def __len__(self) -> int:
        return int(self._valid_indices.size)

    def __getitem__(self, index):
        global_idx = int(self._valid_indices[int(index)])
        return self._dataset[global_idx]

    # Forward common attributes so downstream transforms still work.
    def __getattr__(self, name):
        # Use object.__getattribute__ to avoid recursive __getattr__ calls
        # if _dataset itself isn't set yet.
        try:
            dataset = object.__getattribute__(self, "_dataset")
        except AttributeError:
            raise AttributeError(name)
        return getattr(dataset, name)


# ---------------------------------------------------------------------------
# wrap_lerobot_dataset: single entry point used by data_loader.py
# ---------------------------------------------------------------------------


def wrap_lerobot_dataset(dataset, data_config, action_horizon: int, *, repo_id: str):
    """Apply reward-weighted / filtered / prompt-overridden wrappers in order.

    Returns the wrapped dataset.  When ``data_config.reward_name`` is ``None``
    the dataset is returned unchanged, preserving full backward compatibility
    with existing configs.
    """
    if data_config.reward_name is None:
        return dataset

    # Local import to avoid a circular import (data_loader -> rewards -> data_loader).
    from openpi.training.data_loader import TransformedDataset

    lookup = RewardLookup(
        repo_id=repo_id,
        reward_name=data_config.reward_name,
        action_horizon=action_horizon,
        lerobot_home=data_config.lerobot_home,
        beta=getattr(data_config, "reward_beta", 2.0),
        weight_scheme=getattr(data_config, "weight_scheme", "awr"),
    )

    weight_quantile = getattr(data_config, "weight_quantile", None)
    weight_cutoff = getattr(data_config, "weight_cutoff", None)
    if weight_quantile is not None or weight_cutoff is not None:
        dataset = FilteredLeRobotDataset(
            dataset,
            lookup,
            weight_quantile=weight_quantile,
            weight_cutoff=weight_cutoff,
        )

    transforms: list[_transforms.DataTransformFn] = [AddRewardWeight(lookup=lookup)]
    if getattr(data_config, "override_prompt_from_reward", False):
        if not lookup.reference_instruction:
            raise ValueError(
                f"override_prompt_from_reward=True but reward_name={data_config.reward_name!r} "
                f"has no reference_instruction in config.json. Re-run annotate_rewards_joint.py "
                f"with --reference-instruction."
            )
        transforms.append(OverrideTaskPrompt(prompt=lookup.reference_instruction))

    return TransformedDataset(dataset, transforms)
