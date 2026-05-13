"""Sidecar reward loading for weighted behavior cloning.

Reward arrays produced by ``scripts/annotate_rewards.py`` live under
``<lerobot_home or $HF_LEROBOT_HOME>/<repo_id>/meta/rewards/<reward_name>/episode_*.npy``. This
module reads them at training time and exposes them as a per-sample scalar
``weight`` via a :class:`~openpi.transforms.DataTransformFn`, so the rest of
the data pipeline (collation, normalization, sharding) is untouched.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from lerobot.common.constants import HF_LEROBOT_HOME

import openpi.transforms as _transforms


class RewardLookup:
    """Lazy per-episode reward loader with action-chunk mean aggregation."""

    def __init__(
        self,
        repo_id: str,
        reward_name: str,
        action_horizon: int,
        *,
        lerobot_home: str | Path | None = None,
    ) -> None:
        base = Path(lerobot_home) if lerobot_home is not None else Path(HF_LEROBOT_HOME)
        self._dir = base / repo_id / "meta" / "rewards" / reward_name
        if not self._dir.exists():
            raise FileNotFoundError(
                f"Reward directory {self._dir} not found. Run scripts/annotate_rewards.py first."
            )
        self._action_horizon = int(action_horizon)
        self._cache: dict[int, np.ndarray] = {}

    def _episode(self, episode_index: int) -> np.ndarray:
        rewards = self._cache.get(episode_index)
        if rewards is None:
            path = self._dir / f"episode_{episode_index:06d}.npy"
            if not path.exists():
                raise FileNotFoundError(f"Missing reward file: {path}")
            rewards = np.load(path).astype(np.float32)
            self._cache[episode_index] = rewards
        return rewards

    def weight_for(self, episode_index: int, frame_index: int) -> np.float32:
        rewards = self._episode(int(episode_index))
        t = int(frame_index)
        end = min(t + self._action_horizon, rewards.shape[0])
        if end <= t:
            # frame_index at or past the end; replicate-pad on the last frame.
            return np.float32(rewards[-1])
        chunk = rewards[t:end]
        if chunk.shape[0] < self._action_horizon:
            pad = np.full(self._action_horizon - chunk.shape[0], rewards[-1], dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)
        return np.float32(chunk.mean())


@dataclasses.dataclass(frozen=True)
class AddRewardWeight(_transforms.DataTransformFn):
    """Inject a per-sample scalar ``weight`` from a :class:`RewardLookup`."""

    lookup: RewardLookup

    def __call__(self, data):
        ep = int(np.asarray(data["episode_index"]).item())
        frame = int(np.asarray(data["frame_index"]).item())
        return {**data, "weight": self.lookup.weight_for(ep, frame)}
