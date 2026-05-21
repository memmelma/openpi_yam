"""Sequential action-chunk broker.

Provides ``StandardBroker``, which issues a new inference call only when the
current chunk is exhausted — identical to the original ActionChunkBroker
behaviour.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import tree
from typing_extensions import override

from openpi_client import base_policy as _base_policy


def _slice_step(chunk_result: Dict, step: int) -> Dict:
    """Return a single-step action dict by indexing the chunk dimension."""
    def slicer(x):
        if isinstance(x, np.ndarray):
            return x[step, ...]
        return x
    return tree.map_structure(slicer, chunk_result)


class StandardBroker(_base_policy.BasePolicy):
    """Sequential action-chunk broker — identical to the original ActionChunkBroker.

    A new inference call is made only when the current chunk is exhausted.
    """

    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int) -> None:
        self._policy = policy
        self._action_horizon = action_horizon
        self._last_results: Dict | None = None
        self._cur_step: int = 0

    @override
    def infer(self, obs: Dict) -> Dict:
        if self._last_results is None:
            self._last_results = self._policy.infer(obs)
            self._cur_step = 0

        result = _slice_step(self._last_results, self._cur_step)
        self._cur_step += 1

        if self._cur_step >= self._action_horizon:
            self._last_results = None

        return result

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0
