from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RTCConfig:
    enabled: bool = True
    lookahead_ratio: float = 0.4
    blend_steps: int = 5
    action_ema_alpha: float = 0.4
    safety_margin_steps: int = 2
    latency_window: int = 5


class ActionChunkScheduler:
    def __init__(self, config: RTCConfig) -> None:
        self.config = config
        self._queue: deque[np.ndarray] = deque()
        self._last_action: np.ndarray | None = None
        self._latencies: deque[float] = deque(maxlen=max(1, config.latency_window))

    @property
    def remaining(self) -> int:
        return len(self._queue)

    def should_request_next(self, chunk_size: int) -> bool:
        if not self.config.enabled:
            return self.remaining == 0
        threshold = max(
            self.config.safety_margin_steps,
            int(round(chunk_size * self.config.lookahead_ratio)),
        )
        return self.remaining <= threshold

    def note_request_start(self) -> float:
        return perf_counter()

    def note_request_done(self, start_time: float) -> None:
        self._latencies.append(perf_counter() - start_time)

    def push_chunk(self, actions: Iterable[Iterable[float]] | np.ndarray) -> None:
        chunk = np.asarray(actions, dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)
        if self._queue and self.config.blend_steps > 0:
            self._blend_into_queue(chunk)
            return
        for action in chunk:
            self._queue.append(action)

    def pop_action(self) -> np.ndarray | None:
        if not self._queue:
            return None
        action = self._queue.popleft()
        alpha = self.config.action_ema_alpha
        if self._last_action is not None and alpha < 1:
            action = alpha * action + (1 - alpha) * self._last_action
        self._last_action = action
        return action

    def _blend_into_queue(self, new_chunk: np.ndarray) -> None:
        old = list(self._queue)
        self._queue.clear()
        blend_steps = min(self.config.blend_steps, len(old), len(new_chunk))
        for idx in range(blend_steps):
            weight = (idx + 1) / (blend_steps + 1)
            self._queue.append((1 - weight) * old[idx] + weight * new_chunk[idx])
        for action in new_chunk[blend_steps:]:
            self._queue.append(action)

