"""Rolling performance statistics for the live monitor and the benchmark.

Metric definitions (all per *processed* frame unless noted):

inference_ms   Model forward pass only, as reported by Ultralytics.
detect_ms      Whole detector call: preprocessing, inference, NMS, tracker update.
pipeline_ms    Detector call + anchor conversion + zone/event evaluation.
latency_ms     From the moment the frame was read from the source until its
               events were produced. For a live camera this includes the time
               the frame waited in the capture buffer.
processed_fps  Frames that went through detection, divided by the wall time
               over a sliding window (not 1 / last frame time).
display_fps    Frames shown in the window, over the same kind of window.

The first ``warmup`` processed frames are ignored, because the first
inferences include lazy initialisation and are not representative.
"""

from __future__ import annotations

import statistics
import time
from collections import deque


class RollingStat:
    def __init__(self, size: int):
        self._values: deque[float] = deque(maxlen=size)

    def add(self, value: float) -> None:
        self._values.append(value)

    def __len__(self) -> int:
        return len(self._values)

    @property
    def mean(self) -> float | None:
        return statistics.fmean(self._values) if self._values else None

    @property
    def median(self) -> float | None:
        return statistics.median(self._values) if self._values else None

    def percentile(self, pct: float, min_samples: int = 20) -> float | None:
        """Nearest-rank percentile; None until there are enough samples to mean anything."""
        if len(self._values) < min_samples:
            return None
        return percentile(list(self._values), pct)


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    rank = max(1, round(pct / 100.0 * len(ordered)))  # nearest-rank method
    return ordered[min(rank, len(ordered)) - 1]


class RateCounter:
    """Events per second over the last ``window`` seconds."""

    def __init__(self, window: float = 2.0, clock=time.perf_counter):
        self.window = window
        self._clock = clock
        self._ticks: deque[float] = deque()

    def tick(self) -> None:
        now = self._clock()
        self._ticks.append(now)
        self._trim(now)

    def rate(self) -> float | None:
        now = self._clock()
        self._trim(now)
        if len(self._ticks) < 2:
            return None
        span = self._ticks[-1] - self._ticks[0]
        return (len(self._ticks) - 1) / span if span > 0 else None

    def _trim(self, now: float) -> None:
        while self._ticks and now - self._ticks[0] > self.window:
            self._ticks.popleft()


class PerformanceMonitor:
    def __init__(self, warmup: int = 10, window_size: int = 120, clock=time.perf_counter):
        self.warmup = warmup
        self.processed = 0
        self.inference_ms = RollingStat(window_size)
        self.detect_ms = RollingStat(window_size)
        self.pipeline_ms = RollingStat(window_size)
        self.latency_ms = RollingStat(window_size)
        self.processed_rate = RateCounter(clock=clock)
        self.display_rate = RateCounter(clock=clock)

    def record_processed(self, *, inference_ms: float | None, detect_ms: float, pipeline_ms: float, latency_ms: float) -> None:
        self.processed += 1
        self.processed_rate.tick()
        if self.processed <= self.warmup:
            return
        if inference_ms is not None:
            self.inference_ms.add(inference_ms)
        self.detect_ms.add(detect_ms)
        self.pipeline_ms.add(pipeline_ms)
        self.latency_ms.add(latency_ms)

    def record_displayed(self) -> None:
        self.display_rate.tick()

    @property
    def warming_up(self) -> bool:
        return self.processed <= self.warmup

    def summary(self) -> dict[str, float | None]:
        return {
            "processed_fps": self.processed_rate.rate(),
            "display_fps": self.display_rate.rate(),
            "inference_ms": self.inference_ms.mean,
            "pipeline_ms": self.pipeline_ms.mean,
            "latency_p50_ms": self.latency_ms.median,
            "latency_p95_ms": self.latency_ms.percentile(95),
        }
