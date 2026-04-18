"""The processing core: frame -> detections -> tracked observations -> events.

This module has no UI code, so the live app, headless runs, the benchmark and
the tests all drive exactly the same logic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import numpy as np

from .detector import Detection, DetectionResult
from .event_manager import Event, EventEngine
from .tracker import TrackObservation
from .video_source import Frame
from .zone_manager import Point


class SupportsDetect(Protocol):
    def detect(self, image: np.ndarray, track: bool = True) -> DetectionResult: ...


@dataclass(frozen=True)
class FrameResult:
    frame: Frame
    detections: list[Detection]
    events: list[Event]
    inference_ms: float | None
    detect_ms: float
    pipeline_ms: float
    latency_ms: float


def anchor_point(bbox: tuple[float, float, float, float], width: int, height: int, mode: str) -> Point:
    """The normalized point of a box that is tested against zones.

    ``bottom_center`` approximates where a person's feet touch the floor, which
    is what matters for floor-marked zones seen from an elevated camera; the box
    centre can sit inside a zone while the person is standing in front of it.
    """
    x1, y1, x2, y2 = bbox
    x = (x1 + x2) / 2.0
    y = y2 if mode == "bottom_center" else (y1 + y2) / 2.0
    return (min(max(x / width, 0.0), 1.0), min(max(y / height, 0.0), 1.0))


class MonitorPipeline:
    def __init__(self, detector: SupportsDetect, engine: EventEngine, anchor: str = "bottom_center"):
        self.detector = detector
        self.engine = engine
        self.anchor = anchor

    def process(self, frame: Frame, wall: datetime | None = None) -> FrameResult:
        start = time.perf_counter()
        result = self.detector.detect(frame.image)
        height, width = frame.image.shape[:2]

        # Only tracked detections drive events: without an ID there is no way
        # to tell a new arrival from the same person seen again.
        observations = [
            TrackObservation(d.track_id, d.class_name, d.confidence, anchor_point(d.bbox, width, height, self.anchor))
            for d in result.detections if d.track_id is not None
        ]
        events = self.engine.update(observations, frame.stream_time, wall)
        end = time.perf_counter()
        return FrameResult(
            frame=frame,
            detections=result.detections,
            events=events,
            inference_ms=result.inference_ms,
            detect_ms=result.call_ms,
            pipeline_ms=(end - start) * 1000.0,
            latency_ms=(end - frame.captured_at) * 1000.0,
        )
