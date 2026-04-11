"""YOLOv8 detection with ByteTrack tracking, through Ultralytics.

The model is loaded once in ``Detector.__init__`` and reused for every frame.
Ultralytics is imported lazily so the rest of the package (and the unit tests)
work without PyTorch installed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ModelConfig

log = logging.getLogger(__name__)


class DetectorError(RuntimeError):
    """The model could not be loaded or configured."""


class ModelNotFoundError(DetectorError, FileNotFoundError):
    """The configured weights file does not exist."""


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in source-frame pixels
    confidence: float
    class_id: int
    class_name: str
    track_id: int | None  # None if the tracker hasn't confirmed this object yet


@dataclass(frozen=True)
class DetectionResult:
    detections: list[Detection]
    inference_ms: float | None  # model forward pass only, as measured by Ultralytics
    call_ms: float  # whole call: preprocessing, inference, NMS and tracker update


def check_model_path(path: Path) -> Path:
    """Fail early with a clear message instead of letting Ultralytics try to download weights."""
    path = Path(path)
    if not path.is_file():
        raise ModelNotFoundError(
            f"Model weights not found at {path}. Put yolov8n.pt there or set model.path in the config. "
            "The weights are not downloaded automatically."
        )
    return path


def select_device(requested: str) -> str:
    """Resolve ``auto`` to the best available backend and verify explicit requests."""
    import torch

    cuda = torch.cuda.is_available()
    mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if requested == "auto":
        return "cuda" if cuda else "mps" if mps else "cpu"
    if requested == "cuda" and not cuda:
        raise DetectorError("model.device is 'cuda' but PyTorch reports no CUDA device")
    if requested == "mps" and not mps:
        raise DetectorError("model.device is 'mps' but PyTorch reports MPS is unavailable (needs Apple Silicon + macOS 12.3+)")
    return requested


def resolve_class_ids(class_names: tuple[str, ...], model_names: dict[int, str]) -> list[int]:
    lookup = {name: idx for idx, name in model_names.items()}
    unknown = [c for c in class_names if c not in lookup]
    if unknown:
        raise DetectorError(
            f"Class(es) {unknown} are not in this model. Available: {', '.join(sorted(lookup))}"
        )
    return [lookup[c] for c in class_names]


class Detector:
    def __init__(self, cfg: ModelConfig, tracker: str = "bytetrack.yaml"):
        self.model_path = check_model_path(cfg.path)
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorError("ultralytics is not installed; run `pip install -r requirements.txt`") from exc

        self.device = select_device(cfg.device)
        self.half = cfg.half and self.device == "cuda"
        if cfg.half and not self.half:
            log.warning("model.half is only applied on CUDA; running %s in FP32", self.device)

        try:
            self.model = YOLO(str(self.model_path))
        except Exception as exc:  # Ultralytics raises a variety of types for corrupt files
            raise DetectorError(f"Could not load model {self.model_path}: {exc}") from exc

        self.names: dict[int, str] = dict(self.model.names)
        self.class_ids = resolve_class_ids(cfg.classes, self.names)
        self.confidence = cfg.confidence
        self.iou = cfg.iou
        self.image_size = cfg.image_size
        self.tracker = tracker
        log.info(
            "Loaded %s on %s (imgsz=%d, classes=%s, half=%s)",
            self.model_path.name, self.device, self.image_size, list(cfg.classes), self.half,
        )

    def _common_args(self) -> dict:
        args = dict(
            conf=self.confidence, iou=self.iou, classes=self.class_ids, imgsz=self.image_size,
            device=self.device, verbose=False,
        )
        if self.half:
            # Only passed when enabled: recent Ultralytics releases warn that the
            # argument is deprecated even when it is False.
            args["half"] = True
        return args

    def detect(self, image: np.ndarray, track: bool = True) -> DetectionResult:
        """Run detection (and tracking, by default) on one BGR frame."""
        start = time.perf_counter()
        if track:
            # persist=True keeps ByteTrack's state between calls, which is what
            # makes IDs stable across frames.
            results = self.model.track(image, persist=True, tracker=self.tracker, **self._common_args())
        else:
            results = self.model.predict(image, **self._common_args())
        call_ms = (time.perf_counter() - start) * 1000.0
        result = results[0]

        boxes = result.boxes
        detections: list[Detection] = []
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)
            for box, conf, cls, tid in zip(xyxy, confs, classes, ids, strict=True):
                detections.append(Detection(
                    bbox=tuple(float(v) for v in box),
                    confidence=float(conf),
                    class_id=int(cls),
                    class_name=self.names.get(int(cls), str(cls)),
                    track_id=None if tid is None else int(tid),
                ))
        return DetectionResult(detections, result.speed.get("inference"), call_ms)

    def warmup(self, height: int, width: int, runs: int = 2) -> None:
        """Run a few throwaway predictions so the first real frame isn't slowed by lazy init.

        Uses predict (not track) so the tracker state stays empty.
        """
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(runs):
            self.model.predict(blank, **self._common_args())
