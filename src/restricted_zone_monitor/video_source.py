"""Video input: cameras/streams and video files.

The two kinds of input need different reading strategies:

* **Live sources** (webcam, RTSP/HTTP stream) produce frames whether or not we
  are ready. A background thread reads continuously and keeps only the newest
  frame, so a slow detector processes the most recent picture instead of
  falling further and further behind a growing backlog. Frames that are
  overwritten before being read are counted as dropped.
* **Video files** are read synchronously in the caller's thread, one frame at
  a time, so every frame is analysed and nothing is skipped unintentionally.
  Their stream time comes from the frame index and the file's FPS, which keeps
  dwell times correct no matter how fast the file is processed.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


class VideoSourceError(RuntimeError):
    """The source could not be opened or stopped delivering frames."""


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    index: int  # 0-based count of frames delivered by this source
    stream_time: float  # seconds since the start of the stream / video
    captured_at: float  # time.perf_counter() when the frame was read (for latency)


class FileSource:
    is_live = False

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise VideoSourceError(f"Video file not found: {self.path}")
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            self._cap.release()
            raise VideoSourceError(f"OpenCV could not open {self.path} (unsupported codec or corrupt file?)")
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        # Some containers report 0 or nonsense; fall back to 30 so time still advances.
        self.fps = fps if 1.0 <= fps <= 240.0 else 30.0
        if self.fps != fps:
            log.warning("%s reports FPS=%s; assuming 30", self.path.name, fps)
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.size = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.name = self.path.name
        self.dropped_frames = 0
        self._index = 0

    def read(self) -> Frame | None:
        """Next frame, or None at the end of the file."""
        ok, image = self._cap.read()
        if not ok or image is None:
            return None
        frame = Frame(image, self._index, self._index / self.fps, time.perf_counter())
        self._index += 1
        return frame

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> FileSource:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class LiveSource:
    """Camera index or stream URL, read on a background thread (latest frame wins)."""

    is_live = True
    MAX_CONSECUTIVE_FAILURES = 50

    def __init__(self, source: int | str, width: int | None = None, height: int | None = None):
        self.name = f"camera {source}" if isinstance(source, int) else str(source)
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            self._cap.release()
            hint = (" On macOS, allow camera access for your terminal in "
                    "System Settings > Privacy & Security > Camera.") if isinstance(source, int) else ""
            raise VideoSourceError(f"Could not open {self.name}.{hint}")
        if width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        ok, first = self._cap.read()
        if not ok or first is None:
            self._cap.release()
            raise VideoSourceError(f"{self.name} opened but returned no frame")
        self.size = (first.shape[1], first.shape[0])
        if (width and width != self.size[0]) or (height and height != self.size[1]):
            log.warning("Requested %sx%s, camera delivers %dx%d", width, height, *self.size)
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 0 else None

        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._start = time.perf_counter()
        self._latest: Frame | None = Frame(first, 0, 0.0, self._start)
        self._produced = 1
        self._delivered_index = -1
        self.dropped_frames = 0
        self.error: str | None = None
        self._stopped = False
        self._thread = threading.Thread(target=self._reader, name="capture", daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        failures = 0
        while True:
            with self._lock:
                if self._stopped:
                    return
            ok, image = self._cap.read()  # blocking call, made without holding the lock
            now = time.perf_counter()
            with self._lock:
                if self._stopped:
                    return
                if not ok or image is None:
                    failures += 1
                    if failures >= self.MAX_CONSECUTIVE_FAILURES:
                        self.error = f"{self.name} stopped delivering frames"
                        self._stopped = True
                        self._new_frame.notify_all()
                        return
                else:
                    failures = 0
                    if self._latest is not None and self._latest.index > self._delivered_index:
                        self.dropped_frames += 1  # overwritten before anyone read it
                    self._latest = Frame(image, self._produced, now - self._start, now)
                    self._produced += 1
                    self._new_frame.notify_all()
            if failures:
                time.sleep(0.02)  # back off instead of spinning on a failing device

    def read(self, timeout: float = 2.0) -> Frame | None:
        """Wait for a frame newer than the last one returned.

        Returns None if the source failed or was closed. Raises VideoSourceError
        if no frame arrives within ``timeout`` seconds.
        """
        with self._new_frame:
            got_frame = self._new_frame.wait_for(
                lambda: self._stopped or (self._latest is not None and self._latest.index > self._delivered_index),
                timeout=timeout,
            )
            if self._stopped:
                if self.error:
                    log.error(self.error)
                return None
            if not got_frame:
                raise VideoSourceError(f"No frame from {self.name} for {timeout:.0f}s")
            frame = self._latest
            self._delivered_index = frame.index
            return frame

    def close(self) -> None:
        with self._lock:
            self._stopped = True
            self._new_frame.notify_all()
        # Release only after the reader thread has finished: calling release()
        # while another thread is inside read() can crash some backends.
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            log.warning("Capture thread did not stop within 2s; releasing anyway")
        self._cap.release()

    def __enter__(self) -> LiveSource:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_source(source: int | str, width: int | None = None, height: int | None = None) -> FileSource | LiveSource:
    """Open a camera index (int or digit string), stream URL, or video file path."""
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    if isinstance(source, int) or "://" in str(source):
        return LiveSource(source, width, height)
    return FileSource(source)
