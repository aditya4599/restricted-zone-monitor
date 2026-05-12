import threading
import time

import cv2
import numpy as np
import pytest

from restricted_zone_monitor import video_source
from restricted_zone_monitor.video_source import FileSource, LiveSource, VideoSourceError, open_source


@pytest.fixture
def small_video(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV build cannot write MJPG video")
    for i in range(10):
        writer.write(np.full((48, 64, 3), i * 20, np.uint8))
    writer.release()
    return path


def test_file_source_reads_every_frame_then_ends(small_video):
    with FileSource(small_video) as src:
        assert src.size == (64, 48)
        frames = []
        while (f := src.read()) is not None:
            frames.append(f)
    assert [f.index for f in frames] == list(range(10))
    assert frames[4].stream_time == pytest.approx(4 / 20.0)  # from frame index and FPS, not wall time


def test_missing_and_invalid_files(tmp_path):
    with pytest.raises(VideoSourceError, match="not found"):
        FileSource(tmp_path / "missing.mp4")
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(VideoSourceError):
        FileSource(bad)


def test_open_source_dispatch(small_video, monkeypatch):
    assert isinstance(open_source(str(small_video)), FileSource)
    opened = []
    monkeypatch.setattr(video_source, "LiveSource", lambda s, w, h: opened.append(s) or "live")
    assert open_source("0") == "live" and opened == [0]
    assert open_source("rtsp://cam") == "live"


class FakeCapture:
    """Stands in for cv2.VideoCapture: produces numbered frames at ~200 fps."""

    def __init__(self, source, fail_after=None, opened=True):
        self.n = 0
        self.fail_after = fail_after
        self._opened = opened
        self.released = threading.Event()
        self.reading = False

    def isOpened(self):
        return self._opened

    def set(self, prop, value):
        return True

    def get(self, prop):
        return 30.0

    def read(self):
        assert not self.released.is_set(), "read() after release()"
        time.sleep(0.005)
        self.n += 1
        if self.fail_after is not None and self.n > self.fail_after:
            return False, None
        return True, np.full((4, 4, 3), self.n % 255, np.uint8)

    def release(self):
        self.released.set()


def test_live_source_returns_latest_frames_and_counts_drops(monkeypatch):
    cap = FakeCapture(0)
    monkeypatch.setattr(video_source.cv2, "VideoCapture", lambda s: cap)
    src = LiveSource(0)
    try:
        first = src.read()
        time.sleep(0.1)  # consumer is slow: ~20 frames arrive meanwhile
        second = src.read()
        assert second.index > first.index + 1  # skipped straight to the newest frame
        assert src.dropped_frames > 0
        third = src.read()
        assert third.index > second.index  # never the same frame twice
    finally:
        src.close()
    assert cap.released.is_set()
    assert not src._thread.is_alive()


def test_live_source_reports_failure(monkeypatch):
    monkeypatch.setattr(video_source.cv2, "VideoCapture", lambda s: FakeCapture(0, fail_after=3))
    monkeypatch.setattr(LiveSource, "MAX_CONSECUTIVE_FAILURES", 5)
    with LiveSource(0) as src:
        frames = []
        while (f := src.read(timeout=2.0)) is not None:
            frames.append(f)
        assert src.error and "stopped" in src.error


def test_unavailable_camera(monkeypatch):
    monkeypatch.setattr(video_source.cv2, "VideoCapture", lambda s: FakeCapture(s, opened=False))
    with pytest.raises(VideoSourceError, match="Could not open camera 3"):
        LiveSource(3)
