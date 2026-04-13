"""Bookkeeping for tracked objects.

The actual association of detections across frames is done by ByteTrack
inside Ultralytics (see ``detector.py``). ByteTrack only reports the tracks it
matched in the current frame, so this module keeps the state the event engine
needs between frames: when each track was first and last seen, its latest
position and confidence, and when it should be forgotten.

Track IDs are only meaningful within one monitoring session. They come from a
counter inside the tracker and say nothing about who a person is.
"""

from __future__ import annotations

from dataclasses import dataclass

from .zone_manager import Point


@dataclass(frozen=True)
class TrackObservation:
    """One tracked object in one processed frame."""

    track_id: int
    class_name: str
    confidence: float
    anchor: Point  # normalized point tested against zones


@dataclass
class Track:
    track_id: int
    class_name: str
    first_seen: float
    last_seen: float
    anchor: Point
    confidence: float
    hits: int = 1


class TrackTable:
    """Remembers tracks between frames and drops the ones not seen for ``timeout`` seconds."""

    def __init__(self, timeout: float):
        self.timeout = timeout
        self._tracks: dict[int, Track] = {}

    def update(self, observations: list[TrackObservation], now: float) -> None:
        for obs in observations:
            track = self._tracks.get(obs.track_id)
            if track is None:
                self._tracks[obs.track_id] = Track(
                    obs.track_id, obs.class_name, now, now, obs.anchor, obs.confidence
                )
            else:
                # The tracker can briefly lose an object and recover the same ID
                # (ByteTrack's track_buffer). Within our timeout that is the same
                # track, so it simply continues.
                track.last_seen = now
                track.anchor = obs.anchor
                track.confidence = obs.confidence
                track.class_name = obs.class_name
                track.hits += 1

    def prune(self, now: float) -> list[int]:
        """Forget tracks not seen for longer than the timeout; return their IDs."""
        stale = [tid for tid, t in self._tracks.items() if now - t.last_seen > self.timeout]
        for tid in stale:
            del self._tracks[tid]
        return stale

    def get(self, track_id: int) -> Track | None:
        return self._tracks.get(track_id)

    def clear(self) -> None:
        self._tracks.clear()

    def __len__(self) -> int:
        return len(self._tracks)

    def __contains__(self, track_id: int) -> bool:
        return track_id in self._tracks
