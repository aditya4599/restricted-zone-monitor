"""SQLite event storage, CSV export and import of the old log.csv format.

Timestamps are stored as fixed-width UTC strings (``2026-10-18T09:15:02.123456Z``)
so that string order equals time order and range filters can use the index.

The live monitor never writes to SQLite from the video loop. Events go onto a
queue and ``BackgroundEventWriter`` inserts them from its own thread with its
own connection, so a slow disk can't stall frame processing.
"""

from __future__ import annotations

import csv
import logging
import queue
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from .event_manager import Event, EventType

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id         TEXT PRIMARY KEY,
    timestamp_utc    TEXT NOT NULL,
    session_id       TEXT NOT NULL,
    zone_id          TEXT NOT NULL,
    zone_name        TEXT NOT NULL,
    zone_type        TEXT NOT NULL,
    event_type       TEXT NOT NULL,
    track_id         INTEGER,
    object_class     TEXT,
    confidence       REAL,
    occupancy        INTEGER NOT NULL,
    duration_seconds REAL,
    stream_seconds   REAL,
    detail           TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_time      ON events (timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_events_zone_time ON events (zone_id, timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_events_type_time ON events (event_type, timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_events_track     ON events (session_id, track_id);
"""

COLUMNS = (
    "event_id", "timestamp_utc", "session_id", "zone_id", "zone_name", "zone_type", "event_type",
    "track_id", "object_class", "confidence", "occupancy", "duration_seconds", "stream_seconds", "detail",
)

CSV_COLUMNS = COLUMNS[:2] + ("timestamp_local",) + COLUMNS[2:]


class StorageError(RuntimeError):
    pass


def format_utc(ts: datetime) -> str:
    if ts.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_utc(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class EventFilter:
    since: datetime | None = None  # inclusive, timezone-aware
    until: datetime | None = None  # exclusive, timezone-aware
    zone_id: str | None = None
    event_type: EventType | str | None = None
    track_id: int | None = None
    session_id: str | None = None

    def where(self) -> tuple[str, list]:
        clauses, params = [], []
        if self.since is not None:
            clauses.append("timestamp_utc >= ?")
            params.append(format_utc(self.since))
        if self.until is not None:
            clauses.append("timestamp_utc < ?")
            params.append(format_utc(self.until))
        if self.zone_id is not None:
            clauses.append("zone_id = ?")
            params.append(self.zone_id)
        if self.event_type is not None:
            clauses.append("event_type = ?")
            params.append(EventType(self.event_type).value)
        if self.track_id is not None:
            clauses.append("track_id = ?")
            params.append(self.track_id)
        if self.session_id is not None:
            clauses.append("session_id = ?")
            params.append(self.session_id)
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def _to_row(e: Event) -> tuple:
    return (
        e.event_id, format_utc(e.timestamp), e.session_id, e.zone_id, e.zone_name, e.zone_type,
        EventType(e.event_type).value, e.track_id, e.object_class, e.confidence, e.occupancy,
        e.duration_seconds, e.stream_seconds, e.detail,
    )


def _from_row(row: sqlite3.Row) -> Event:
    return Event(
        event_id=row["event_id"], timestamp=parse_utc(row["timestamp_utc"]), session_id=row["session_id"],
        zone_id=row["zone_id"], zone_name=row["zone_name"], zone_type=row["zone_type"],
        event_type=EventType(row["event_type"]), track_id=row["track_id"], object_class=row["object_class"],
        confidence=row["confidence"], occupancy=row["occupancy"], duration_seconds=row["duration_seconds"],
        stream_seconds=row["stream_seconds"], detail=row["detail"],
    )


class EventStore:
    """One SQLite connection. Use it from a single thread (or one store per thread)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, timeout=10.0)
            self._conn.row_factory = sqlite3.Row
            # WAL lets the CLI read while a monitor is writing.
            self._conn.execute("PRAGMA journal_mode=WAL")
            version = self._conn.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise StorageError(f"{self.path} was created by a newer version (schema {version})")
            with self._conn:
                self._conn.executescript(SCHEMA)
                self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        except (sqlite3.Error, OSError) as exc:
            raise StorageError(f"Cannot open event database {self.path}: {exc}") from exc

    def add(self, events: list[Event], ignore_duplicates: bool = False) -> int:
        """Insert events; returns how many rows were written.

        Plain INSERT (not REPLACE): an existing event is never overwritten. A
        duplicate ID raises, unless ``ignore_duplicates`` is set (used by the
        importer so re-running an import doesn't double the data).
        """
        if not events:
            return 0
        verb = "INSERT OR IGNORE" if ignore_duplicates else "INSERT"
        sql = f"{verb} INTO events ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})"
        try:
            with self._conn:
                before = self._conn.total_changes
                self._conn.executemany(sql, [_to_row(e) for e in events])
                return self._conn.total_changes - before
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to store {len(events)} event(s): {exc}") from exc

    def query(self, filters: EventFilter = EventFilter(), limit: int | None = None, newest_first: bool = False) -> list[Event]:
        where, params = filters.where()
        sql = f"SELECT * FROM events{where} ORDER BY timestamp_utc {'DESC' if newest_first else 'ASC'}, rowid"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        try:
            return [_from_row(r) for r in self._conn.execute(sql, params)]
        except sqlite3.Error as exc:
            raise StorageError(f"Query failed: {exc}") from exc

    def count(self, filters: EventFilter = EventFilter()) -> int:
        where, params = filters.where()
        return self._conn.execute(f"SELECT COUNT(*) FROM events{where}", params).fetchone()[0]

    def export_csv(self, path: str | Path, filters: EventFilter = EventFilter(), overwrite: bool = False) -> int:
        """Write matching events to CSV; returns the number of rows. Refuses to overwrite by default."""
        path = Path(path)
        if path.exists() and not overwrite:
            raise StorageError(f"{path} already exists (pass overwrite=True / --overwrite to replace it)")
        path.parent.mkdir(parents=True, exist_ok=True)
        events = self.query(filters)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(CSV_COLUMNS)
            for e in events:
                row = list(_to_row(e))
                row.insert(2, e.timestamp.astimezone().isoformat(timespec="seconds"))
                writer.writerow(["" if v is None else v for v in row])
        return len(events)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> EventStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class BackgroundEventWriter:
    """Persists events from a worker thread so the video loop never waits on disk I/O."""

    _STOP = object()

    def __init__(self, path: str | Path, batch_size: int = 100):
        self.path = Path(path)
        EventStore(self.path).close()  # open once here so a bad path fails at startup, not later
        self.batch_size = batch_size
        self.written = 0
        self.failed = 0
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="event-writer", daemon=True)
        self._thread.start()

    def write(self, events: list[Event]) -> None:
        for event in events:
            self._queue.put(event)

    def _run(self) -> None:
        with EventStore(self.path) as store:
            stopping = False
            while not stopping:
                item = self._queue.get()
                batch = []
                if item is self._STOP:
                    stopping = True
                else:
                    batch.append(item)
                while len(batch) < self.batch_size:
                    try:
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is self._STOP:
                        stopping = True
                        continue
                    batch.append(item)
                if batch:
                    self._persist(store, batch)

    def _persist(self, store: EventStore, batch: list[Event]) -> None:
        try:
            self.written += store.add(batch)
            return
        except StorageError as exc:
            log.warning("Batch insert failed (%s); retrying events one by one", exc)
        # The transaction was rolled back, so retry individually: one bad row
        # should not cost the other events in the batch.
        for event in batch:
            try:
                self.written += store.add([event])
            except StorageError:
                self.failed += 1
                log.exception("Could not persist event %s", event.event_id)

    def close(self, timeout: float = 10.0) -> None:
        """Flush everything still queued, then stop the worker."""
        self._queue.put(self._STOP)
        self._thread.join(timeout)
        if self._thread.is_alive():
            log.error("Event writer did not finish within %.0fs; some events may be lost", timeout)

    def __enter__(self) -> BackgroundEventWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --- legacy import ------------------------------------------------------------

LEGACY_SESSION = "legacy_csv"
_LEGACY_NAMESPACE = uuid.UUID("5c1f2a8e-3a59-4c1e-9d3e-7f4b0c2d9a11")


def import_legacy_csv(store: EventStore, csv_path: Path, day: date, zone_name: str = "Legacy zone") -> tuple[int, int]:
    """Import the ``Entry Time,Exit Time,Duration (sec)`` log written by the original main.py.

    That format stored only local times of day, so the date must be supplied.
    Each row becomes an entry and an exit event. Event IDs are derived from the
    file name, date and row number, so importing the same file twice inserts
    nothing new. Returns (rows read, events inserted).
    """
    csv_path = Path(csv_path)
    local_tz = datetime.now().astimezone().tzinfo
    events: list[Event] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        expected = {"Entry Time", "Exit Time", "Duration (sec)"}
        if not reader.fieldnames or not expected.issubset(reader.fieldnames):
            raise StorageError(f"{csv_path} does not look like the legacy log (expected columns {sorted(expected)})")
        rows = list(reader)
    for i, row in enumerate(rows):
        try:
            entry_t = time.fromisoformat(row["Entry Time"])
            duration = float(row["Duration (sec)"])
        except ValueError as exc:
            raise StorageError(f"{csv_path} row {i + 2}: {exc}") from exc
        entry = datetime.combine(day, entry_t, tzinfo=local_tz)
        # The old format truncated both times to whole seconds; deriving the exit
        # from entry + duration keeps the timestamps consistent with the duration.
        exit_ = entry + timedelta(seconds=duration)
        for kind, ts, occ, dur in (
            (EventType.ENTRY, entry, 1, None),
            (EventType.EXIT, exit_, 0, duration),
        ):
            events.append(Event(
                event_id=uuid.uuid5(_LEGACY_NAMESPACE, f"{csv_path.name}|{day}|{i}|{kind.value}").hex,
                timestamp=ts, session_id=LEGACY_SESSION, zone_id="legacy", zone_name=zone_name,
                zone_type="restricted", event_type=kind, track_id=None, object_class=None,
                confidence=None, occupancy=occ, duration_seconds=dur, stream_seconds=None,
                detail="imported from legacy log.csv",
            ))
    inserted = store.add(events, ignore_duplicates=True)
    return len(rows), inserted
