"""Command-line entry point (``python main.py <command>``)."""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import sys
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, AppConfig, ConfigError, load_config
from .event_manager import EventType
from .storage import EventFilter, EventStore, StorageError, import_legacy_csv
from .zone_manager import ZoneError, load_zones

log = logging.getLogger("restricted_zone_monitor")


def setup_logging(cfg: AppConfig) -> None:
    root = logging.getLogger()
    root.setLevel(cfg.logging.level)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)
    if cfg.logging.file:
        cfg.logging.file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(cfg.logging.file, maxBytes=1_000_000, backupCount=3,
                                                       encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        root.addHandler(handler)


def parse_when(text: str, end_of_day: bool = False) -> datetime:
    """Parse ``YYYY-MM-DD`` or an ISO datetime. Naive values are local time.

    A bare date used as an upper bound means "through the end of that day".
    """
    try:
        if len(text) == 10:
            day = date.fromisoformat(text)
            if end_of_day:
                day += timedelta(days=1)
            return datetime.combine(day, time()).astimezone()
        value = datetime.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date/time {text!r} (use YYYY-MM-DD or YYYY-MM-DDTHH:MM)") from exc
    return value if value.tzinfo else value.astimezone()


def _filters(args) -> EventFilter:
    return EventFilter(
        since=parse_when(args.since) if args.since else None,
        until=parse_when(args.until, end_of_day=True) if args.until else None,
        zone_id=args.zone,
        event_type=args.type,
        track_id=args.track,
        session_id=args.session,
    )


def _add_filter_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--since", help="start date/time, local (YYYY-MM-DD or YYYY-MM-DDTHH:MM)")
    p.add_argument("--until", help="end date/time, local; a bare date includes that whole day")
    p.add_argument("--zone", help="zone id")
    p.add_argument("--type", choices=[t.value for t in EventType], help="event type")
    p.add_argument("--track", type=int, help="tracking id (only meaningful together with --session)")
    p.add_argument("--session", help="session id")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py", description="Restricted Zone Monitor")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="YAML config file")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="start monitoring")
    run.add_argument("--source", help="camera index, video file or stream URL (overrides video.source)")
    run.add_argument("--zones", type=Path, help="zones JSON file (overrides zones.file)")
    run.add_argument("--headless", action="store_true", help="no window; log progress instead")

    zones = sub.add_parser("zones", help="list or edit zones")
    zones_sub = zones.add_subparsers(dest="zones_command", required=True)
    edit = zones_sub.add_parser("edit", help="draw and edit zones on a frame from the source")
    edit.add_argument("--source", help="camera index, video file or image to draw on")
    edit.add_argument("--zones", type=Path, help="zones JSON file to edit")
    show = zones_sub.add_parser("list", help="print the configured zones")
    show.add_argument("--zones", type=Path)

    events = sub.add_parser("events", help="show stored events (newest first)")
    _add_filter_args(events)
    events.add_argument("--limit", type=int, default=50)

    export = sub.add_parser("export", help="export events to CSV")
    _add_filter_args(export)
    export.add_argument("--output", type=Path, help="CSV path (default: <export_dir>/events_<timestamp>.csv)")
    export.add_argument("--overwrite", action="store_true", help="replace the output file if it exists")

    report = sub.add_parser("report", help="summary statistics from stored events")
    report.add_argument("--since")
    report.add_argument("--until")
    report.add_argument("--zone")
    report.add_argument("--charts", action="store_true", help="also save PNG charts to the export directory")

    imp = sub.add_parser("import-legacy", help="import a log.csv written by the original version")
    imp.add_argument("csv", type=Path)
    imp.add_argument("--date", required=True, type=date.fromisoformat,
                     help="local date the log was recorded on (the old format has no dates)")
    imp.add_argument("--zone-name", default="Legacy zone")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    if getattr(args, "zones", None):
        cfg = replace(cfg, zones=replace(cfg.zones, file=args.zones.resolve()))
    if getattr(args, "source", None):
        source = args.source if args.source.isdigit() or "://" in args.source else str(Path(args.source).resolve())
        cfg = replace(cfg, video=replace(cfg.video, source=source))
    setup_logging(cfg)

    try:
        if args.command == "run":
            return cmd_run(cfg, args)
        if args.command == "zones":
            return cmd_zones(cfg, args)
        if args.command == "events":
            return cmd_events(cfg, args)
        if args.command == "export":
            return cmd_export(cfg, args)
        if args.command == "report":
            return cmd_report(cfg, args)
        if args.command == "import-legacy":
            return cmd_import(cfg, args)
    except (ZoneError, StorageError) as exc:
        log.error("%s", exc)
        return 1
    return 0


def cmd_run(cfg: AppConfig, args) -> int:
    # Must be set before torch is imported: lets ops that MPS doesn't implement
    # fall back to the CPU instead of raising.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    from .app import run_monitor
    from .detector import DetectorError
    from .video_source import VideoSourceError

    try:
        run_monitor(cfg, show_window=not args.headless)
    except (DetectorError, VideoSourceError) as exc:
        log.error("%s", exc)
        return 1
    return 0


def cmd_zones(cfg: AppConfig, args) -> int:
    if args.zones_command == "list":
        zones = load_zones(cfg.zones.file)
        print(f"{cfg.zones.file} ({len(zones)} zone(s), normalized coordinates)")
        for z in zones:
            extra = f", loiter after {z.loiter_seconds:g}s" if z.loiter_seconds else ""
            print(f"  {z.zone_id:<22} {z.name:<24} {z.zone_type.value}{extra}, {len(z.polygon)} vertices")
        return 0

    import cv2

    from .app import WINDOW
    from .video_source import VideoSourceError, open_source
    from .zone_editor import ZoneEditor

    source = cfg.video.source
    is_image = isinstance(source, str) and Path(source).suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    image = cv2.imread(str(source)) if is_image else None
    if image is None:
        try:
            with open_source(source, cfg.video.width, cfg.video.height) as src:
                frame = src.read()
        except VideoSourceError as exc:
            log.error("%s", exc)
            return 1
        if frame is None:
            log.error("Could not read a frame from %s", source)
            return 1
        image = frame.image
    zones = load_zones(cfg.zones.file)
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    try:
        saved = ZoneEditor(WINDOW, image, zones, cfg.zones.file, cfg.display.width).run()
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    print(f"Saved {len(saved)} zone(s) to {cfg.zones.file}" if saved is not None else "No changes saved.")
    return 0


def cmd_events(cfg: AppConfig, args) -> int:
    with EventStore(cfg.storage.database) as store:
        events = store.query(_filters(args), limit=args.limit, newest_first=True)
        total = store.count(_filters(args))
    if not events:
        print("No matching events.")
        return 0
    print(f"{'local time':<20}{'event':<19}{'zone':<22}{'track':>6}  {'class':<8}{'conf':>5}{'occ':>5}{'dur(s)':>8}  "
          f"{'session':<13}detail")
    for e in events:
        print(f"{e.timestamp.astimezone():%Y-%m-%d %H:%M:%S} {e.event_type.value:<19}{e.zone_name[:21]:<22}"
              f"{'' if e.track_id is None else e.track_id:>6}  {e.object_class or '':<8}"
              f"{'' if e.confidence is None else f'{e.confidence:.2f}':>5}{e.occupancy:>5}"
              f"{'' if e.duration_seconds is None else f'{e.duration_seconds:.1f}':>8}  {e.session_id:<13}{e.detail or ''}")
    print(f"\nShowing {len(events)} of {total} matching event(s).")
    return 0


def cmd_export(cfg: AppConfig, args) -> int:
    output = args.output or cfg.storage.export_dir / f"events_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with EventStore(cfg.storage.database) as store:
        rows = store.export_csv(output, _filters(args), overwrite=args.overwrite)
    print(f"Exported {rows} event(s) to {output}")
    return 0


def cmd_report(cfg: AppConfig, args) -> int:
    from .analytics import format_report, write_charts

    since = parse_when(args.since) if args.since else None
    until = parse_when(args.until, end_of_day=True) if args.until else None
    with EventStore(cfg.storage.database) as store:
        events = store.query(EventFilter(since=since, until=until, zone_id=args.zone))
    print(format_report(events, since, until))
    if args.charts:
        paths = write_charts(events, cfg.storage.export_dir, f"{datetime.now():%Y%m%d_%H%M%S}")
        for p in paths:
            print(f"Chart saved: {p}")
        if not paths:
            print("No entries in this period, so no charts were written.")
    return 0


def cmd_import(cfg: AppConfig, args) -> int:
    with EventStore(cfg.storage.database) as store:
        rows, inserted = import_legacy_csv(store, args.csv, args.date, args.zone_name)
    print(f"Read {rows} row(s); inserted {inserted} event(s) "
          f"({rows * 2 - inserted} already present) into {cfg.storage.database}")
    return 0
