"""The live monitoring loop: read frames, run the pipeline, persist events, draw the window."""

from __future__ import annotations

import logging
import time
from collections import Counter, deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import AppConfig
from .detector import Detector
from .event_manager import Event, EventEngine, EventType, new_session_id
from .performance import PerformanceMonitor
from .pipeline import FrameResult, MonitorPipeline
from .storage import BackgroundEventWriter, EventStore
from .video_source import Frame, open_source
from .visualization import (
    ALERT,
    BANNER_HEIGHT,
    WARN,
    PanelState,
    draw_banner,
    draw_detections,
    draw_panel,
    draw_zones,
    latest_alert,
    scale_frame,
)
from .zone_editor import ZoneEditor
from .zone_manager import ZoneType, load_zones

log = logging.getLogger(__name__)

WINDOW = "Restricted Zone Monitor"
HEADLESS_LOG_INTERVAL = 5.0


@dataclass
class RunSummary:
    session_id: str
    frames_read: int = 0
    frames_processed: int = 0
    events: Counter = field(default_factory=Counter)
    storage_failures: int = 0
    perf: dict = field(default_factory=dict)
    wall_seconds: float = 0.0
    ended_by: str = ""


def run_monitor(cfg: AppConfig, show_window: bool = True, detector=None) -> RunSummary:
    zones = load_zones(cfg.zones.file)
    if zones:
        log.info("Loaded %d zone(s) from %s", len(zones), cfg.zones.file)
    else:
        log.warning("No zones in %s - %s", cfg.zones.file,
                    "press z in the window to draw some" if show_window else "nothing will be recorded")

    detector = detector or Detector(cfg.model, cfg.tracking.tracker)
    summary = RunSummary(session_id=new_session_id())
    log.info("Session %s started", summary.session_id)

    with EventStore(cfg.storage.database) as store:
        recorded_before = store.count()

    with open_source(cfg.video.source, cfg.video.width, cfg.video.height) as source, \
            BackgroundEventWriter(cfg.storage.database) as writer:
        width, height = source.size
        log.info("Source %s: %dx%d%s", source.name, width, height,
                 f" @ {source.fps:.1f} fps" if source.fps else "")
        if hasattr(detector, "warmup"):
            detector.warmup(height, width)

        engine = EventEngine(
            zones,
            session_id=summary.session_id,
            entry_confirm_seconds=cfg.tracking.entry_confirm_seconds,
            exit_grace_seconds=cfg.tracking.exit_grace_seconds,
            track_timeout_seconds=cfg.tracking.track_timeout_seconds,
        )
        pipeline = MonitorPipeline(detector, engine, cfg.zones.anchor)
        perf = PerformanceMonitor(cfg.performance.warmup_frames, cfg.performance.window_size)
        recent: deque[Event] = deque(maxlen=cfg.display.recent_events)

        def emit(events: list[Event]) -> None:
            if not events:
                return
            writer.write(events)
            recent.extend(events)
            for e in events:
                summary.events[e.event_type.value] += 1
                log.info("%s | %s | track %s | occupancy %d%s", e.event_type.value, e.zone_name, e.track_id,
                         e.occupancy, f" | {e.duration_seconds:.1f}s" if e.duration_seconds is not None else "")

        paused = False
        last_frame: Frame | None = None
        last_result: FrameResult | None = None
        next_progress_log = time.monotonic() + HEADLESS_LOG_INTERVAL
        if show_window:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

        started = time.perf_counter()
        try:
            while True:
                frame = source.read()
                if frame is None:
                    summary.ended_by = "end of video" if not source.is_live else "source stopped"
                    break
                last_frame = frame
                summary.frames_read += 1

                if not paused and frame.index % cfg.video.process_every_n_frames == 0:
                    last_result = pipeline.process(frame)
                    summary.frames_processed += 1
                    perf.record_processed(
                        inference_ms=last_result.inference_ms, detect_ms=last_result.detect_ms,
                        pipeline_ms=last_result.pipeline_ms, latency_ms=last_result.latency_ms,
                    )
                    emit(last_result.events)

                if not show_window:
                    if time.monotonic() >= next_progress_log:
                        next_progress_log += HEADLESS_LOG_INTERVAL
                        _log_progress(summary, perf, source)
                    continue

                status = "PAUSED" if paused else "WARMING UP" if perf.warming_up else "MONITORING"
                panel = PanelState(
                    source_name=source.name, device=getattr(detector, "device", "?"), status=status,
                    perf=perf.summary(), dropped_frames=source.dropped_frames, zone_status=engine.zone_status(),
                    events_recorded=recorded_before + writer.written, storage_failures=writer.failed,
                    recent_events=list(recent),
                )
                cv2.imshow(WINDOW, render(frame, None if paused else last_result, engine, panel, cfg, list(recent)))
                perf.record_displayed()

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    summary.ended_by = "user"
                    break
                if key == ord(" "):
                    paused = not paused
                    if paused:
                        # Close open visits so the pause isn't counted as time in a zone.
                        emit(engine.close_all(frame.stream_time, reason="monitoring_paused"))
                        last_result = None
                    log.info("Monitoring %s", "paused" if paused else "resumed")
                elif key == ord("z"):
                    emit(engine.close_all(frame.stream_time, reason="monitoring_paused"))
                    last_result = None
                    edited = ZoneEditor(WINDOW, frame.image, engine.zones, cfg.zones.file, cfg.display.width).run()
                    if edited is not None:
                        emit(engine.set_zones(edited, frame.stream_time))
                        log.info("Zones updated: %s", ", ".join(z.name for z in edited) or "none")
        except KeyboardInterrupt:
            summary.ended_by = "interrupted"
        finally:
            if last_frame is not None:
                emit(engine.close_all(last_frame.stream_time, reason="session_end"))
            if show_window:
                cv2.destroyAllWindows()
                cv2.waitKey(1)  # lets the window actually close on macOS
            summary.wall_seconds = time.perf_counter() - started
            summary.perf = perf.summary()
            summary.perf.update(
                inference_ms_median=perf.inference_ms.median,
                pipeline_ms_median=perf.pipeline_ms.median,
                dropped_frames=source.dropped_frames,
            )
        # Leaving the `with` block flushed the writer, so its counters are final now.
    summary.storage_failures = writer.failed
    _log_summary(summary)
    return summary


def render(frame: Frame, result: FrameResult | None, engine: EventEngine, panel: PanelState,
           cfg: AppConfig, recent: list[Event]) -> np.ndarray:
    video, scale = scale_frame(frame.image, cfg.display.width)
    status = {s.zone.zone_id: s for s in panel.zone_status}
    draw_zones(video, engine.zones, status)

    banner = None
    if panel.status == "PAUSED":
        banner = ("MONITORING PAUSED - press space to resume", WARN)
    elif (alert := latest_alert(recent)) is not None:
        if alert.event_type is EventType.LOITERING:
            banner = (f"LOITERING  {alert.zone_name}  track #{alert.track_id}", WARN)
        else:
            banner = (f"INTRUSION  {alert.zone_name}  track #{alert.track_id}", ALERT)

    if result is not None:
        restricted = {t for s in panel.zone_status if s.zone.zone_type is ZoneType.RESTRICTED for t in s.track_ids}
        loitering = {t for s in panel.zone_status for t in s.loitering_track_ids}
        draw_detections(video, result.detections, scale, restricted, loitering, cfg.zones.anchor,
                        top_margin=BANNER_HEIGHT if banner else 0)
    if banner:
        draw_banner(video, *banner)
    side = draw_panel(video.shape[0], cfg.display.panel_width, panel)
    return np.hstack([video, side])


def _log_progress(summary: RunSummary, perf: PerformanceMonitor, source) -> None:
    p = perf.summary()
    total = f"/{source.frame_count}" if getattr(source, "frame_count", 0) else ""
    log.info("frames %d%s | processed %s fps | inference %s ms | events %d",
             summary.frames_read, total, _f(p["processed_fps"]), _f(p["inference_ms"]), sum(summary.events.values()))


def _log_summary(s: RunSummary) -> None:
    p = s.perf
    overall = s.frames_processed / s.wall_seconds if s.wall_seconds > 0 else None
    log.info("Session %s ended (%s): %d frames read, %d processed in %.1fs (%s processed fps overall)",
             s.session_id, s.ended_by or "?", s.frames_read, s.frames_processed, s.wall_seconds, _f(overall))
    log.info("Events: %s", dict(s.events) or "none")
    log.info("Timing (after warm-up): inference mean %s / median %s ms, pipeline mean %s ms, latency p50 %s / p95 %s ms",
             _f(p.get("inference_ms")), _f(p.get("inference_ms_median")), _f(p.get("pipeline_ms")),
             _f(p.get("latency_p50_ms")), _f(p.get("latency_p95_ms")))
    if s.storage_failures:
        log.error("%d event(s) could not be saved to the database", s.storage_failures)


def _f(value) -> str:
    return "-" if value is None else f"{value:.1f}"
