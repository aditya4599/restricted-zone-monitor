"""Drawing the monitor window with OpenCV: video overlay on the left, status panel on the right.

Everything is drawn on a copy of the frame scaled to the display width. Zone
vertices are normalized, detections are in source pixels; both are converted
to display pixels here and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import cv2
import numpy as np

from .detector import Detection
from .event_manager import Event, EventType, ZoneStatus
from .zone_manager import Zone, ZoneType, to_pixels

FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR colours
ZONE_COLORS = {
    ZoneType.RESTRICTED: (60, 60, 220),
    ZoneType.LOITERING: (40, 170, 240),
    ZoneType.OCCUPANCY: (200, 150, 60),
}
BOX_DEFAULT = (90, 200, 90)
BOX_UNTRACKED = (150, 150, 150)
ALERT = (60, 60, 230)
WARN = (40, 170, 240)
OK = (90, 200, 90)
PANEL_BG = (34, 32, 30)
TEXT = (230, 230, 230)
MUTED = (150, 150, 150)
RULE = (70, 70, 70)

ALERT_BANNER_SECONDS = 4.0


def scale_frame(image: np.ndarray, display_width: int) -> tuple[np.ndarray, float]:
    """Resize keeping the aspect ratio. Returns the image and the scale factor."""
    h, w = image.shape[:2]
    scale = display_width / w
    if abs(scale - 1.0) < 1e-3:
        return image.copy(), 1.0
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, (display_width, round(h * scale)), interpolation=interp), scale


def draw_label(img: np.ndarray, text: str, x: int, y: int, color: tuple, scale: float = 0.45) -> None:
    """Text on a filled background box whose bottom-left corner is (x, y)."""
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
    y = max(y, th + base + 2)
    cv2.rectangle(img, (x, y - th - base - 2), (x + tw + 4, y), color, -1)
    cv2.putText(img, text, (x + 2, y - base), FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_zones(img: np.ndarray, zones: list[Zone], status: dict[str, ZoneStatus] | None = None,
               selected: str | None = None) -> None:
    h, w = img.shape[:2]
    fill_layer = img.copy()
    filled = False
    for zone in zones:
        pts = np.array(to_pixels(zone.polygon, w, h), dtype=np.int32)
        color = ZONE_COLORS[zone.zone_type]
        st = status.get(zone.zone_id) if status else None
        if st and st.occupancy:
            cv2.fillPoly(fill_layer, [pts], color)
            filled = True
    if filled:
        cv2.addWeighted(fill_layer, 0.25, img, 0.75, 0, dst=img)
    for zone in zones:
        pts = np.array(to_pixels(zone.polygon, w, h), dtype=np.int32)
        color = ZONE_COLORS[zone.zone_type]
        thickness = 3 if zone.zone_id == selected else 2
        cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)
        st = status.get(zone.zone_id) if status else None
        label = zone.name if st is None else f"{zone.name}  {st.occupancy}"
        top = pts[np.argmin(pts[:, 1])]
        draw_label(img, label, int(top[0]), int(top[1]) - 4, color)
        if zone.zone_id == selected:
            for p in pts:
                cv2.circle(img, tuple(int(v) for v in p), 6, (255, 255, 255), 2, cv2.LINE_AA)


def draw_detections(img: np.ndarray, detections: list[Detection], scale: float,
                    restricted_ids: set[int], loitering_ids: set[int], anchor: str, top_margin: int = 0) -> None:
    """Boxes with "#id class conf" labels. Labels that would sit under a banner
    of height ``top_margin`` are moved inside the box instead."""
    for d in detections:
        x1, y1, x2, y2 = (round(v * scale) for v in d.bbox)
        if d.track_id is None:
            color = BOX_UNTRACKED
        elif d.track_id in restricted_ids:
            color = ALERT
        elif d.track_id in loitering_ids:
            color = WARN
        else:
            color = BOX_DEFAULT
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        ax, ay = (x1 + x2) // 2, (y2 if anchor == "bottom_center" else (y1 + y2) // 2)
        cv2.circle(img, (ax, ay), 4, color, -1)
        tid = f"#{d.track_id} " if d.track_id is not None else ""
        label_y = y1 - 2 if y1 - 22 >= top_margin else max(y1, top_margin) + 20
        draw_label(img, f"{tid}{d.class_name} {d.confidence:.2f}", x1, label_y, color)


BANNER_HEIGHT = 30


def draw_banner(img: np.ndarray, text: str, color: tuple) -> None:
    cv2.rectangle(img, (0, 0), (img.shape[1], BANNER_HEIGHT), color, -1)
    cv2.putText(img, text, (10, 21), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


@dataclass
class PanelState:
    source_name: str
    device: str
    status: str  # MONITORING / PAUSED / WARMING UP / ENDED
    perf: dict[str, float | None]
    dropped_frames: int
    zone_status: list[ZoneStatus]
    events_recorded: int
    storage_failures: int
    recent_events: list[Event]


def _fmt(value: float | None, unit: str, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f} {unit}"


def _fit(text: str, max_width: int, scale: float) -> str:
    if cv2.getTextSize(text, FONT, scale, 1)[0][0] <= max_width:
        return text
    while text and cv2.getTextSize(text + "..", FONT, scale, 1)[0][0] > max_width:
        text = text[:-1]
    return text + ".."


def draw_panel(height: int, width: int, state: PanelState) -> np.ndarray:
    panel = np.full((height, width, 3), PANEL_BG, dtype=np.uint8)
    x, y, lh = 14, 26, 19
    inner = width - 2 * x

    def text(s: str, color=TEXT, scale=0.45, dx=0):
        cv2.putText(panel, _fit(s, inner - dx, scale), (x + dx, y), FONT, scale, color, 1, cv2.LINE_AA)

    def kv(key: str, value: str, color=TEXT):
        cv2.putText(panel, key, (x, y), FONT, 0.45, MUTED, 1, cv2.LINE_AA)
        cv2.putText(panel, value, (x + 110, y), FONT, 0.45, color, 1, cv2.LINE_AA)

    def section(title: str):
        nonlocal y
        y += 8
        cv2.line(panel, (x, y - 12), (width - x, y - 12), RULE, 1)
        text(title.upper(), MUTED, 0.4)
        y += lh

    status_color = {"MONITORING": OK, "PAUSED": WARN, "WARMING UP": WARN}.get(state.status, MUTED)
    text("Restricted Zone Monitor", TEXT, 0.55)
    y += lh + 2
    text(f"{state.source_name}  |  {state.device}", MUTED)
    y += lh
    text(state.status, status_color, 0.5)
    y += lh + 4

    section("Performance")
    p = state.perf
    kv("processed", _fmt(p.get("processed_fps"), "fps"))
    y += lh
    kv("displayed", _fmt(p.get("display_fps"), "fps"))
    y += lh
    kv("inference", _fmt(p.get("inference_ms"), "ms"))
    y += lh
    kv("pipeline", _fmt(p.get("pipeline_ms"), "ms"))
    y += lh
    p50, p95 = p.get("latency_p50_ms"), p.get("latency_p95_ms")
    kv("latency", f"p50 {_fmt(p50, 'ms', 0)}" + (f" / p95 {p95:.0f}" if p95 is not None else ""))
    y += lh
    kv("dropped", str(state.dropped_frames))
    y += lh

    section("Zones")
    if not state.zone_status:
        text("No zones defined - press z", MUTED)
        y += lh
    for st in state.zone_status:
        color = ZONE_COLORS[st.zone.zone_type]
        cv2.rectangle(panel, (x, y - 10), (x + 10, y), color, -1)
        alert, alert_color = "", TEXT
        if st.zone.zone_type is ZoneType.RESTRICTED and st.occupancy:
            alert, alert_color = "INTRUSION", ALERT
        elif st.loitering_track_ids:
            alert, alert_color = "LOITERING", WARN
        right = f"{st.occupancy}" + (f"  {alert}" if alert else "")
        rw = cv2.getTextSize(right, FONT, 0.45, 1)[0][0]
        cv2.putText(panel, _fit(st.zone.name, inner - rw - 30, 0.45), (x + 18, y), FONT, 0.45, TEXT, 1, cv2.LINE_AA)
        cv2.putText(panel, right, (width - x - rw, y), FONT, 0.45, alert_color, 1, cv2.LINE_AA)
        y += lh

    section("Events")
    kv("recorded", str(state.events_recorded))
    y += lh
    if state.storage_failures:
        kv("not saved", str(state.storage_failures), ALERT)
        y += lh

    keys_y = height - 14
    remaining = (keys_y - 2 * lh - y) // lh
    if remaining > 0 and state.recent_events:
        y += 2
        for e in list(reversed(state.recent_events))[:remaining]:
            color = WARN if e.event_type is EventType.LOITERING else ALERT if _is_alert(e) else TEXT
            text(event_line(e), color, 0.4)
            y += lh - 2

    cv2.line(panel, (x, keys_y - 16), (width - x, keys_y - 16), RULE, 1)
    cv2.putText(panel, "space pause   z zones   q quit", (x, keys_y), FONT, 0.4, MUTED, 1, cv2.LINE_AA)
    return panel


def _is_alert(e: Event) -> bool:
    return e.event_type is EventType.LOITERING or (
        e.event_type is EventType.ENTRY and e.zone_type == ZoneType.RESTRICTED.value
    )


def event_line(e: Event) -> str:
    t = e.timestamp.astimezone().strftime("%H:%M:%S")
    kind = {
        EventType.ENTRY: "entry", EventType.EXIT: "exit",
        EventType.LOITERING: "loiter", EventType.OCCUPANCY: "count",
    }[e.event_type]
    who = f"#{e.track_id}" if e.track_id is not None else f"= {e.occupancy}"
    extra = f" {e.duration_seconds:.0f}s" if e.event_type is EventType.EXIT and e.duration_seconds else ""
    return f"{t} {kind:<6} {who} {e.zone_name}{extra}"


def latest_alert(events: list[Event], now: datetime | None = None) -> Event | None:
    now = now or datetime.now(timezone.utc)
    for e in reversed(events):
        if _is_alert(e) and (now - e.timestamp).total_seconds() <= ALERT_BANNER_SECONDS:
            return e
    return None
