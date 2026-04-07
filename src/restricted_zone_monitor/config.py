"""Loading and validating the YAML configuration.

Every setting has a default here, so a config file only needs the values it
changes. Unknown keys are rejected so a typo like ``confidance`` fails loudly
instead of being silently ignored.

Relative paths in the config are resolved against the project root (the
directory that contains ``main.py``), not the current working directory.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"

BUILTIN_TRACKERS = ("bytetrack.yaml", "botsort.yaml")
DEVICES = ("auto", "cpu", "cuda", "mps")
ANCHORS = ("bottom_center", "center")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


class ConfigError(ValueError):
    """Raised when the configuration file is missing, malformed or invalid."""


@dataclass(frozen=True)
class ModelConfig:
    path: Path
    device: str = "auto"
    confidence: float = 0.25
    iou: float = 0.5
    classes: tuple[str, ...] = ("person",)
    image_size: int = 640
    half: bool = False


@dataclass(frozen=True)
class TrackingConfig:
    tracker: str = "bytetrack.yaml"
    entry_confirm_seconds: float = 0.3
    exit_grace_seconds: float = 1.5
    track_timeout_seconds: float = 5.0


@dataclass(frozen=True)
class VideoConfig:
    source: int | str = 0
    width: int | None = None
    height: int | None = None
    process_every_n_frames: int = 1


@dataclass(frozen=True)
class ZonesConfig:
    file: Path
    anchor: str = "bottom_center"


@dataclass(frozen=True)
class StorageConfig:
    database: Path
    export_dir: Path


@dataclass(frozen=True)
class DisplayConfig:
    width: int = 960
    panel_width: int = 320
    recent_events: int = 8


@dataclass(frozen=True)
class PerformanceConfig:
    warmup_frames: int = 10
    window_size: int = 120


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    file: Path | None = None


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig
    tracking: TrackingConfig
    video: VideoConfig
    zones: ZonesConfig
    storage: StorageConfig
    display: DisplayConfig
    performance: PerformanceConfig
    logging: LoggingConfig


def load_config(path: str | Path | None = None, base_dir: Path = PROJECT_ROOT) -> AppConfig:
    """Load a config file (or the defaults when ``path`` is None) and validate it."""
    raw: dict[str, Any] = {}
    if path is not None:
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"Config file not found: {path}")
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"Could not parse {path}: {exc}") from exc
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"{path} must contain a mapping of sections, got {type(loaded).__name__}")
        raw = loaded
    return parse_config(raw, base_dir)


def parse_config(raw: dict[str, Any], base_dir: Path = PROJECT_ROOT) -> AppConfig:
    sections = {f.name for f in fields(AppConfig)}
    unknown = set(raw) - sections
    if unknown:
        raise ConfigError(f"Unknown config section(s): {', '.join(sorted(unknown))}")

    def section(name: str, cls: type) -> dict[str, Any]:
        values = raw.get(name) or {}
        if not isinstance(values, dict):
            raise ConfigError(f"Section '{name}' must be a mapping")
        allowed = {f.name for f in fields(cls)}
        bad = set(values) - allowed
        if bad:
            raise ConfigError(f"Unknown key(s) in '{name}': {', '.join(sorted(bad))}")
        return values

    def resolve(value: str | Path) -> Path:
        p = Path(value).expanduser()
        return p if p.is_absolute() else (base_dir / p)

    m = section("model", ModelConfig)
    classes = m.get("classes", ["person"])
    if isinstance(classes, str):
        classes = [classes]
    model = ModelConfig(
        path=resolve(m.get("path", "models/yolov8n.pt")),
        device=_choice("model.device", str(m.get("device", "auto")).lower(), DEVICES),
        confidence=_fraction("model.confidence", m.get("confidence", 0.25)),
        iou=_fraction("model.iou", m.get("iou", 0.5)),
        classes=_class_list(classes),
        image_size=_image_size(m.get("image_size", 640)),
        half=_bool("model.half", m.get("half", False)),
    )

    t = section("tracking", TrackingConfig)
    tracking = TrackingConfig(
        tracker=_tracker(t.get("tracker", "bytetrack.yaml"), resolve),
        entry_confirm_seconds=_seconds("tracking.entry_confirm_seconds", t.get("entry_confirm_seconds", 0.3)),
        exit_grace_seconds=_seconds("tracking.exit_grace_seconds", t.get("exit_grace_seconds", 1.5)),
        track_timeout_seconds=_seconds("tracking.track_timeout_seconds", t.get("track_timeout_seconds", 5.0)),
    )
    if tracking.track_timeout_seconds < tracking.exit_grace_seconds:
        # A track must not be forgotten before its presences have had the chance
        # to expire, otherwise the engine would lose the exit event.
        raise ConfigError("tracking.track_timeout_seconds must be >= tracking.exit_grace_seconds")

    v = section("video", VideoConfig)
    source = v.get("source", 0)
    if isinstance(source, bool) or not isinstance(source, (int, str)) or source == "":
        raise ConfigError("video.source must be a camera index or a file path / stream URL")
    if isinstance(source, str) and not source.isdigit() and "://" not in source:
        source = str(resolve(source))
    video = VideoConfig(
        source=source,
        width=_optional_positive_int("video.width", v.get("width")),
        height=_optional_positive_int("video.height", v.get("height")),
        process_every_n_frames=_positive_int("video.process_every_n_frames", v.get("process_every_n_frames", 1)),
    )

    z = section("zones", ZonesConfig)
    zones = ZonesConfig(
        file=resolve(z.get("file", "data/zones/default.json")),
        anchor=_choice("zones.anchor", z.get("anchor", "bottom_center"), ANCHORS),
    )

    s = section("storage", StorageConfig)
    storage = StorageConfig(
        database=resolve(s.get("database", "data/events.db")),
        export_dir=resolve(s.get("export_dir", "reports")),
    )

    d = section("display", DisplayConfig)
    display = DisplayConfig(
        width=_positive_int("display.width", d.get("width", 960)),
        panel_width=_positive_int("display.panel_width", d.get("panel_width", 320)),
        recent_events=_positive_int("display.recent_events", d.get("recent_events", 8)),
    )

    p = section("performance", PerformanceConfig)
    performance = PerformanceConfig(
        warmup_frames=_non_negative_int("performance.warmup_frames", p.get("warmup_frames", 10)),
        window_size=_positive_int("performance.window_size", p.get("window_size", 120)),
    )

    lg = section("logging", LoggingConfig)
    log_file = lg.get("file")
    logging_cfg = LoggingConfig(
        level=_choice("logging.level", str(lg.get("level", "INFO")).upper(), LOG_LEVELS),
        file=resolve(log_file) if log_file else None,
    )

    return AppConfig(model, tracking, video, zones, storage, display, performance, logging_cfg)


# --- small validators -------------------------------------------------------

def _number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number (got {value!r})")
    return float(value)


def _fraction(name: str, value: Any) -> float:
    number = _number(name, value)
    if not 0.0 < number < 1.0:
        raise ConfigError(f"{name} must be between 0 and 1 (got {value})")
    return number


def _seconds(name: str, value: Any) -> float:
    number = _number(name, value)
    if number < 0:
        raise ConfigError(f"{name} must not be negative (got {value})")
    return number


def _int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer (got {value!r})")
    return value


def _positive_int(name: str, value: Any) -> int:
    number = _int(name, value)
    if number <= 0:
        raise ConfigError(f"{name} must be greater than 0 (got {value})")
    return number


def _non_negative_int(name: str, value: Any) -> int:
    number = _int(name, value)
    if number < 0:
        raise ConfigError(f"{name} must not be negative (got {value})")
    return number


def _optional_positive_int(name: str, value: Any) -> int | None:
    return None if value is None else _positive_int(name, value)


def _bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be true or false (got {value!r})")
    return value


def _choice(name: str, value: Any, options: tuple[str, ...]) -> str:
    if value not in options:
        raise ConfigError(f"{name} must be one of {', '.join(options)} (got {value!r})")
    return value


def _image_size(value: Any) -> int:
    size = _positive_int("model.image_size", value)
    # YOLOv8 downsamples by a factor of 32; Ultralytics would silently round
    # other values, which makes benchmark numbers confusing.
    if size % 32:
        raise ConfigError(f"model.image_size must be a multiple of 32 (got {size})")
    return size


def _class_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(c, str) and c for c in value):
        raise ConfigError("model.classes must be a non-empty list of class names, e.g. [person]")
    return tuple(value)


def _tracker(value: Any, resolve) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError("tracking.tracker must be a tracker YAML name or path")
    path = resolve(value)
    if path.is_file():
        return str(path)
    if value in BUILTIN_TRACKERS:
        return value
    raise ConfigError(f"tracking.tracker: {value!r} is neither an existing file nor one of {BUILTIN_TRACKERS}")
