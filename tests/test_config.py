from pathlib import Path

import pytest

from restricted_zone_monitor.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, ConfigError, load_config, parse_config


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


def test_defaults_without_file():
    cfg = load_config(None, base_dir=Path("/project"))
    assert cfg.model.path == Path("/project/models/yolov8n.pt")
    assert cfg.model.classes == ("person",)
    assert cfg.model.device == "auto"
    assert cfg.video.source == 0
    assert cfg.tracking.tracker == "bytetrack.yaml"
    assert cfg.logging.file is None


def test_shipped_default_config_loads():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert cfg.model.path == PROJECT_ROOT / "models" / "yolov8n.pt"
    assert cfg.tracking.tracker == str(PROJECT_ROOT / "config" / "bytetrack.yaml")
    assert cfg.zones.file == PROJECT_ROOT / "data" / "zones" / "default.json"


def test_valid_overrides(tmp_path):
    cfg = load_config(write(tmp_path, """
model:
  confidence: 0.6
  classes: [person, truck]
  device: CPU
video:
  source: clips/dock.mp4
  process_every_n_frames: 2
logging:
  level: debug
"""), base_dir=tmp_path)
    assert cfg.model.confidence == 0.6
    assert cfg.model.classes == ("person", "truck")
    assert cfg.model.device == "cpu"
    assert cfg.video.source == str(tmp_path / "clips" / "dock.mp4")
    assert cfg.video.process_every_n_frames == 2
    assert cfg.logging.level == "DEBUG"
    assert cfg.model.iou == 0.5  # untouched keys keep their defaults


def test_missing_optional_sections_and_empty_file(tmp_path):
    cfg = load_config(write(tmp_path, ""), base_dir=tmp_path)
    assert cfg.display.width == 960
    cfg = load_config(write(tmp_path, "model:\n"), base_dir=tmp_path)
    assert cfg.model.confidence == 0.25


def test_camera_index_and_stream_url_are_not_paths():
    assert parse_config({"video": {"source": 1}}).video.source == 1
    assert parse_config({"video": {"source": "rtsp://cam/1"}}).video.source == "rtsp://cam/1"


@pytest.mark.parametrize("raw, message", [
    ({"model": {"confidence": 1.5}}, "model.confidence must be between 0 and 1"),
    ({"model": {"confidence": "high"}}, "must be a number"),
    ({"model": {"iou": 0}}, "model.iou"),
    ({"model": {"image_size": 500}}, "multiple of 32"),
    ({"model": {"device": "tpu"}}, "model.device must be one of"),
    ({"model": {"classes": []}}, "non-empty list"),
    ({"model": {"half": "yes"}}, "true or false"),
    ({"tracking": {"exit_grace_seconds": -1}}, "must not be negative"),
    ({"tracking": {"exit_grace_seconds": 10, "track_timeout_seconds": 5}}, "track_timeout_seconds must be >="),
    ({"tracking": {"tracker": "missing.yaml"}}, "neither an existing file"),
    ({"video": {"process_every_n_frames": 0}}, "greater than 0"),
    ({"video": {"source": True}}, "video.source"),
    ({"zones": {"anchor": "top"}}, "zones.anchor"),
    ({"logging": {"level": "LOUD"}}, "logging.level"),
    ({"model": {"confidance": 0.5}}, "Unknown key(s) in 'model': confidance"),
    ({"modle": {}}, "Unknown config section"),
    ({"model": "yolov8n.pt"}, "must be a mapping"),
])
def test_invalid_values(raw, message):
    with pytest.raises(ConfigError, match=message.replace("(", r"\(").replace(")", r"\)")):
        parse_config(raw)


def test_missing_and_malformed_files(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
    with pytest.raises(ConfigError, match="Could not parse"):
        load_config(write(tmp_path, "model: [unclosed"))
    with pytest.raises(ConfigError, match="mapping of sections"):
        load_config(write(tmp_path, "- just\n- a list\n"))
