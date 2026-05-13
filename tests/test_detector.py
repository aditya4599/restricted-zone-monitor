import pytest

from restricted_zone_monitor.config import PROJECT_ROOT, ModelConfig
from restricted_zone_monitor.detector import (
    Detector,
    DetectorError,
    ModelNotFoundError,
    check_model_path,
    resolve_class_ids,
)

COCO_SUBSET = {0: "person", 2: "car", 7: "truck"}


def test_missing_model_file_fails_without_download(tmp_path):
    with pytest.raises(ModelNotFoundError, match="not downloaded automatically"):
        check_model_path(tmp_path / "yolov8n.pt")
    with pytest.raises(ModelNotFoundError):
        Detector(ModelConfig(path=tmp_path / "missing.pt"))
    assert not (tmp_path / "missing.pt").exists()


def test_class_names_resolved_to_ids():
    assert resolve_class_ids(("person",), COCO_SUBSET) == [0]
    assert resolve_class_ids(("truck", "person"), COCO_SUBSET) == [7, 0]


def test_unknown_class_lists_available_ones():
    with pytest.raises(DetectorError, match="forklift.*Available: car, person, truck"):
        resolve_class_ids(("forklift",), COCO_SUBSET)


def test_device_selection(monkeypatch):
    torch = pytest.importorskip("torch")
    from restricted_zone_monitor.detector import select_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert select_device("auto") == "cpu"
    with pytest.raises(DetectorError, match="CUDA"):
        select_device("cuda")
    with pytest.raises(DetectorError, match="MPS"):
        select_device("mps")

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert select_device("auto") == "mps"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert select_device("auto") == "cuda"
    assert select_device("cpu") == "cpu"


def test_bundled_weights_present():
    assert (PROJECT_ROOT / "models" / "yolov8n.pt").stat().st_size > 1_000_000
