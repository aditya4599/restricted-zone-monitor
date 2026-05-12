import pytest

from restricted_zone_monitor.tracker import TrackObservation
from restricted_zone_monitor.zone_manager import Zone, ZoneType


def square(x0: float, y0: float, x1: float, y1: float) -> list[tuple[float, float]]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def obs(track_id: int, x: float, y: float, conf: float = 0.9, cls: str = "person") -> TrackObservation:
    return TrackObservation(track_id, cls, conf, (x, y))


@pytest.fixture
def restricted_zone() -> Zone:
    return Zone("machinery_area", "Machinery Area", ZoneType.RESTRICTED, square(0.0, 0.0, 0.5, 0.5))


@pytest.fixture
def loitering_zone() -> Zone:
    return Zone("equipment_room", "Equipment Room", ZoneType.LOITERING, square(0.5, 0.5, 1.0, 1.0), loiter_seconds=5.0)


@pytest.fixture
def occupancy_zone() -> Zone:
    return Zone("warehouse_entrance", "Warehouse Entrance", ZoneType.OCCUPANCY, square(0.0, 0.5, 0.5, 1.0))
