import json

import pytest

from conftest import square
from restricted_zone_monitor.zone_manager import (
    Zone,
    ZoneError,
    ZoneType,
    load_zones,
    make_zone_id,
    point_in_polygon,
    save_zones,
    to_normalized,
    to_pixels,
    validate_polygon,
)

TRIANGLE = [(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)]
# An L-shape: concave, so a naive bounding-box test would be wrong for (0.75, 0.75).
L_SHAPE = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)]


class TestValidation:
    def test_valid_polygons(self):
        assert validate_polygon(TRIANGLE) == []
        assert validate_polygon(L_SHAPE) == []

    def test_too_few_vertices(self):
        assert "at least 3" in validate_polygon([(0.1, 0.1), (0.2, 0.2)])[0]

    def test_out_of_range(self):
        errors = validate_polygon([(0.1, 0.1), (1.5, 0.1), (0.5, 0.9)])
        assert any("0..1" in e for e in errors)

    def test_collinear_points_have_no_area(self):
        assert any("zero area" in e for e in validate_polygon([(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)]))

    def test_self_intersecting_bow_tie(self):
        bow_tie = [(0.1, 0.1), (0.9, 0.9), (0.9, 0.1), (0.1, 0.9)]
        assert any("cross" in e for e in validate_polygon(bow_tie))

    def test_duplicate_vertices(self):
        assert any("duplicate" in e for e in validate_polygon([(0.1, 0.1), (0.1, 0.1), (0.9, 0.1), (0.5, 0.9)]))

    def test_zone_constructor_rejects_invalid_polygon(self):
        with pytest.raises(ZoneError):
            Zone("bad", "Bad", ZoneType.RESTRICTED, [(0.1, 0.1), (0.2, 0.2)])

    def test_loitering_zone_needs_threshold(self):
        with pytest.raises(ZoneError, match="loiter_seconds"):
            Zone("room", "Room", ZoneType.LOITERING, TRIANGLE)

    def test_zone_id_format(self):
        with pytest.raises(ZoneError, match="lowercase"):
            Zone("Bad Id", "Room", ZoneType.RESTRICTED, TRIANGLE)


class TestPointInPolygon:
    @pytest.mark.parametrize("point", [(0.5, 0.3), (0.5, 0.11), (0.3, 0.2)])
    def test_inside_triangle(self, point):
        assert point_in_polygon(point, TRIANGLE)

    @pytest.mark.parametrize("point", [(0.05, 0.05), (0.95, 0.5), (0.5, 0.95), (0.15, 0.8)])
    def test_outside_triangle(self, point):
        assert not point_in_polygon(point, TRIANGLE)

    @pytest.mark.parametrize("point", [(0.5, 0.1), (0.1, 0.1), (0.3, 0.5), (0.9, 0.1)])
    def test_boundary_counts_as_inside(self, point):
        # edge midpoint, vertex, point on a slanted edge, another vertex
        assert point_in_polygon(point, TRIANGLE)

    def test_concave_polygon(self):
        assert point_in_polygon((0.25, 0.25), L_SHAPE)
        assert point_in_polygon((0.75, 0.75), L_SHAPE)
        assert not point_in_polygon((0.75, 0.25), L_SHAPE)  # the notch

    def test_ray_through_vertex_is_counted_once(self):
        # The horizontal ray from (0.2, 0.5) passes exactly through the vertex (0.5, 0.5) of the L.
        assert point_in_polygon((0.2, 0.5), L_SHAPE)
        diamond = [(0.5, 0.1), (0.9, 0.5), (0.5, 0.9), (0.1, 0.5)]
        assert point_in_polygon((0.5, 0.5), diamond)
        assert not point_in_polygon((0.05, 0.5), diamond)  # ray passes through two vertices

    def test_zone_contains_uses_bounding_box_and_polygon(self, restricted_zone):
        assert restricted_zone.contains((0.25, 0.25))
        assert restricted_zone.contains((0.5, 0.5))  # corner
        assert not restricted_zone.contains((0.51, 0.25))


class TestCoordinates:
    def test_round_trip(self):
        pts = [(100, 50), (640, 360), (1279, 719)]
        norm = to_normalized(pts, 1280, 720)
        assert to_pixels(norm, 1280, 720) == pts

    def test_same_zone_at_different_resolutions(self):
        """A point at the same relative position must give the same answer at any frame size."""
        zone = Zone("z", "Z", ZoneType.RESTRICTED, square(0.25, 0.25, 0.75, 0.75))
        for w, h in [(320, 240), (1280, 720), (1920, 1080), (1024, 576)]:
            inside_px = (0.5 * w, 0.6 * h)
            outside_px = (0.8 * w, 0.6 * h)
            assert zone.contains(to_normalized([inside_px], w, h)[0])
            assert not zone.contains(to_normalized([outside_px], w, h)[0])

    def test_display_click_maps_back_to_source(self):
        # A click on a 960-wide display of a 1920x1080 frame at (480, 270) is the frame centre.
        assert to_normalized([(480, 270)], 960, 540)[0] == (0.5, 0.5)
        assert to_pixels([(0.5, 0.5)], 1920, 1080)[0] == (960, 540)

    def test_clicks_outside_image_are_clamped(self):
        assert to_normalized([(-5, 900)], 640, 480)[0] == (0.0, 1.0)


class TestPersistence:
    def test_save_and_load_round_trip(self, tmp_path, restricted_zone, loitering_zone, occupancy_zone):
        path = tmp_path / "zones.json"
        save_zones(path, [restricted_zone, loitering_zone, occupancy_zone])
        loaded = load_zones(path)
        assert [z.zone_id for z in loaded] == ["machinery_area", "equipment_room", "warehouse_entrance"]
        assert loaded[1].zone_type is ZoneType.LOITERING
        assert loaded[1].loiter_seconds == 5.0
        assert loaded[0].polygon == restricted_zone.polygon
        assert not (tmp_path / "zones.json.tmp").exists()

    def test_missing_file_means_no_zones(self, tmp_path):
        assert load_zones(tmp_path / "nope.json") == []

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "zones.json"
        path.write_text("{not json")
        with pytest.raises(ZoneError, match="not valid JSON"):
            load_zones(path)

    def test_invalid_polygon_in_file_is_rejected(self, tmp_path):
        path = tmp_path / "zones.json"
        path.write_text(json.dumps({"zones": [{"id": "a", "name": "A", "type": "restricted",
                                               "polygon": [[0.1, 0.1], [0.2, 0.2]]}]}))
        with pytest.raises(ZoneError, match="at least 3"):
            load_zones(path)

    def test_unknown_type_and_missing_field(self, tmp_path):
        path = tmp_path / "zones.json"
        path.write_text(json.dumps({"zones": [{"id": "a", "name": "A", "type": "forbidden", "polygon": TRIANGLE}]}))
        with pytest.raises(ZoneError, match="Unknown zone type"):
            load_zones(path)
        path.write_text(json.dumps({"zones": [{"id": "a", "type": "restricted", "polygon": TRIANGLE}]}))
        with pytest.raises(ZoneError, match="missing"):
            load_zones(path)

    def test_duplicate_ids_rejected(self, tmp_path, restricted_zone):
        with pytest.raises(ZoneError, match="Duplicate"):
            save_zones(tmp_path / "z.json", [restricted_zone, restricted_zone])

    def test_shipped_example_zones_are_valid(self):
        from restricted_zone_monitor.config import PROJECT_ROOT
        zones = load_zones(PROJECT_ROOT / "data" / "zones" / "default.json")
        assert {z.zone_type for z in zones} == set(ZoneType)


def test_make_zone_id():
    assert make_zone_id("Machinery Area", set()) == "machinery_area"
    assert make_zone_id("Machinery Area", {"machinery_area"}) == "machinery_area_2"
    assert make_zone_id("!!!", set()) == "zone"
