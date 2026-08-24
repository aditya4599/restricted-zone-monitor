"""Zone editor logic, driven by synthetic mouse/key events (no window is opened)."""

import cv2
import numpy as np
import pytest

from restricted_zone_monitor.zone_editor import ZoneEditor
from restricted_zone_monitor.zone_manager import ZoneType, load_zones

ENTER, TAB = 13, 9


@pytest.fixture
def editor(tmp_path):
    frame = np.zeros((1080, 1920, 3), np.uint8)  # shown at 960x540: display = source / 2
    return ZoneEditor("test", frame, [], tmp_path / "zones.json", display_width=960)


def click(ed, x, y):
    ed._on_mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)


def type_text(ed, text):
    for ch in text:
        ed._handle_text_key(ord(ch))


def draw_zone(ed, points, name, type_key, seconds=None):
    for x, y in points:
        click(ed, x, y)
    ed._handle_key(ENTER)
    type_text(ed, name)
    ed._handle_text_key(ENTER)
    ed._handle_text_key(ord(type_key))
    if seconds is not None:
        ed._handle_text_key(8)  # clear the default "10"
        ed._handle_text_key(8)
        type_text(ed, seconds)
        ed._handle_text_key(ENTER)


def test_draw_name_type_and_save(editor, tmp_path):
    draw_zone(editor, [(96, 54), (480, 54), (480, 270), (96, 270)], "Machinery Area", "1")
    draw_zone(editor, [(500, 300), (900, 300), (700, 500)], "Equipment Room", "2", seconds="15")
    assert [z.zone_type for z in editor.zones] == [ZoneType.RESTRICTED, ZoneType.LOITERING]
    assert editor.zones[1].loiter_seconds == 15.0
    # Display clicks are stored normalized: (96, 54) on a 960x540 display = (0.1, 0.1) of the frame.
    assert editor.zones[0].polygon[0] == pytest.approx((0.1, 0.1))

    editor._handle_key(ord("s"))
    saved = load_zones(tmp_path / "zones.json")
    assert [z.zone_id for z in saved] == ["machinery_area", "equipment_room"]
    assert editor.saved and not editor.dirty


def test_incomplete_polygon_is_not_accepted(editor):
    click(editor, 10, 10)
    click(editor, 100, 10)
    editor._handle_key(ENTER)
    assert editor._text_mode is None
    assert "at least 3" in editor.message
    assert editor.zones == []


def test_self_intersecting_polygon_rejected(editor):
    for p in [(100, 100), (400, 400), (400, 100), (100, 400)]:
        click(editor, *p)
    editor._handle_key(ENTER)
    assert "cross" in editor.message and editor._text_mode is None


def test_right_click_undo_and_clear(editor):
    for p in [(10, 10), (100, 10), (100, 100)]:
        click(editor, *p)
    editor._on_mouse(cv2.EVENT_RBUTTONDOWN, 0, 0, 0, None)
    assert len(editor.draft) == 2
    editor._handle_key(ord("c"))
    assert editor.draft == []


def test_drag_vertex_and_invalid_drag_is_reverted(editor):
    draw_zone(editor, [(100, 100), (300, 100), (300, 300), (100, 300)], "Room", "3")
    assert editor.selected == 0
    editor._on_mouse(cv2.EVENT_LBUTTONDOWN, 300, 300, 0, None)  # grab the bottom-right vertex
    editor._on_mouse(cv2.EVENT_MOUSEMOVE, 480, 270, 0, None)
    editor._on_mouse(cv2.EVENT_LBUTTONUP, 480, 270, 0, None)
    assert editor.zones[0].polygon[2] == pytest.approx((0.5, 0.5))

    before = editor.zones[0].polygon
    # Dragging a vertex across the opposite edge would make the polygon cross itself.
    editor._on_mouse(cv2.EVENT_LBUTTONDOWN, 100, 100, 0, None)
    editor._on_mouse(cv2.EVENT_LBUTTONUP, 400, 500, 0, None)
    assert editor.zones[0].polygon == before
    assert editor.message.startswith("Not moved")


def test_select_retype_rename_delete(editor):
    draw_zone(editor, [(100, 100), (300, 100), (300, 300)], "A", "1")
    editor._handle_key(TAB)
    assert editor.selected is None  # tab cycles through "no selection"
    editor._handle_key(TAB)
    editor._handle_key(ord("t"))
    assert editor.zones[0].zone_type is ZoneType.LOITERING and editor.zones[0].loiter_seconds == 10.0
    editor._handle_key(ord("+"))
    assert editor.zones[0].loiter_seconds == 15.0
    editor._handle_key(ord("n"))
    editor._handle_text_key(8)
    type_text(editor, "Dock")
    editor._handle_text_key(ENTER)
    assert editor.zones[0].name == "Dock" and editor.zones[0].zone_id == "a"  # id stays stable
    editor._handle_key(ord("x"))
    assert editor.zones == []


def test_reset_needs_confirmation_and_quit_warns_about_unsaved(editor):
    draw_zone(editor, [(100, 100), (300, 100), (300, 300)], "A", "1")
    editor._handle_key(ord("R"))
    assert len(editor.zones) == 1
    editor._handle_key(ord("R"))
    assert editor.zones == []
    assert editor._handle_key(ord("q")) is True  # first q only warns
    assert editor._handle_key(ord("q")) is False


def test_escape_cancels_naming(editor):
    for p in [(100, 100), (300, 100), (300, 300)]:
        click(editor, *p)
    editor._handle_key(ENTER)
    editor._handle_text_key(27)
    assert editor._text_mode is None and editor.zones == []
    assert len(editor.draft) == 3  # the outline is kept so it can be finished again


def test_render_produces_display_sized_image(editor):
    draw_zone(editor, [(100, 100), (300, 100), (300, 300)], "A", "1")
    click(editor, 500, 100)
    assert editor._render().shape == (540, 960, 3)
