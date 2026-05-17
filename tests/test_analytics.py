from datetime import datetime, timedelta, timezone

from restricted_zone_monitor.analytics import (
    daily_summary,
    events_per_bucket,
    format_report,
    summarize_zones,
    write_charts,
)
from restricted_zone_monitor.event_manager import Event, EventType

T0 = datetime(2026, 10, 18, 9, 0, tzinfo=timezone.utc)


def ev(n, kind, zone="machinery_area", name="Machinery Area", occupancy=1, duration=None, minutes=0, track=1):
    return Event(f"e{n}", T0 + timedelta(minutes=minutes), "s1", zone, name, "restricted", kind, track,
                 "person", 0.9, occupancy, duration, None, None)


EVENTS = [
    ev(1, EventType.ENTRY, minutes=0),
    ev(2, EventType.ENTRY, occupancy=2, minutes=1, track=2),
    ev(3, EventType.EXIT, occupancy=1, duration=30.0, minutes=2),
    ev(4, EventType.EXIT, occupancy=0, duration=90.0, minutes=3, track=2),
    ev(5, EventType.ENTRY, zone="equipment_room", name="Equipment Room", minutes=70),
    ev(6, EventType.LOITERING, zone="equipment_room", name="Equipment Room", minutes=71, duration=10.0),
    ev(7, EventType.ENTRY, minutes=60 * 24 + 5),  # next day
]


def test_zone_summary_counts_and_derived_values():
    summaries = {s.zone_id: s for s in summarize_zones(EVENTS)}
    m = summaries["machinery_area"]
    assert (m.entries, m.exits, m.loitering_alerts, m.max_occupancy) == (3, 2, 0, 2)
    assert m.average_duration == 60.0 and m.median_duration == 60.0
    e = summaries["equipment_room"]
    assert (e.entries, e.loitering_alerts) == (1, 1)
    assert e.average_duration is None  # no completed visits, so no average


def test_most_triggered_zone_first():
    assert [s.zone_id for s in summarize_zones(EVENTS)] == ["machinery_area", "equipment_room"]


def test_entries_per_hour_and_day():
    per_hour = events_per_bucket(EVENTS, "hour")
    assert sum(per_hour.values()) == 4  # only entries are counted by default
    per_day = events_per_bucket(EVENTS, "day")
    assert len(per_day) >= 1 and sum(per_day.values()) == 4
    assert sum(events_per_bucket(EVENTS, "hour", event_type=None).values()) == len(EVENTS)


def test_daily_summary_totals():
    days = daily_summary(EVENTS)
    assert sum(sum(c.values()) for c in days.values()) == len(EVENTS)
    assert sum(c["zone_entry"] for c in days.values()) == 4


def test_report_text():
    text = format_report(EVENTS)
    assert "Recorded per zone" in text and "Calculated per zone" in text
    assert "Machinery Area" in text and "60.0s" in text
    assert "1.0m" not in text.split("Calculated")[0]  # durations only in the calculated section


def test_empty_data_is_handled(tmp_path):
    assert summarize_zones([]) == []
    assert format_report([]) == "No events recorded for the selected period."
    assert write_charts([], tmp_path, "x") == []


def test_charts_written(tmp_path):
    paths = write_charts(EVENTS, tmp_path, "test")
    assert [p.name for p in paths] == ["entries_by_zone_test.png", "entries_per_hour_test.png"]
    assert all(p.stat().st_size > 1000 for p in paths)
