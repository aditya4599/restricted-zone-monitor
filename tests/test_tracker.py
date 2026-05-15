from conftest import obs
from restricted_zone_monitor.tracker import TrackTable


def test_new_ids_create_tracks():
    table = TrackTable(timeout=2.0)
    table.update([obs(1, 0.1, 0.1), obs(2, 0.5, 0.5)], now=0.0)
    assert len(table) == 2
    assert table.get(1).first_seen == 0.0


def test_existing_track_is_updated_not_duplicated():
    table = TrackTable(timeout=2.0)
    table.update([obs(1, 0.1, 0.1, conf=0.5)], now=0.0)
    table.update([obs(1, 0.2, 0.3, conf=0.8)], now=0.1)
    track = table.get(1)
    assert len(table) == 1
    assert (track.first_seen, track.last_seen, track.hits) == (0.0, 0.1, 2)
    assert track.anchor == (0.2, 0.3) and track.confidence == 0.8


def test_disappeared_track_kept_until_timeout():
    table = TrackTable(timeout=2.0)
    table.update([obs(1, 0.1, 0.1)], now=0.0)
    table.update([], now=1.5)
    assert table.prune(now=1.5) == []
    assert 1 in table


def test_lost_track_recovered_with_same_id_continues():
    table = TrackTable(timeout=2.0)
    table.update([obs(5, 0.1, 0.1)], now=0.0)
    table.update([obs(5, 0.15, 0.1)], now=1.9)  # ByteTrack re-associated it after a gap
    assert table.prune(now=1.9) == []
    assert table.get(5).first_seen == 0.0 and table.get(5).hits == 2


def test_stale_tracks_removed():
    table = TrackTable(timeout=2.0)
    table.update([obs(1, 0.1, 0.1), obs(2, 0.2, 0.2)], now=0.0)
    table.update([obs(2, 0.2, 0.2)], now=2.5)
    assert table.prune(now=2.5) == [1]
    assert 1 not in table and 2 in table


def test_reused_id_after_timeout_starts_fresh():
    table = TrackTable(timeout=1.0)
    table.update([obs(3, 0.1, 0.1)], now=0.0)
    table.prune(now=5.0)
    table.update([obs(3, 0.9, 0.9)], now=5.0)
    assert table.get(3).first_seen == 5.0
