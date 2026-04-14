"""Historical analytics over stored events.

Everything here is computed from events loaded from the database. The report
separates values that are *recorded* (counted straight from event rows, or a
field stored on an event) from values that are *calculated* (averages and
rates derived from those rows).
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from .event_manager import Event, EventType


@dataclass
class ZoneSummary:
    zone_id: str
    zone_name: str
    zone_type: str
    entries: int = 0
    exits: int = 0
    loitering_alerts: int = 0
    max_occupancy: int = 0
    durations: list[float] = field(default_factory=list, repr=False)

    @property
    def average_duration(self) -> float | None:
        return statistics.fmean(self.durations) if self.durations else None

    @property
    def median_duration(self) -> float | None:
        return statistics.median(self.durations) if self.durations else None


def summarize_zones(events: list[Event]) -> list[ZoneSummary]:
    """Per-zone totals, most frequently entered zones first."""
    zones: dict[str, ZoneSummary] = {}
    for e in events:
        s = zones.get(e.zone_id)
        if s is None:
            s = zones[e.zone_id] = ZoneSummary(e.zone_id, e.zone_name, e.zone_type)
        s.zone_name = e.zone_name  # keep the most recent name if a zone was renamed
        if e.event_type is EventType.ENTRY:
            s.entries += 1
        elif e.event_type is EventType.EXIT:
            s.exits += 1
            if e.duration_seconds is not None:
                s.durations.append(e.duration_seconds)
        elif e.event_type is EventType.LOITERING:
            s.loitering_alerts += 1
        s.max_occupancy = max(s.max_occupancy, e.occupancy)
    return sorted(zones.values(), key=lambda s: (-s.entries, s.zone_name))


def daily_summary(events: list[Event]) -> dict[date, Counter]:
    """Event counts per local calendar day and event type."""
    days: dict[date, Counter] = defaultdict(Counter)
    for e in events:
        days[e.timestamp.astimezone().date()][e.event_type.value] += 1
    return dict(sorted(days.items()))


def events_per_bucket(
    events: list[Event], bucket: str = "hour", event_type: EventType | None = EventType.ENTRY
) -> dict[datetime, int]:
    """Count events per local hour or day (only buckets that have events)."""
    counts: Counter = Counter()
    for e in events:
        if event_type is not None and e.event_type is not event_type:
            continue
        local = e.timestamp.astimezone()
        key = local.replace(minute=0, second=0, microsecond=0)
        if bucket == "day":
            key = key.replace(hour=0)
        counts[key] += 1
    return dict(sorted(counts.items()))


def format_report(events: list[Event], since: datetime | None = None, until: datetime | None = None) -> str:
    if not events:
        return "No events recorded for the selected period."

    lines: list[str] = []
    first = events[0].timestamp.astimezone()
    last = events[-1].timestamp.astimezone()
    period = f"{since.astimezone():%Y-%m-%d %H:%M} to {until.astimezone():%Y-%m-%d %H:%M}" if since and until else \
        f"{first:%Y-%m-%d %H:%M} to {last:%Y-%m-%d %H:%M}"
    sessions = len({e.session_id for e in events})
    lines.append(f"Events: {len(events)} across {sessions} session(s), {period} (local time)")
    lines.append("")

    summaries = summarize_zones(events)
    lines.append("Recorded per zone (counted from stored events)")
    header = f"  {'Zone':<24}{'Type':<11}{'Entries':>8}{'Exits':>7}{'Loiter':>8}{'Max occ.':>9}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for s in summaries:
        lines.append(f"  {s.zone_name[:23]:<24}{s.zone_type:<11}{s.entries:>8}{s.exits:>7}"
                     f"{s.loitering_alerts:>8}{s.max_occupancy:>9}")
    lines.append("")

    lines.append("Calculated per zone (derived from exit durations)")
    header = f"  {'Zone':<24}{'Visits':>7}{'Avg stay':>10}{'Median':>9}{'Longest':>9}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for s in summaries:
        if not s.durations:
            lines.append(f"  {s.zone_name[:23]:<24}{0:>7}{'-':>10}{'-':>9}{'-':>9}")
            continue
        lines.append(f"  {s.zone_name[:23]:<24}{len(s.durations):>7}{_secs(s.average_duration):>10}"
                     f"{_secs(s.median_duration):>9}{_secs(max(s.durations)):>9}")
    lines.append("")

    triggered = [s for s in summaries if s.entries]
    if triggered:
        total_entries = sum(s.entries for s in triggered)
        lines.append("Most frequently entered zones")
        for s in triggered[:5]:
            lines.append(f"  {s.zone_name:<24}{s.entries:>6} entries ({s.entries / total_entries:.0%})")
        lines.append("")

    lines.append("Daily summary (local dates)")
    types = [t.value for t in EventType]
    lines.append(f"  {'Date':<12}" + "".join(f"{t:>19}" for t in types))
    for day, counts in daily_summary(events).items():
        lines.append(f"  {day.isoformat():<12}" + "".join(f"{counts.get(t, 0):>19}" for t in types))
    return "\n".join(lines)


def _secs(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}s" if value < 120 else f"{value / 60:.1f}m"


def write_charts(events: list[Event], out_dir: Path, stamp: str) -> list[Path]:
    """Save PNG charts for entries per zone and entries over time. Returns the written paths."""
    summaries = [s for s in summarize_zones(events) if s.entries]
    if not summaries:
        return []

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    bar_color = "#3a6ea5"
    written: list[Path] = []

    def style(ax, title: str) -> None:
        ax.set_title(title, loc="left", fontsize=11)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("#999999")
        ax.spines["bottom"].set_color("#999999")
        ax.tick_params(colors="#444444", labelsize=9)

    fig, ax = plt.subplots(figsize=(7, 0.5 * len(summaries) + 1.2))
    names = [s.zone_name for s in reversed(summaries)]
    values = [s.entries for s in reversed(summaries)]
    ax.barh(names, values, color=bar_color, height=0.6)
    for y, v in enumerate(values):
        ax.text(v, y, f" {v}", va="center", fontsize=9, color="#333333")
    ax.xaxis.grid(True, color="#e5e5e5")
    ax.set_axisbelow(True)
    ax.set_xlabel("Entries")
    style(ax, "Entries per zone")
    fig.tight_layout()
    path = out_dir / f"entries_by_zone_{stamp}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(path)

    span_days = (events[-1].timestamp - events[0].timestamp).total_seconds() / 86400
    bucket = "day" if span_days > 2 else "hour"
    series = events_per_bucket(events, bucket)
    if series:
        fig, ax = plt.subplots(figsize=(8, 3.2))
        width = 0.8 if bucket == "day" else 0.8 / 24
        ax.bar(list(series.keys()), list(series.values()), width=width, color=bar_color)
        ax.yaxis.grid(True, color="#e5e5e5")
        ax.set_axisbelow(True)
        ax.set_ylabel("Entries")
        style(ax, f"Entries per {bucket} (all zones, local time)")
        fig.autofmt_xdate()
        fig.tight_layout()
        path = out_dir / f"entries_per_{bucket}_{stamp}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
    return written
