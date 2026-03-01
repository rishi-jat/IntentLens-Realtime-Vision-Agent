"""
IntentLens — SessionMemory.

Rolling short-term memory for the scene.  Stores time-stamped SceneState
snapshots and per-person timeline events.  Designed as an injectable
dependency — no global variables.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from config import (
    MEMORY_DEFAULT_WINDOW_SECS,
    MEMORY_MAX_SNAPSHOTS,
    MEMORY_MAX_TIMELINE_EVENTS,
    REENTRY_COOLDOWN_SECS,
)
from models import SceneState, TrackedPerson

logger = logging.getLogger(__name__)


@dataclass
class _PersonMemory:
    """Internal mutable state for a single tracked person."""

    track_id: int
    first_seen: float
    last_seen: float
    zones_entered: list[str] = field(default_factory=list)
    repeated_approaches: int = 0
    timeline: list[str] = field(default_factory=list)
    last_zone: Optional[str] = None
    zone_exit_times: dict[str, float] = field(default_factory=dict)
    last_center: Optional[tuple[float, float]] = None
    last_timestamp: Optional[float] = None
    cached_risk: Optional[dict] = None
    last_llm_call: float = 0.0
    # Velocity EMA for smoothing (prevents noisy spikes)
    velocity_ema: float = 0.0
    # Previous wrist positions for joint-based velocity
    last_left_wrist: Optional[tuple[float, float]] = None
    last_right_wrist: Optional[tuple[float, float]] = None


class SessionMemory:
    """Injectable rolling memory for scene states and per-person history."""

    def __init__(
        self,
        max_snapshots: int = MEMORY_MAX_SNAPSHOTS,
        max_timeline_events: int = MEMORY_MAX_TIMELINE_EVENTS,
    ) -> None:
        self._snapshots: deque[SceneState] = deque(maxlen=max_snapshots)
        self._persons: dict[int, _PersonMemory] = {}
        self._max_timeline: int = max_timeline_events
        self._transcripts: deque[tuple[float, str]] = deque(maxlen=50)
        self._last_scene_summary: str = ""

    # ------------------------------------------------------------------
    # Snapshot storage
    # ------------------------------------------------------------------

    def store_snapshot(self, scene: SceneState) -> None:
        """Append a new scene snapshot to the rolling buffer."""
        self._snapshots.append(scene)

    def get_recent_window(self, seconds: float = MEMORY_DEFAULT_WINDOW_SECS) -> list[SceneState]:
        """Return snapshots from the last *seconds*."""
        if not self._snapshots:
            return []
        cutoff = self._snapshots[-1].timestamp - seconds
        return [s for s in self._snapshots if s.timestamp >= cutoff]

    # ------------------------------------------------------------------
    # Per-person state
    # ------------------------------------------------------------------

    def get_or_create_person(self, track_id: int, timestamp: float) -> _PersonMemory:
        """Return existing person memory or create a fresh one."""
        if track_id not in self._persons:
            self._persons[track_id] = _PersonMemory(
                track_id=track_id,
                first_seen=timestamp,
                last_seen=timestamp,
            )
        return self._persons[track_id]

    def update_person_zone(
        self,
        mem: _PersonMemory,
        zone: str,
        timestamp: float,
    ) -> None:
        """Update zone tracking, detect re-entry, record timeline events."""
        if zone == mem.last_zone:
            return

        # Re-entry detection
        if zone in mem.zone_exit_times:
            elapsed = timestamp - mem.zone_exit_times[zone]
            if elapsed > REENTRY_COOLDOWN_SECS:
                mem.repeated_approaches += 1
                self._add_timeline(
                    mem,
                    f"t={round(timestamp, 1)}: re-entered {zone} after {round(elapsed, 1)}s",
                )

        # Record exit from previous zone
        if mem.last_zone is not None:
            mem.zone_exit_times[mem.last_zone] = timestamp

        if zone not in mem.zones_entered:
            mem.zones_entered.append(zone)
            self._add_timeline(mem, f"t={round(timestamp, 1)}: entered {zone}")

        mem.last_zone = zone

    def add_velocity_event(
        self,
        mem: _PersonMemory,
        velocity_label: str,
        timestamp: float,
    ) -> None:
        """Record a velocity change event (skip if redundant)."""
        if velocity_label == "stationary":
            return
        tag = f"velocity: {velocity_label}"
        if mem.timeline and tag in mem.timeline[-1]:
            return
        self._add_timeline(mem, f"t={round(timestamp, 1)}: {tag}")

    def summarize_person(self, track_id: int) -> Optional[dict]:
        """Return a plain-dict summary for the given person, or None."""
        mem = self._persons.get(track_id)
        if mem is None:
            return None
        return {
            "track_id": mem.track_id,
            "dwell_time": round(mem.last_seen - mem.first_seen, 2),
            "zones_entered": list(mem.zones_entered),
            "repeated_approaches": mem.repeated_approaches,
            "timeline": list(mem.timeline[-self._max_timeline:]),
        }

    def prune_stale(self, current_time: float, timeout_secs: float) -> None:
        """Remove person memories not updated for *timeout_secs*."""
        stale = [
            pid
            for pid, pm in self._persons.items()
            if current_time - pm.last_seen > timeout_secs
        ]
        for pid in stale:
            logger.debug("Pruning stale person memory: track_id=%d", pid)
            del self._persons[pid]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _add_timeline(self, mem: _PersonMemory, event: str) -> None:
        mem.timeline.append(event)
        if len(mem.timeline) > self._max_timeline:
            mem.timeline = mem.timeline[-self._max_timeline:]

    # ------------------------------------------------------------------
    # Transcript memory
    # ------------------------------------------------------------------

    def store_transcript(self, timestamp: float, text: str) -> None:
        """Store a user voice transcript."""
        self._transcripts.append((timestamp, text))

    def get_recent_transcripts(self, seconds: float = 60.0) -> list[tuple[float, str]]:
        """Return transcripts from the last *seconds*."""
        if not self._transcripts:
            return []
        cutoff = self._transcripts[-1][0] - seconds
        return [(t, txt) for t, txt in self._transcripts if t >= cutoff]

    # ------------------------------------------------------------------
    # Scene context summary (for voice queries)
    # ------------------------------------------------------------------

    def build_context_summary(self) -> str:
        """Build a plain-text summary of current scene state for LLM context."""
        if not self._snapshots:
            return "No scene data available yet."

        latest = self._snapshots[-1]
        parts = [
            f"Current scene: {len(latest.people)} person(s) tracked, "
            f"{len(latest.objects)} other object(s).",
        ]

        for p in latest.people:
            person_mem = self._persons.get(p.track_id)
            dwell = p.dwell_time
            parts.append(
                f"  Person #{p.track_id}: zone={p.zone}, "
                f"velocity={p.velocity_label}, dwell={dwell:.1f}s, "
                f"zones_visited={len(p.zones_entered)}, "
                f"re-entries={p.repeated_approaches}"
            )
            if person_mem and person_mem.timeline:
                parts.append(f"    Recent: {'; '.join(person_mem.timeline[-4:])}")

        recent_tx = self.get_recent_transcripts(30.0)
        if recent_tx:
            parts.append("\nRecent conversation:")
            for t, txt in recent_tx[-5:]:
                parts.append(f"  User: {txt}")

        return "\n".join(parts)
