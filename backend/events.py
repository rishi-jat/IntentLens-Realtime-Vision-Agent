"""
IntentLens — Event system.

Produces structured events that drive the "alive" feeling:
- new_person       : first time a track ID appears
- person_departed  : person pruned from memory
- zone_breach      : person enters a monitored zone
- gesture          : gesture detected (raised hand, waving, etc.)
- risk_change      : scene or person risk level changed
- voice_input      : user spoke (transcript received)

Events are stored in a rolling buffer and included in API responses.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Maximum events retained
MAX_EVENTS: int = 100

# Alert zones — if person enters these zones, fire zone_breach
# Format: set of zone strings. Empty = no restricted zones by default.
RESTRICTED_ZONES: set[str] = {"zone_0_0", "zone_0_2"}  # top-left, top-right corners


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """A single event in the event stream."""

    kind: str           # event type
    timestamp: float
    message: str        # human-readable description (for AI to speak)
    person_id: Optional[int] = None
    severity: str = "info"  # "info" | "warning" | "alert"
    data: Optional[dict] = None
    speakable: bool = False  # if True, frontend should auto-speak this via TTS


class EventBus:
    """Accumulates events across frames and provides them to the API layer."""

    def __init__(self, max_events: int = MAX_EVENTS) -> None:
        self._events: deque[AgentEvent] = deque(maxlen=max_events)
        self._known_tracks: set[int] = set()
        self._person_zones: dict[int, str] = {}
        self._last_scene_risk: str = "low"
        self._person_risks: dict[int, str] = {}

    @property
    def recent_events(self) -> list[AgentEvent]:
        return list(self._events)

    def get_events_since(self, since: float) -> list[AgentEvent]:
        """Return events with timestamp > since."""
        return [e for e in self._events if e.timestamp > since]

    def clear(self) -> None:
        self._events.clear()

    # ------------------------------------------------------------------
    # Event producers
    # ------------------------------------------------------------------

    def check_new_persons(self, active_ids: set[int], timestamp: float) -> list[AgentEvent]:
        """Fire new_person events for IDs not previously seen."""
        events: list[AgentEvent] = []
        for tid in active_ids:
            if tid not in self._known_tracks:
                self._known_tracks.add(tid)
                total = len(active_ids)
                if total == 1:
                    msg = f"I see someone new. Tracking as Person {tid}."
                elif total == 2:
                    msg = f"Another person just appeared — now tracking {total} individuals. Welcome, Person {tid}."
                else:
                    msg = f"Person {tid} just entered. I'm now monitoring {total} people in the scene."
                ev = AgentEvent(
                    kind="new_person",
                    timestamp=timestamp,
                    message=msg,
                    person_id=tid,
                    severity="info",
                    speakable=True,
                )
                events.append(ev)
                self._events.append(ev)
                logger.info("Event: new_person #%d (total: %d)", tid, total)
        return events

    def check_departed(self, active_ids: set[int], timestamp: float) -> list[AgentEvent]:
        """Fire person_departed for IDs that disappeared."""
        events: list[AgentEvent] = []
        departed = self._known_tracks - active_ids
        for tid in departed:
            ev = AgentEvent(
                kind="person_departed",
                timestamp=timestamp,
                message=f"Person {tid} has left the scene.",
                person_id=tid,
                severity="info",
            )
            events.append(ev)
            self._events.append(ev)
            self._known_tracks.discard(tid)
            self._person_zones.pop(tid, None)
            self._person_risks.pop(tid, None)
            logger.info("Event: person_departed #%d", tid)
        return events

    def check_zone_breach(
        self,
        person_id: int,
        zone: Optional[str],
        timestamp: float,
    ) -> list[AgentEvent]:
        """Fire zone_breach if person enters a restricted zone."""
        events: list[AgentEvent] = []
        if zone is None:
            return events

        prev = self._person_zones.get(person_id)
        self._person_zones[person_id] = zone

        if zone != prev and zone in RESTRICTED_ZONES:
            ev = AgentEvent(
                kind="zone_breach",
                timestamp=timestamp,
                message=f"Heads up — Person {person_id} just entered a restricted zone.",
                person_id=person_id,
                severity="alert",
                data={"zone": zone},
                speakable=True,
            )
            events.append(ev)
            self._events.append(ev)
            logger.warning("Event: zone_breach #%d → %s", person_id, zone)
        return events

    def add_gesture_events(
        self,
        gestures: list,
        timestamp: float,
    ) -> list[AgentEvent]:
        """Convert Gesture objects into AgentEvents."""
        events: list[AgentEvent] = []
        for g in gestures:
            sev = "warning" if g.kind in ("rapid_movement", "crouching") else "info"
            ev = AgentEvent(
                kind=f"gesture_{g.kind}",
                timestamp=timestamp,
                message=g.description,
                person_id=g.person_id,
                severity=sev,
                data={"gesture": g.kind, "confidence": g.confidence},
                speakable=True,
            )
            events.append(ev)
            self._events.append(ev)
        return events

    def check_risk_change(
        self,
        scene_risk: str,
        person_risks: dict[int, str],
        timestamp: float,
    ) -> list[AgentEvent]:
        """Fire risk_change if scene or person risk level escalated."""
        events: list[AgentEvent] = []
        _order = {"low": 0, "medium": 1, "high": 2}

        # Scene-level
        if _order.get(scene_risk, 0) > _order.get(self._last_scene_risk, 0):
            ev = AgentEvent(
                kind="risk_change",
                timestamp=timestamp,
                message=f"Risk level is now {scene_risk}. Stay alert.",
                severity="alert" if scene_risk == "high" else "warning",
                data={"previous": self._last_scene_risk, "current": scene_risk},
                speakable=True,
            )
            events.append(ev)
            self._events.append(ev)
        self._last_scene_risk = scene_risk

        # Per-person
        for pid, risk in person_risks.items():
            prev = self._person_risks.get(pid, "low")
            if _order.get(risk, 0) > _order.get(prev, 0):
                ev = AgentEvent(
                    kind="risk_change",
                    timestamp=timestamp,
                    message=f"Person {pid} risk elevated to {risk}.",
                    person_id=pid,
                    severity="alert" if risk == "high" else "warning",
                    speakable=True,
                )
                events.append(ev)
                self._events.append(ev)
            self._person_risks[pid] = risk

        return events

    def add_voice_event(self, transcript: str, timestamp: float) -> AgentEvent:
        """Record a voice input event."""
        ev = AgentEvent(
            kind="voice_input",
            timestamp=timestamp,
            message=f'User said: "{transcript}"',
            severity="info",
            data={"transcript": transcript},
        )
        self._events.append(ev)
        logger.info("Event: voice_input — %s", transcript[:80])
        return ev
