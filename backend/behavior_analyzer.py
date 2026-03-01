"""
IntentLens — BehaviorAnalyzer.

Single responsibility: examine a ``TrackedPerson`` and its memory summary
to produce heuristic *behavioural signals* — precursors for LLM reasoning.

No LLM calls here. Pure deterministic logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from models import TrackedPerson

logger = logging.getLogger(__name__)

# Dwell thresholds (seconds)
_SHORT_DWELL: float = 5.0
_MEDIUM_DWELL: float = 30.0
_LONG_DWELL: float = 120.0

# Repeated-approach threshold for flagging
_HIGH_REENTRY_COUNT: int = 3


@dataclass(frozen=True, slots=True)
class BehaviorSignals:
    """Heuristic behavioural signals derived from tracking + memory."""

    dwell_category: str  # "brief" | "short" | "medium" | "extended"
    is_pacing: bool
    is_loitering: bool
    movement_intensity: str  # "still" | "low" | "moderate" | "high"
    zone_diversity: int  # number of distinct zones visited
    reentry_flag: bool  # True if repeated_approaches >= threshold
    summary: str  # one-line human-readable summary


class BehaviorAnalyzer:
    """Produces deterministic behavioural signals from tracked-person state."""

    def analyze(self, person: TrackedPerson) -> BehaviorSignals:
        """Derive heuristic signals for a single person.

        Parameters
        ----------
        person : enriched TrackedPerson from SceneStateBuilder.

        Returns
        -------
        Frozen ``BehaviorSignals``.
        """
        dwell_cat = self._categorise_dwell(person.dwell_time)
        is_pacing = self._detect_pacing(person)
        is_loitering = dwell_cat in ("medium", "extended") and person.velocity_label == "stationary"
        movement = self._movement_intensity(person.velocity_label)
        zone_div = len(person.zones_entered)
        reentry_flag = person.repeated_approaches >= _HIGH_REENTRY_COUNT

        summary = self._build_summary(
            person, dwell_cat, is_pacing, is_loitering, reentry_flag
        )

        signals = BehaviorSignals(
            dwell_category=dwell_cat,
            is_pacing=is_pacing,
            is_loitering=is_loitering,
            movement_intensity=movement,
            zone_diversity=zone_div,
            reentry_flag=reentry_flag,
            summary=summary,
        )

        logger.debug(
            "BehaviorSignals track_id=%d: dwell=%s pacing=%s loiter=%s",
            person.track_id,
            dwell_cat,
            is_pacing,
            is_loitering,
        )
        return signals

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _categorise_dwell(dwell: float) -> str:
        if dwell < _SHORT_DWELL:
            return "brief"
        if dwell < _MEDIUM_DWELL:
            return "short"
        if dwell < _LONG_DWELL:
            return "medium"
        return "extended"

    @staticmethod
    def _detect_pacing(person: TrackedPerson) -> bool:
        """Heuristic: pacing = re-entered zones >= 2 times AND not stationary."""
        return (
            person.repeated_approaches >= 2
            and person.velocity_label not in ("stationary",)
        )

    @staticmethod
    def _movement_intensity(vel_label: str) -> str:
        mapping = {
            "stationary": "still",
            "slow": "low",
            "moderate": "moderate",
            "fast": "high",
        }
        return mapping.get(vel_label, "low")

    @staticmethod
    def _build_summary(
        person: TrackedPerson,
        dwell_cat: str,
        is_pacing: bool,
        is_loitering: bool,
        reentry_flag: bool,
    ) -> str:
        parts: list[str] = [
            f"Person #{person.track_id}",
            f"dwell={person.dwell_time:.1f}s ({dwell_cat})",
            f"movement={person.velocity_label}",
            f"zones={len(person.zones_entered)}",
        ]
        if is_pacing:
            parts.append("PACING")
        if is_loitering:
            parts.append("LOITERING")
        if reentry_flag:
            parts.append(f"HIGH_REENTRY({person.repeated_approaches})")
        return " | ".join(parts)
