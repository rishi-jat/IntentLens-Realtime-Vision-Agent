"""
IntentLens — GestureDetector (v2 — Keypoint-based).

Precision gesture recognition using COCO pose keypoints with
temporal confirmation.  No false gestures.

Detectable gestures:
- raised_hand     : wrist above shoulder for 6+ consecutive frames
- waving          : wrist lateral oscillation with 3+ direction reversals
- crouching       : hip near knee level confirmed over 4+ frames
- rapid_movement  : velocity spike above threshold (from tracking data)
- pacing          : already flagged by BehaviorAnalyzer, surfaced here

Every gesture requires multi-frame temporal validation.
No single-frame triggers.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from config import KEYPOINT_CONFIDENCE_THRESHOLD, VELOCITY_MODERATE
from models import PersonKeypoints, TrackedPerson

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Minimum velocity (px/s) to flag as "rapid movement"
RAPID_MOVEMENT_THRESHOLD: float = 50.0

# Raised hand: wrist must be above shoulder for this many consecutive detections
RAISED_HAND_CONSECUTIVE_FRAMES: int = 6

# Waving: minimum direction reversals in the window
WAVE_MIN_REVERSALS: int = 3
WAVE_WINDOW_FRAMES: int = 12
WAVE_MIN_LATERAL_PX: float = 8.0

# Crouching: confirmed over this many frames
CROUCH_CONSECUTIVE_FRAMES: int = 4

# Cooldown: don't re-fire the same gesture within this many seconds
GESTURE_COOLDOWN_SECS: float = 6.0

# COCO keypoint indices
_LEFT_SHOULDER = 5
_RIGHT_SHOULDER = 6
_LEFT_ELBOW = 7
_RIGHT_ELBOW = 8
_LEFT_WRIST = 9
_RIGHT_WRIST = 10
_LEFT_HIP = 11
_RIGHT_HIP = 12
_LEFT_KNEE = 13
_RIGHT_KNEE = 14


@dataclass(frozen=True, slots=True)
class Gesture:
    """A single detected gesture."""

    kind: str          # "raised_hand" | "waving" | "crouching" | "rapid_movement" | "pacing"
    person_id: int
    confidence: float  # 0..1 heuristic confidence
    description: str   # human-readable sentence


def _kp_valid(kp_list: list, idx: int) -> bool:
    """Check if keypoint at index is valid."""
    if idx >= len(kp_list):
        return False
    kp = kp_list[idx]
    return kp.confidence >= KEYPOINT_CONFIDENCE_THRESHOLD and (kp.x > 0 or kp.y > 0)


@dataclass
class _PersonGestureState:
    """Rolling state for gesture detection on one person."""

    track_id: int
    # Temporal counters for multi-frame confirmation
    raised_hand_frames: int = 0
    crouch_frames: int = 0
    # Wrist x-position history for wave detection
    wrist_x_history: deque = field(default_factory=lambda: deque(maxlen=WAVE_WINDOW_FRAMES))
    # Rapid movement tracking
    high_velocity_frames: int = 0
    # Cooldowns
    last_gesture_times: dict[str, float] = field(default_factory=dict)


class GestureDetector:
    """Stateful per-person gesture detector using keypoint data + temporal confirmation."""

    def __init__(self) -> None:
        self._states: dict[int, _PersonGestureState] = {}

    def detect(
        self,
        person: TrackedPerson,
        timestamp: float,
        is_pacing: bool = False,
    ) -> list[Gesture]:
        """Analyse a tracked person's keypoints and return newly-detected gestures.

        Parameters
        ----------
        person : enriched TrackedPerson with keypoints from SceneStateBuilder
        timestamp : current frame timestamp
        is_pacing : from BehaviorAnalyzer
        """
        state = self._get_state(person.track_id)
        gestures: list[Gesture] = []

        kp = person.keypoints

        # --- Keypoint-based gestures (only if we have pose data) ---
        if kp is not None:
            kps = kp.keypoints

            # --- Raised hand (keypoint-based with temporal confirmation) ---
            hand_raised = self._check_raised_hand_frame(kps)
            if hand_raised:
                state.raised_hand_frames += 1
            else:
                state.raised_hand_frames = max(0, state.raised_hand_frames - 1)

            if state.raised_hand_frames >= RAISED_HAND_CONSECUTIVE_FRAMES:
                conf = min(1.0, state.raised_hand_frames / (RAISED_HAND_CONSECUTIVE_FRAMES * 2))
                g = self._maybe_fire(
                    state, "raised_hand", timestamp,
                    Gesture(
                        kind="raised_hand",
                        person_id=person.track_id,
                        confidence=conf,
                        description=f"Person {person.track_id} is raising their hand.",
                    ),
                )
                if g:
                    gestures.append(g)
                    state.raised_hand_frames = 0  # Reset after firing

            # --- Waving (wrist lateral oscillation) ---
            wrist_x = self._get_wrist_x(kps)
            if wrist_x is not None:
                state.wrist_x_history.append(wrist_x)

            if len(state.wrist_x_history) >= 6:
                g = self._check_waving(state, person, timestamp)
                if g:
                    gestures.append(g)

            # --- Crouching (keypoint-based) ---
            is_crouching = self._check_crouch_frame(kps)
            if is_crouching:
                state.crouch_frames += 1
            else:
                state.crouch_frames = max(0, state.crouch_frames - 1)

            if state.crouch_frames >= CROUCH_CONSECUTIVE_FRAMES:
                conf = min(1.0, state.crouch_frames / (CROUCH_CONSECUTIVE_FRAMES * 2))
                g = self._maybe_fire(
                    state, "crouching", timestamp,
                    Gesture(
                        kind="crouching",
                        person_id=person.track_id,
                        confidence=conf,
                        description=f"Person {person.track_id} appears to be crouching.",
                    ),
                )
                if g:
                    gestures.append(g)
                    state.crouch_frames = 0

        # --- Rapid movement (velocity-based, temporal) ---
        if person.velocity >= RAPID_MOVEMENT_THRESHOLD:
            state.high_velocity_frames += 1
        else:
            state.high_velocity_frames = max(0, state.high_velocity_frames - 1)

        if state.high_velocity_frames >= 3:
            g = self._maybe_fire(
                state, "rapid_movement", timestamp,
                Gesture(
                    kind="rapid_movement",
                    person_id=person.track_id,
                    confidence=min(1.0, person.velocity / (RAPID_MOVEMENT_THRESHOLD * 2)),
                    description=f"Person {person.track_id} just picked up speed — moving fast.",
                ),
            )
            if g:
                gestures.append(g)
                state.high_velocity_frames = 0

        # --- Pacing (pass-through from BehaviorAnalyzer) ---
        if is_pacing:
            g = self._maybe_fire(
                state, "pacing", timestamp,
                Gesture(
                    kind="pacing",
                    person_id=person.track_id,
                    confidence=0.8,
                    description=f"Person {person.track_id} is pacing back and forth.",
                ),
            )
            if g:
                gestures.append(g)

        return gestures

    def prune_stale(self, active_ids: set[int]) -> None:
        """Remove state for persons no longer tracked."""
        stale = [k for k in self._states if k not in active_ids]
        for k in stale:
            del self._states[k]

    # ------------------------------------------------------------------
    # Internal — keypoint checks (single-frame predicates)
    # ------------------------------------------------------------------

    @staticmethod
    def _check_raised_hand_frame(kps: list) -> bool:
        """Single-frame check: is either wrist above its shoulder AND elbow extended?"""
        for wrist_idx, elbow_idx, shoulder_idx in [
            (_LEFT_WRIST, _LEFT_ELBOW, _LEFT_SHOULDER),
            (_RIGHT_WRIST, _RIGHT_ELBOW, _RIGHT_SHOULDER),
        ]:
            if not (_kp_valid(kps, wrist_idx) and _kp_valid(kps, shoulder_idx)):
                continue

            wrist = kps[wrist_idx]
            shoulder = kps[shoulder_idx]

            # Wrist must be above shoulder (y decreases upward in image coords)
            if wrist.y >= shoulder.y:
                continue

            # Optional: check elbow is also elevated
            if _kp_valid(kps, elbow_idx):
                elbow = kps[elbow_idx]
                if elbow.y > shoulder.y:
                    continue  # Elbow below shoulder — hand not truly raised

            return True
        return False

    @staticmethod
    def _get_wrist_x(kps: list) -> Optional[float]:
        """Get the x-position of the most active (highest confidence) wrist."""
        best_x = None
        best_conf = 0.0
        for idx in [_LEFT_WRIST, _RIGHT_WRIST]:
            if _kp_valid(kps, idx) and kps[idx].confidence > best_conf:
                best_x = kps[idx].x
                best_conf = kps[idx].confidence
        return best_x

    @staticmethod
    def _check_crouch_frame(kps: list) -> bool:
        """Single-frame check: are hips close to knee level?"""
        hip_y_vals = []
        knee_y_vals = []
        shoulder_y_vals = []

        for idx in [_LEFT_HIP, _RIGHT_HIP]:
            if _kp_valid(kps, idx):
                hip_y_vals.append(kps[idx].y)
        for idx in [_LEFT_KNEE, _RIGHT_KNEE]:
            if _kp_valid(kps, idx):
                knee_y_vals.append(kps[idx].y)
        for idx in [_LEFT_SHOULDER, _RIGHT_SHOULDER]:
            if _kp_valid(kps, idx):
                shoulder_y_vals.append(kps[idx].y)

        if not hip_y_vals or not knee_y_vals or not shoulder_y_vals:
            return False

        avg_hip = sum(hip_y_vals) / len(hip_y_vals)
        avg_knee = sum(knee_y_vals) / len(knee_y_vals)
        avg_shoulder = sum(shoulder_y_vals) / len(shoulder_y_vals)

        torso_len = abs(avg_hip - avg_shoulder)
        hip_knee_dist = abs(avg_knee - avg_hip)

        if torso_len > 10 and hip_knee_dist < torso_len * 0.5:
            return True

        return False

    def _check_waving(
        self,
        state: _PersonGestureState,
        person: TrackedPerson,
        timestamp: float,
    ) -> Optional[Gesture]:
        """Check for wrist lateral oscillation with direction reversals."""
        xs = list(state.wrist_x_history)
        if len(xs) < 6:
            return None

        reversals = 0
        for i in range(2, len(xs)):
            d1 = xs[i - 1] - xs[i - 2]
            d2 = xs[i] - xs[i - 1]
            if abs(d1) >= WAVE_MIN_LATERAL_PX and abs(d2) >= WAVE_MIN_LATERAL_PX:
                if (d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0):
                    reversals += 1

        if reversals >= WAVE_MIN_REVERSALS:
            return self._maybe_fire(
                state, "waving", timestamp,
                Gesture(
                    kind="waving",
                    person_id=person.track_id,
                    confidence=min(1.0, reversals / (WAVE_MIN_REVERSALS * 2)),
                    description=f"Person {person.track_id} appears to be waving.",
                ),
            )
        return None

    # ------------------------------------------------------------------
    # Internal — state management
    # ------------------------------------------------------------------

    def _get_state(self, track_id: int) -> _PersonGestureState:
        if track_id not in self._states:
            self._states[track_id] = _PersonGestureState(track_id=track_id)
        return self._states[track_id]

    @staticmethod
    def _maybe_fire(
        state: _PersonGestureState,
        kind: str,
        timestamp: float,
        gesture: Gesture,
    ) -> Optional[Gesture]:
        """Fire gesture only if cooldown has elapsed."""
        last = state.last_gesture_times.get(kind, 0.0)
        if timestamp - last < GESTURE_COOLDOWN_SECS:
            return None
        state.last_gesture_times[kind] = timestamp
        logger.info("Gesture detected: %s for person %d", kind, state.track_id)
        return gesture
