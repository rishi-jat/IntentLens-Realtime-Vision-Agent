"""
IntentLens — SceneGraph builder.

Aggregates all per-person structured data (tracking, behaviour, attributes,
gestures) into a single ``SceneGraph`` object that is the sole input to the
LLM reasoning layer.  No raw bbox data reaches the LLM — everything is
structured and labelled.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from behavior_analyzer import BehaviorSignals
from gesture_detector import Gesture
from models import (
    DetectedObject,
    PersonAttributes,
    SceneGraph,
    SceneGraphPerson,
    SceneState,
    TrackedPerson,
)

logger = logging.getLogger(__name__)


def _classify_activity(people: Sequence[TrackedPerson]) -> str:
    """Classify overall scene activity level."""
    n = len(people)
    if n == 0:
        return "quiet"

    fast_count = sum(1 for p in people if p.velocity_label in ("moderate", "fast"))
    ratio = fast_count / n

    if n >= 5 or ratio > 0.5:
        return "busy"
    if n >= 2 or ratio > 0.2:
        return "moderate"
    return "quiet"


def _classify_motion_intensity(person: TrackedPerson) -> str:
    """Classify per-person motion intensity from velocity."""
    v = person.velocity
    if v < 3.0:
        return "none"
    if v < 20.0:
        return "low"
    if v < 60.0:
        return "moderate"
    return "high"


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Compute IoU between two (x1, y1, x2, y2) boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _detect_object_in_hand(
    person: TrackedPerson,
    objects: Sequence[DetectedObject],
    iou_threshold: float = 0.05,
    wrist_region_px: float = 60.0,
) -> Optional[str]:
    """Check if any non-person object bbox overlaps a wrist region bbox.

    Creates a square region around each wrist keypoint and computes IoU
    with each object's bounding box. This is more robust than simple
    distance-based proximity.

    Returns the label of the best-overlapping object, or None.
    """
    kps = person.keypoints
    if kps is None:
        return None

    left_wrist = kps.get("left_wrist")
    right_wrist = kps.get("right_wrist")

    wrist_regions: list[tuple[float, float, float, float]] = []
    if left_wrist and left_wrist.confidence > 0.4:
        half = wrist_region_px / 2
        wrist_regions.append((
            left_wrist.x - half, left_wrist.y - half,
            left_wrist.x + half, left_wrist.y + half,
        ))
    if right_wrist and right_wrist.confidence > 0.4:
        half = wrist_region_px / 2
        wrist_regions.append((
            right_wrist.x - half, right_wrist.y - half,
            right_wrist.x + half, right_wrist.y + half,
        ))

    if not wrist_regions or not objects:
        return None

    best_label: Optional[str] = None
    best_iou = iou_threshold

    for obj in objects:
        for wrist_box in wrist_regions:
            overlap = _iou(wrist_box, obj.bbox)
            if overlap > best_iou:
                best_iou = overlap
                best_label = obj.label

    return best_label


class SceneGraphBuilder:
    """Builds a SceneGraph from perception + analysis outputs."""

    def build(
        self,
        scene: SceneState,
        person_signals: list[tuple[TrackedPerson, BehaviorSignals]],
        person_gestures: dict[int, list[Gesture]],
        person_attributes: dict[int, PersonAttributes],
    ) -> SceneGraph:
        """Construct a fully structured SceneGraph.

        Parameters
        ----------
        scene : current SceneState snapshot
        person_signals : list of (TrackedPerson, BehaviorSignals) tuples
        person_gestures : map of track_id → list of active gestures
        person_attributes : map of track_id → extracted attributes

        Returns
        -------
        A frozen SceneGraph for LLM consumption.
        """
        persons: list[SceneGraphPerson] = []

        for person, signals in person_signals:
            attrs = person_attributes.get(person.track_id)
            gestures = person_gestures.get(person.track_id, [])

            # Object-in-hand detection
            held_object = _detect_object_in_hand(person, scene.objects)

            # Compute detection confidence from average keypoint confidence
            det_confidence = 0.5  # base confidence for tracked person
            if person.keypoints is not None:
                valid_confs = [
                    k.confidence for k in person.keypoints.keypoints
                    if k.confidence > 0.1 and (k.x > 0 or k.y > 0)
                ]
                if valid_confs:
                    det_confidence = round(sum(valid_confs) / len(valid_confs), 3)

            persons.append(
                SceneGraphPerson(
                    person_id=person.track_id,
                    zone=person.zone,
                    dwell_time=person.dwell_time,
                    velocity_label=person.velocity_label,
                    posture=attrs.posture if attrs else "upright",
                    dominant_color=attrs.dominant_color if attrs else "unknown",
                    head_tilt=attrs.head_tilt if attrs else "neutral",
                    gaze_direction=attrs.gaze_direction if attrs else "forward",
                    gesture_state=[g.kind for g in gestures],
                    is_pacing=signals.is_pacing,
                    is_loitering=signals.is_loitering,
                    zone_diversity=signals.zone_diversity,
                    repeated_approaches=person.repeated_approaches,
                    confidence=det_confidence,
                    object_in_hand=held_object,
                    motion_intensity=_classify_motion_intensity(person),
                )
            )

        object_labels = [obj.label for obj in scene.objects[:15]]

        graph = SceneGraph(
            timestamp=scene.timestamp,
            total_persons=len(persons),
            activity_level=_classify_activity(scene.people),
            persons=persons,
            objects=object_labels,
        )

        logger.debug(
            "SceneGraph built: %d persons, activity=%s",
            graph.total_persons,
            graph.activity_level,
        )
        return graph
