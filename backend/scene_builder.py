"""
IntentLens — SceneStateBuilder.

Single responsibility: take raw VisionProcessor + TrackingManager outputs
and combine them into a single ``SceneState`` snapshot, enriching tracked
persons with spatial zone info, velocity via SessionMemory, and pose keypoints.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from config import (
    KEYPOINT_CONFIDENCE_THRESHOLD,
    VELOCITY_MODERATE,
    VELOCITY_SLOW,
    VELOCITY_STATIONARY,
    ZONE_GRID_COLS,
    ZONE_GRID_ROWS,
)
from memory import SessionMemory
from models import (
    DetectedObject,
    PersonAttributes,
    PersonKeypoints,
    SceneState,
    TrackedPerson,
)
from scene_attribute_engine import SceneAttributeEngine
from tracking import ConfirmedTrack

logger = logging.getLogger(__name__)

_attribute_engine = SceneAttributeEngine()


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _compute_zone(cx: float, cy: float, frame_w: int, frame_h: int) -> str:
    col = min(int(cx / (frame_w / ZONE_GRID_COLS)), ZONE_GRID_COLS - 1)
    row = min(int(cy / (frame_h / ZONE_GRID_ROWS)), ZONE_GRID_ROWS - 1)
    return f"zone_{row}_{col}"


def _classify_velocity(speed: float) -> str:
    if speed < VELOCITY_STATIONARY:
        return "stationary"
    if speed < VELOCITY_SLOW:
        return "slow"
    if speed < VELOCITY_MODERATE:
        return "moderate"
    return "fast"


class SceneStateBuilder:
    """Assembles a ``SceneState`` from perception + tracking outputs."""

    def build(
        self,
        tracks: Sequence[ConfirmedTrack],
        objects: Sequence[DetectedObject],
        frame_width: int,
        frame_height: int,
        timestamp: float,
        memory: SessionMemory,
        frame: Optional[NDArray[np.uint8]] = None,
    ) -> SceneState:
        """Construct a fully enriched SceneState.

        For each confirmed track the builder:
        1. Computes zone from bbox centre
        2. Calculates instantaneous velocity from memory
        3. Updates memory (zone entry, re-entry, velocity event)
        4. Derives dwell time from memory first_seen
        5. Attaches keypoints and extracts attributes

        Parameters
        ----------
        tracks : confirmed tracker output (with keypoints)
        objects : non-person detected objects
        frame_width, frame_height : current frame dimensions
        timestamp : current frame timestamp
        memory : session memory (mutated in place)
        frame : BGR image for attribute extraction (optional)

        Returns
        -------
        A frozen ``SceneState`` snapshot.
        """
        people: list[TrackedPerson] = []

        for track in tracks:
            cx, cy = _bbox_center(track.bbox)
            zone = _compute_zone(cx, cy, frame_width, frame_height)

            mem = memory.get_or_create_person(track.track_id, timestamp)
            mem.last_seen = timestamp

            # Keypoints from tracking (must be assigned before velocity block uses it)
            kp: Optional[PersonKeypoints] = track.keypoints

            # Velocity — hybrid: use joint-based (wrist) if available, else bbox center.
            # Smoothed with exponential moving average (alpha=0.4) to prevent spikes.
            EMA_ALPHA = 0.4
            velocity: float = 0.0
            velocity_label: str = "stationary"
            if mem.last_center is not None and mem.last_timestamp is not None:
                dt = timestamp - mem.last_timestamp
                if dt > 0:
                    # Primary: bbox center velocity
                    dx = cx - mem.last_center[0]
                    dy = cy - mem.last_center[1]
                    bbox_vel = math.sqrt(dx * dx + dy * dy) / dt

                    # Secondary: joint-based velocity from wrists (more accurate for gestures)
                    joint_vel = None
                    if kp is not None:
                        lw = kp.get("left_wrist")
                        rw = kp.get("right_wrist")
                        joint_vels = []
                        if lw and lw.confidence >= KEYPOINT_CONFIDENCE_THRESHOLD and mem.last_left_wrist is not None:
                            jdx = lw.x - mem.last_left_wrist[0]
                            jdy = lw.y - mem.last_left_wrist[1]
                            joint_vels.append(math.sqrt(jdx * jdx + jdy * jdy) / dt)
                        if rw and rw.confidence >= KEYPOINT_CONFIDENCE_THRESHOLD and mem.last_right_wrist is not None:
                            jdx = rw.x - mem.last_right_wrist[0]
                            jdy = rw.y - mem.last_right_wrist[1]
                            joint_vels.append(math.sqrt(jdx * jdx + jdy * jdy) / dt)
                        if joint_vels:
                            joint_vel = max(joint_vels)  # Use max wrist velocity

                    # Blend: prefer joint velocity when available, but cap wild outliers
                    raw_vel = joint_vel if joint_vel is not None else bbox_vel
                    raw_vel = min(raw_vel, 500.0)  # Hard cap at 500 px/s to reject noise

                    # EMA smoothing
                    velocity = EMA_ALPHA * raw_vel + (1 - EMA_ALPHA) * mem.velocity_ema
                    mem.velocity_ema = velocity
                    velocity_label = _classify_velocity(velocity)

            # Store wrist positions for next frame
            if kp is not None:
                lw = kp.get("left_wrist")
                rw = kp.get("right_wrist")
                mem.last_left_wrist = (lw.x, lw.y) if lw and lw.confidence >= KEYPOINT_CONFIDENCE_THRESHOLD else None
                mem.last_right_wrist = (rw.x, rw.y) if rw and rw.confidence >= KEYPOINT_CONFIDENCE_THRESHOLD else None
            else:
                mem.last_left_wrist = None
                mem.last_right_wrist = None

            # Update memory
            memory.update_person_zone(mem, zone, timestamp)
            memory.add_velocity_event(mem, velocity_label, timestamp)
            mem.last_center = (cx, cy)
            mem.last_timestamp = timestamp

            dwell_time = round(timestamp - mem.first_seen, 2)

            # Extract structured attributes if we have keypoints and frame
            attrs: Optional[PersonAttributes] = None
            if kp is not None and frame is not None:
                try:
                    attrs = _attribute_engine.extract(kp, frame)
                except Exception as exc:
                    logger.debug("Attribute extraction failed for track %d: %s", track.track_id, exc)

            people.append(
                TrackedPerson(
                    track_id=track.track_id,
                    bbox=track.bbox,
                    center=(round(cx, 2), round(cy, 2)),
                    velocity=round(velocity, 2),
                    velocity_label=velocity_label,
                    zone=zone,
                    dwell_time=dwell_time,
                    zones_entered=list(mem.zones_entered),
                    repeated_approaches=mem.repeated_approaches,
                    timeline=list(mem.timeline),
                    keypoints=kp,
                    attributes=attrs,
                )
            )

        scene = SceneState(
            timestamp=timestamp,
            frame_width=frame_width,
            frame_height=frame_height,
            people=people,
            objects=list(objects),
        )

        memory.store_snapshot(scene)

        logger.debug(
            "SceneState built: %d people, %d objects",
            len(people),
            len(objects),
        )
        return scene
