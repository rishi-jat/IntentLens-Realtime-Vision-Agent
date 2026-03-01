"""
IntentLens — TrackingManager.

Single responsibility: maintain DeepSORT tracker, convert raw detections into
confirmed track IDs with bounding boxes and associated keypoints.
Handles duplicate suppression and bbox area filtering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from numpy.typing import NDArray

from config import (
    TRACKER_DUPLICATE_IOU_THRESHOLD,
    TRACKER_MAX_AGE,
    TRACKER_MAX_IOU_DISTANCE,
    TRACKER_MIN_BBOX_AREA,
    TRACKER_N_INIT,
    TRACKER_NMS_MAX_OVERLAP,
)
from models import DetectedObject, PersonKeypoints

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ConfirmedTrack:
    """Output of the tracking stage — one per confirmed person."""

    track_id: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    keypoints: Optional[PersonKeypoints] = None


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Compute Intersection-over-Union between two (x1, y1, x2, y2) boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = _bbox_area(a)
    area_b = _bbox_area(b)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


class TrackingManager:
    """Wraps DeepSORT with duplicate suppression and area filtering."""

    def __init__(self) -> None:
        logger.info(
            "Initialising DeepSORT (max_age=%d, n_init=%d, max_iou=%.2f)",
            TRACKER_MAX_AGE,
            TRACKER_N_INIT,
            TRACKER_MAX_IOU_DISTANCE,
        )
        self._tracker = DeepSort(
            max_age=TRACKER_MAX_AGE,
            n_init=TRACKER_N_INIT,
            max_iou_distance=TRACKER_MAX_IOU_DISTANCE,
            nms_max_overlap=TRACKER_NMS_MAX_OVERLAP,
        )

    def update(
        self,
        person_poses: Sequence[PersonKeypoints],
        frame: NDArray[np.uint8],
    ) -> list[ConfirmedTrack]:
        """Feed person pose detections and return confirmed, deduplicated tracks.

        Parameters
        ----------
        person_poses:
            Person detections with keypoints from VisionProcessor.
        frame:
            The BGR frame (passed to DeepSORT for feature extraction).

        Returns
        -------
        List of ``ConfirmedTrack`` with unique track IDs and associated keypoints.
        """
        # Pre-filter tiny bboxes
        filtered = [
            p for p in person_poses if _bbox_area(p.bbox) >= TRACKER_MIN_BBOX_AREA
        ]

        # Format for DeepSORT: ([left, top, w, h], confidence, class_name)
        ds_input: list[tuple[list[float], float, str]] = []
        for pp in filtered:
            x1, y1, x2, y2 = pp.bbox
            # Use average keypoint confidence as detection confidence
            avg_conf = sum(k.confidence for k in pp.keypoints) / max(len(pp.keypoints), 1)
            # Floor confidence at 0.50 to match YOLO threshold
            ds_input.append(([x1, y1, x2 - x1, y2 - y1], max(avg_conf, 0.50), "person"))

        raw_tracks = self._tracker.update_tracks(ds_input, frame=frame)

        confirmed: list[ConfirmedTrack] = []
        for track in raw_tracks:
            if not track.is_confirmed():
                continue

            # STABILISATION: Only include tracks that have a current-frame
            # detection match (IoU > 0.25). This prevents ghost/stale tracks
            # from being counted when the person is no longer visible.
            ltrb = track.to_ltrb().tolist()
            bbox = (round(ltrb[0], 2), round(ltrb[1], 2), round(ltrb[2], 2), round(ltrb[3], 2))
            if _bbox_area(bbox) < TRACKER_MIN_BBOX_AREA:
                continue

            # Associate keypoints: find the detection with highest IoU to this track
            best_kp: Optional[PersonKeypoints] = None
            best_iou = 0.0
            for pp in filtered:
                overlap = _iou(bbox, pp.bbox)
                if overlap > best_iou:
                    best_iou = overlap
                    best_kp = pp

            # Require minimum IoU match with a current detection to avoid ghosts
            if best_iou < 0.25:
                logger.debug(
                    "Dropping stale track %d (best IoU=%.2f, no current detection match)",
                    track.track_id,
                    best_iou,
                )
                continue

            confirmed.append(ConfirmedTrack(track_id=track.track_id, bbox=bbox, keypoints=best_kp))

        # Duplicate suppression: if two confirmed tracks overlap heavily, keep
        # the one with the lower (older) ID — it's more established.
        deduplicated = self._suppress_duplicates(confirmed)

        if len(person_poses) != len(deduplicated):
            logger.info(
                "Tracking: %d YOLO → %d area-ok → %d raw → %d IoU-matched → %d deduped (ghost-filtered: %d)",
                len(person_poses),
                len(filtered),
                len(raw_tracks),
                len(confirmed),
                len(deduplicated),
                len(confirmed) - len(deduplicated),
            )
        else:
            logger.debug(
                "Tracking: %d YOLO → %d confirmed → %d final",
                len(person_poses),
                len(confirmed),
                len(deduplicated),
            )
        return deduplicated

    @staticmethod
    def _suppress_duplicates(tracks: list[ConfirmedTrack]) -> list[ConfirmedTrack]:
        if len(tracks) <= 1:
            return tracks

        keep: list[bool] = [True] * len(tracks)
        for i in range(len(tracks)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(tracks)):
                if not keep[j]:
                    continue
                if _iou(tracks[i].bbox, tracks[j].bbox) > TRACKER_DUPLICATE_IOU_THRESHOLD:
                    # Suppress the newer (higher) track ID
                    if tracks[i].track_id < tracks[j].track_id:
                        keep[j] = False
                        logger.debug(
                            "Suppressed duplicate track %d (overlaps %d)",
                            tracks[j].track_id,
                            tracks[i].track_id,
                        )
                    else:
                        keep[i] = False
                        logger.debug(
                            "Suppressed duplicate track %d (overlaps %d)",
                            tracks[i].track_id,
                            tracks[j].track_id,
                        )

        return [t for t, k in zip(tracks, keep) if k]
