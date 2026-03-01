"""
IntentLens — VisionProcessor.

Single responsibility: load YOLOv8-pose model once and convert raw frames
into a list of ``DetectedObject`` with optional pose keypoints.
No tracking, no behaviour, no LLM.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO

from config import (
    KEYPOINT_CONFIDENCE_THRESHOLD,
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_MODEL_NAME,
    YOLO_PERSON_CLASS_ID,
)
from models import DetectedObject, Keypoint, PersonKeypoints

logger = logging.getLogger(__name__)


class VisionProcessor:
    """Loads a YOLOv8-pose model once and exposes detection + pose methods."""

    def __init__(self, model_name: str = YOLO_MODEL_NAME) -> None:
        logger.info("Loading YOLO model: %s", model_name)
        self._model: YOLO = YOLO(model_name)
        logger.info("YOLO model loaded successfully")

    def detect(
        self,
        frame: NDArray[np.uint8],
        confidence_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
        person_only: bool = False,
    ) -> list[DetectedObject]:
        """Run inference and return structured ``DetectedObject`` list.

        Parameters
        ----------
        frame:
            BGR uint8 image (H x W x 3).
        confidence_threshold:
            Minimum confidence to include a detection.
        person_only:
            When *True* only COCO ``person`` (class 0) detections are kept.

        Returns
        -------
        Immutable list of ``DetectedObject`` instances.
        """
        results = self._model(frame, verbose=False)

        detections: list[DetectedObject] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                conf: float = float(boxes.conf[i])
                if conf < confidence_threshold:
                    continue

                cls_id: int = int(boxes.cls[i])
                if person_only and cls_id != YOLO_PERSON_CLASS_ID:
                    continue

                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                class_name: str = self._model.names.get(cls_id, str(cls_id))

                detections.append(
                    DetectedObject(
                        label=class_name,
                        confidence=round(conf, 4),
                        bbox=(round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)),
                        class_id=cls_id,
                    )
                )

        logger.debug("VisionProcessor detected %d objects", len(detections))
        return detections

    def detect_persons_with_pose(
        self,
        frame: NDArray[np.uint8],
        confidence_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
    ) -> list[PersonKeypoints]:
        """Detect persons and extract 17-point COCO pose keypoints.

        Uses the YOLOv8-pose model's native keypoint output.
        Returns one PersonKeypoints per detected person.
        """
        results = self._model(frame, verbose=False)
        person_poses: list[PersonKeypoints] = []

        for result in results:
            boxes = result.boxes
            kpts_data = result.keypoints  # ultralytics Keypoints object

            if boxes is None or kpts_data is None:
                continue

            for i in range(len(boxes)):
                conf: float = float(boxes.conf[i])
                if conf < confidence_threshold:
                    continue

                # Pose model only detects persons, but check just in case
                cls_id: int = int(boxes.cls[i])
                if cls_id != YOLO_PERSON_CLASS_ID:
                    continue

                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                bbox = (round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2))

                # Extract 17 keypoints: xy coords + confidence
                xy = kpts_data.xy[i]  # shape (17, 2)
                kp_conf = kpts_data.conf[i] if kpts_data.conf is not None else None  # shape (17,)

                keypoints: list[Keypoint] = []
                for j in range(17):
                    kx = float(xy[j][0])
                    ky = float(xy[j][1])
                    kc = float(kp_conf[j]) if kp_conf is not None else 0.0
                    keypoints.append(Keypoint(x=round(kx, 2), y=round(ky, 2), confidence=round(kc, 3)))

                person_poses.append(PersonKeypoints(bbox=bbox, keypoints=keypoints))

        logger.debug("VisionProcessor detected %d persons with pose", len(person_poses))
        return person_poses

    def detect_persons(
        self,
        frame: NDArray[np.uint8],
        confidence_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
    ) -> list[DetectedObject]:
        """Convenience: detect only persons."""
        return self.detect(frame, confidence_threshold=confidence_threshold, person_only=True)

    def detect_all(
        self,
        frame: NDArray[np.uint8],
        confidence_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
    ) -> tuple[list[PersonKeypoints], list[DetectedObject]]:
        """Detect all: persons with pose + non-person objects.

        Returns
        -------
        (person_poses, objects) — person poses with keypoints, other objects.
        """
        results = self._model(frame, verbose=False)
        person_poses: list[PersonKeypoints] = []
        objects: list[DetectedObject] = []

        for result in results:
            boxes = result.boxes
            kpts_data = result.keypoints

            if boxes is None:
                continue

            for i in range(len(boxes)):
                conf: float = float(boxes.conf[i])
                if conf < confidence_threshold:
                    continue

                cls_id: int = int(boxes.cls[i])
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                bbox_t = (round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2))

                if cls_id == YOLO_PERSON_CLASS_ID and kpts_data is not None:
                    # Person with pose
                    xy = kpts_data.xy[i]
                    kp_conf = kpts_data.conf[i] if kpts_data.conf is not None else None

                    keypoints: list[Keypoint] = []
                    for j in range(17):
                        kx = float(xy[j][0])
                        ky = float(xy[j][1])
                        kc = float(kp_conf[j]) if kp_conf is not None else 0.0
                        keypoints.append(Keypoint(x=round(kx, 2), y=round(ky, 2), confidence=round(kc, 3)))

                    person_poses.append(PersonKeypoints(bbox=bbox_t, keypoints=keypoints))
                elif cls_id != YOLO_PERSON_CLASS_ID:
                    class_name = self._model.names.get(cls_id, str(cls_id))
                    objects.append(
                        DetectedObject(
                            label=class_name,
                            confidence=round(conf, 4),
                            bbox=bbox_t,
                            class_id=cls_id,
                        )
                    )

        logger.debug("VisionProcessor: %d person poses, %d objects", len(person_poses), len(objects))
        return person_poses, objects

