"""
IntentLens — SceneAttributeEngine.

Extracts structured per-person attributes from pose keypoints and frame crops:
- Posture (upright / leaning / crouching) from torso + leg angles
- Dominant shirt color from torso crop
- Head tilt from eye-line angle
- Gaze direction approximation from nose + eye geometry
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from config import KEYPOINT_CONFIDENCE_THRESHOLD
from models import Keypoint, PersonAttributes, PersonKeypoints

logger = logging.getLogger(__name__)

# COCO keypoint indices
_NOSE = 0
_LEFT_EYE = 1
_RIGHT_EYE = 2
_LEFT_EAR = 3
_RIGHT_EAR = 4
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
_LEFT_ANKLE = 15
_RIGHT_ANKLE = 16


def _valid(kp: Optional[Keypoint], threshold: float = KEYPOINT_CONFIDENCE_THRESHOLD) -> bool:
    """Check if a keypoint is valid (detected with sufficient confidence)."""
    return kp is not None and kp.confidence >= threshold and (kp.x > 0 or kp.y > 0)


def _midpoint(a: Keypoint, b: Keypoint) -> tuple[float, float]:
    """Midpoint between two keypoints."""
    return ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)


def _angle_deg(ax: float, ay: float, bx: float, by: float) -> float:
    """Angle in degrees of the vector from (ax,ay) to (bx,by) relative to vertical."""
    dx = bx - ax
    dy = by - ay
    return math.degrees(math.atan2(dx, -dy))  # 0° = straight up


class SceneAttributeEngine:
    """Extracts structured attributes from keypoints + frame crop."""

    def extract(
        self,
        keypoints: PersonKeypoints,
        frame: NDArray[np.uint8],
    ) -> PersonAttributes:
        """Extract attributes for a single person.

        Parameters
        ----------
        keypoints : PersonKeypoints with 17 COCO keypoints
        frame : BGR image for color extraction

        Returns
        -------
        PersonAttributes with posture, color, head tilt, gaze
        """
        posture = self._compute_posture(keypoints)
        dominant_color = self._extract_torso_color(keypoints, frame)
        head_tilt = self._compute_head_tilt(keypoints)
        gaze = self._compute_gaze(keypoints)

        return PersonAttributes(
            posture=posture,
            dominant_color=dominant_color,
            head_tilt=head_tilt,
            gaze_direction=gaze,
        )

    def _compute_posture(self, kp: PersonKeypoints) -> str:
        """Determine posture from shoulder-hip-knee geometry.

        - upright: torso roughly vertical (< 25° from vertical)
        - leaning: torso tilted (25°-50°)
        - crouching: hips close to knees vertically
        """
        ls = kp.keypoints[_LEFT_SHOULDER]
        rs = kp.keypoints[_RIGHT_SHOULDER]
        lh = kp.keypoints[_LEFT_HIP]
        rh = kp.keypoints[_RIGHT_HIP]
        lk = kp.keypoints[_LEFT_KNEE]
        rk = kp.keypoints[_RIGHT_KNEE]

        # Need at least one shoulder and one hip
        shoulders_valid = _valid(ls) or _valid(rs)
        hips_valid = _valid(lh) or _valid(rh)

        if not shoulders_valid or not hips_valid:
            return "upright"  # default when insufficient data

        # Shoulder midpoint
        if _valid(ls) and _valid(rs):
            shoulder_mid = _midpoint(ls, rs)
        elif _valid(ls):
            shoulder_mid = (ls.x, ls.y)
        else:
            shoulder_mid = (rs.x, rs.y)

        # Hip midpoint
        if _valid(lh) and _valid(rh):
            hip_mid = _midpoint(lh, rh)
        elif _valid(lh):
            hip_mid = (lh.x, lh.y)
        else:
            hip_mid = (rh.x, rh.y)

        # Check crouching: hip close to knee level
        if (_valid(lk) or _valid(rk)):
            knee_y = 0.0
            knee_count = 0
            if _valid(lk):
                knee_y += lk.y
                knee_count += 1
            if _valid(rk):
                knee_y += rk.y
                knee_count += 1
            knee_y /= max(knee_count, 1)

            torso_len = abs(hip_mid[1] - shoulder_mid[1])
            hip_knee_dist = abs(knee_y - hip_mid[1])

            # If hip-knee distance is small relative to torso → crouching
            if torso_len > 0 and hip_knee_dist / torso_len < 0.5:
                return "crouching"

        # Torso angle from vertical
        torso_angle = abs(_angle_deg(shoulder_mid[0], shoulder_mid[1], hip_mid[0], hip_mid[1]))
        if torso_angle > 50:
            return "crouching"
        if torso_angle > 25:
            return "leaning"
        return "upright"

    def _extract_torso_color(
        self,
        kp: PersonKeypoints,
        frame: NDArray[np.uint8],
    ) -> str:
        """Extract dominant color from the torso region (shoulder to hip crop)."""
        ls = kp.keypoints[_LEFT_SHOULDER]
        rs = kp.keypoints[_RIGHT_SHOULDER]
        lh = kp.keypoints[_LEFT_HIP]
        rh = kp.keypoints[_RIGHT_HIP]

        # Need at least one shoulder and one hip for torso crop
        if not ((_valid(ls) or _valid(rs)) and (_valid(lh) or _valid(rh))):
            # Fall back to center of bbox
            x1, y1, x2, y2 = kp.bbox
            crop_x1 = int(x1 + (x2 - x1) * 0.25)
            crop_x2 = int(x1 + (x2 - x1) * 0.75)
            crop_y1 = int(y1 + (y2 - y1) * 0.2)
            crop_y2 = int(y1 + (y2 - y1) * 0.6)
        else:
            # Use keypoints for precise torso crop
            xs = [p.x for p in [ls, rs, lh, rh] if _valid(p)]
            ys = [p.y for p in [ls, rs, lh, rh] if _valid(p)]
            crop_x1 = int(min(xs))
            crop_x2 = int(max(xs))
            crop_y1 = int(min(ys))
            crop_y2 = int(max(ys))

        h, w = frame.shape[:2]
        crop_x1 = max(0, crop_x1)
        crop_y1 = max(0, crop_y1)
        crop_x2 = min(w, crop_x2)
        crop_y2 = min(h, crop_y2)

        if crop_x2 - crop_x1 < 5 or crop_y2 - crop_y1 < 5:
            return "unknown"

        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        return self._classify_color(crop)

    @staticmethod
    def _classify_color(crop: NDArray[np.uint8]) -> str:
        """Classify dominant color using K-means clustering on torso crop.
        
        Returns color name only if confidence is high, otherwise "unknown".
        Uses K-means with k=3 to find dominant cluster, then names it.
        """
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return "unknown"
        
        # Reshape crop to list of pixels
        pixels = crop.reshape(-1, 3).astype(np.float32)
        
        # K-means clustering with k=3
        from sklearn.cluster import KMeans
        try:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10, max_iter=100)
            kmeans.fit(pixels)
            
            # Find largest cluster
            labels = kmeans.labels_
            unique, counts = np.unique(labels, return_counts=True)
            dominant_cluster_idx = unique[np.argmax(counts)]
            dominant_cluster_size = counts[np.argmax(counts)]
            
            # Confidence check: dominant cluster must be > 40% of pixels
            if dominant_cluster_size / len(labels) < 0.4:
                return "unknown"
            
            # Extract dominant color (BGR)
            dominant_bgr = kmeans.cluster_centers_[dominant_cluster_idx]
            
            # Convert to HSV for better color classification
            dominant_rgb = np.array([[dominant_bgr[::-1]]], dtype=np.uint8)  # BGR → RGB
            dominant_hsv = cv2.cvtColor(dominant_rgb, cv2.COLOR_RGB2HSV)[0][0]
            
            h, s, v = float(dominant_hsv[0]), float(dominant_hsv[1]), float(dominant_hsv[2])
            
            # Low saturation → achromatic
            if s < 50:
                if v < 80:
                    return "black"
                if v > 200:
                    return "white"
                return "gray"
            
            # Classify by hue (0-179 in OpenCV)
            if h < 10 or h > 170:
                return "red"
            if h < 25:
                return "orange"
            if h < 35:
                return "yellow"
            if h < 80:
                return "green"
            if h < 130:
                return "blue"
            if h < 155:
                return "purple"
            if h <= 170:
                return "pink"
            
            return "unknown"
        
        except Exception as e:
            logger.warning(f"K-means color extraction failed: {e}")
            return "unknown"

    def _compute_head_tilt(self, kp: PersonKeypoints) -> str:
        """Compute head tilt from eye-line angle."""
        le = kp.keypoints[_LEFT_EYE]
        re = kp.keypoints[_RIGHT_EYE]
        nose = kp.keypoints[_NOSE]

        if _valid(le) and _valid(re):
            # Angle of eye line from horizontal
            dx = re.x - le.x
            dy = re.y - le.y
            angle = math.degrees(math.atan2(dy, dx))

            if angle > 15:
                return "tilted_right"
            if angle < -15:
                return "tilted_left"
            return "neutral"

        # If only nose + one eye, estimate from vertical alignment
        if _valid(nose) and (_valid(le) or _valid(re)):
            eye = le if _valid(le) else re
            # If nose is well below eyes → looking down
            if nose.y - eye.y > 30:
                return "looking_down"

        return "neutral"

    def _compute_gaze(self, kp: PersonKeypoints) -> str:
        """Approximate gaze direction from nose position relative to eyes and ears."""
        nose = kp.keypoints[_NOSE]
        le = kp.keypoints[_LEFT_EYE]
        re = kp.keypoints[_RIGHT_EYE]
        l_ear = kp.keypoints[_LEFT_EAR]
        r_ear = kp.keypoints[_RIGHT_EAR]

        if not _valid(nose):
            return "forward"

        # Method 1: nose vs eye midpoint
        if _valid(le) and _valid(re):
            eye_mid_x = (le.x + re.x) / 2.0
            eye_span = abs(re.x - le.x)

            if eye_span > 5:  # enough separation
                nose_offset = (nose.x - eye_mid_x) / eye_span
                if nose_offset > 0.3:
                    return "right"
                if nose_offset < -0.3:
                    return "left"

            # Check for looking down
            eye_mid_y = (le.y + re.y) / 2.0
            if nose.y - eye_mid_y > eye_span * 1.5:
                return "down"

            return "forward"

        # Method 2: nose vs ear visibility
        if _valid(l_ear) and not _valid(r_ear):
            return "left"
        if _valid(r_ear) and not _valid(l_ear):
            return "right"

        return "forward"
