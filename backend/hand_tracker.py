"""
IntentLens — Hand Tracking Module (MediaPipe Hands).

Provides 21-point hand landmarks for precise gesture recognition.
Replaces body-level gesture heuristics with true hand-level detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# MediaPipe Hands configuration
_HAND_DETECTION_CONFIDENCE = 0.6
_HAND_TRACKING_CONFIDENCE = 0.5
_MAX_HANDS = 2


@dataclass(frozen=True)
class HandLandmark:
    """Single hand landmark point (21 points per hand)."""
    x: float  # normalized [0, 1]
    y: float  # normalized [0, 1]
    z: float  # depth (relative)
    

@dataclass(frozen=True)
class HandDetection:
    """Detected hand with 21 landmarks."""
    landmarks: list[HandLandmark]  # 21 points
    handedness: str  # "Left" or "Right"
    confidence: float
    

@dataclass(frozen=True)
class HandGesture:
    """Recognized hand gesture."""
    name: str  # "open_palm", "fist", "raised_hand", "wave", "pointing"
    confidence: float
    hand: str  # "Left" or "Right"


class HandTracker:
    """MediaPipe-based hand tracker for precise gesture detection."""
    
    def __init__(self):
        """Initialize MediaPipe Hands model."""
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=_MAX_HANDS,
            min_detection_confidence=_HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=_HAND_TRACKING_CONFIDENCE,
        )
        
        # Temporal validation buffers
        self._gesture_history: dict[str, list[str]] = {}  # hand -> list of recent gestures
        self._history_window = 6  # frames to accumulate
        
        logger.info("HandTracker initialized with MediaPipe Hands")
    
    def detect_hands(self, frame: NDArray[np.uint8]) -> list[HandDetection]:
        """Detect hands in frame and return 21-point landmarks per hand.
        
        Parameters
        ----------
        frame : BGR image from OpenCV
        
        Returns
        -------
        List of HandDetection objects (up to 2 hands)
        """
        # MediaPipe requires RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        
        if not results.multi_hand_landmarks or not results.multi_handedness:
            return []
        
        detections: list[HandDetection] = []
        
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Extract 21 landmarks
            landmarks = [
                HandLandmark(x=lm.x, y=lm.y, z=lm.z)
                for lm in hand_landmarks.landmark
            ]
            
            hand_label = handedness.classification[0].label  # "Left" or "Right"
            hand_conf = handedness.classification[0].score
            
            detections.append(
                HandDetection(
                    landmarks=landmarks,
                    handedness=hand_label,
                    confidence=hand_conf,
                )
            )
        
        return detections
    
    def recognize_gestures(
        self,
        hands: list[HandDetection],
        frame_width: int,
        frame_height: int,
    ) -> list[HandGesture]:
        """Recognize gestures from hand landmarks with temporal validation.
        
        Parameters
        ----------
        hands : Detected hands with landmarks
        frame_width : Frame width in pixels
        frame_height : Frame height in pixels
        
        Returns
        -------
        List of recognized gestures (only those with high confidence + temporal validation)
        """
        gestures: list[HandGesture] = []
        
        for hand in hands:
            key = hand.handedness
            
            # Detect gesture from current frame
            current_gesture = self._classify_hand_gesture(hand, frame_width, frame_height)
            
            if current_gesture:
                # Temporal validation: accumulate history
                if key not in self._gesture_history:
                    self._gesture_history[key] = []
                
                self._gesture_history[key].append(current_gesture.name)
                
                # Keep only recent history
                if len(self._gesture_history[key]) > self._history_window:
                    self._gesture_history[key].pop(0)
                
                # Require gesture to appear in at least 4 of last 6 frames
                recent = self._gesture_history[key]
                if len(recent) >= 4 and recent.count(current_gesture.name) >= 4:
                    gestures.append(current_gesture)
        
        return gestures
    
    def _classify_hand_gesture(
        self,
        hand: HandDetection,
        frame_w: int,
        frame_h: int,
    ) -> Optional[HandGesture]:
        """Classify gesture from 21 hand landmarks.
        
        Detects:
        - open_palm: all fingers extended
        - fist: all fingers curled
        - raised_hand: hand elevated above shoulder
        - wave: wrist oscillation (requires multi-frame)
        - pointing: index extended, others curled
        """
        lm = hand.landmarks
        
        # Key indices:
        # 0 = wrist
        # 4 = thumb tip, 8 = index tip, 12 = middle tip, 16 = ring tip, 20 = pinky tip
        # 5 = index base, 9 = middle base, 13 = ring base, 17 = pinky base
        
        wrist = lm[0]
        thumb_tip = lm[4]
        index_tip = lm[8]
        index_mcp = lm[5]
        middle_tip = lm[12]
        middle_mcp = lm[9]
        ring_tip = lm[16]
        ring_mcp = lm[13]
        pinky_tip = lm[20]
        pinky_mcp = lm[17]
        
        # Finger extension check: tip is significantly higher than base
        def is_extended(tip: HandLandmark, base: HandLandmark) -> bool:
            return tip.y < base.y - 0.03  # y decreases upward
        
        fingers_extended = [
            is_extended(index_tip, index_mcp),
            is_extended(middle_tip, middle_mcp),
            is_extended(ring_tip, ring_mcp),
            is_extended(pinky_tip, pinky_mcp),
        ]
        
        num_extended = sum(fingers_extended)
        
        # Open palm: all 4 fingers extended
        if num_extended == 4:
            return HandGesture(name="open_palm", confidence=0.9, hand=hand.handedness)
        
        # Fist: no fingers extended
        if num_extended == 0:
            return HandGesture(name="fist", confidence=0.85, hand=hand.handedness)
        
        # Pointing: only index extended
        if fingers_extended == [True, False, False, False]:
            return HandGesture(name="pointing", confidence=0.8, hand=hand.handedness)
        
        # Raised hand: wrist elevated (y < 0.3 means upper region of frame)
        if wrist.y < 0.3:
            return HandGesture(name="raised_hand", confidence=0.75, hand=hand.handedness)
        
        return None
    
    def cleanup(self):
        """Release MediaPipe resources."""
        if hasattr(self, '_hands'):
            self._hands.close()
