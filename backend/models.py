"""
IntentLens — Domain models.

Strongly-typed dataclasses consumed by every layer downstream of perception.
No raw YOLO output leaks past VisionProcessor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True, slots=True)
class DetectedObject:
    """A single YOLO detection, already filtered and normalised."""

    label: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    class_id: int


# ---------------------------------------------------------------------------
# Pose / Keypoint data
# ---------------------------------------------------------------------------

# COCO 17-keypoint indices
KEYPOINT_NAMES: list[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


@dataclass(frozen=True, slots=True)
class Keypoint:
    """A single pose keypoint."""
    x: float
    y: float
    confidence: float


@dataclass(frozen=True, slots=True)
class PersonKeypoints:
    """17 COCO keypoints for a detected person, plus their bbox."""
    bbox: tuple[float, float, float, float]
    keypoints: list[Keypoint]  # length 17, indexed by KEYPOINT_NAMES

    def get(self, name: str) -> Optional[Keypoint]:
        """Get a keypoint by name, returns None if not found or below threshold."""
        try:
            idx = KEYPOINT_NAMES.index(name)
            return self.keypoints[idx]
        except (ValueError, IndexError):
            return None


@dataclass(frozen=True, slots=True)
class PersonAttributes:
    """Structured attributes extracted from pose + frame crop."""
    posture: str  # "upright" | "leaning" | "crouching"
    dominant_color: str  # e.g. "red", "blue", "dark", "light"
    head_tilt: str  # "neutral" | "tilted_left" | "tilted_right" | "looking_down"
    gaze_direction: str  # "forward" | "left" | "right" | "down"


# ---------------------------------------------------------------------------
# Tracked person
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TrackedPerson:
    """Enriched representation of a person being tracked across frames."""

    track_id: int
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    velocity: float = 0.0
    velocity_label: str = "stationary"
    zone: Optional[str] = None
    dwell_time: float = 0.0
    zones_entered: list[str] = field(default_factory=list)
    repeated_approaches: int = 0
    timeline: list[str] = field(default_factory=list)
    keypoints: Optional[PersonKeypoints] = None
    attributes: Optional[PersonAttributes] = None


@dataclass(frozen=True, slots=True)
class SceneState:
    """Complete snapshot of a single analysed frame.

    All downstream reasoning (BehaviorAnalyzer, LLMReasoner) consumes this —
    never raw YOLO tensors.
    """

    timestamp: float
    frame_width: int
    frame_height: int
    people: list[TrackedPerson]
    objects: list[DetectedObject]


# ---------------------------------------------------------------------------
# Scene Graph (structured input for LLM)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SceneGraphPerson:
    """Structured per-person data for the scene graph consumed by LLM."""
    person_id: int
    zone: Optional[str]
    dwell_time: float
    velocity_label: str
    posture: str
    dominant_color: str
    head_tilt: str
    gaze_direction: str
    gesture_state: list[str]  # currently active gestures
    is_pacing: bool
    is_loitering: bool
    zone_diversity: int
    repeated_approaches: int
    confidence: float = 0.0  # detection confidence (0-1)
    object_in_hand: Optional[str] = None  # detected object near wrist
    motion_intensity: str = "none"  # "none" | "low" | "moderate" | "high"


@dataclass(frozen=True, slots=True)
class SceneGraph:
    """Structured scene representation for LLM reasoning."""
    timestamp: float
    total_persons: int
    activity_level: str  # "quiet" | "moderate" | "busy"
    persons: list[SceneGraphPerson]
    objects: list[str]  # simple label list


# ---------------------------------------------------------------------------
# Analysis results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Structured output from the LLM reasoner."""

    risk_level: str  # "low" | "medium" | "high"
    explanation: str
    alerts: list[str]
    recommended_action: str


@dataclass(frozen=True, slots=True)
class PersonAnalysisResult:
    """Per-person analysis bundle returned by the orchestrator."""

    person_id: int
    bbox: tuple[float, float, float, float]
    behavior: TrackedPerson
    intent: AnalysisResult


@dataclass(frozen=True, slots=True)
class FrameResult:
    """Top-level result for a single frame, returned by the /analyze endpoint."""

    scene: SceneState
    persons: list[PersonAnalysisResult]
    analysis: AnalysisResult
    latency_ms: float
    events: list[dict] = field(default_factory=list)
    scene_graph: Optional[SceneGraph] = None


@dataclass(frozen=True, slots=True)
class VoiceQueryResult:
    """Response from the conversational voice query pipeline."""

    response: str
    events: list[dict]
    latency_ms: float
