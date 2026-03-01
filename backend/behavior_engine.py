"""
IntentLens — BehaviorEngine (orchestrator).

This is a **thin orchestration layer** that wires together:
- VisionProcessor  (YOLOv8-pose detection + keypoints)
- TrackingManager  (DeepSORT tracking + duplicate suppression)
- SessionMemory    (rolling per-person memory + transcript storage)
- SceneStateBuilder (zone / velocity / attribute enrichment → SceneState)
- BehaviorAnalyzer (heuristic signals)
- GestureDetector  (keypoint-based gesture recognition w/ temporal confirmation)
- SceneGraphBuilder (structured SceneGraph for LLM)
- LLMReasoner     (structured intent classification + voice query)
- EventBus        (real-time event system)

No business logic lives here — only composition and pipeline glue.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from behavior_analyzer import BehaviorAnalyzer
from config import TRACKER_STALE_TIMEOUT_SECS
from events import EventBus
from gesture_detector import Gesture, GestureDetector
from hand_tracker import HandTracker
from llm_reasoner import LLMReasoner
from memory import SessionMemory
from models import (
    AnalysisResult,
    DetectedObject,
    FrameResult,
    PersonAnalysisResult,
    PersonAttributes,
    SceneState,
    TrackedPerson,
    VoiceQueryResult,
)
from scene_builder import SceneStateBuilder
from scene_graph import SceneGraphBuilder
from tracking import TrackingManager
from vision import VisionProcessor

logger = logging.getLogger(__name__)


class BehaviorEngine:
    """Top-level orchestrator. Instantiate once at startup via lifespan."""

    def __init__(self, openai_api_key: str) -> None:
        self._vision = VisionProcessor()
        self._tracker = TrackingManager()
        self._memory = SessionMemory()
        self._scene_builder = SceneStateBuilder()
        self._analyzer = BehaviorAnalyzer()
        self._gesture = GestureDetector()
        self._hand_tracker = HandTracker()
        self._scene_graph_builder = SceneGraphBuilder()
        self._reasoner = LLMReasoner(openai_api_key=openai_api_key)
        self._event_bus = EventBus()
        self._last_frame: Optional[NDArray[np.uint8]] = None
        self._held_objects: dict[int, str] = {}  # track_id → last known held object
        logger.info("BehaviorEngine initialised with hand tracking (all sub-components ready)")

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    @property
    def memory(self) -> SessionMemory:
        return self._memory

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def process_frame(self, frame: NDArray[np.uint8], timestamp: float) -> FrameResult:
        """Run the full perception → tracking → reasoning pipeline.

        Parameters
        ----------
        frame : BGR uint8 image.
        timestamp : frame timestamp (epoch seconds).

        Returns
        -------
        ``FrameResult`` with scene, per-person analysis, scene-level analysis,
        events, scene graph, and latency measurement.
        """
        t0 = time.perf_counter()
        frame_h, frame_w = frame.shape[:2]
        self._last_frame = frame

        # 1. Perception — detect persons with pose + objects
        person_poses, object_dets = self._vision.detect_all(frame)

        # 2. Hand detection — only run when people detected (perf optimization)
        hand_detections = []
        hand_gestures = []
        if person_poses:
            hand_detections = self._hand_tracker.detect_hands(frame)
            hand_gestures = self._hand_tracker.recognize_gestures(hand_detections, frame_w, frame_h)
        
        # Log detected hand gestures
        if hand_gestures:
            for hg in hand_gestures:
                logger.debug(f"Hand gesture: {hg.name} ({hg.hand}, conf={hg.confidence:.2f})")

        # 3. Tracking — stable IDs with associated keypoints
        tracks = self._tracker.update(person_poses, frame)

        # 4. Scene state — enrich with zones, velocity, memory, attributes
        scene = self._scene_builder.build(
            tracks=tracks,
            objects=object_dets,
            frame_width=frame_w,
            frame_height=frame_h,
            timestamp=timestamp,
            memory=self._memory,
            frame=frame,
        )

        # 5. Events — new person / departed
        active_ids = {p.track_id for p in scene.people}
        frame_events = []
        frame_events.extend(self._event_bus.check_new_persons(active_ids, timestamp))
        frame_events.extend(self._event_bus.check_departed(active_ids, timestamp))
        
        # Add hand gesture events
        for hg in hand_gestures:
            frame_events.append({
                "kind": f"hand_{hg.name}",
                "timestamp": timestamp,
                "message": f"{hg.hand} hand: {hg.name.replace('_', ' ')}",
                "person_id": None,
                "severity": "info",
                "data": {"hand": hg.hand, "confidence": hg.confidence},
                "speakable": hg.name in ["raised_hand", "wave"],
            })

        # 6. Behaviour analysis + gesture detection per person
        person_results: list[PersonAnalysisResult] = []
        all_signals: list[tuple[TrackedPerson, Any]] = []
        person_gestures: dict[int, list[Gesture]] = {}
        person_attributes: dict[int, PersonAttributes] = {}
        person_risks: dict[int, str] = {}

        for person in scene.people:
            signals = self._analyzer.analyze(person)
            all_signals.append((person, signals))

            # Collect attributes
            if person.attributes is not None:
                person_attributes[person.track_id] = person.attributes

            # Keypoint-based gesture detection with temporal confirmation
            gestures = self._gesture.detect(person, timestamp, is_pacing=signals.is_pacing)
            if gestures:
                person_gestures[person.track_id] = gestures
                frame_events.extend(self._event_bus.add_gesture_events(gestures, timestamp))

            # Zone breach check
            frame_events.extend(
                self._event_bus.check_zone_breach(person.track_id, person.zone, timestamp)
            )

            # LLM intent classification
            intent = self._reasoner.classify_person(person, signals, timestamp)
            person_risks[person.track_id] = intent.risk_level
            person_results.append(
                PersonAnalysisResult(
                    person_id=person.track_id,
                    bbox=person.bbox,
                    behavior=person,
                    intent=intent,
                )
            )

        # 7. Build structured SceneGraph for LLM
        scene_graph = self._scene_graph_builder.build(
            scene=scene,
            person_signals=all_signals,
            person_gestures=person_gestures,
            person_attributes=person_attributes,
        )

        # 7b. Object-in-hand proactive events
        for sg_person in scene_graph.persons:
            if sg_person.object_in_hand:
                prev = self._held_objects.get(sg_person.person_id)
                if prev != sg_person.object_in_hand:
                    self._held_objects[sg_person.person_id] = sg_person.object_in_hand
                    from events import AgentEvent
                    ev = AgentEvent(
                        kind="object_detected",
                        timestamp=timestamp,
                        message=f"Person {sg_person.person_id} appears to be holding a {sg_person.object_in_hand}.",
                        person_id=sg_person.person_id,
                        severity="info",
                        data={"object": sg_person.object_in_hand},
                        speakable=True,
                    )
                    frame_events.append(ev)

        # 8. Scene-level analysis (using SceneGraph)
        scene_analysis = self._reasoner.classify_scene(scene, all_signals, timestamp, scene_graph)

        # 9. Risk change events
        frame_events.extend(
            self._event_bus.check_risk_change(scene_analysis.risk_level, person_risks, timestamp)
        )

        # 10. Cleanup
        self._gesture.prune_stale(active_ids)
        self._memory.prune_stale(timestamp, TRACKER_STALE_TIMEOUT_SECS)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(
            "Frame processed: %d people, %d objects, %d events, latency=%.1fms",
            len(scene.people),
            len(scene.objects),
            len(frame_events),
            elapsed_ms,
        )

        # Serialize events for transport (handle both AgentEvent objects and dicts)
        events_out = []
        for e in frame_events:
            if isinstance(e, dict):
                events_out.append(e)
            else:
                events_out.append({
                    "kind": e.kind,
                    "timestamp": e.timestamp,
                    "message": e.message,
                    "person_id": e.person_id,
                    "severity": e.severity,
                    "data": e.data or {},
                    "speakable": e.speakable,
                })

        return FrameResult(
            scene=scene,
            persons=person_results,
            analysis=scene_analysis,
            latency_ms=elapsed_ms,
            events=events_out,
            scene_graph=scene_graph,
        )

    # ------------------------------------------------------------------
    # Visual Q&A (unchanged)
    # ------------------------------------------------------------------

    def visual_qa(self, frame: NDArray[np.uint8], question: str) -> str:
        return self._reasoner.visual_qa(frame, question)

    # ------------------------------------------------------------------
    # Voice query (conversational)
    # ------------------------------------------------------------------

    def voice_query(
        self,
        transcript: str,
        frame: Optional[NDArray[np.uint8]] = None,
        timestamp: Optional[float] = None,
    ) -> VoiceQueryResult:
        """Process a voice transcript with scene context.

        Parameters
        ----------
        transcript : what the user said
        frame : optional current frame for visual grounding
        timestamp : current time (defaults to now)
        """
        t0 = time.perf_counter()
        ts = timestamp or time.time()

        # Store transcript in memory
        self._memory.store_transcript(ts, transcript)
        self._event_bus.add_voice_event(transcript, ts)

        # Build context from memory
        context = self._memory.build_context_summary()

        # Use the last captured frame if none provided
        use_frame = frame if frame is not None else self._last_frame

        # Get conversational response
        response = self._reasoner.voice_query(transcript, context, use_frame)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        # Get recent events for the frontend
        recent = self._event_bus.get_events_since(ts - 5.0)
        events_out = []
        for e in recent:
            if isinstance(e, dict):
                events_out.append(e)
            else:
                events_out.append({
                    "kind": e.kind,
                    "timestamp": e.timestamp,
                    "message": e.message,
                    "person_id": e.person_id,
                    "severity": e.severity,
                    "data": e.data or {},
                    "speakable": e.speakable,
                })

        return VoiceQueryResult(
            response=response,
            events=events_out,
            latency_ms=elapsed_ms,
        )

