"""
IntentLens — Unit tests.

Covers:
- SceneStateBuilder zone / velocity logic
- TrackingManager duplicate suppression
- SessionMemory window + zone tracking
- LLMReasoner JSON parsing / validation
- BehaviorAnalyzer heuristic signals
"""

from __future__ import annotations

import json
import time

import pytest

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

from models import (
    AnalysisResult,
    DetectedObject,
    SceneState,
    TrackedPerson,
)


# ---------------------------------------------------------------------------
# SceneStateBuilder
# ---------------------------------------------------------------------------

class TestSceneStateBuilder:
    """Tests for zone computation and velocity classification."""

    def test_compute_zone_center(self):
        from scene_builder import _compute_zone

        # Centre of a 900×900 frame → zone_1_1
        assert _compute_zone(450.0, 450.0, 900, 900) == "zone_1_1"

    def test_compute_zone_top_left(self):
        from scene_builder import _compute_zone

        assert _compute_zone(10.0, 10.0, 900, 900) == "zone_0_0"

    def test_compute_zone_bottom_right(self):
        from scene_builder import _compute_zone

        assert _compute_zone(890.0, 890.0, 900, 900) == "zone_2_2"

    def test_classify_velocity_stationary(self):
        from scene_builder import _classify_velocity

        assert _classify_velocity(0.5) == "stationary"

    def test_classify_velocity_slow(self):
        from scene_builder import _classify_velocity

        assert _classify_velocity(10.0) == "slow"

    def test_classify_velocity_moderate(self):
        from scene_builder import _classify_velocity

        assert _classify_velocity(40.0) == "moderate"

    def test_classify_velocity_fast(self):
        from scene_builder import _classify_velocity

        assert _classify_velocity(100.0) == "fast"

    def test_bbox_center(self):
        from scene_builder import _bbox_center

        cx, cy = _bbox_center((10.0, 20.0, 110.0, 120.0))
        assert cx == pytest.approx(60.0)
        assert cy == pytest.approx(70.0)


# ---------------------------------------------------------------------------
# TrackingManager (duplicate suppression)
# ---------------------------------------------------------------------------

class TestTrackingManagerDuplicates:
    """Tests for IoU-based duplicate suppression logic."""

    def test_no_suppression_distant_tracks(self):
        from tracking import ConfirmedTrack, TrackingManager

        tracks = [
            ConfirmedTrack(track_id=1, bbox=(0.0, 0.0, 50.0, 50.0)),
            ConfirmedTrack(track_id=2, bbox=(200.0, 200.0, 250.0, 250.0)),
        ]
        result = TrackingManager._suppress_duplicates(tracks)
        assert len(result) == 2

    def test_suppresses_overlapping_newer_track(self):
        from tracking import ConfirmedTrack, TrackingManager

        tracks = [
            ConfirmedTrack(track_id=1, bbox=(0.0, 0.0, 100.0, 100.0)),
            ConfirmedTrack(track_id=5, bbox=(5.0, 5.0, 105.0, 105.0)),
        ]
        result = TrackingManager._suppress_duplicates(tracks)
        assert len(result) == 1
        assert result[0].track_id == 1  # older ID kept

    def test_single_track_unchanged(self):
        from tracking import ConfirmedTrack, TrackingManager

        tracks = [ConfirmedTrack(track_id=1, bbox=(0.0, 0.0, 50.0, 50.0))]
        result = TrackingManager._suppress_duplicates(tracks)
        assert len(result) == 1

    def test_iou_computation(self):
        from tracking import _iou

        # Identical boxes → IoU = 1.0
        assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

        # Non-overlapping → IoU = 0.0
        assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == pytest.approx(0.0)

        # Partial overlap
        iou = _iou((0, 0, 10, 10), (5, 5, 15, 15))
        assert 0.0 < iou < 1.0


# ---------------------------------------------------------------------------
# SessionMemory
# ---------------------------------------------------------------------------

class TestSessionMemory:
    """Tests for rolling memory window and zone tracking."""

    def test_store_and_retrieve_window(self):
        from memory import SessionMemory

        mem = SessionMemory(max_snapshots=100)
        t_base = time.time()

        for i in range(5):
            scene = SceneState(
                timestamp=t_base + i * 10,
                frame_width=640,
                frame_height=480,
                people=[],
                objects=[],
            )
            mem.store_snapshot(scene)

        # Last 25 seconds → should get 3 snapshots (t+20, t+30, t+40)
        recent = mem.get_recent_window(25.0)
        assert len(recent) == 3

    def test_zone_reentry_detection(self):
        from memory import SessionMemory

        mem = SessionMemory()
        t = time.time()

        pm = mem.get_or_create_person(1, t)

        # Enter zone_0_0
        mem.update_person_zone(pm, "zone_0_0", t)
        assert pm.zones_entered == ["zone_0_0"]
        assert pm.repeated_approaches == 0

        # Move to zone_1_1
        mem.update_person_zone(pm, "zone_1_1", t + 1)

        # Re-enter zone_0_0 after cooldown
        mem.update_person_zone(pm, "zone_0_0", t + 10)
        assert pm.repeated_approaches == 1

    def test_prune_stale_persons(self):
        from memory import SessionMemory

        mem = SessionMemory()
        t = time.time()

        pm1 = mem.get_or_create_person(1, t)
        pm1.last_seen = t - 200  # stale

        pm2 = mem.get_or_create_person(2, t)
        pm2.last_seen = t  # fresh

        mem.prune_stale(t, timeout_secs=100.0)

        assert mem.summarize_person(1) is None
        assert mem.summarize_person(2) is not None

    def test_summarize_person(self):
        from memory import SessionMemory

        mem = SessionMemory()
        t = time.time()

        pm = mem.get_or_create_person(42, t)
        pm.last_seen = t + 5
        mem.update_person_zone(pm, "zone_0_0", t)

        summary = mem.summarize_person(42)
        assert summary is not None
        assert summary["track_id"] == 42
        assert summary["dwell_time"] == pytest.approx(5.0, abs=0.1)
        assert "zone_0_0" in summary["zones_entered"]


# ---------------------------------------------------------------------------
# LLMReasoner (JSON parsing)
# ---------------------------------------------------------------------------

class TestLLMReasonerParsing:
    """Tests for LLM response parsing and validation."""

    def test_valid_json(self):
        from llm_reasoner import LLMReasoner

        raw = json.dumps({
            "risk_level": "medium",
            "explanation": "Person loitering in zone_1_1",
            "alerts": ["Dwell time > 60s"],
            "recommended_action": "Monitor closely",
        })
        result = LLMReasoner._parse_response(raw)
        assert result.risk_level == "medium"
        assert len(result.alerts) == 1
        assert result.recommended_action == "Monitor closely"

    def test_invalid_json_returns_default(self):
        from llm_reasoner import LLMReasoner

        result = LLMReasoner._parse_response("not json at all")
        assert result.risk_level == "low"
        assert result.explanation == "Insufficient data to perform analysis."

    def test_markdown_fenced_json(self):
        from llm_reasoner import LLMReasoner

        raw = '```json\n{"risk_level":"high","explanation":"Suspicious","alerts":[],"recommended_action":"Alert"}\n```'
        result = LLMReasoner._parse_response(raw)
        assert result.risk_level == "high"

    def test_unknown_risk_level_defaults_to_low(self):
        from llm_reasoner import LLMReasoner

        raw = json.dumps({
            "risk_level": "extreme",
            "explanation": "test",
            "alerts": [],
            "recommended_action": "none",
        })
        result = LLMReasoner._parse_response(raw)
        assert result.risk_level == "low"

    def test_missing_fields_handled(self):
        from llm_reasoner import LLMReasoner

        raw = json.dumps({"risk_level": "high"})
        result = LLMReasoner._parse_response(raw)
        assert result.risk_level == "high"
        assert result.explanation == ""
        assert result.alerts == []


# ---------------------------------------------------------------------------
# BehaviorAnalyzer
# ---------------------------------------------------------------------------

class TestBehaviorAnalyzer:
    """Tests for heuristic behaviour classification."""

    def _make_person(self, **kwargs) -> TrackedPerson:
        defaults = dict(
            track_id=1,
            bbox=(100.0, 100.0, 200.0, 300.0),
            center=(150.0, 200.0),
            velocity=0.0,
            velocity_label="stationary",
            zone="zone_1_1",
            dwell_time=0.0,
            zones_entered=["zone_1_1"],
            repeated_approaches=0,
            timeline=[],
        )
        defaults.update(kwargs)
        return TrackedPerson(**defaults)

    def test_brief_dwell(self):
        from behavior_analyzer import BehaviorAnalyzer

        ba = BehaviorAnalyzer()
        person = self._make_person(dwell_time=2.0)
        signals = ba.analyze(person)
        assert signals.dwell_category == "brief"

    def test_loitering_detection(self):
        from behavior_analyzer import BehaviorAnalyzer

        ba = BehaviorAnalyzer()
        person = self._make_person(dwell_time=35.0, velocity_label="stationary")
        signals = ba.analyze(person)
        assert signals.is_loitering is True

    def test_pacing_detection(self):
        from behavior_analyzer import BehaviorAnalyzer

        ba = BehaviorAnalyzer()
        person = self._make_person(
            repeated_approaches=3,
            velocity_label="slow",
        )
        signals = ba.analyze(person)
        assert signals.is_pacing is True

    def test_no_pacing_when_stationary(self):
        from behavior_analyzer import BehaviorAnalyzer

        ba = BehaviorAnalyzer()
        person = self._make_person(
            repeated_approaches=5,
            velocity_label="stationary",
        )
        signals = ba.analyze(person)
        assert signals.is_pacing is False

    def test_reentry_flag(self):
        from behavior_analyzer import BehaviorAnalyzer

        ba = BehaviorAnalyzer()
        person = self._make_person(repeated_approaches=4)
        signals = ba.analyze(person)
        assert signals.reentry_flag is True

    def test_movement_intensity_mapping(self):
        from behavior_analyzer import BehaviorAnalyzer

        ba = BehaviorAnalyzer()
        for vel, expected in [
            ("stationary", "still"),
            ("slow", "low"),
            ("moderate", "moderate"),
            ("fast", "high"),
        ]:
            person = self._make_person(velocity_label=vel)
            signals = ba.analyze(person)
            assert signals.movement_intensity == expected, f"Failed for {vel}"
