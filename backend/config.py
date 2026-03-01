"""
IntentLens — Centralised configuration.

All tuneable parameters live here. No magic numbers elsewhere.
Environment variables are loaded once at import time.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------

STREAM_API_KEY: str = os.environ.get("STREAM_API_KEY", "")
STREAM_API_SECRET: str = os.environ.get("STREAM_API_SECRET", "")
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# YOLO / Vision
# ---------------------------------------------------------------------------

YOLO_MODEL_NAME: str = "yolov8n-pose.pt"
YOLO_CONFIDENCE_THRESHOLD: float = 0.50        # Accuracy > sensitivity — eliminates ghost detections
YOLO_PERSON_CLASS_ID: int = 0

# Pose keypoint confidence threshold — ignore low-confidence joints
KEYPOINT_CONFIDENCE_THRESHOLD: float = 0.40     # Raised for accuracy

# ---------------------------------------------------------------------------
# DeepSORT / Tracking
# ---------------------------------------------------------------------------

TRACKER_MAX_AGE: int = 30           # Reduced — prevent counting stale/departed tracks
TRACKER_N_INIT: int = 2             # Fast confirmation (keep)
TRACKER_MAX_IOU_DISTANCE: float = 0.7
TRACKER_NMS_MAX_OVERLAP: float = 0.8

# Minimum bbox area (px²) to accept a detection — filters noise
TRACKER_MIN_BBOX_AREA: float = 1200.0           # Raised — reject tiny/partial detections

# Maximum allowed IoU between two *confirmed* tracks to suppress duplicates
TRACKER_DUPLICATE_IOU_THRESHOLD: float = 0.60   # Slightly tighter duplicate suppression

# Seconds after which a lost person state is pruned
TRACKER_STALE_TIMEOUT_SECS: float = 45.0        # Prune faster

# ---------------------------------------------------------------------------
# Zone grid
# ---------------------------------------------------------------------------

ZONE_GRID_COLS: int = 3
ZONE_GRID_ROWS: int = 3

# ---------------------------------------------------------------------------
# Behaviour analysis
# ---------------------------------------------------------------------------

# Velocity classification thresholds (px / s)
VELOCITY_STATIONARY: float = 2.0
VELOCITY_SLOW: float = 20.0
VELOCITY_MODERATE: float = 60.0

# Re-entry cooldown
REENTRY_COOLDOWN_SECS: float = 5.0

# ---------------------------------------------------------------------------
# Session memory
# ---------------------------------------------------------------------------

MEMORY_MAX_SNAPSHOTS: int = 300
MEMORY_DEFAULT_WINDOW_SECS: float = 30.0
MEMORY_MAX_TIMELINE_EVENTS: int = 60

# ---------------------------------------------------------------------------
# LLM reasoning
# ---------------------------------------------------------------------------

LLM_MODEL: str = "gpt-4o-mini"
LLM_TEMPERATURE: float = 0.15
LLM_MAX_TOKENS: int = 300
LLM_CALL_INTERVAL_SECS: float = 4.0
LLM_MIN_DWELL_FOR_CALL: float = 1.5

# ---------------------------------------------------------------------------
# Visual QA
# ---------------------------------------------------------------------------

VLM_MODEL: str = "gpt-4o"
VLM_MAX_TOKENS: int = 512

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
