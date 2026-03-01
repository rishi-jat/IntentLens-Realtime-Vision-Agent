"""
IntentLens Backend — FastAPI application entry point.

Provides:
- ``GET  /health``     — health check
- ``POST /token``      — Stream Video token generation
- ``POST /analyze``    — full frame analysis pipeline
- ``POST /visual_qa``  — on-demand visual question answering

No business logic here — route handlers delegate to BehaviorEngine.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import sys
import time
from contextlib import asynccontextmanager
from functools import partial
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import LOG_LEVEL, OPENAI_API_KEY, STREAM_API_KEY, STREAM_API_SECRET

# ---------------------------------------------------------------------------
# Structured logging — configured once before anything else logs
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("intentlens")

# ---------------------------------------------------------------------------
# Engine singleton (initialised in lifespan)
# ---------------------------------------------------------------------------

from behavior_engine import BehaviorEngine  # noqa: E402

_engine: BehaviorEngine | None = None

# Server-side voice rate limiter
_last_voice_query_ts: float = 0.0
VOICE_COOLDOWN_SECS: float = 3.0


def _get_engine() -> BehaviorEngine:
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised")
    return _engine


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _engine
    logger.info("Starting IntentLens engine…")
    _engine = BehaviorEngine(openai_api_key=OPENAI_API_KEY)
    logger.info("Engine ready")
    yield
    _engine = None
    logger.info("Engine shut down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="IntentLens API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class TokenRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=128)


class TokenResponse(BaseModel):
    token: str
    user_id: str


class HealthResponse(BaseModel):
    status: str
    timestamp: float


# --- /analyze models ---


class DetectionOut(BaseModel):
    label: str
    confidence: float
    bbox: list[float]
    class_id: int


class KeypointOut(BaseModel):
    x: float
    y: float
    confidence: float


class PersonKeypointsOut(BaseModel):
    bbox: list[float]
    keypoints: list[KeypointOut]


class PersonAttributesOut(BaseModel):
    posture: str
    dominant_color: str
    head_tilt: str
    gaze_direction: str


class TrackedPersonOut(BaseModel):
    track_id: int
    bbox: list[float]
    center: list[float]
    velocity: float
    velocity_label: str
    zone: Optional[str]
    dwell_time: float
    zones_entered: list[str]
    repeated_approaches: int
    timeline: list[str]
    keypoints: Optional[PersonKeypointsOut] = None
    attributes: Optional[PersonAttributesOut] = None


class SceneOut(BaseModel):
    timestamp: float
    frame_width: int
    frame_height: int
    people: list[TrackedPersonOut]
    objects: list[DetectionOut]


class AnalysisOut(BaseModel):
    risk_level: str
    explanation: str
    alerts: list[str]
    recommended_action: str


class PersonAnalysisOut(BaseModel):
    person_id: int
    bbox: list[float]
    behavior: TrackedPersonOut
    intent: AnalysisOut


class SceneGraphPersonOut(BaseModel):
    person_id: int
    zone: Optional[str]
    dwell_time: float
    velocity_label: str
    posture: str
    dominant_color: str
    head_tilt: str
    gaze_direction: str
    gesture_state: list[str]
    is_pacing: bool
    is_loitering: bool
    zone_diversity: int
    repeated_approaches: int
    confidence: float = 0.0
    object_in_hand: Optional[str] = None
    motion_intensity: str = "none"


class SceneGraphOut(BaseModel):
    timestamp: float
    total_persons: int
    activity_level: str
    persons: list[SceneGraphPersonOut]
    objects: list[str]


class AnalyzeRequest(BaseModel):
    frame: str = Field(..., description="Base64-encoded JPEG/PNG frame")


class AnalyzeResponse(BaseModel):
    scene: SceneOut
    persons: list[PersonAnalysisOut]
    analysis: AnalysisOut
    latency_ms: float
    events: list[dict] = []
    scene_graph: Optional[SceneGraphOut] = None


# --- /visual_qa models ---


class VisualQARequest(BaseModel):
    frame: str = Field(..., description="Base64-encoded JPEG/PNG frame")
    question: str = Field(..., min_length=1, max_length=1024)


class VisualQAResponse(BaseModel):
    answer: str
    latency_ms: float


# --- /voice_query models ---


class VoiceQueryRequest(BaseModel):
    transcript: str = Field(..., min_length=1, max_length=2048)
    frame: Optional[str] = Field(None, description="Optional base64-encoded frame for visual grounding")


class VoiceQueryResponse(BaseModel):
    response: str
    events: list[dict] = []
    latency_ms: float


# ---------------------------------------------------------------------------
# Helper: decode base64 frame
# ---------------------------------------------------------------------------


def _decode_frame(b64: str) -> np.ndarray:
    try:
        raw = base64.b64decode(b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("cv2.imdecode returned None")
        return frame
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid frame data: {exc}") from exc


# ---------------------------------------------------------------------------
# Serialisation helpers (dataclass → Pydantic)
# ---------------------------------------------------------------------------


def _serialise_frame_result(fr) -> AnalyzeResponse:
    """Convert ``FrameResult`` dataclass tree into Pydantic response."""
    scene = fr.scene

    def _serialise_person_keypoints(kp):
        """Convert PersonKeypoints to Pydantic model."""
        if kp is None:
            return None
        return PersonKeypointsOut(
            bbox=list(kp.bbox),
            keypoints=[
                KeypointOut(x=k.x, y=k.y, confidence=k.confidence)
                for k in kp.keypoints
            ],
        )

    def _serialise_attributes(attrs):
        """Convert PersonAttributes to Pydantic model."""
        if attrs is None:
            return None
        return PersonAttributesOut(
            posture=attrs.posture,
            dominant_color=attrs.dominant_color,
            head_tilt=attrs.head_tilt,
            gaze_direction=attrs.gaze_direction,
        )

    def _serialise_tracked_person(p):
        return TrackedPersonOut(
            track_id=p.track_id,
            bbox=list(p.bbox),
            center=list(p.center),
            velocity=p.velocity,
            velocity_label=p.velocity_label,
            zone=p.zone,
            dwell_time=p.dwell_time,
            zones_entered=p.zones_entered,
            repeated_approaches=p.repeated_approaches,
            timeline=p.timeline,
            keypoints=_serialise_person_keypoints(p.keypoints),
            attributes=_serialise_attributes(p.attributes),
        )

    scene_out = SceneOut(
        timestamp=scene.timestamp,
        frame_width=scene.frame_width,
        frame_height=scene.frame_height,
        people=[_serialise_tracked_person(p) for p in scene.people],
        objects=[
            DetectionOut(
                label=o.label,
                confidence=o.confidence,
                bbox=list(o.bbox),
                class_id=o.class_id,
            )
            for o in scene.objects
        ],
    )

    analysis_out = AnalysisOut(
        risk_level=fr.analysis.risk_level,
        explanation=fr.analysis.explanation,
        alerts=fr.analysis.alerts,
        recommended_action=fr.analysis.recommended_action,
    )

    persons_out: list[PersonAnalysisOut] = []
    for pa in fr.persons:
        persons_out.append(
            PersonAnalysisOut(
                person_id=pa.person_id,
                bbox=list(pa.bbox),
                behavior=_serialise_tracked_person(pa.behavior),
                intent=AnalysisOut(
                    risk_level=pa.intent.risk_level,
                    explanation=pa.intent.explanation,
                    alerts=pa.intent.alerts,
                    recommended_action=pa.intent.recommended_action,
                ),
            )
        )

    # Serialise scene graph
    scene_graph_out = None
    if fr.scene_graph is not None:
        sg = fr.scene_graph
        scene_graph_out = SceneGraphOut(
            timestamp=sg.timestamp,
            total_persons=sg.total_persons,
            activity_level=sg.activity_level,
            persons=[
                SceneGraphPersonOut(
                    person_id=sp.person_id,
                    zone=sp.zone,
                    dwell_time=sp.dwell_time,
                    velocity_label=sp.velocity_label,
                    posture=sp.posture,
                    dominant_color=sp.dominant_color,
                    head_tilt=sp.head_tilt,
                    gaze_direction=sp.gaze_direction,
                    gesture_state=sp.gesture_state,
                    is_pacing=sp.is_pacing,
                    is_loitering=sp.is_loitering,
                    zone_diversity=sp.zone_diversity,
                    repeated_approaches=sp.repeated_approaches,
                    confidence=sp.confidence,
                    object_in_hand=sp.object_in_hand,
                    motion_intensity=sp.motion_intensity,
                )
                for sp in sg.persons
            ],
            objects=sg.objects,
        )

    return AnalyzeResponse(
        scene=scene_out,
        persons=persons_out,
        analysis=analysis_out,
        latency_ms=fr.latency_ms,
        events=fr.events,
        scene_graph=scene_graph_out,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Lightweight health check."""
    return HealthResponse(status="ok", timestamp=time.time())


@app.post("/token", response_model=TokenResponse)
async def create_token(request: TokenRequest) -> TokenResponse:
    """Generate a Stream Video user token."""
    if not STREAM_API_KEY or not STREAM_API_SECRET:
        raise HTTPException(status_code=500, detail="Stream API credentials not configured")

    try:
        from stream_chat import StreamChat

        client = StreamChat(api_key=STREAM_API_KEY, api_secret=STREAM_API_SECRET)
        token: str = client.create_token(request.user_id)
        return TokenResponse(token=token, user_id=request.user_id)
    except Exception as exc:
        logger.error("Token generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Token generation failed: {exc}") from exc


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_frame(request: AnalyzeRequest) -> AnalyzeResponse:
    """Full frame analysis: detect → track → reason → respond."""
    frame = _decode_frame(request.frame)
    engine = _get_engine()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, partial(engine.process_frame, frame=frame, timestamp=time.time())
    )
    return _serialise_frame_result(result)


@app.post("/visual_qa", response_model=VisualQAResponse)
async def visual_qa(request: VisualQARequest) -> VisualQAResponse:
    """On-demand visual question answering (does NOT run tracking pipeline)."""
    frame = _decode_frame(request.frame)
    engine = _get_engine()
    t0 = time.perf_counter()
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(
        None, partial(engine.visual_qa, frame=frame, question=request.question)
    )
    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    return VisualQAResponse(answer=answer, latency_ms=elapsed)


@app.post("/voice_query", response_model=VoiceQueryResponse)
async def voice_query(request: VoiceQueryRequest) -> VoiceQueryResponse:
    """Conversational voice query with scene context (rate-limited)."""
    global _last_voice_query_ts
    now = time.time()
    if now - _last_voice_query_ts < VOICE_COOLDOWN_SECS:
        raise HTTPException(
            status_code=429,
            detail="Voice cooldown active — please wait a few seconds.",
        )
    _last_voice_query_ts = now

    engine = _get_engine()
    frame = _decode_frame(request.frame) if request.frame else None
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        partial(
            engine.voice_query,
            transcript=request.transcript,
            frame=frame,
            timestamp=time.time(),
        ),
    )
    return VoiceQueryResponse(
        response=result.response,
        events=result.events,
        latency_ms=result.latency_ms,
    )

