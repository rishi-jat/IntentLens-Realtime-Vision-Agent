"""
IntentLens — LLMReasoner.

Single responsibility: consume structured behavioural data and produce
a validated ``AnalysisResult`` via OpenAI chat completions.

Also provides the Visual Q&A capability (``visual_qa``).
"""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from openai import OpenAI

from behavior_analyzer import BehaviorSignals
from config import (
    LLM_CALL_INTERVAL_SECS,
    LLM_MAX_TOKENS,
    LLM_MIN_DWELL_FOR_CALL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    VLM_MAX_TOKENS,
    VLM_MODEL,
)
from models import AnalysisResult, SceneGraph, SceneState, TrackedPerson

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are IntentLens — an advanced real-time environmental awareness AI. "
    "You speak with a calm, tactical, and precise tone. You are observant and measured. "
    "Never sound robotic — be clear, concise, and slightly futuristic.\n\n"
    "You receive a structured scene graph with per-person data including:\n"
    "- Posture (upright/leaning/crouching)\n"
    "- Dominant clothing color\n"
    "- Head tilt and gaze direction\n"
    "- Active gestures (raised_hand, waving, crouching, etc.)\n"
    "- Movement patterns (velocity, pacing, loitering)\n"
    "- Spatial data (zone, dwell time, zone diversity)\n"
    "- Confidence score (0-1) for each person's detection\n\n"
    "CRITICAL ACCURACY RULES:\n"
    "1. ONLY describe what the structured data EXPLICITLY tells you. NEVER invent, assume, or hallucinate details.\n"
    "2. If a field says 'unknown', DO NOT guess — say you cannot determine it.\n"
    "3. If confidence is below 0.5, say 'I am not confident enough to determine that.'\n"
    "4. NEVER describe objects, clothing, or actions not present in the data.\n"
    "5. When in doubt, say less. Accuracy over impressiveness.\n\n"
    "Your task: classify risk, explain your reasoning, and recommend action.\n\n"
    "Respond ONLY with valid JSON:\n"
    "{\n"
    '  "risk_level": "low" | "medium" | "high",\n'
    '  "explanation": "<evidence-based, natural-sounding assessment>",\n'
    '  "alerts": ["<specific observation from data>", ...],\n'
    '  "recommended_action": "<actionable guidance>"\n'
    "}\n\n"
    "Tone: calm, precise, slightly futuristic. Reference observable attributes from the data only.\n"
    "- low: normal behaviour, environment stable\n"
    "- medium: unusual patterns warranting attention\n"
    "- high: strongly suspicious patterns requiring immediate response\n"
    "- alerts: list ONLY observations directly supported by the data. Empty list if none.\n"
    "- Be precise, evidence-based, and never speculate beyond the data."
)

_DEFAULT_RESULT = AnalysisResult(
    risk_level="low",
    explanation="Environment stable. No anomalies detected. All activity within normal parameters.",
    alerts=[],
    recommended_action="Continue monitoring.",
)

_VQA_SYSTEM_PROMPT = (
    "You are IntentLens — a calm, tactical AI that can see through a live camera. "
    "You are given a camera frame and a user question. Answer conversationally but concisely. "
    "Sound like a composed surveillance analyst, not a chatbot.\n\n"
    "ACCURACY RULES:\n"
    "- ONLY describe what you can clearly see in the image. Never fabricate details.\n"
    "- If you cannot determine the answer from the image, say 'I am not confident enough to determine that.'\n"
    "- Do NOT guess colors, objects, or actions you cannot clearly identify.\n"
    "- Accuracy is more important than sounding impressive."
)

_VOICE_SYSTEM_PROMPT = (
    "You are IntentLens — a real-time AI that sees through a live camera and talks to people. "
    "You are calm, witty, and concise — like a sharp-eyed friend, not a robot.\n\n"
    "CRITICAL RULES:\n"
    "1. ANSWER THE USER'S ACTUAL QUESTION. Do NOT just list scene data or numbers.\n"
    "2. If a camera image is provided, LOOK AT IT and describe what you actually see.\n"
    "3. Keep responses to 1-2 natural sentences. Talk like a person, not a database.\n"
    "4. NEVER say 'velocity', 'zone_1_2', 'dwell time', 'px/s', or any technical tracking jargon.\n"
    "5. If asked 'what's in my hand?' or 'what am I holding?' — LOOK at the image and describe the object.\n"
    "6. If asked 'what do you see?' — describe the person and scene naturally (appearance, posture, surroundings).\n"
    "7. If you genuinely can't tell, say 'I can't quite make that out' — don't dump raw data.\n"
    "8. Never output JSON. Speak naturally.\n"
    "9. Sound alive and aware. Be slightly witty. Never robotic.\n"
    "10. If conversation history is provided, stay coherent."
)


class LLMReasoner:
    """Rate-limited LLM reasoning over structured behavioural data."""

    # Global call budget: max 10 LLM calls per 60 seconds
    _global_call_timestamps: list[float] = []
    MAX_CALLS_PER_MINUTE: int = 10

    def __init__(self, openai_api_key: str) -> None:
        self._client: Optional[OpenAI] = (
            OpenAI(api_key=openai_api_key) if openai_api_key else None
        )
        self._last_call_per_person: dict[int, float] = {}
        self._cached_per_person: dict[int, AnalysisResult] = {}
        self._last_scene_call: float = 0.0
        self._cached_scene: AnalysisResult = _DEFAULT_RESULT
        # Rolling conversation history for voice queries (last 6 messages = 3 exchanges)
        self._conversation_history: list[dict[str, str]] = []
        self._max_conversation_turns: int = 6

    @property
    def available(self) -> bool:
        return self._client is not None

    def _check_global_budget(self) -> bool:
        """Return True if we are within the global LLM call budget."""
        now = time.time()
        # Prune old timestamps
        LLMReasoner._global_call_timestamps = [
            ts for ts in LLMReasoner._global_call_timestamps if now - ts < 60.0
        ]
        return len(LLMReasoner._global_call_timestamps) < self.MAX_CALLS_PER_MINUTE

    def _record_call(self) -> None:
        """Record that an LLM call was made."""
        LLMReasoner._global_call_timestamps.append(time.time())

    # ------------------------------------------------------------------
    # Per-person intent (rate-limited)
    # ------------------------------------------------------------------

    def classify_person(
        self,
        person: TrackedPerson,
        signals: BehaviorSignals,
        timestamp: float,
    ) -> AnalysisResult:
        """Classify a single person's intent. Returns cached result if rate-limited.

        Parameters
        ----------
        person : the tracked person
        signals : heuristic behavioural signals
        timestamp : current frame time

        Returns
        -------
        Validated ``AnalysisResult``.
        """
        cached = self._cached_per_person.get(person.track_id)
        if cached is not None:
            last_call = self._last_call_per_person.get(person.track_id, 0.0)
            if timestamp - last_call < LLM_CALL_INTERVAL_SECS:
                return cached

        if not self.available:
            return cached or _DEFAULT_RESULT

        if person.dwell_time < LLM_MIN_DWELL_FOR_CALL and not person.zones_entered:
            return cached or _DEFAULT_RESULT

        if not self._check_global_budget():
            logger.debug("Global LLM budget exhausted — returning cached for person %d", person.track_id)
            return cached or _DEFAULT_RESULT

        prompt = self._build_person_prompt(person, signals)
        result = self._call_llm(prompt)

        self._cached_per_person[person.track_id] = result
        self._last_call_per_person[person.track_id] = timestamp
        return result

    # ------------------------------------------------------------------
    # Scene-level analysis (rate-limited)
    # ------------------------------------------------------------------

    def classify_scene(
        self,
        scene: SceneState,
        all_signals: list[tuple[TrackedPerson, BehaviorSignals]],
        timestamp: float,
        scene_graph: Optional[SceneGraph] = None,
    ) -> AnalysisResult:
        """Produce a scene-level risk assessment.

        Called once per frame. Rate-limited to avoid API spam.
        """
        if timestamp - self._last_scene_call < LLM_CALL_INTERVAL_SECS:
            return self._cached_scene

        if not self.available or not all_signals:
            return self._cached_scene

        if not self._check_global_budget():
            logger.debug("Global LLM budget exhausted — returning cached scene")
            return self._cached_scene

        prompt = self._build_scene_prompt(scene, all_signals, scene_graph)
        result = self._call_llm(prompt)

        self._cached_scene = result
        self._last_scene_call = timestamp
        return result

    # ------------------------------------------------------------------
    # Visual Q&A
    # ------------------------------------------------------------------

    def visual_qa(
        self,
        frame: NDArray[np.uint8],
        question: str,
    ) -> str:
        """Answer a user question about a single frame using vision LLM.

        Parameters
        ----------
        frame : BGR uint8 image
        question : user's natural-language question

        Returns
        -------
        Answer string.
        """
        if not self.available:
            return "LLM not available — OpenAI API key not configured."

        import cv2

        success, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not success:
            return "Failed to encode frame for VLM."

        b64_image = base64.b64encode(buf.tobytes()).decode("utf-8")

        try:
            start = time.perf_counter()
            response = self._client.chat.completions.create(  # type: ignore[union-attr]
                model=VLM_MODEL,
                messages=[
                    {"role": "system", "content": _VQA_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}",
                                    "detail": "low",
                                },
                            },
                        ],
                    },
                ],
                max_tokens=VLM_MAX_TOKENS,
                temperature=0.3,
            )
            elapsed = (time.perf_counter() - start) * 1000
            logger.info("Visual QA completed in %.0fms", elapsed)
            return response.choices[0].message.content or "No answer generated."
        except Exception as exc:
            logger.error("Visual QA failed: %s", exc)
            return f"Visual QA error: {exc}"

    # ------------------------------------------------------------------
    # Voice query (conversational, context-aware)
    # ------------------------------------------------------------------

    def voice_query(
        self,
        transcript: str,
        scene_context: str,
        frame: Optional[NDArray[np.uint8]] = None,
    ) -> str:
        """Answer a voice query using scene context and optional frame.

        Uses smart model selection:
        - Visual questions ("what do you see", "what color", "describe", "look at", etc.)
            → gpt-4o with frame
        - All other questions → gpt-4o-mini with text context only (fast + cheap)

        Parameters
        ----------
        transcript : what the user said
        scene_context : plain-text scene summary from SessionMemory
        frame : optional BGR image for visual grounding

        Returns
        -------
        Conversational response string.
        """
        if not self.available:
            return "I'm currently unable to process voice queries — API key not configured."

        # Determine if this is a visual question that needs the frame
        # BROAD list — when in doubt, use vision. It's more accurate than guessing.
        visual_keywords = [
            "see", "look", "color", "wearing", "holding", "hold", "held",
            "describe", "show", "hand", "carry", "carrying", "grab",
            "what is", "who is", "how many", "where", "what are", "appear",
            "visible", "image", "camera", "screen", "watch", "observe",
            "doing", "happen", "going on", "tell me", "around", "behind",
            "shirt", "wearing", "clothes", "face", "hair", "object",
            "room", "background", "front", "near", "next to",
        ]
        transcript_lower = transcript.lower()
        # Always use vision when frame is available — user is in front of the camera,
        # every question is contextual. Only skip vision for pure meta questions.
        meta_only_keywords = ["your name", "who are you", "what are you", "help", "reset", "stop"]
        is_meta = any(kw in transcript_lower for kw in meta_only_keywords)
        needs_vision = frame is not None and not is_meta

        # Keep scene context brief — don't overwhelm the LLM with raw data
        # Trim to avoid the LLM just parroting velocities and zone IDs
        brief_context = scene_context[:300] if scene_context else "No scene data."
        user_text = f"Brief scene context: {brief_context}\n\nThe user asks: \"{transcript}\"\n\nAnswer the user's question directly. If a camera frame is attached, USE IT to answer. Do NOT just list tracking data."
        user_content: str | list = user_text

        if needs_vision:
            import cv2
            # Good enough quality for GPT-4o to actually see what's happening
            success, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if success:
                b64_image = base64.b64encode(buf.tobytes()).decode("utf-8")
                user_content = [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}",
                            "detail": "low",
                        },
                    },
                ]

        model = VLM_MODEL if isinstance(user_content, list) else LLM_MODEL

        # No retry loop — fail fast on errors, respect rate limits
        try:
            if not self._check_global_budget():
                logger.warning("Global LLM budget exhausted for voice query")
                return "I'm taking a short break to manage my resources. Try again in a moment."

            start = time.perf_counter()

            messages: list[dict] = [
                {"role": "system", "content": _VOICE_SYSTEM_PROMPT},
            ]
            messages.extend(self._conversation_history[-self._max_conversation_turns:])
            messages.append({"role": "user", "content": user_content})

            response = self._client.chat.completions.create(  # type: ignore[union-attr]
                model=model,
                messages=messages,
                max_tokens=200,
                temperature=0.4,
                timeout=10.0,
            )
            self._record_call()
            elapsed = (time.perf_counter() - start) * 1000
            logger.info("Voice query completed in %.0fms (model=%s)", elapsed, model)
            answer = response.choices[0].message.content or "I didn't catch that."

            # Store in conversation history
            self._conversation_history.append({"role": "user", "content": transcript})
            self._conversation_history.append({"role": "assistant", "content": answer})
            if len(self._conversation_history) > self._max_conversation_turns * 2:
                self._conversation_history = self._conversation_history[-self._max_conversation_turns:]

            return answer

        except Exception as exc:
            exc_str = str(exc).lower()
            logger.warning("Voice query failed: %s", exc)

            # Quota exhausted — fail immediately, no retry
            if "insufficient_quota" in exc_str or "billing" in exc_str:
                logger.error("OpenAI quota exhausted — cannot make LLM calls")
                return "I can't respond right now — my AI service quota has been exceeded."

            # Rate limited — fail immediately, no retry
            if "rate" in exc_str or "429" in exc_str:
                logger.warning("Rate limited — returning fallback")
                return "I'm being rate limited. Please wait a moment before asking again."

            return self._build_fallback_voice_response(scene_context, transcript)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_llm(self, user_prompt: str) -> AnalysisResult:
        """Execute a chat completion and parse the structured JSON response."""
        try:
            start = time.perf_counter()
            response = self._client.chat.completions.create(  # type: ignore[union-attr]
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
            self._record_call()
            elapsed = (time.perf_counter() - start) * 1000
            logger.info("LLM call completed in %.0fms", elapsed)

            content = response.choices[0].message.content
            if not content:
                logger.warning("LLM returned empty content")
                return _DEFAULT_RESULT

            return self._parse_response(content)

        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return _DEFAULT_RESULT

    @staticmethod
    def _parse_response(raw: str) -> AnalysisResult:
        """Parse and validate the JSON response from the LLM."""
        try:
            # Strip markdown fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()

            data = json.loads(cleaned)

            risk = str(data.get("risk_level", "low")).lower()
            if risk not in ("low", "medium", "high"):
                risk = "low"

            explanation = str(data.get("explanation", ""))
            alerts_raw = data.get("alerts", [])
            alerts = [str(a) for a in alerts_raw] if isinstance(alerts_raw, list) else []
            action = str(data.get("recommended_action", "Continue monitoring."))

            return AnalysisResult(
                risk_level=risk,
                explanation=explanation,
                alerts=alerts,
                recommended_action=action,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Failed to parse LLM response: %s — raw: %s", exc, raw[:200])
            return _DEFAULT_RESULT

    @staticmethod
    def _build_person_prompt(person: TrackedPerson, signals: BehaviorSignals) -> str:
        lines = [
            "Analyse the following tracked person's behaviour:",
            "",
            f"Track ID: {person.track_id}",
            f"Dwell time: {person.dwell_time:.1f}s ({signals.dwell_category})",
            f"Current zone: {person.zone or 'unknown'}",
            f"Zones visited: {', '.join(person.zones_entered) or 'none'}",
            f"Zone diversity: {signals.zone_diversity}",
            f"Repeated zone entries: {person.repeated_approaches}",
            f"Velocity: {person.velocity:.1f} px/s ({person.velocity_label})",
            f"Movement intensity: {signals.movement_intensity}",
            f"Pacing detected: {signals.is_pacing}",
            f"Loitering detected: {signals.is_loitering}",
            f"Re-entry flag: {signals.reentry_flag}",
        ]

        # Structured attributes from pose/vision
        if person.attributes is not None:
            attrs = person.attributes
            lines.append("")
            lines.append("Observed attributes:")
            lines.append(f"  Posture: {attrs.posture}")
            lines.append(f"  Clothing color: {attrs.dominant_color}")
            lines.append(f"  Head tilt: {attrs.head_tilt}")
            lines.append(f"  Gaze direction: {attrs.gaze_direction}")

        lines.append("")
        lines.append("Recent timeline:")
        for event in person.timeline[-12:]:
            lines.append(f"  - {event}")
        if not person.timeline:
            lines.append("  - (no events yet)")
        return "\n".join(lines)

    @staticmethod
    def _build_scene_prompt(
        scene: SceneState,
        all_signals: list[tuple[TrackedPerson, BehaviorSignals]],
        scene_graph: Optional[SceneGraph] = None,
    ) -> str:
        # Use SceneGraph if available for structured data
        if scene_graph is not None:
            lines = [
                "Analyse the overall scene from the following structured scene graph:",
                "",
                f"Total people: {scene_graph.total_persons}",
                f"Activity level: {scene_graph.activity_level}",
                "",
            ]
            for sp in scene_graph.persons:
                lines.append(f"--- Person #{sp.person_id} (confidence: {sp.confidence:.2f}) ---")
                lines.append(f"  Zone: {sp.zone}, velocity: {sp.velocity_label}")
                lines.append(f"  Dwell: {sp.dwell_time:.1f}s, Motion intensity: {sp.motion_intensity}")
                lines.append(f"  Posture: {sp.posture}")
                lines.append(f"  Clothing: {sp.dominant_color}")
                lines.append(f"  Head tilt: {sp.head_tilt}, Gaze: {sp.gaze_direction}")
                if sp.gesture_state:
                    lines.append(f"  Active gestures: {', '.join(sp.gesture_state)}")
                if sp.object_in_hand:
                    lines.append(f"  Holding: {sp.object_in_hand}")
                lines.append(f"  Pacing: {sp.is_pacing}, Loitering: {sp.is_loitering}")
                lines.append(f"  Zone diversity: {sp.zone_diversity}, Re-entries: {sp.repeated_approaches}")
                lines.append("")

            lines.append("IMPORTANT: Only reference data above. If confidence < 0.5, note uncertainty.")

            if scene_graph.objects:
                lines.append("Objects in scene:")
                for obj_label in scene_graph.objects[:10]:
                    lines.append(f"  - {obj_label}")

            return "\n".join(lines)

        # Fallback to raw signals
        lines = [
            "Analyse the overall scene:",
            "",
            f"Total people: {len(scene.people)}",
            f"Detected objects (non-person): {len(scene.objects)}",
            "",
        ]
        for person, signals in all_signals:
            lines.append(f"--- Person #{person.track_id} ---")
            lines.append(f"  Dwell: {person.dwell_time:.1f}s ({signals.dwell_category})")
            lines.append(f"  Zone: {person.zone}, velocity: {person.velocity_label}")
            lines.append(f"  Pacing: {signals.is_pacing}, Loitering: {signals.is_loitering}")
            lines.append(f"  Re-entries: {person.repeated_approaches}")
            if person.attributes:
                lines.append(f"  Posture: {person.attributes.posture}, Color: {person.attributes.dominant_color}")
            lines.append(f"  Summary: {signals.summary}")
            lines.append("")
        if scene.objects:
            lines.append("Non-person objects in scene:")
            for obj in scene.objects[:10]:
                lines.append(f"  - {obj.label} (conf={obj.confidence:.2f})")
        return "\n".join(lines)

    @staticmethod
    def _build_fallback_voice_response(scene_context: str, transcript: str) -> str:
        """Build a clean fallback when the LLM is unavailable or timed out."""
        return "Sorry, I'm having trouble connecting right now. Try asking again in a moment."
