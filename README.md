# ◉ IntentLens — Real-Time Multi-Modal Vision Agent

**Understand human intent in real time through computer vision, pose estimation, behavioral AI, and LLM reasoning.**

[![Live Demo](https://img.shields.io/badge/Live_Demo-Vercel-black?style=for-the-badge&logo=vercel)](https://intentlens-theta.vercel.app)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=white)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![GPT-4o](https://img.shields.io/badge/GPT--4o-OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)

</div>

---

IntentLens is a **real-time AI surveillance agent** that watches a live camera feed, tracks every person with persistent IDs, analyses their body language, gestures, movement patterns, and scene context — then uses GPT-4o to reason about **intent and risk**. You can also talk to it: ask what the camera sees, and it answers with visual grounding.

> Think of it as a Jarvis-style heads-up display for real-time human behavior understanding.

---

## ✨ Features at a Glance

| Capability | What It Does |
|---|---|
| 🎯 **Person Detection** | YOLOv8n-pose detects people + 17-point COCO skeleton keypoints in a single inference pass |
| 🔗 **Persistent Tracking** | DeepSORT assigns stable IDs across frames with re-identification and ghost suppression |
| 🖐️ **Hand Gesture Recognition** | MediaPipe 21-point hand landmarks detect open palm, fist, pointing, raised hand with temporal validation |
| 🏃 **Body Gesture Detection** | Keypoint-based detection of raised hand, waving, crouching, rapid movement, and pacing with multi-frame confirmation |
| 🧠 **Behavioral Analysis** | Deterministic heuristics detect loitering, pacing, dwell time, zone violations, and re-entry patterns |
| 🤖 **LLM Intent Reasoning** | GPT-4o-mini classifies per-person intent and scene-level risk from a structured Scene Graph |
| 👁️ **Visual Q&A** | Send a live frame + question to GPT-4o for on-demand visual question answering |
| 🎙️ **Voice Interaction** | Ask the agent questions using speech — it sees the frame, reasons, and speaks back |
| 🗣️ **Proactive TTS** | The agent auto-narrates important events (new arrivals, zone breaches, gestures) |
| 🎨 **Jarvis-Style Overlay** | Neon bounding boxes, energy skeletons, motion trails, arc-reactor center pulse, risk-colored HUD |
| 📊 **Event Timeline** | Auto-scrolling event feed with severity-coded entries and live counters |
| ⚡ **Rate Limiting & Safety** | 3-layer rate control (STT debounce → client cooldown → server 429) + global LLM budget with voice reservation |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     React 19 + Vite Frontend                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐  │
│  │VideoFeed │ │Detection │ │EventFeed │ │  VoicePanel   │  │
│  │ + Camera │ │ Overlay  │ │(timeline)│ │ (STT + TTS)   │  │
│  └─────┬────┘ └──────────┘ └──────────┘ └───────┬───────┘  │
│        │  base64 JPEG @ 1 fps                    │ voice    │
│        ▼                                         ▼ query    │
├────────────────── FastAPI Backend ───────────────────────────┤
│  ┌───────────── BehaviorEngine (orchestrator) ───────────┐  │
│  │                                                       │  │
│  │  Vision ──► Tracker ──► SceneBuilder ──► Behavior     │  │
│  │  (YOLO)    (DeepSORT)   (state)          Analyzer     │  │
│  │                                              │        │  │
│  │  HandTracker ──► GestureDetector             │        │  │
│  │  (MediaPipe)     (temporal keypoints)        ▼        │  │
│  │                              SceneAttributeEngine     │  │
│  │                              (posture, color, gaze)   │  │
│  │                                       │               │  │
│  │                              SceneGraphBuilder        │  │
│  │                              (structured repr)        │  │
│  │                                       │               │  │
│  │                              LLM Reasoner ◄── Memory  │  │
│  │                              (GPT-4o / mini)   Store  │  │
│  │                                       │               │  │
│  │                              EventBus (100-event buf) │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔍 Deep Dive: What IntentLens Can Do

### 1. Vision & Detection

- **YOLOv8n-pose** runs a single forward pass to produce both bounding boxes (all 80 COCO classes) and 17-point skeleton keypoints per person
- Confidence threshold at **0.50** — eliminates ghost detections from partial views or noise
- Per-joint keypoint confidence filtering at **0.40**
- Accepts base64-encoded JPEG/PNG frames via API

### 2. Multi-Person Tracking (DeepSORT)

- Persistent person IDs across frames with appearance-based re-identification
- Fast confirmation: only **2 frames** needed to confirm a new track (`n_init=2`)
- Configurable track age (`max_age=30`), IoU distance, NMS overlap
- **Duplicate suppression** — IoU-based deduplication (threshold 0.60) keeps older, more established tracks
- **Ghost track elimination** — stale tracks with no detection match (IoU < 0.25) are dropped
- Minimum bounding box area (**1200 px²**) rejects noise and partial detections

### 3. Hand Gesture Recognition (MediaPipe)

- **21-point hand landmarks** for up to 2 hands simultaneously
- Classified gestures: **open palm** (0.9), **fist** (0.85), **pointing** (0.8), **raised hand** (0.75)
- **Temporal validation** — gesture must appear in 4 of last 6 frames before firing
- Performance-optimized: hand detection only runs when persons are detected

### 4. Body Gesture Detection (Keypoints)

All gestures use **multi-frame temporal confirmation** — no single-frame false positives:

| Gesture | Detection Logic |
|---------|----------------|
| **Raised Hand** | Wrist above shoulder + elbow elevated for 6+ consecutive frames |
| **Waving** | Wrist lateral oscillation with 3+ direction reversals in a 12-frame window |
| **Crouching** | Hips near knee level (< 50% torso length) confirmed over 4+ frames |
| **Rapid Movement** | Velocity > 50 px/s sustained for 3+ frames |
| **Pacing** | Passed through from BehaviorAnalyzer (re-entered zones ≥ 2 times) |

- **6-second cooldown** per gesture per person prevents spamming
- Stale state pruned when a person leaves the scene

### 5. Behavioral Intelligence

Pure deterministic heuristics — no LLM required:

- **Dwell categorization** — brief (< 5s), short (< 30s), medium (< 120s), extended (> 120s)
- **Pacing detection** — re-entered zones ≥ 2 times AND not stationary
- **Loitering detection** — medium/extended dwell AND stationary velocity
- **Movement intensity** — still, low, moderate, high (from velocity)
- **Zone diversity** — how many distinct zones a person has visited
- **Re-entry detection** — fires when re-entries ≥ 3

### 6. Scene Understanding

- **3×3 zone grid** — frame divided into 9 zones; restricted zones (top-left, top-right) trigger zone breach alerts
- **Velocity computation** — hybrid bbox-center + wrist-keypoint velocity with EMA smoothing (α=0.4) and 500 px/s hard cap
- **Velocity classification** — stationary (< 2 px/s), slow (< 20), moderate (< 60), fast (≥ 60)
- **Scene activity level** — "quiet", "moderate", "busy" based on person count and fast-movers ratio
- **Posture classification** — upright, leaning, crouching (from shoulder-hip-knee geometry)
- **Dominant clothing color** — K-means clustering on torso crop, HSV classification (10 colors)
- **Head tilt** — neutral, tilted left/right, looking down (from eye-line angle)
- **Gaze direction** — forward, left, right, down (from nose/eye/ear geometry)
- **Object-in-hand** — IoU overlap between wrist region and non-person object bounding boxes

### 7. Scene Graph → LLM Reasoning

The **SceneGraphBuilder** aggregates all per-person data into a structured `SceneGraph` — no raw bounding boxes reach the LLM:

**Per-person attributes:** zone, dwell time, velocity label, posture, dominant color, head tilt, gaze direction, gesture state, pacing/loitering flags, zone diversity, re-entries, confidence, object in hand, motion intensity

**LLM outputs:**
- Per-person: `risk_level`, `explanation`, `alerts[]`, `recommended_action`
- Scene-level: overall risk assessment consuming the full graph
- Models: **GPT-4o-mini** for classification (temperature 0.15), **GPT-4o** for visual Q&A (0.3) and voice (0.4)
- Anti-hallucination rules enforced in all prompts — "never describe what data doesn't show"

### 8. Voice Interaction

A full bidirectional voice pipeline:

1. **Speech-to-Text** — Web Speech API with 1-second silence debounce, accumulated final transcripts
2. **Smart routing** — visual keywords (40+ words like "see", "wearing", "color") send the live frame to GPT-4o; meta questions ("who are you") use GPT-4o-mini text-only
3. **Conversational memory** — 3-turn rolling history for coherent multi-turn dialogue
4. **Text-to-Speech** — Web Speech Synthesis with voice selection (Samantha, Google UK Female, etc.)
5. **Proactive narration** — events marked `speakable` are auto-spoken (new arrivals, zone breaches, gestures) with 8-second cooldown

### 9. Rate Limiting & Quota Safety

Three-layer protection to prevent API cost explosions:

| Layer | Mechanism |
|-------|-----------|
| **STT Debounce** | 1s silence timer — prevents rapid-fire transcripts |
| **Client Cooldown** | 4s cooldown after each voice query — mic blocked, UI indicator shown |
| **Server Cooldown** | 3s cooldown between voice requests — returns HTTP 429 |
| **Global Budget** | Max 10 LLM calls per 60-second sliding window |
| **Voice Reservation** | 3 of 10 budget slots reserved exclusively for voice queries |
| **Per-person Cache** | Intent classification cached for 4 seconds per person |
| **Dwell Gate** | LLM not called until a person has been present for 1.5 seconds |
| **Quota Detection** | `insufficient_quota` error permanently disables all LLM calls with clear user message |

### 10. Event System

A rolling 100-event bus powers the event feed and proactive TTS:

| Event | Trigger | Severity |
|-------|---------|----------|
| `new_person` | First appearance of a track ID | info |
| `person_departed` | Person pruned from scene | info |
| `zone_breach` | Person enters a restricted zone | alert |
| `gesture_raised_hand` | Raised hand confirmed | info |
| `gesture_waving` | Waving confirmed | info |
| `gesture_crouching` | Crouching confirmed | warning |
| `gesture_rapid_movement` | Rapid movement detected | warning |
| `risk_change` | Risk level escalated | warning/alert |
| `object_detected` | Person holding a new object | info |
| `hand_open_palm` / `hand_fist` / `hand_pointing` | Hand gesture detected | info |

Events marked `speakable=True` are automatically narrated by the TTS engine.

---

## 🖥️ UI: Jarvis-Style HUD

The frontend renders a cinematic heads-up display over the webcam feed:

- **Corner-bracket bounding boxes** — neon cyan (low risk), amber (medium), red (high) with SVG glow filters
- **Energy skeleton** — 12 limb-pair connections with velocity-reactive opacity and glow
- **Joint energy dots** — shoulders, elbows, wrists, hips with velocity-reactive size
- **Motion trails** — wrist position history (8 points) rendered as glowing paths
- **Arc-reactor center pulse** — animated center dot per person, larger when moving
- **ID + Risk label** — monospace HUD tag at top of each person's bounding box
- **Info tags** — active gestures, velocity, pacing, loitering, dominant color shown at bottom
- **High-risk pulse** — outer pulsing red border on high-risk persons
- **Status bar** — LIVE / SCANNING / LISTENING / SPEAKING indicators with animated dots
- **Event feed** — auto-scrolling timeline with severity color-coding and icons (◈ / ⚠ / ⚡)
- **Voice panel** — mic toggle, live transcript, agent response, waveform during TTS

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Video Platform** | **Vision Agents by Stream** (`@stream-io/video-react-sdk`, `@stream-io/video-client`) |
| **Video Tokens** | **Stream Chat SDK** (Python `stream-chat` — server-side token generation) |
| Detection | YOLOv8n-pose (Ultralytics) |
| Tracking | DeepSORT (appearance re-ID) |
| Hand Tracking | MediaPipe Hands (21 landmarks) |
| LLM | OpenAI GPT-4o / GPT-4o-mini |
| Backend | Python 3.11 · FastAPI · Uvicorn |
| Frontend | React 19 · TypeScript · Vite 7 |
| STT | Web Speech API (SpeechRecognition) |
| TTS | Web Speech Synthesis API |

---

## 📁 Project Structure

```
backend/
  main.py                    # FastAPI app, endpoints (/analyze, /voice_query, /visual_qa)
  config.py                  # Centralized configuration (all thresholds + env vars)
  vision.py                  # YOLOv8 detection + keypoint extraction
  tracking.py                # DeepSORT tracker integration
  scene_builder.py           # Build per-frame scene state from detections
  behavior_analyzer.py       # Deterministic behavioral signal extraction
  behavior_engine.py         # Pipeline orchestrator — wires 9 sub-components
  llm_reasoner.py            # LLM reasoning, voice queries, VQA, rate limiting
  memory.py                  # Per-person rolling memory + session snapshots
  scene_graph.py             # Scene graph construction for LLM consumption
  scene_attribute_engine.py  # Posture, clothing color, head tilt, gaze
  gesture_detector.py        # Temporal keypoint-based gesture recognition
  hand_tracker.py            # MediaPipe hand landmark detection + classification
  models.py                  # Pydantic + dataclass domain models
  events.py                  # Event bus (100-event rolling buffer)

frontend/intentlens-frontend/
  src/
    App.tsx                  # Main app layout + theme
    api.ts                   # Axios-based backend API client (15s timeout)
    types.ts                 # TypeScript type definitions
    components/
      VideoFeed.tsx          # Webcam + frame capture + composite layout
      DetectionOverlay.tsx   # Jarvis-style SVG overlay (skeletons, trails, HUD)
      StatusBar.tsx          # LIVE / SCANNING / LISTENING / SPEAKING indicators
      EventFeed.tsx          # Auto-scrolling event timeline
      VoicePanel.tsx         # Mic toggle, transcript, agent response, cooldown UI
      ExplanationPanel.tsx   # Risk gauge, per-person cards, scene overview
    hooks/
      useFrameAnalyzer.ts    # Captures frames at 1 fps, sends to /analyze
      useSpeechRecognition.ts # STT with silence debounce + accumulated transcripts
      useSpeechSynthesis.ts  # TTS with voice selection + proactive speaking
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- OpenAI API key with GPT-4o access
- Webcam

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```env
OPENAI_API_KEY=sk-your-openai-key
STREAM_API_KEY=your-stream-key        # optional — for Stream Chat
STREAM_API_SECRET=your-stream-secret  # optional — for Stream Chat
```

Start the server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend Setup

```bash
cd frontend/intentlens-frontend
npm install
npm run dev
```

The app opens at `http://localhost:5173`. Grant camera and microphone permissions when prompted.

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/analyze` | Full frame analysis pipeline (detect → track → reason → respond) |
| `POST` | `/voice_query` | Conversational voice query with scene context + optional frame |
| `POST` | `/visual_qa` | On-demand visual question answering with GPT-4o |
| `POST` | `/token` | Stream Video token generation |

---

## 🌐 Deployment

- **Frontend**: Deployed on Vercel — [intentlens-theta.vercel.app](https://intentlens-theta.vercel.app)
- **Backend**: Run locally or deploy to any server with Python 3.11+ and GPU (optional but recommended for YOLO inference speed)

---

## 🏆 Hackathon Alignment

How IntentLens maps to the **Vision Possible: Agent Protocol** judging criteria:

| Criteria | How IntentLens Delivers |
|----------|------------------------|
| **Potential Impact** | Real-time behavioral understanding for security, retail analytics, smart spaces, and accessibility |
| **Creativity & Innovation** | Combines pose estimation + gesture recognition + behavioral heuristics + LLM reasoning + voice interaction into one cohesive agent — goes far beyond simple object detection |
| **Technical Excellence** | 9-component pipeline with temporal gesture confirmation, EMA-smoothed velocity, structured Scene Graph, anti-hallucination LLM prompts, and 3-layer rate limiting |
| **Real-Time Performance** | Stream's edge network for <30ms video latency + 1 fps frame analysis + instant event narration |
| **User Experience** | Jarvis-style neon HUD, voice conversation, proactive TTS narration, severity-colored event timeline |
| **Best Use of Vision Agents** | Uses `@stream-io/video-react-sdk` + `@stream-io/video-client` for the video pipeline, `stream-chat` for token auth, leverages Stream's low-latency edge network |

---

## 👤 Author

**Rishi Jat** — [GitHub](https://github.com/rishi-jat)

Built with ☕ for the [Vision Possible: Agent Protocol](https://www.wemakedevs.org/events/hackathons/vision-possible) hackathon by WeMakeDevs × Vision Agents by Stream.

## 📄 License

MIT
