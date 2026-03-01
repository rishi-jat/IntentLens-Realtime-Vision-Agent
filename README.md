# IntentLens — Real-Time Vision Agent

A real-time multi-modal AI agent that uses computer vision, pose estimation, and LLM reasoning to understand human behavior through a live camera feed.

## Features

- **YOLOv8 Pose Detection** — Real-time person detection with skeleton keypoints
- **DeepSORT Tracking** — Persistent multi-person tracking across frames
- **Behavioral Analysis** — Velocity, pacing, loitering, zone monitoring
- **LLM Reasoning** — GPT-4o powered intent classification and risk assessment
- **Voice Interaction** — Ask questions about what the camera sees
- **Visual Q&A** — Send a frame to GPT-4o for visual grounding
- **Scene Graph** — Structured scene understanding with object-in-hand detection
- **React Dashboard** — Real-time visualization with bounding boxes, skeletons, and event feed

## Tech Stack

| Layer | Tech |
|-------|------|
| Detection | YOLOv8n-pose (Ultralytics) |
| Tracking | DeepSORT |
| Pose | MediaPipe / YOLOv8 keypoints |
| LLM | OpenAI GPT-4o / GPT-4o-mini |
| Backend | FastAPI + Uvicorn |
| Frontend | React 19 + TypeScript + Vite |
| Streaming | Stream Chat SDK |

## Project Structure

```
backend/
  main.py              # FastAPI app, endpoints
  config.py            # Centralized configuration
  vision.py            # YOLOv8 detection + keypoint extraction
  tracking.py          # DeepSORT tracker integration
  scene_builder.py     # Build scene state from detections
  behavior_analyzer.py # Behavioral signal extraction
  behavior_engine.py   # Pipeline orchestrator
  llm_reasoner.py      # LLM reasoning + voice + VQA
  memory.py            # Per-person rolling memory
  scene_graph.py       # Scene graph construction
  models.py            # Pydantic data models
  events.py            # Event bus for agent events
  gesture_detector.py  # Gesture recognition
  hand_tracker.py      # Hand/wrist tracking
  scene_attribute_engine.py # Scene attribute extraction

frontend/intentlens-frontend/
  src/
    App.tsx            # Main app layout
    api.ts             # Backend API client
    types.ts           # TypeScript types
    components/        # UI components
    hooks/             # Custom React hooks
```

## Setup

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # Add your API keys
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd frontend/intentlens-frontend
npm install
cp .env.example .env   # Add your Stream API key
npm run dev
```

### Environment Variables

**Backend** (`.env`):
- `OPENAI_API_KEY` — OpenAI API key for GPT-4o
- `STREAM_API_KEY` — Stream Chat API key
- `STREAM_API_SECRET` — Stream Chat secret

**Frontend** (`.env`):
- `VITE_STREAM_API_KEY` — Stream Chat client key

## License

MIT
