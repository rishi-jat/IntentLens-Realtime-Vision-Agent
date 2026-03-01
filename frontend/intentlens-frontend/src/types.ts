/**
 * IntentLens — Shared TypeScript types (v4 — Final Evolution).
 *
 * Mirrors the Pydantic response models in backend/main.py.
 */

// --- Primitives ---

export interface DetectionOut {
  label: string;
  confidence: number;
  bbox: number[];
  class_id: number;
}

export interface KeypointOut {
  x: number;
  y: number;
  confidence: number;
}

export interface PersonKeypointsOut {
  bbox: number[];
  keypoints: KeypointOut[];
}

export interface PersonAttributesOut {
  posture: string | null;
  dominant_color: string | null;
  head_tilt: string | null;
  gaze_direction: string | null;
}

export interface TrackedPersonOut {
  track_id: number;
  bbox: number[];
  center: number[];
  velocity: number;
  velocity_label: string;
  zone: string | null;
  dwell_time: number;
  zones_entered: string[];
  repeated_approaches: number;
  timeline: string[];
  keypoints: PersonKeypointsOut | null;
  attributes: PersonAttributesOut | null;
}

export interface AnalysisOut {
  risk_level: string; // "low" | "medium" | "high"
  explanation: string;
  alerts: string[];
  recommended_action: string;
}

// --- Events ---

export type EventSeverity = "info" | "warning" | "alert";

export interface AgentEvent {
  kind: string;
  timestamp: number;
  message: string;
  person_id: number | null;
  severity: EventSeverity;
  data: Record<string, unknown>;
  speakable?: boolean;
}

// --- Scene Graph ---

export interface SceneGraphPersonOut {
  person_id: number;
  zone: string | null;
  dwell_time: number;
  velocity_label: string;
  posture: string | null;
  dominant_color: string | null;
  head_tilt: string | null;
  gaze_direction: string | null;
  gesture_state: string | null;
  is_pacing: boolean;
  is_loitering: boolean;
  zone_diversity: number;
  repeated_approaches: number;
  confidence: number;
  object_in_hand: string | null;
  motion_intensity: string;
}

export interface SceneGraphOut {
  timestamp: number;
  total_persons: number;
  activity_level: string;
  persons: SceneGraphPersonOut[];
  objects: string[];
}

// --- Composite ---

export interface SceneOut {
  timestamp: number;
  frame_width: number;
  frame_height: number;
  people: TrackedPersonOut[];
  objects: DetectionOut[];
}

export interface PersonAnalysisOut {
  person_id: number;
  bbox: number[];
  behavior: TrackedPersonOut;
  intent: AnalysisOut;
}

// --- Endpoint responses ---

export interface AnalyzeResponse {
  scene: SceneOut;
  persons: PersonAnalysisOut[];
  analysis: AnalysisOut;
  latency_ms: number;
  events: AgentEvent[];
  scene_graph: SceneGraphOut | null;
}

export interface VisualQAResponse {
  answer: string;
  latency_ms: number;
}

export interface VoiceQueryResponse {
  response: string;
  events: AgentEvent[];
  latency_ms: number;
}

export interface TokenResponse {
  token: string;
  user_id: string;
}

export interface HealthResponse {
  status: string;
  timestamp: number;
}
