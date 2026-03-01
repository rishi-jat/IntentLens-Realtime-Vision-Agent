/**
 * IntentLens — API service layer (v3 — Hackathon Edition).
 *
 * All backend communication lives here. No API calls inside components.
 */

import axios, { type AxiosInstance } from "axios";
import type {
  AnalyzeResponse,
  VisualQAResponse,
  VoiceQueryResponse,
  TokenResponse,
  HealthResponse,
} from "./types";

const BASE_URL = "http://localhost:8000";

const client: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  timeout: 15_000,
  headers: { "Content-Type": "application/json" },
});

export async function fetchToken(userId: string): Promise<TokenResponse> {
  const { data } = await client.post<TokenResponse>("/token", {
    user_id: userId,
  });
  return data;
}

export async function analyzeFrame(
  base64Frame: string
): Promise<AnalyzeResponse> {
  const { data } = await client.post<AnalyzeResponse>("/analyze", {
    frame: base64Frame,
  });
  return data;
}

export async function visualQA(
  base64Frame: string,
  question: string
): Promise<VisualQAResponse> {
  const { data } = await client.post<VisualQAResponse>("/visual_qa", {
    frame: base64Frame,
    question,
  });
  return data;
}

export async function voiceQuery(
  transcript: string,
  base64Frame?: string
): Promise<VoiceQueryResponse> {
  const { data } = await client.post<VoiceQueryResponse>("/voice_query", {
    transcript,
    frame: base64Frame ?? null,
  });
  return data;
}

export async function healthCheck(): Promise<HealthResponse> {
  const { data } = await client.get<HealthResponse>("/health");
  return data;
}
