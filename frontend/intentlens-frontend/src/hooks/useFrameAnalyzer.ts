/**
 * useFrameAnalyzer — captures frames from a <video> element at a set interval
 * and sends them to the backend for analysis.
 *
 * Returns the latest AnalyzeResponse so the overlay can render it.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { AnalyzeResponse } from "../types";
import { analyzeFrame } from "../api";

/** Milliseconds between frame captures — balanced for responsiveness + backend capacity */
const CAPTURE_INTERVAL_MS = 500;

/** JPEG quality (0–1) for the canvas export */
const JPEG_QUALITY = 0.55;

interface UseFrameAnalyzerOptions {
  /** Whether analysis is currently enabled */
  enabled: boolean;
  /** Interval override in ms */
  intervalMs?: number;
}

interface UseFrameAnalyzerReturn {
  /** Ref to attach to the <video> element */
  videoRef: React.RefObject<HTMLVideoElement | null>;
  /** Latest analysis result */
  analysis: AnalyzeResponse | null;
  /** Whether an analysis request is in-flight */
  isAnalyzing: boolean;
  /** Last error, if any */
  error: string | null;
}

export function useFrameAnalyzer(
  options: UseFrameAnalyzerOptions
): UseFrameAnalyzerReturn {
  const { enabled, intervalMs = CAPTURE_INTERVAL_MS } = options;

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const inFlightRef = useRef(false);

  const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const captureAndAnalyze = useCallback(async () => {
    if (inFlightRef.current) return;
    const video = videoRef.current;
    if (!video || video.readyState < 2) return;

    // Lazily create an offscreen canvas
    if (!canvasRef.current) {
      canvasRef.current = document.createElement("canvas");
    }

    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg", JPEG_QUALITY);
    // Strip the data:image/jpeg;base64, prefix
    const base64 = dataUrl.split(",")[1];
    if (!base64) return;

    inFlightRef.current = true;
    setIsAnalyzing(true);
    setError(null);

    try {
      const result = await analyzeFrame(base64);
      setAnalysis(result);
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : "Analysis request failed";
      setError(message);
    } finally {
      inFlightRef.current = false;
      setIsAnalyzing(false);
    }
  }, []);

  useEffect(() => {
    if (!enabled) return;

    const id = setInterval(() => {
      void captureAndAnalyze();
    }, intervalMs);

    return () => clearInterval(id);
  }, [enabled, intervalMs, captureAndAnalyze]);

  return { videoRef, analysis, isAnalyzing, error };
}
