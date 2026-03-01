/**
 * VideoFeed — Main composite component (v4 — Final Push).
 *
 * Layout: StatusBar on top, then [Video + ExplanationPanel], then [EventFeed + VoicePanel].
 * Lifts TTS to this level for proactive event speaking + voice response speaking.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useFrameAnalyzer } from "../hooks/useFrameAnalyzer";
import { useSpeechSynthesis } from "../hooks/useSpeechSynthesis";
import { DetectionOverlay } from "./DetectionOverlay";
import { EventFeed } from "./EventFeed";
import { StatusBar } from "./StatusBar";
import { VoicePanel } from "./VoicePanel";
import type { AgentEvent } from "../types";

const JPEG_QUALITY = 0.6;
/** Minimum seconds between proactive TTS utterances */
const PROACTIVE_COOLDOWN_SECS = 8;

export function VideoFeed() {
  const [cameraReady, setCameraReady] = useState(false);
  const [videoDimensions, setVideoDimensions] = useState({
    width: 0,
    height: 0,
  });
  const streamRef = useRef<MediaStream | null>(null);
  const [allEvents, setAllEvents] = useState<AgentEvent[]>([]);
  const [isListening, setIsListening] = useState(false);

  // Lift TTS to this level so both VoicePanel and proactive events share it
  const { speak, stop: stopSpeech, isSpeaking, isSupported: ttsSupported } =
    useSpeechSynthesis();

  const { videoRef, analysis, isAnalyzing, error } = useFrameAnalyzer({
    enabled: cameraReady,
    intervalMs: 1_000,
  });

  // Proactive TTS: auto-speak events marked as speakable
  const lastProactiveRef = useRef(0);
  useEffect(() => {
    if (!analysis?.events || analysis.events.length === 0) return;

    const speakableEvents = analysis.events.filter(
      (e) => e.speakable && e.message
    );
    if (speakableEvents.length === 0) return;

    // Don't speak if currently listening or already speaking
    if (isListening || isSpeaking) return;

    const now = Date.now() / 1000;
    if (now - lastProactiveRef.current < PROACTIVE_COOLDOWN_SECS) return;

    // Speak the first speakable event
    speak(speakableEvents[0].message);
    lastProactiveRef.current = now;
  }, [analysis?.events, isListening, isSpeaking, speak]);

  // Accumulate events from each analysis response
  useEffect(() => {
    if (analysis?.events && analysis.events.length > 0) {
      setAllEvents((prev) => {
        const next = [...prev, ...analysis.events];
        return next.length > 100 ? next.slice(-100) : next;
      });
    }
  }, [analysis]);

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Camera access denied";
      console.error("Camera error:", message);
    }
  }, [videoRef]);

  useEffect(() => {
    void startCamera();

    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, [startCamera]);

  const handleLoadedMetadata = useCallback(() => {
    const video = videoRef.current;
    if (video) {
      setVideoDimensions({
        width: video.videoWidth,
        height: video.videoHeight,
      });
      setCameraReady(true);
    }
  }, [videoRef]);

  // Provide current frame to VoicePanel for visual context
  const getFrame = useCallback((): string | undefined => {
    const video = videoRef.current;
    if (!video || video.readyState < 2) return undefined;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return undefined;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg", JPEG_QUALITY);
    return dataUrl.split(",")[1];
  }, [videoRef]);

  const persons = analysis?.persons ?? [];

  return (
    <div className="video-feed-container">
      <StatusBar
        isConnected={cameraReady}
        isAnalyzing={isAnalyzing}
        analysisError={error}
        personCount={persons.length}
        latencyMs={analysis?.latency_ms}
        isListening={isListening}
        isSpeaking={isSpeaking}
      />

      <div className="video-feed-main">
        <div className="video-feed-left">
          <div className="video-container">
            <video
              ref={videoRef}
              onLoadedMetadata={handleLoadedMetadata}
              muted
              playsInline
              className="video-element"
            />
            <DetectionOverlay
              persons={persons}
              videoWidth={videoDimensions.width}
              videoHeight={videoDimensions.height}
              sceneGraph={analysis?.scene_graph ?? null}
            />
          </div>

          {/* Bottom bar: Events + Voice */}
          <div className="bottom-bar">
            <EventFeed events={allEvents} />
            <VoicePanel
              getFrame={getFrame}
              speak={speak}
              stopSpeech={stopSpeech}
              isSpeaking={isSpeaking}
              ttsSupported={ttsSupported}
              onListeningChange={setIsListening}
            />
          </div>
        </div>

        {/* Minimal ExplanationPanel - hidden by default, can toggle */}
        {/* Uncomment to show: */}
        {/* <ExplanationPanel
          persons={persons}
          sceneAnalysis={analysis?.analysis}
          latencyMs={analysis?.latency_ms}
        /> */}
      </div>
    </div>
  );
}
