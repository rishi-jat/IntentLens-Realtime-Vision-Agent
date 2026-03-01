/**
 * VoicePanel — Voice interaction UI with waveform animation (v4).
 *
 * TTS is lifted to VideoFeed level to share with proactive events.
 * Shows: mic button, live transcript, AI response, speaking waveform.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useSpeechRecognition } from "../hooks/useSpeechRecognition";
import { voiceQuery } from "../api";

interface VoicePanelProps {
  /** Base64 frame to optionally attach for visual context */
  getFrame?: () => string | undefined;
  /** TTS speak function (lifted from VideoFeed) */
  speak: (text: string) => void;
  /** Stop TTS */
  stopSpeech: () => void;
  /** Whether TTS is currently speaking */
  isSpeaking: boolean;
  /** Whether TTS is supported */
  ttsSupported: boolean;
  /** Report listening state changes to parent */
  onListeningChange?: (listening: boolean) => void;
}

export function VoicePanel({
  getFrame,
  speak,
  stopSpeech,
  isSpeaking,
  ttsSupported,
  onListeningChange,
}: VoicePanelProps) {
  const {
    isListening,
    transcript,
    interimTranscript,
    isComplete,
    startListening,
    stopListening,
    isSupported: sttSupported,
    error: sttError,
  } = useSpeechRecognition();

  const [agentResponse, setAgentResponse] = useState("");
  const [isThinking, setIsThinking] = useState(false);
  const [isCoolingDown, setIsCoolingDown] = useState(false);
  const sentTranscriptRef = useRef("");
  const cooldownTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  /** Cooldown between consecutive voice queries (ms) */
  const VOICE_COOLDOWN_MS = 4000;

  // Report listening state changes to parent (VideoFeed → StatusBar)
  useEffect(() => {
    onListeningChange?.(isListening);
  }, [isListening, onListeningChange]);

  // When listening stops and transcript is complete, send to backend
  useEffect(() => {
    if (!isComplete || !transcript || transcript === sentTranscriptRef.current) return;
    if (isCoolingDown) return; // Reject during cooldown
    sentTranscriptRef.current = transcript;

    const sendQuery = async () => {
      setIsThinking(true);
      setAgentResponse("");
      try {
        const frame = getFrame?.();
        const result = await voiceQuery(transcript, frame);
        setAgentResponse(result.response);
        if (ttsSupported) {
          speak(result.response);
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Voice query failed";
        // On 429, show friendly message
        if (msg.includes("429") || msg.includes("rate") || msg.includes("cooldown")) {
          setAgentResponse("I'm processing your last question. Please wait a moment.");
        } else {
          console.error("Voice query error:", err);
          setAgentResponse(`Voice error: ${msg}`);
        }
      } finally {
        setIsThinking(false);
        // Start cooldown
        setIsCoolingDown(true);
        cooldownTimerRef.current = setTimeout(() => {
          setIsCoolingDown(false);
        }, VOICE_COOLDOWN_MS);
      }
    };

    void sendQuery();
  }, [isComplete, transcript, getFrame, speak, ttsSupported, isCoolingDown]);

  // Cleanup cooldown timer
  useEffect(() => {
    return () => {
      if (cooldownTimerRef.current) clearTimeout(cooldownTimerRef.current);
    };
  }, []);

  const toggleMic = useCallback(() => {
    if (isCoolingDown) return; // Block mic during cooldown
    if (isListening) {
      stopListening();
    } else {
      stopSpeech();
      startListening();
    }
  }, [isListening, isCoolingDown, startListening, stopListening, stopSpeech]);

  if (!sttSupported) {
    return (
      <div className="voice-panel">
        <div className="voice-unsupported">
          Voice not supported in this browser. Use Chrome for full experience.
        </div>
      </div>
    );
  }

  return (
    <div className="voice-panel">
      {/* Mic button */}
      <button
        className={`voice-mic-btn ${isListening ? "listening" : ""} ${isSpeaking ? "speaking" : ""} ${isCoolingDown ? "cooldown" : ""}`}
        onClick={toggleMic}
        title={isCoolingDown ? "Please wait…" : isListening ? "Stop listening" : "Start talking to IntentLens"}
        disabled={isCoolingDown}
      >
        {isListening ? (
          <MicActiveIcon />
        ) : isSpeaking ? (
          <SpeakerIcon />
        ) : (
          <MicIcon />
        )}
      </button>

      {/* Status line */}
      <div className="voice-status">
        {isListening && (
          <span className="voice-status-listening">
            <span className="voice-pulse-dot" />
            Listening…
          </span>
        )}
        {isThinking && (
          <span className="voice-status-thinking">Thinking…</span>
        )}
        {isSpeaking && (
          <span className="voice-status-speaking">
            <WaveformMini />
            Speaking
          </span>
        )}
        {!isListening && !isThinking && !isSpeaking && agentResponse && !isCoolingDown && (
          <span className="voice-status-idle">Ready</span>
        )}
        {isCoolingDown && !isThinking && !isSpeaking && (
          <span className="voice-status-cooldown">Cooldown…</span>
        )}
      </div>

      {/* Transcript / Response area */}
      <div className="voice-transcript-area">
        {isListening && interimTranscript && (
          <div className="voice-interim">"{interimTranscript}"</div>
        )}
        {transcript && !isListening && (
          <div className="voice-user-text">
            <span className="voice-label">You:</span> {transcript}
          </div>
        )}
        {agentResponse && (
          <div className="voice-agent-text">
            <span className="voice-label">IntentLens:</span> {agentResponse}
          </div>
        )}
      </div>

      {sttError && <div className="voice-error">{sttError}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Inline SVG icons
// ---------------------------------------------------------------------------

function MicIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
      <line x1="12" y1="19" x2="12" y2="23" />
      <line x1="8" y1="23" x2="16" y2="23" />
    </svg>
  );
}

function MicActiveIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" strokeWidth="1">
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2" fill="none" strokeWidth="2" />
      <line x1="12" y1="19" x2="12" y2="23" strokeWidth="2" />
      <line x1="8" y1="23" x2="16" y2="23" strokeWidth="2" />
    </svg>
  );
}

function SpeakerIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
      <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
      <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
    </svg>
  );
}

function WaveformMini() {
  return (
    <svg className="voice-waveform-mini" width="24" height="14" viewBox="0 0 24 14">
      <rect className="waveform-bar bar-1" x="1" y="5" width="2" height="4" rx="1" fill="currentColor" />
      <rect className="waveform-bar bar-2" x="5" y="3" width="2" height="8" rx="1" fill="currentColor" />
      <rect className="waveform-bar bar-3" x="9" y="1" width="2" height="12" rx="1" fill="currentColor" />
      <rect className="waveform-bar bar-4" x="13" y="3" width="2" height="8" rx="1" fill="currentColor" />
      <rect className="waveform-bar bar-5" x="17" y="4" width="2" height="6" rx="1" fill="currentColor" />
      <rect className="waveform-bar bar-6" x="21" y="5" width="2" height="4" rx="1" fill="currentColor" />
    </svg>
  );
}
