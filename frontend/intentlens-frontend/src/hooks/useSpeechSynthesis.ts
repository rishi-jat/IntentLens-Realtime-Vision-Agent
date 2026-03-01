/**
 * useSpeechSynthesis — Browser TTS hook.
 *
 * Wraps the browser's SpeechSynthesis API with a React-friendly interface.
 * Uses a calm, natural voice when available.
 */

import { useCallback, useEffect, useRef, useState } from "react";

export interface UseSpeechSynthesisReturn {
  /** Speak the given text */
  speak: (text: string) => void;
  /** Stop any current speech */
  stop: () => void;
  /** Whether the agent is currently speaking */
  isSpeaking: boolean;
  /** Whether the browser supports speech synthesis */
  isSupported: boolean;
}

/** Preferred voice names — we pick the first match available */
const PREFERRED_VOICES = [
  "Samantha",          // macOS
  "Google UK English Female",
  "Google US English",
  "Microsoft Zira",    // Windows
  "English (America)", // Android
];

function pickVoice(voices: SpeechSynthesisVoice[]): SpeechSynthesisVoice | null {
  for (const name of PREFERRED_VOICES) {
    const match = voices.find((v) => v.name.includes(name));
    if (match) return match;
  }
  // Fallback: first English voice
  const english = voices.find((v) => v.lang.startsWith("en"));
  return english ?? voices[0] ?? null;
}

export function useSpeechSynthesis(): UseSpeechSynthesisReturn {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const voiceRef = useRef<SpeechSynthesisVoice | null>(null);
  const isSupported =
    typeof window !== "undefined" && "speechSynthesis" in window;

  // Resolve voice once voices are loaded
  useEffect(() => {
    if (!isSupported) return;

    const resolve = () => {
      const voices = speechSynthesis.getVoices();
      if (voices.length > 0) {
        voiceRef.current = pickVoice(voices);
      }
    };

    resolve(); // voices may already be loaded
    speechSynthesis.addEventListener("voiceschanged", resolve);
    return () => speechSynthesis.removeEventListener("voiceschanged", resolve);
  }, [isSupported]);

  const speak = useCallback(
    (text: string) => {
      if (!isSupported || !text.trim()) return;

      // Cancel any current speech first
      speechSynthesis.cancel();

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      utterance.volume = 0.9;

      if (voiceRef.current) {
        utterance.voice = voiceRef.current;
      }

      utterance.onstart = () => setIsSpeaking(true);
      utterance.onend = () => setIsSpeaking(false);
      utterance.onerror = () => setIsSpeaking(false);

      speechSynthesis.speak(utterance);
    },
    [isSupported]
  );

  const stop = useCallback(() => {
    if (!isSupported) return;
    speechSynthesis.cancel();
    setIsSpeaking(false);
  }, [isSupported]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (isSupported) speechSynthesis.cancel();
    };
  }, [isSupported]);

  return { speak, stop, isSpeaking, isSupported };
}
