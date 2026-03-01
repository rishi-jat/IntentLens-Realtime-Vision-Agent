/**
 * useSpeechRecognition — Web Speech API STT hook (v2 — Stabilized).
 *
 * Key design:
 * - Only emits final transcripts (ignores partial for triggering).
 * - 1000ms silence debounce before marking complete.
 * - Clean state management, no duplicate triggers.
 */

import { useCallback, useEffect, useRef, useState } from "react";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type SpeechRecognitionInstance = any;

function getSpeechRecognitionCtor(): (new () => SpeechRecognitionInstance) | null {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const w = window as any;
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

export interface UseSpeechRecognitionReturn {
  isListening: boolean;
  transcript: string;
  interimTranscript: string;
  isComplete: boolean;
  startListening: () => void;
  stopListening: () => void;
  isSupported: boolean;
  error: string | null;
}

/** Silence before auto-stop (ms) */
const SILENCE_DEBOUNCE_MS = 1000;

export function useSpeechRecognition(): UseSpeechRecognitionReturn {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [interimTranscript, setInterimTranscript] = useState("");
  const [isComplete, setIsComplete] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const recognitionRef = useRef<SpeechRecognitionInstance | null>(null);
  const silenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const accumulatedRef = useRef("");

  const isSupported =
    typeof window !== "undefined" && getSpeechRecognitionCtor() !== null;

  const clearSilenceTimer = useCallback(() => {
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
  }, []);

  const getRecognition = useCallback(() => {
    if (recognitionRef.current) return recognitionRef.current;

    const Ctor = getSpeechRecognitionCtor();
    if (!Ctor) return null;

    const recognition = new Ctor();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";
    recognition.maxAlternatives = 1;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    recognition.onresult = (event: any) => {
      let finalText = "";
      let interimText = "";

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        if (result.isFinal) {
          finalText += result[0].transcript;
        } else {
          interimText += result[0].transcript;
        }
      }

      if (finalText) {
        const trimmed = finalText.trim();
        accumulatedRef.current = accumulatedRef.current
          ? accumulatedRef.current + " " + trimmed
          : trimmed;
        setTranscript(accumulatedRef.current);
        setInterimTranscript("");

        // Reset silence timer — user just spoke
        clearSilenceTimer();
        silenceTimerRef.current = setTimeout(() => {
          if (recognitionRef.current) {
            try { recognitionRef.current.stop(); } catch { /* */ }
          }
        }, SILENCE_DEBOUNCE_MS);
      } else if (interimText) {
        setInterimTranscript(interimText);
      }
    };

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    recognition.onerror = (event: any) => {
      if (event.error === "no-speech" || event.error === "aborted") return;
      setError(event.error);
      setIsListening(false);
    };

    recognition.onend = () => {
      setIsListening(false);
      setInterimTranscript("");
      // Only complete if we have a final transcript
      if (accumulatedRef.current.trim()) {
        setIsComplete(true);
      }
    };

    recognitionRef.current = recognition;
    return recognition;
  }, [clearSilenceTimer]);

  const startListening = useCallback(() => {
    setError(null);
    setTranscript("");
    setInterimTranscript("");
    setIsComplete(false);
    accumulatedRef.current = "";
    clearSilenceTimer();

    const recognition = getRecognition();
    if (!recognition) {
      setError("Speech recognition not supported");
      return;
    }

    try {
      recognition.start();
      setIsListening(true);
    } catch {
      // Already started
    }
  }, [getRecognition, clearSilenceTimer]);

  const stopListening = useCallback(() => {
    clearSilenceTimer();
    const recognition = recognitionRef.current;
    if (recognition) {
      try { recognition.stop(); } catch { /* */ }
    }
    setIsListening(false);
  }, [clearSilenceTimer]);

  useEffect(() => {
    return () => {
      clearSilenceTimer();
      if (recognitionRef.current) {
        try { recognitionRef.current.stop(); } catch { /* */ }
      }
    };
  }, [clearSilenceTimer]);

  return {
    isListening,
    transcript,
    interimTranscript,
    isComplete,
    startListening,
    stopListening,
    isSupported,
    error,
  };
}
