/**
 * useSpeechRecognition — Web Speech API STT hook.
 *
 * Provides continuous speech-to-text via the browser's built-in
 * SpeechRecognition API (free, no external service needed).
 */

import { useCallback, useEffect, useRef, useState } from "react";

// Web Speech API — vendor-prefixed in most browsers
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type SpeechRecognitionInstance = any;

function getSpeechRecognitionCtor(): (new () => SpeechRecognitionInstance) | null {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const w = window as any;
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

export interface UseSpeechRecognitionReturn {
  /** Whether we are currently listening */
  isListening: boolean;
  /** The latest finalised transcript */
  transcript: string;
  /** Interim (live) partial transcript while speaking */
  interimTranscript: string;
  /** True when listening stopped and transcript is ready to send */
  isComplete: boolean;
  /** Start listening */
  startListening: () => void;
  /** Stop listening */
  stopListening: () => void;
  /** Whether the browser supports the Web Speech API */
  isSupported: boolean;
  /** Last error, if any */
  error: string | null;
}

export function useSpeechRecognition(): UseSpeechRecognitionReturn {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [interimTranscript, setInterimTranscript] = useState("");
  const [isComplete, setIsComplete] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const recognitionRef = useRef<SpeechRecognitionInstance | null>(null);
  const isSupported = typeof window !== "undefined" && getSpeechRecognitionCtor() !== null;
  const sendTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const silenceTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  
  /** Auto-send transcript after 1 second of silence */
  const SILENCE_DURATION_MS = 1000;

  // Create recognition instance lazily
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
        // Accumulate final segments instead of overwriting
        setTranscript((prev) => (prev ? prev + " " + finalText.trim() : finalText.trim()));
        setInterimTranscript("");
        
        // Reset silence timer on new speech
        if (silenceTimeoutRef.current) {
          clearTimeout(silenceTimeoutRef.current);
        }
        
        // Auto-stop after 1s silence
        silenceTimeoutRef.current = setTimeout(() => {
          if (recognitionRef.current) {
            recognitionRef.current.stop();
          }
        }, SILENCE_DURATION_MS);
      } else {
        setInterimTranscript(interimText);
      }
    };

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    recognition.onerror = (event: any) => {
      // "no-speech" and "aborted" are not real errors
      if (event.error === "no-speech" || event.error === "aborted") return;
      setError(event.error);
      setIsListening(false);
    };

    recognition.onend = () => {
      setIsListening(false);
      // Signal that accumulated transcript is ready to send
      setIsComplete(true);
    };

    recognitionRef.current = recognition;
    return recognition;
  }, []);

  const startListening = useCallback(() => {
    setError(null);
    setTranscript("");
    setInterimTranscript("");
    setIsComplete(false);

    const recognition = getRecognition();
    if (!recognition) {
      setError("Speech recognition not supported in this browser");
      return;
    }

    try {
      recognition.start();
      setIsListening(true);
    } catch {
      // Already started — ignore
    }
  }, [getRecognition]);

  const stopListening = useCallback(() => {
    const recognition = recognitionRef.current;
    if (recognition) {
      try {
        recognition.stop();
      } catch {
        // Already stopped
      }
    }
    setIsListening(false);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (recognitionRef.current) {
        try { recognitionRef.current.stop(); } catch { /* noop */ }
      }
      if (sendTimeoutRef.current) clearTimeout(sendTimeoutRef.current);
      if (silenceTimeoutRef.current) clearTimeout(silenceTimeoutRef.current);
    };
  }, []);

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
