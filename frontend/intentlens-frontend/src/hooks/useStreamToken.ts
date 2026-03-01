/**
 * useStreamToken — Fetches a Stream Video token for the given user ID.
 */

import { useEffect, useState } from "react";
import { fetchToken } from "../api";

interface UseStreamTokenReturn {
  token: string | null;
  loading: boolean;
  error: string | null;
}

export function useStreamToken(userId: string): UseStreamTokenReturn {
  const [token, setToken] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetchToken(userId)
      .then((res) => {
        if (!cancelled) setToken(res.token);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          const message =
            err instanceof Error ? err.message : "Failed to fetch token";
          setError(message);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [userId]);

  return { token, loading, error };
}
