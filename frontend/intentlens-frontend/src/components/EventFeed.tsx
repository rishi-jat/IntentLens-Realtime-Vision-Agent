/**
 * EventFeed — Scrolling real-time event timeline.
 *
 * Renders agent events (new person, departure, gesture, zone breach, risk change)
 * with severity-based color coding and auto-scroll.
 */

import { useEffect, useRef } from "react";
import type { AgentEvent, EventSeverity } from "../types";

interface EventFeedProps {
  events: AgentEvent[];
}

const SEVERITY_COLORS: Record<EventSeverity, string> = {
  info: "#00f0ff",    // cyan neon
  warning: "#f59e0b", // amber
  alert: "#ff0066",   // neon red
};

const SEVERITY_BG: Record<EventSeverity, string> = {
  info: "rgba(0,240,255,0.08)",
  warning: "rgba(245,158,11,0.08)",
  alert: "rgba(255,0,102,0.1)",
};

const SEVERITY_ICON: Record<EventSeverity, string> = {
  info: "◈",
  warning: "⚠",
  alert: "⚡",
};

function formatTime(ts: number): string {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function EventFeed({ events }: EventFeedProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new events arrive
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events.length]);

  return (
    <div className="event-feed">
      <div className="event-feed-header">
        <span className="event-feed-dot" />
        <span>Live Events</span>
        <span className="event-feed-count">{events.length}</span>
      </div>

      <div className="event-feed-scroll">
        {events.length === 0 ? (
          <div className="event-feed-empty">Waiting for events…</div>
        ) : (
          events.map((event, i) => (
            <div
              key={`${event.timestamp}-${i}`}
              className="event-item"
              style={{
                borderLeftColor: SEVERITY_COLORS[event.severity],
                background: SEVERITY_BG[event.severity],
              }}
            >
              <div className="event-item-header">
                <span
                  className="event-severity-icon"
                  style={{ color: SEVERITY_COLORS[event.severity] }}
                >
                  {SEVERITY_ICON[event.severity]}
                </span>
                <span
                  className="event-kind"
                  style={{ color: SEVERITY_COLORS[event.severity] }}
                >
                  {event.kind.replace(/_/g, " ")}
                </span>
                <span className="event-time">{formatTime(event.timestamp)}</span>
              </div>
              <div className="event-message">{event.message}</div>
            </div>
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
