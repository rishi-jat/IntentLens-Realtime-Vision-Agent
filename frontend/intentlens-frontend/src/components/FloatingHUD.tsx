/**
 * FloatingHUD — Minimal contextual overlay near each person.
 *
 * Replaces heavy debug panel with subtle floating cards that show:
 * - Person ID
 * - Risk level
 * - Movement state
 * - Gesture (if active)
 * - Shirt color (if detected)
 * - Gaze direction
 *
 * No heavy panels. No debug artifacts. Production-grade UI.
 */

import type { PersonAnalysisOut } from "../types";

interface FloatingHUDProps {
  persons: PersonAnalysisOut[];
  videoWidth: number;
  videoHeight: number;
}

const RISK_COLORS: Record<string, string> = {
  low: "#00f0ff",
  medium: "#f59e0b",
  high: "#ff0066",
};

function getRiskColor(risk: string): string {
  return RISK_COLORS[risk.toLowerCase()] ?? "#94a3b8";
}

export function FloatingHUD({
  persons,
  videoWidth,
  videoHeight,
}: FloatingHUDProps) {
  if (videoWidth === 0 || videoHeight === 0 || persons.length === 0) return null;

  return (
    <div className="floating-hud-container">
      {persons.map((person) => {
        const [x1, y1, x2, y2] = person.bbox;
        const bboxWidth = x2 - x1;
        const bboxHeight = y2 - y1;
        
        // Position HUD to the right of the person
        const hudX = x2 + 10;
        const hudY = y1;
        
        const risk = person.intent.risk_level.toLowerCase();
        const color = getRiskColor(risk);
        const attrs = person.behavior.attributes;
        
        return (
          <div
            key={person.person_id}
            className="floating-hud-card"
            style={{
              left: `${(hudX / videoWidth) * 100}%`,
              top: `${(hudY / videoHeight) * 100}%`,
              borderColor: color,
            }}
          >
            {/* Header */}
            <div className="hud-header" style={{ background: color }}>
              <span className="hud-id">#{person.person_id}</span>
              <span className="hud-risk">{risk.toUpperCase()}</span>
            </div>

            {/* Body */}
            <div className="hud-body">
              {/* Movement */}
              <div className="hud-row">
                <span className="hud-icon">🚶</span>
                <span className="hud-value">{person.behavior.velocity_label}</span>
              </div>

              {/* Shirt color (only if detected) */}
              {attrs?.dominant_color && attrs.dominant_color !== "unknown" && (
                <div className="hud-row">
                  <span className="hud-icon">👕</span>
                  <span className="hud-value">{attrs.dominant_color}</span>
                </div>
              )}

              {/* Posture */}
              {attrs?.posture && (
                <div className="hud-row">
                  <span className="hud-icon">🧍</span>
                  <span className="hud-value">{attrs.posture}</span>
                </div>
              )}

              {/* Gaze */}
              {attrs?.gaze_direction && (
                <div className="hud-row">
                  <span className="hud-icon">👁️</span>
                  <span className="hud-value">gaze {attrs.gaze_direction}</span>
                </div>
              )}

              {/* Dwell time */}
              {person.behavior.dwell_time > 2 && (
                <div className="hud-row">
                  <span className="hud-icon">⏱️</span>
                  <span className="hud-value">{person.behavior.dwell_time.toFixed(0)}s</span>
                </div>
              )}

              {/* Active alerts */}
              {person.intent.alerts.length > 0 && (
                <div className="hud-alerts">
                  {person.intent.alerts.slice(0, 2).map((alert, i) => (
                    <div key={i} className="hud-alert">⚠ {alert}</div>
                  ))}
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
