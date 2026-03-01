/**
 * ExplanationPanel — Cinematic glassmorphism analysis panel (v3).
 *
 * Shows scene analysis, per-person risk cards, and a RiskGauge.
 */

import type { PersonAnalysisOut, AnalysisOut } from "../types";

interface ExplanationPanelProps {
  persons: PersonAnalysisOut[];
  sceneAnalysis?: AnalysisOut | null;
  latencyMs?: number | null;
}

export function ExplanationPanel({
  persons,
  sceneAnalysis,
  latencyMs,
}: ExplanationPanelProps) {
  const overallRisk = sceneAnalysis?.risk_level?.toLowerCase() ?? "low";

  return (
    <div className="explanation-panel">
      <div className="panel-header-row">
        <h3 className="panel-title">
          <span className="panel-title-dot" />
          Intent Analysis
        </h3>
        {latencyMs != null && (
          <span className="panel-latency">{latencyMs.toFixed(0)}ms</span>
        )}
      </div>

      {/* Risk gauge */}
      <RiskGauge risk={overallRisk} />

      {/* Scene-level analysis */}
      {sceneAnalysis && (
        <div className="analysis-card scene-card">
          <div className="card-header">
            <span className="card-label">Scene Overview</span>
            <RiskBadge risk={sceneAnalysis.risk_level} />
          </div>
          <p className="card-explanation">{sceneAnalysis.explanation}</p>
          {sceneAnalysis.alerts.length > 0 && (
            <AlertList alerts={sceneAnalysis.alerts} />
          )}
          {sceneAnalysis.recommended_action && (
            <div className="card-action">
              <strong>Action:</strong> {sceneAnalysis.recommended_action}
            </div>
          )}
        </div>
      )}

      {persons.length === 0 ? (
        <p className="panel-empty">
          No persons detected. Point camera at people to begin analysis.
        </p>
      ) : (
        <div className="person-card-list">
          {persons.map((person) => (
            <PersonCard key={person.person_id} person={person} />
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function RiskGauge({ risk }: { risk: string }) {
  const levels = ["low", "medium", "high"];
  const activeIdx = levels.indexOf(risk);

  return (
    <div className="risk-gauge">
      {levels.map((level, i) => (
        <div
          key={level}
          className={`risk-gauge-seg ${i <= activeIdx ? `risk-gauge-active risk-${level}` : ""}`}
        >
          <div className="risk-gauge-bar" />
          <span className="risk-gauge-label">{level}</span>
        </div>
      ))}
    </div>
  );
}

function RiskBadge({ risk }: { risk: string }) {
  return (
    <span className={`risk-badge risk-${risk.toLowerCase()}`}>
      {risk.toUpperCase()}
    </span>
  );
}

function PersonCard({ person }: { person: PersonAnalysisOut }) {
  const { behavior, intent } = person;
  const attrs = behavior.attributes;

  return (
    <div className="analysis-card person-card">
      <div className="card-header">
        <span className="card-label">Person #{person.person_id}</span>
        <RiskBadge risk={intent.risk_level} />
      </div>

      <p className="card-explanation">{intent.explanation}</p>

      {intent.alerts.length > 0 && <AlertList alerts={intent.alerts} />}

      {intent.recommended_action && (
        <div className="card-action">
          <strong>Action:</strong> {intent.recommended_action}
        </div>
      )}

      {/* Structured attribute tags */}
      {attrs && (
        <div className="attr-tags">
          {attrs.posture && <span className="attr-tag attr-posture">{attrs.posture}</span>}
          {attrs.dominant_color && <span className="attr-tag attr-color">{attrs.dominant_color}</span>}
          {attrs.head_tilt && attrs.head_tilt !== "neutral" && (
            <span className="attr-tag attr-tilt">{attrs.head_tilt.replace("_", " ")}</span>
          )}
          {attrs.gaze_direction && (
            <span className="attr-tag attr-gaze">gaze: {attrs.gaze_direction}</span>
          )}
        </div>
      )}

      <div className="stats-grid">
        <StatItem label="Dwell" value={`${behavior.dwell_time.toFixed(1)}s`} />
        <StatItem label="Zones" value={String(behavior.zones_entered.length)} />
        <StatItem label="Re-entries" value={String(behavior.repeated_approaches)} />
        <StatItem label="Movement" value={behavior.velocity_label} />
      </div>

      {behavior.timeline.length > 0 && (
        <details className="timeline-details">
          <summary className="timeline-summary">
            Timeline ({behavior.timeline.length} events)
          </summary>
          <ul className="timeline-list">
            {behavior.timeline.slice(-8).map((event, i) => (
              <li key={i} className="timeline-item">{event}</li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

function AlertList({ alerts }: { alerts: string[] }) {
  return (
    <ul className="alert-list">
      {alerts.map((alert, i) => (
        <li key={i} className="alert-item">⚠ {alert}</li>
      ))}
    </ul>
  );
}

function StatItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat-item">
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}
