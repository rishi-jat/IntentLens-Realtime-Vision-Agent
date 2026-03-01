/**
 * StatusBar — Minimal ambient status bar for embodied agent feel.
 */

interface StatusBarProps {
  isConnected: boolean;
  isAnalyzing: boolean;
  analysisError: string | null;
  personCount: number;
  latencyMs?: number | null;
  isListening?: boolean;
  isSpeaking?: boolean;
}

export function StatusBar({
  isConnected,
  isAnalyzing,
  analysisError,
  personCount,
  isListening,
  isSpeaking,
}: StatusBarProps) {
  return (
    <div className="status-bar">
      <div className="status-bar-left">
        <span className="status-brand">◉ INTENTLENS</span>
        <StatusDot active={isConnected} label="LIVE" />
        {isAnalyzing && <StatusDot active label="SCANNING" pulse />}
        {isListening && <StatusDot active color="var(--neon-red)" label="LISTENING" pulse />}
        {isSpeaking && <StatusDot active color="var(--neon-green)" label="SPEAKING" />}
        <span className="status-count">
          {personCount} {personCount === 1 ? "SUBJECT" : "SUBJECTS"}
        </span>
      </div>

      {analysisError && <span className="status-error">{analysisError}</span>}
    </div>
  );
}

function StatusDot({
  active,
  label,
  pulse,
  color,
}: {
  active: boolean;
  label: string;
  pulse?: boolean;
  color?: string;
}) {
  return (
    <div className="status-dot-group">
      <div
        className={`status-dot ${active ? "active" : ""} ${active && pulse ? "pulse" : ""}`}
        style={active && color ? { background: color, boxShadow: `0 0 8px ${color}` } : undefined}
      />
      <span className="status-label">{label}</span>
    </div>
  );
}
