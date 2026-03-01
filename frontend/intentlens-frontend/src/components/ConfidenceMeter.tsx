/**
 * ConfidenceMeter — Circular or bar-style confidence indicator.
 */

interface ConfidenceMeterProps {
  confidence: number;
  label?: string;
}

function getConfidenceColor(confidence: number): string {
  if (confidence >= 75) return "#22c55e";
  if (confidence >= 50) return "#f59e0b";
  if (confidence >= 25) return "#f97316";
  return "#ef4444";
}

export function ConfidenceMeter({ confidence, label }: ConfidenceMeterProps) {
  const color = getConfidenceColor(confidence);
  const pct = Math.max(0, Math.min(100, confidence));

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      {label && (
        <span
          style={{
            fontSize: 12,
            color: "#94a3b8",
            fontWeight: 600,
            textTransform: "uppercase",
            letterSpacing: 0.5,
          }}
        >
          {label}
        </span>
      )}
      <div
        style={{
          flex: 1,
          height: 8,
          borderRadius: 4,
          background: "#1e293b",
          overflow: "hidden",
          minWidth: 80,
        }}
      >
        <div
          style={{
            width: `${pct}%`,
            height: "100%",
            background: color,
            borderRadius: 4,
            transition: "width 0.3s ease, background 0.3s ease",
          }}
        />
      </div>
      <span
        style={{
          fontSize: 14,
          fontWeight: 700,
          color,
          fontFamily: "monospace",
          minWidth: 40,
          textAlign: "right",
        }}
      >
        {pct}%
      </span>
    </div>
  );
}
