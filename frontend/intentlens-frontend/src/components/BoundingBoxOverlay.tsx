/**
 * BoundingBoxOverlay — Ethereal SVG bounding boxes with soft glow (v4).
 *
 * Thinner strokes, larger glow radius, subtle corner accents, translucent labels.
 */

import type { PersonAnalysisOut } from "../types";

interface BoundingBoxOverlayProps {
  persons: PersonAnalysisOut[];
  videoWidth: number;
  videoHeight: number;
}

const RISK_COLORS: Record<string, string> = {
  low: "#00f0ff",     // cyan neon
  medium: "#f59e0b",  // amber
  high: "#ff0066",    // neon red
};

function getRiskColor(risk: string): string {
  return RISK_COLORS[risk.toLowerCase()] ?? "#94a3b8";
}

export function BoundingBoxOverlay({
  persons,
  videoWidth,
  videoHeight,
}: BoundingBoxOverlayProps) {
  if (videoWidth === 0 || videoHeight === 0) return null;

  return (
    <svg
      viewBox={`0 0 ${videoWidth} ${videoHeight}`}
      className="bbox-overlay"
    >
      <defs>
        {/* Soft glow filters — larger blur, lower opacity for ethereal feel */}
        <filter id="glow-low" x="-40%" y="-40%" width="180%" height="180%">
          <feGaussianBlur stdDeviation="8" result="blur" />
          <feFlood floodColor="#00f0ff" floodOpacity="0.35" result="color" />
          <feComposite in="color" in2="blur" operator="in" result="glow" />
          <feMerge>
            <feMergeNode in="glow" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <filter id="glow-medium" x="-40%" y="-40%" width="180%" height="180%">
          <feGaussianBlur stdDeviation="8" result="blur" />
          <feFlood floodColor="#f59e0b" floodOpacity="0.4" result="color" />
          <feComposite in="color" in2="blur" operator="in" result="glow" />
          <feMerge>
            <feMergeNode in="glow" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <filter id="glow-high" x="-40%" y="-40%" width="180%" height="180%">
          <feGaussianBlur stdDeviation="10" result="blur" />
          <feFlood floodColor="#ff0066" floodOpacity="0.5" result="color" />
          <feComposite in="color" in2="blur" operator="in" result="glow" />
          <feMerge>
            <feMergeNode in="glow" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {persons.map((person) => {
        const [x1, y1, x2, y2] = person.bbox;
        const w = x2 - x1;
        const h = y2 - y1;
        const risk = person.intent.risk_level.toLowerCase();
        const color = getRiskColor(risk);
        const filterId = `glow-${risk === "low" ? "low" : risk === "medium" ? "medium" : "high"}`;
        const cornerLen = 10;

        return (
          <g key={person.person_id}>
            {/* Soft bounding rect */}
            <rect
              x={x1}
              y={y1}
              width={w}
              height={h}
              fill="none"
              stroke={color}
              strokeWidth={1.2}
              strokeOpacity={0.6}
              rx={6}
              filter={`url(#${filterId})`}
              className={risk === "high" ? "bbox-pulse" : ""}
            />

            {/* Corner accents — small and subtle */}
            <path d={`M${x1},${y1 + cornerLen} L${x1},${y1} L${x1 + cornerLen},${y1}`} fill="none" stroke={color} strokeWidth={2} strokeOpacity={0.8} />
            <path d={`M${x2 - cornerLen},${y1} L${x2},${y1} L${x2},${y1 + cornerLen}`} fill="none" stroke={color} strokeWidth={2} strokeOpacity={0.8} />
            <path d={`M${x1},${y2 - cornerLen} L${x1},${y2} L${x1 + cornerLen},${y2}`} fill="none" stroke={color} strokeWidth={2} strokeOpacity={0.8} />
            <path d={`M${x2 - cornerLen},${y2} L${x2},${y2} L${x2},${y2 - cornerLen}`} fill="none" stroke={color} strokeWidth={2} strokeOpacity={0.8} />

            {/* Minimal label — translucent pill */}
            <rect
              x={x1}
              y={y1 - 22}
              width={Math.max(w * 0.6, 100)}
              height={20}
              fill={color}
              fillOpacity={0.45}
              rx={10}
            />
            <text
              x={x1 + 8}
              y={y1 - 8}
              fill="#fff"
              fillOpacity={0.9}
              fontSize={10}
              fontFamily="'JetBrains Mono', monospace"
              fontWeight="500"
            >
              #{person.person_id} · {risk.toUpperCase()}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
