/**
 * DetectionOverlay — Jarvis-style motion-reactive detection rendering.
 *
 * Each detected person gets:
 * - Corner-bracket bounding box (neon color based on risk)
 * - Energy lines along limb connections (keypoints)
 * - Motion trails for wrists when moving
 * - Arc-reactor style center pulse
 * - Velocity-reactive glow intensity
 * - Minimal HUD tag: ID + risk + gesture
 */

import { useRef } from "react";
import type { PersonAnalysisOut, SceneGraphOut, KeypointOut } from "../types";

interface DetectionOverlayProps {
  persons: PersonAnalysisOut[];
  videoWidth: number;
  videoHeight: number;
  sceneGraph?: SceneGraphOut | null;
}

const RISK_COLORS: Record<string, string> = {
  low: "#00f0ff",
  medium: "#f59e0b",
  high: "#ff0066",
};

function getRiskColor(risk: string): string {
  return RISK_COLORS[risk.toLowerCase()] ?? "#94a3b8";
}

/** COCO 17-keypoint skeleton connections */
const SKELETON_PAIRS: [number, number][] = [
  [5, 6],   // shoulders
  [5, 7],   // left shoulder → left elbow
  [7, 9],   // left elbow → left wrist
  [6, 8],   // right shoulder → right elbow
  [8, 10],  // right elbow → right wrist
  [5, 11],  // left shoulder → left hip
  [6, 12],  // right shoulder → right hip
  [11, 12], // hips
  [11, 13], // left hip → left knee
  [13, 15], // left knee → left ankle
  [12, 14], // right hip → right knee
  [14, 16], // right knee → right ankle
];

const CONF_THRESHOLD = 0.25;
const CORNER_FRAC = 0.15;
const MIN_CORNER = 12;
const STROKE_W = 2;

export function DetectionOverlay({
  persons,
  videoWidth,
  videoHeight,
  sceneGraph,
}: DetectionOverlayProps) {
  // Track previous wrist positions for motion trails
  const trailsRef = useRef<Map<number, { lw: [number, number][]; rw: [number, number][] }>>(new Map());
  const MAX_TRAIL = 8;

  if (videoWidth === 0 || videoHeight === 0) return null;

  // Build gesture lookup  
  const gestureMap = new Map<number, string[]>();
  const sgPersonMap = new Map<number, { is_pacing: boolean; is_loitering: boolean }>();
  if (sceneGraph?.persons) {
    for (const sp of sceneGraph.persons) {
      if (sp.gesture_state && Array.isArray(sp.gesture_state) && sp.gesture_state.length > 0) {
        gestureMap.set(sp.person_id, sp.gesture_state);
      }
      sgPersonMap.set(sp.person_id, { is_pacing: sp.is_pacing, is_loitering: sp.is_loitering });
    }
  }

  // Update motion trails
  const currentTrails = trailsRef.current;
  const activeIds = new Set(persons.map(p => p.person_id));
  for (const id of currentTrails.keys()) {
    if (!activeIds.has(id)) currentTrails.delete(id);
  }

  return (
    <svg
      viewBox={`0 0 ${videoWidth} ${videoHeight}`}
      className="detection-overlay"
    >
      <defs>
        <filter id="glow-cyan" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feFlood floodColor="#00f0ff" floodOpacity="0.5" result="color" />
          <feComposite in="color" in2="blur" operator="in" result="glow" />
          <feMerge><feMergeNode in="glow" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="glow-amber" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feFlood floodColor="#f59e0b" floodOpacity="0.5" result="color" />
          <feComposite in="color" in2="blur" operator="in" result="glow" />
          <feMerge><feMergeNode in="glow" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="glow-red" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="4" result="blur" />
          <feFlood floodColor="#ff0066" floodOpacity="0.6" result="color" />
          <feComposite in="color" in2="blur" operator="in" result="glow" />
          <feMerge><feMergeNode in="glow" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="energy-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="2.5" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
      </defs>

      {persons.map((person) => {
        const [x1, y1, x2, y2] = person.bbox;
        const w = x2 - x1;
        const h = y2 - y1;
        if (w < 10 || h < 10) return null;

        const risk = person.intent.risk_level.toLowerCase();
        const color = getRiskColor(risk);
        const glowFilter = risk === "high" ? "url(#glow-red)" : risk === "medium" ? "url(#glow-amber)" : "url(#glow-cyan)";

        const cw = Math.max(MIN_CORNER, w * CORNER_FRAC);
        const ch = Math.max(MIN_CORNER, h * CORNER_FRAC);

        const vel = person.behavior.velocity;
        const isMoving = person.behavior.velocity_label !== "stationary";
        const energyOpacity = Math.min(0.85, 0.25 + (vel / 80) * 0.6);

        // Keypoints for energy lines
        const kps = person.behavior.keypoints?.keypoints ?? [];
        const hasKeypoints = kps.length === 17;

        // Update trails
        if (hasKeypoints) {
          if (!currentTrails.has(person.person_id)) {
            currentTrails.set(person.person_id, { lw: [], rw: [] });
          }
          const t = currentTrails.get(person.person_id)!;
          const lw = kps[9];
          const rw = kps[10];
          if (lw && lw.confidence > CONF_THRESHOLD) {
            t.lw.push([lw.x, lw.y]);
            if (t.lw.length > MAX_TRAIL) t.lw.shift();
          }
          if (rw && rw.confidence > CONF_THRESHOLD) {
            t.rw.push([rw.x, rw.y]);
            if (t.rw.length > MAX_TRAIL) t.rw.shift();
          }
        }

        // Build info tag
        const gestures = gestureMap.get(person.person_id);
        const sgPerson = sgPersonMap.get(person.person_id);
        const tags: string[] = [];
        if (gestures?.length) tags.push(...gestures.map(g => g.replace(/_/g, " ")));
        if (isMoving) tags.push(person.behavior.velocity_label);
        if (sgPerson?.is_pacing) tags.push("pacing");
        if (sgPerson?.is_loitering) tags.push("loitering");
        const attrs = person.behavior.attributes;
        if (attrs?.dominant_color && attrs.dominant_color !== "unknown") tags.push(attrs.dominant_color);

        const centerX = (x1 + x2) / 2;
        const centerY = (y1 + y2) / 2;

        return (
          <g key={person.person_id}>
            {/* Faint bbox fill */}
            <rect
              x={x1} y={y1} width={w} height={h}
              fill={color} fillOpacity={0.03} stroke="none"
            />

            {/* Corner brackets */}
            <polyline points={`${x1},${y1 + ch} ${x1},${y1} ${x1 + cw},${y1}`}
              fill="none" stroke={color} strokeWidth={STROKE_W}
              strokeLinecap="round" strokeLinejoin="round" filter={glowFilter} />
            <polyline points={`${x2 - cw},${y1} ${x2},${y1} ${x2},${y1 + ch}`}
              fill="none" stroke={color} strokeWidth={STROKE_W}
              strokeLinecap="round" strokeLinejoin="round" filter={glowFilter} />
            <polyline points={`${x1},${y2 - ch} ${x1},${y2} ${x1 + cw},${y2}`}
              fill="none" stroke={color} strokeWidth={STROKE_W}
              strokeLinecap="round" strokeLinejoin="round" filter={glowFilter} />
            <polyline points={`${x2 - cw},${y2} ${x2},${y2} ${x2},${y2 - ch}`}
              fill="none" stroke={color} strokeWidth={STROKE_W}
              strokeLinecap="round" strokeLinejoin="round" filter={glowFilter} />

            {/* Energy skeleton lines */}
            {hasKeypoints && SKELETON_PAIRS.map(([i, j], idx) => {
              const a = kps[i];
              const b = kps[j];
              if (!a || !b || a.confidence < CONF_THRESHOLD || b.confidence < CONF_THRESHOLD) return null;
              return (
                <line
                  key={`skel-${idx}`}
                  x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                  stroke={color}
                  strokeWidth={isMoving ? 1.8 : 1.2}
                  strokeOpacity={energyOpacity}
                  strokeLinecap="round"
                  filter="url(#energy-glow)"
                />
              );
            })}

            {/* Joint energy dots */}
            {hasKeypoints && kps.map((kp: KeypointOut, idx: number) => {
              if (kp.confidence < CONF_THRESHOLD) return null;
              if (![5, 6, 7, 8, 9, 10, 11, 12].includes(idx)) return null;
              return (
                <circle
                  key={`kp-${idx}`}
                  cx={kp.x} cy={kp.y}
                  r={idx === 9 || idx === 10 ? 3.5 : 2.5}
                  fill={color} fillOpacity={energyOpacity}
                  filter="url(#energy-glow)"
                />
              );
            })}

            {/* Motion trails on wrists */}
            {isMoving && currentTrails.has(person.person_id) && (() => {
              const t = currentTrails.get(person.person_id)!;
              const trails: React.ReactElement[] = [];
              for (const [key, points] of [["lw", t.lw], ["rw", t.rw]] as const) {
                if (points.length < 3) continue;
                const d = points.map((p, i) => `${i === 0 ? "M" : "L"}${p[0]},${p[1]}`).join(" ");
                trails.push(
                  <path
                    key={`trail-${key}`}
                    d={d}
                    fill="none" stroke={color}
                    strokeWidth={1.5} strokeOpacity={0.4}
                    strokeLinecap="round"
                    filter="url(#energy-glow)"
                    className="motion-trail"
                  />
                );
              }
              return trails;
            })()}

            {/* Arc-reactor center */}
            <circle cx={centerX} cy={centerY}
              r={isMoving ? 6 : 4}
              fill="none" stroke={color}
              strokeWidth={1} strokeOpacity={0.35}
              className="arc-reactor-ring"
            />
            <circle cx={centerX} cy={centerY}
              r={2} fill={color} fillOpacity={0.5}
              className={isMoving ? "arc-reactor-core active" : "arc-reactor-core"}
            />

            {/* ID label */}
            <rect x={x1} y={y1 - 18} width={Math.min(68, w)} height={14}
              fill={color} fillOpacity={0.9} rx={3} />
            <text x={x1 + 4} y={y1 - 8}
              fill="#0a0f19" fontSize={8.5}
              fontFamily="'JetBrains Mono', monospace"
              fontWeight="700" letterSpacing="0.5"
            >
              ID:{person.person_id} {risk.toUpperCase()}
            </text>

            {/* Info tags */}
            {tags.length > 0 && (
              <>
                <rect x={x1} y={y2 + 3}
                  width={Math.min(tags.join(" · ").length * 5.8 + 12, w + 40)}
                  height={13}
                  fill="rgba(10,15,25,0.88)" rx={3}
                  stroke={color} strokeWidth={0.5} strokeOpacity={0.3}
                />
                <text x={x1 + 5} y={y2 + 12}
                  fill={color} fontSize={8}
                  fontFamily="'JetBrains Mono', monospace"
                  fontWeight="500" fillOpacity={0.85}
                >
                  {tags.join(" · ")}
                </text>
              </>
            )}

            {/* High risk pulse */}
            {risk === "high" && (
              <rect x={x1 - 3} y={y1 - 3} width={w + 6} height={h + 6}
                fill="none" stroke="#ff0066"
                strokeWidth={1.5} strokeOpacity={0.5} rx={4}
                className="detection-pulse"
              />
            )}
          </g>
        );
      })}
    </svg>
  );
}
