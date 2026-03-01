/**
 * SkeletonOverlay — Cinematic SVG skeleton renderer with motion trails (v2).
 *
 * Replaces BoundingBoxOverlay. Draws 17-point COCO pose skeletons with
 * soft glow joints, connecting limb lines, motion trails (last 10 frames),
 * and minimal risk labels. Falls back to a thin bbox when no keypoints are available.
 */

import { useRef } from "react";
import type { PersonAnalysisOut, KeypointOut } from "../types";

interface SkeletonOverlayProps {
  persons: PersonAnalysisOut[];
  videoWidth: number;
  videoHeight: number;
}

/** Minimum keypoint confidence to render a joint or connection. */
const CONF_THRESHOLD = 0.3;

/** Motion trail history depth (frames) */
const TRAIL_FRAMES = 10;

/** Key joints to show trails for (wrists, ankles, nose for maximum effect) */
const TRAIL_JOINTS = [0, 9, 10, 15, 16]; // nose, left_wrist, right_wrist, left_ankle, right_ankle

/** COCO-17 skeleton connections: pairs of keypoint indices. */
const SKELETON_CONNECTIONS: [number, number][] = [
  // Head
  [0, 1],   // nose → left_eye
  [0, 2],   // nose → right_eye
  [1, 3],   // left_eye → left_ear
  [2, 4],   // right_eye → right_ear
  // Torso
  [5, 6],   // left_shoulder → right_shoulder
  [5, 11],  // left_shoulder → left_hip
  [6, 12],  // right_shoulder → right_hip
  [11, 12], // left_hip → right_hip
  // Left arm
  [5, 7],   // left_shoulder → left_elbow
  [7, 9],   // left_elbow → left_wrist
  // Right arm
  [6, 8],   // right_shoulder → right_elbow
  [8, 10],  // right_elbow → right_wrist
  // Left leg
  [11, 13], // left_hip → left_knee
  [13, 15], // left_knee → left_ankle
  // Right leg
  [12, 14], // right_hip → right_knee
  [14, 16], // right_knee → right_ankle
];

const RISK_COLORS: Record<string, string> = {
  low: "#00f0ff",
  medium: "#f59e0b",
  high: "#ff0066",
};

function getRiskColor(risk: string): string {
  return RISK_COLORS[risk.toLowerCase()] ?? "#94a3b8";
}

function getRiskFilter(risk: string): string {
  const key = risk.toLowerCase();
  if (key === "medium") return "url(#skel-glow-medium)";
  if (key === "high") return "url(#skel-glow-high)";
  return "url(#skel-glow-low)";
}

function isVisible(kp: KeypointOut): boolean {
  return kp.confidence >= CONF_THRESHOLD && (kp.x !== 0 || kp.y !== 0);
}

export function SkeletonOverlay({
  persons,
  videoWidth,
  videoHeight,
}: SkeletonOverlayProps) {
  if (videoWidth === 0 || videoHeight === 0) return null;
  
  // Track motion trails: person_id -> joint_idx -> history of [x, y]
  const trailHistoryRef = useRef<Map<number, Map<number, Array<[number, number]>>>>(new Map());
  
  // Update trails for current frame
  persons.forEach((person) => {
    const kps = person.behavior.keypoints?.keypoints;
    if (!kps || kps.length !== 17) return;
    
    if (!trailHistoryRef.current.has(person.person_id)) {
      trailHistoryRef.current.set(person.person_id, new Map());
    }
    
    const personTrails = trailHistoryRef.current.get(person.person_id)!;
    
    TRAIL_JOINTS.forEach((jointIdx) => {
      const kp = kps[jointIdx];
      if (!isVisible(kp)) return;
      
      if (!personTrails.has(jointIdx)) {
        personTrails.set(jointIdx, []);
      }
      
      const trail = personTrails.get(jointIdx)!;
      trail.push([kp.x, kp.y]);
      
      // Keep only last N frames
      if (trail.length > TRAIL_FRAMES) {
        trail.shift();
      }
    });
  });
  
  // Clean up trails for departed persons
  const activePids = new Set(persons.map(p => p.person_id));
  for (const pid of trailHistoryRef.current.keys()) {
    if (!activePids.has(pid)) {
      trailHistoryRef.current.delete(pid);
    }
  }

  return (
    <svg
      viewBox={`0 0 ${videoWidth} ${videoHeight}`}
      className="skeleton-overlay"
    >
      <defs>
        {/* Glow filters for skeleton joints & limbs */}
        <filter id="skel-glow-low" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="4" result="blur" />
          <feFlood floodColor="#00f0ff" floodOpacity="0.5" result="color" />
          <feComposite in="color" in2="blur" operator="in" result="glow" />
          <feMerge>
            <feMergeNode in="glow" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <filter id="skel-glow-medium" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="4" result="blur" />
          <feFlood floodColor="#f59e0b" floodOpacity="0.55" result="color" />
          <feComposite in="color" in2="blur" operator="in" result="glow" />
          <feMerge>
            <feMergeNode in="glow" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <filter id="skel-glow-high" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="5" result="blur" />
          <feFlood floodColor="#ff0066" floodOpacity="0.6" result="color" />
          <feComposite in="color" in2="blur" operator="in" result="glow" />
          <feMerge>
            <feMergeNode in="glow" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {persons.map((person) => {
        const risk = person.intent.risk_level.toLowerCase();
        const color = getRiskColor(risk);
        const filter = getRiskFilter(risk);
        const kps = person.behavior.keypoints?.keypoints;

        // Render skeleton if keypoints available, otherwise thin bbox fallback
        if (kps && kps.length === 17) {
          const personTrails = trailHistoryRef.current.get(person.person_id);
          
          return (
            <g key={person.person_id}>
              {/* Motion trails for key joints */}
              {personTrails && TRAIL_JOINTS.map((jointIdx) => {
                const trail = personTrails.get(jointIdx);
                if (!trail || trail.length < 2) return null;
                
                // Generate polyline path from trail history
                const points = trail.map(([x, y]) => `${x},${y}`).join(' ');
                
                return (
                  <polyline
                    key={`trail-${jointIdx}`}
                    points={points}
                    fill="none"
                    stroke={color}
                    strokeWidth={1.5}
                    strokeOpacity={0.2}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    filter={filter}
                    className="skeleton-trail"
                  />
                );
              })}
              
              {/* Limb connections */}
              {SKELETON_CONNECTIONS.map(([i, j], idx) => {
                const a = kps[i];
                const b = kps[j];
                if (!isVisible(a) || !isVisible(b)) return null;
                return (
                  <line
                    key={`limb-${idx}`}
                    x1={a.x}
                    y1={a.y}
                    x2={b.x}
                    y2={b.y}
                    stroke={color}
                    strokeWidth={2}
                    strokeOpacity={0.7}
                    strokeLinecap="round"
                    filter={filter}
                    className="skeleton-limb"
                  />
                );
              })}

              {/* Joints */}
              {kps.map((kp, idx) => {
                if (!isVisible(kp)) return null;
                // Head keypoints (0-4) get smaller radius
                const isHead = idx <= 4;
                const r = isHead ? 2.5 : 3.5;
                return (
                  <circle
                    key={`joint-${idx}`}
                    cx={kp.x}
                    cy={kp.y}
                    r={r}
                    fill={color}
                    fillOpacity={0.9}
                    filter={filter}
                    className="skeleton-joint"
                  />
                );
              })}

              {/* Minimal label pill near top of skeleton */}
              {(() => {
                // Find topmost visible keypoint for label placement
                let minY = Infinity;
                let labelX = 0;
                for (const kp of kps) {
                  if (isVisible(kp) && kp.y < minY) {
                    minY = kp.y;
                    labelX = kp.x;
                  }
                }
                if (minY === Infinity) return null;
                return (
                  <g>
                    <rect
                      x={labelX - 30}
                      y={minY - 24}
                      width={60}
                      height={18}
                      fill={color}
                      fillOpacity={0.35}
                      rx={9}
                    />
                    <text
                      x={labelX}
                      y={minY - 12}
                      fill="#fff"
                      fillOpacity={0.9}
                      fontSize={9}
                      fontFamily="'JetBrains Mono', monospace"
                      fontWeight="600"
                      textAnchor="middle"
                    >
                      #{person.person_id} · {risk.toUpperCase()}
                    </text>
                  </g>
                );
              })()}

              {/* Pulse ring on high-risk skeletons at center of mass */}
              {risk === "high" && (() => {
                const visible = kps.filter(isVisible);
                if (visible.length === 0) return null;
                const cx = visible.reduce((s, k) => s + k.x, 0) / visible.length;
                const cy = visible.reduce((s, k) => s + k.y, 0) / visible.length;
                return (
                  <circle
                    cx={cx}
                    cy={cy}
                    r={20}
                    fill="none"
                    stroke="#ff0066"
                    strokeWidth={1.5}
                    strokeOpacity={0.6}
                    className="skeleton-pulse"
                  />
                );
              })()}
            </g>
          );
        }

        // Fallback: thin bbox when no keypoints
        const [x1, y1, x2, y2] = person.bbox;
        const w = x2 - x1;
        const h = y2 - y1;
        return (
          <g key={person.person_id}>
            <rect
              x={x1}
              y={y1}
              width={w}
              height={h}
              fill="none"
              stroke={color}
              strokeWidth={1}
              strokeOpacity={0.4}
              rx={4}
              strokeDasharray="6 4"
            />
            <rect
              x={x1}
              y={y1 - 20}
              width={60}
              height={16}
              fill={color}
              fillOpacity={0.35}
              rx={8}
            />
            <text
              x={x1 + 30}
              y={y1 - 8}
              fill="#fff"
              fillOpacity={0.9}
              fontSize={9}
              fontFamily="'JetBrains Mono', monospace"
              fontWeight="600"
              textAnchor="middle"
            >
              #{person.person_id} · {risk.toUpperCase()}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
