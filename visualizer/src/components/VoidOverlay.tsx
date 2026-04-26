/**
 * VoidOverlay.tsx
 */

import React, { useMemo, useState, useEffect } from "react";
import type { ViewTransform, CrossPollinationAnim, PairLink } from "../types";
import type { Void, SelectedPaper } from "../hooks/useVoidData";

interface VoidOverlayProps {
  voids: Void[];
  selectedVoidId: number | null;
  showVoidLabels: boolean;
  transform: ViewTransform;
  width: number;
  height: number;
  onVoidClick: (id: number) => void;
  researchDoneVoidIds?: Set<number>;
  crossAnim?: CrossPollinationAnim;
}

// ── Geometry helpers ──────────────────────────────────────────────────────────

function normToScreen(
  nx: number,
  ny: number,
  t: ViewTransform,
): [number, number] {
  return [nx * t.scale + t.offsetX, ny * t.scale + t.offsetY];
}

function smoothHullPath(pts: [number, number][], tension = 0.38): string {
  const n = pts.length;
  if (n < 3) return "";

  if (tension <= 0) {
    return (
      pts
        .map(
          ([x, y], i) =>
            `${i === 0 ? "M" : "L"} ${x.toFixed(2)},${y.toFixed(2)}`,
        )
        .join(" ") + " Z"
    );
  }

  const cp = (
    prev: [number, number],
    cur: [number, number],
    next: [number, number],
  ): [number, number] => {
    const dx = next[0] - prev[0];
    const dy = next[1] - prev[1];
    return [cur[0] + (dx * tension) / 3, cur[1] + (dy * tension) / 3];
  };

  let d = `M ${pts[0][0].toFixed(2)},${pts[0][1].toFixed(2)}`;
  for (let i = 0; i < n; i++) {
    const p0 = pts[(i - 1 + n) % n];
    const p1 = pts[i];
    const p2 = pts[(i + 1) % n];
    const p3 = pts[(i + 2) % n];
    const c1 = cp(p0, p1, p2);
    const c2x = p2[0] - ((p3[0] - p1[0]) * tension) / 3;
    const c2y = p2[1] - ((p3[1] - p1[1]) * tension) / 3;
    d +=
      ` C ${c1[0].toFixed(2)},${c1[1].toFixed(2)}` +
      ` ${c2x.toFixed(2)},${c2y.toFixed(2)}` +
      ` ${p2[0].toFixed(2)},${p2[1].toFixed(2)}`;
  }
  return d + " Z";
}

function hullScreenPoints(
  nv: [number, number][],
  t: ViewTransform,
): [number, number][] {
  return nv.map(([nx, ny]) => normToScreen(nx, ny, t));
}

function hullNormBounds(nv: [number, number][]) {
  let minX = Infinity,
    maxX = -Infinity,
    minY = Infinity,
    maxY = -Infinity;
  for (const [x, y] of nv) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  return { minX, maxX, minY, maxY };
}

function hullScreenBounds(nv: [number, number][], t: ViewTransform) {
  const b = hullNormBounds(nv);
  const [x0, y0] = normToScreen(b.minX, b.minY, t);
  const [x1, y1] = normToScreen(b.maxX, b.maxY, t);
  return { minX: x0, maxX: x1, minY: y0, maxY: y1 };
}

function aabbOverlap(
  a: { minX: number; maxX: number; minY: number; maxY: number },
  b: { minX: number; maxX: number; minY: number; maxY: number },
): boolean {
  return (
    a.maxX > b.minX && a.minX < b.maxX && a.maxY > b.minY && a.minY < b.maxY
  );
}

// ── Graph coloring ────────────────────────────────────────────────────────────

const PALETTE_HUES = [55, 42, 68, 35, 78] as const;
const N_COLORS = PALETTE_HUES.length;

function assignColors(voids: Void[]): Map<number, number> {
  const boxes = new Map<number, ReturnType<typeof hullNormBounds>>();
  for (const v of voids) {
    if ((v.shape?.nvertices?.length ?? 0) >= 3) {
      boxes.set(v.void_id, hullNormBounds(v.shape.nvertices));
    }
  }

  const assignment = new Map<number, number>();
  for (const v of voids) {
    const box = boxes.get(v.void_id);
    if (!box) continue;
    const forbidden = new Set<number>();
    for (const [otherId, otherBox] of boxes) {
      if (otherId === v.void_id) continue;
      if (!assignment.has(otherId)) continue;
      if (aabbOverlap(box, otherBox)) forbidden.add(assignment.get(otherId)!);
    }
    let chosen = 0;
    while (forbidden.has(chosen) && chosen < N_COLORS - 1) chosen++;
    assignment.set(v.void_id, chosen);
  }
  return assignment;
}

// ── Color helpers ─────────────────────────────────────────────────────────────

function makeColor(hue: number, alpha: number): string {
  return `oklch(0.70 0.17 ${hue} / ${alpha})`;
}
function makeSolid(hue: number): string {
  return `oklch(0.88 0.19 ${hue})`;
}
function makeGlow(hue: number): string {
  return `oklch(0.74 0.13 ${hue} / 0.18)`;
}

const LABEL_SHADOW = "rgba(0,0,0,0.45)";

// Pair highlight colors (4 vivid, distinct hues)
const PAIR_COLORS = ["#ef4444", "#3b82f6", "#22c55e", "#a855f7"] as const;

type ActivePairEntry = {
  fromPaper: SelectedPaper;
  toPaper: SelectedPaper;
  color: string;
};

function buildActivePairs(
  v: Void,
  crossAnim: CrossPollinationAnim,
  pairCycleIdx: number,
): ActivePairEntry[] {
  if (crossAnim.phase !== "cross_pollinating") return [];

  const paperByDoi = new Map(v.selected_papers.map((p) => [p.DOI, p]));

  // Use explicit cross-idea pairs when available; fall back to auto-generated pairs
  let links: PairLink[] = crossAnim.pairLinks.filter(
    (l) => paperByDoi.has(l.fromDoi) && paperByDoi.has(l.toDoi),
  );

  if (!links.length && crossAnim.selectedDois.length >= 2) {
    const dois = crossAnim.selectedDois.filter((d) => paperByDoi.has(d));
    for (let i = 0; i < dois.length; i++) {
      for (let j = i + 1; j < dois.length; j++) {
        links.push({ fromDoi: dois[i], toDoi: dois[j] });
      }
    }
  }

  if (!links.length) return [];

  const batchSize = 4;
  const totalBatches = Math.max(1, Math.ceil(links.length / batchSize));
  const idx = pairCycleIdx % totalBatches;
  const batch = links.slice(idx * batchSize, (idx + 1) * batchSize);

  return batch.map((link, i) => ({
    fromPaper: paperByDoi.get(link.fromDoi)!,
    toPaper: paperByDoi.get(link.toDoi)!,
    color: PAIR_COLORS[i % PAIR_COLORS.length],
  }));
}

// ── Animation CSS ─────────────────────────────────────────────────────────────

const ANIM_CSS = `
@keyframes voidOrchestrateGlow {
  0%, 100% { opacity: 0; }
  50%      { opacity: 0.16; }
}
@keyframes voidBuildPulse {
  0%, 100% { opacity: 0.06; }
  50%      { opacity: 0.82; }
}
@keyframes voidGoldGlow {
  0%   { opacity: 0; }
  12%  { opacity: 0.88; }
  72%  { opacity: 0.88; }
  100% { opacity: 0; }
}
@keyframes absAttenPulse {
  0%, 100% { opacity: 0.9; }
  50%      { opacity: 0.4; }
}
`;

// ── Component ─────────────────────────────────────────────────────────────────

export const VoidOverlay: React.FC<VoidOverlayProps> = ({
  voids,
  selectedVoidId,
  showVoidLabels,
  transform,
  width,
  height,
  onVoidClick,
  researchDoneVoidIds,
  crossAnim,
}) => {
  const colorAssignment = useMemo(() => assignColors(voids), [voids]);

  // Pair cycling timer — increments every 2 s while cross-pollinating
  const [pairCycleIdx, setPairCycleIdx] = useState(0);

  useEffect(() => {
    if (crossAnim?.phase !== "cross_pollinating") {
      setPairCycleIdx(0);
      return;
    }
    const id = setInterval(() => setPairCycleIdx((i) => i + 1), 2000);
    return () => clearInterval(id);
  }, [crossAnim?.phase]);

  const rendered = useMemo(() => {
    return voids
      .map((v) => {
        const isSelected = v.void_id === selectedVoidId;
        const isAnimated = crossAnim?.voidId === v.void_id;
        const animPhase = isAnimated ? crossAnim!.phase : "idle";
        const nvertices = v.shape?.nvertices ?? [];
        if (nvertices.length < 3) return null;

        // Viewport cull
        const sb = hullScreenBounds(nvertices, transform);
        const pad = 60;
        if (
          sb.maxX < -pad ||
          sb.minX > width + pad ||
          sb.maxY < -pad ||
          sb.minY > height + pad
        )
          return null;

        const edgeStyle = (v.shape as any)?.edge_style;
        const strokeJoin = edgeStyle === "angular" ? "miter" : "round";
        const pathD = smoothHullPath(
          hullScreenPoints(nvertices, transform),
          edgeStyle === "angular" ? 0 : 0.38,
        );
        const [cx, cy] = normToScreen(v.ncx, v.ncy, transform);

        const slot = colorAssignment.get(v.void_id) ?? 0;
        const hue = PALETTE_HUES[slot];

        const fillNormal = makeColor(hue, 0.09);
        const fillSelected = makeColor(hue, 0.2);
        const strokeSolid = makeSolid(hue);
        const strokeDim = makeColor(hue, 0.6);
        const glowColor = makeGlow(hue);

        const rawName = v.name ?? `Void ${v.void_rank}`;
        const labelText =
          rawName.length > 38 ? rawName.slice(0, 36) + "…" : rawName;
        const showLabel = showVoidLabels || isSelected;

        // ── Pair connection lines (cross-pollinating phase) ──
        const activePairs: ActivePairEntry[] = isAnimated
          ? buildActivePairs(v, crossAnim!, pairCycleIdx)
          : [];

        // Quick lookup: doi → pair color for diamond overrides
        const pairColorByDoi = new Map<string, string>();
        for (const ap of activePairs) {
          pairColorByDoi.set(ap.fromPaper.DOI, ap.color);
          pairColorByDoi.set(ap.toPaper.DOI, ap.color);
        }

        const selectedDOIs = new Set(
          (v.selected_papers ?? []).map((p) => p.DOI),
        );

        // ── Border paper dots ──
        const borderOnlyDots = v.border_papers
          .filter((p) => !selectedDOIs.has(p.DOI))
          .map((p, i) => {
            const [px, py] = normToScreen(p.nx, p.ny, transform);
            return (
              <circle
                key={`b-${i}`}
                cx={px}
                cy={py}
                r={isSelected ? 3.5 : 2.5}
                fill={makeColor(hue, isSelected ? 0.7 : 0.45)}
                style={{ pointerEvents: "none" }}
              />
            );
          });

        // ── Selected paper markers ──
        const selectedMarkers = (v.selected_papers ?? []).map((p, i) => {
          const [px, py] = normToScreen(p.nx, p.ny, transform);
          const r = isSelected ? 5.5 : 4;
          const size = p.rank === 0 ? r * 1.25 : r;

          // During cross-pollinating use pair color; proposals_complete → back to gold
          const diamondColor =
            animPhase === "cross_pollinating" && pairColorByDoi.has(p.DOI)
              ? pairColorByDoi.get(p.DOI)!
              : strokeSolid;

          return (
            <g key={`s-${i}`} style={{ pointerEvents: "none" }}>
              <circle
                cx={px}
                cy={py}
                r={size + 3}
                fill={diamondColor}
                opacity={isSelected ? 0.18 : 0.1}
              />
              <rect
                x={px - size * 0.62}
                y={py - size * 0.62}
                width={size * 1.24}
                height={size * 1.24}
                fill={diamondColor}
                opacity={isSelected ? 0.92 : 0.6}
                stroke="rgba(255,255,255,0.95)"
                strokeWidth={isSelected ? 1.8 : 1.2}
                strokeLinejoin="round"
                transform={`rotate(45 ${px} ${py})`}
              />
              {isSelected && size >= 5 && (
                <text
                  x={px}
                  y={py}
                  textAnchor="middle"
                  dominantBaseline="central"
                  fontFamily="'JetBrains Mono', monospace"
                  fontSize={size * 0.85}
                  fontWeight={700}
                  fill="rgba(0,0,0,0.55)"
                  style={{ pointerEvents: "none", userSelect: "none" }}
                >
                  {p.rank + 1}
                </text>
              )}
            </g>
          );
        });

        // ── AbsAttenNet reveal ──
        const showAbsAttenNet =
          v.name === "Transformer-Augmented Vision Adaptation Gap" &&
          (researchDoneVoidIds?.has(v.void_id) ||
            (isAnimated && animPhase === "done"));

        return (
          <g
            key={v.void_id}
            className="cursor-pointer"
            style={{ pointerEvents: "all" }}
            onClick={() => onVoidClick(v.void_id)}
          >
            {/* Hull fill + glow + dashed stroke */}
            {pathD && (
              <>
                <path
                  d={pathD}
                  fill={isSelected ? fillSelected : fillNormal}
                  stroke="none"
                  style={{ pointerEvents: "fill" }}
                />
                <path
                  d={pathD}
                  fill="none"
                  stroke={glowColor}
                  strokeWidth={isSelected ? 16 : 9}
                  strokeLinejoin={strokeJoin}
                  style={{ pointerEvents: "none" }}
                />
                <path
                  d={pathD}
                  fill="none"
                  stroke={isSelected ? strokeSolid : strokeDim}
                  strokeWidth={isSelected ? 2 : 1.2}
                  strokeDasharray={isSelected ? "7 4" : "4 5"}
                  strokeLinejoin={strokeJoin}
                  style={{ pointerEvents: "none" }}
                >
                  {isSelected && (
                    <animate
                      attributeName="stroke-dashoffset"
                      from="0"
                      to="-44"
                      dur="2s"
                      repeatCount="indefinite"
                    />
                  )}
                </path>
                {/* Orchestrating / cross-pollinating / proposals-complete: soft golden pulse */}
                {isAnimated &&
                  (animPhase === "orchestrating" ||
                    animPhase === "cross_pollinating" ||
                    animPhase === "proposals_complete") && (
                    <path
                      d={pathD}
                      fill="#FFD700"
                      stroke="none"
                      style={{
                        pointerEvents: "none",
                        animation:
                          "voidOrchestrateGlow 2.5s ease-in-out infinite",
                        opacity: 0,
                      }}
                    />
                  )}
                Building: white brightness pulse
                {isAnimated && animPhase === "building" && (
                  <path
                    d={pathD}
                    fill="#ff990038"
                    stroke="none"
                    style={{
                      pointerEvents: "none",
                      animation: "voidBuildPulse 1.8s ease-in-out infinite",
                      opacity: 0.06,
                    }}
                  />
                )}
                {/* Glowing: 1-second gold flash */}
                {isAnimated && animPhase === "glowing" && (
                  <path
                    d={pathD}
                    fill="#FFD700"
                    stroke="none"
                    style={{
                      pointerEvents: "none",
                      animation: "voidGoldGlow 1.2s ease-out forwards",
                      opacity: 0,
                    }}
                  />
                )}
              </>
            )}

            {/* Pair connection lines */}
            {activePairs.map((ap, i) => {
              const [fx, fy] = normToScreen(
                ap.fromPaper.nx,
                ap.fromPaper.ny,
                transform,
              );
              const [tx, ty] = normToScreen(
                ap.toPaper.nx,
                ap.toPaper.ny,
                transform,
              );
              return (
                <line
                  key={`pair-${i}`}
                  x1={fx}
                  y1={fy}
                  x2={tx}
                  y2={ty}
                  stroke={ap.color}
                  strokeWidth={2}
                  strokeOpacity={0.75}
                  strokeLinecap="round"
                  style={{ pointerEvents: "none" }}
                />
              );
            })}

            {/* Border-only dots */}
            {borderOnlyDots}

            {/* Selected paper diamonds */}
            {selectedMarkers}

            {/* Void centroid marker when selected */}
            {isSelected && (
              <g
                transform={`translate(${cx},${cy})`}
                style={{ pointerEvents: "none" }}
              >
                <rect
                  x={-4}
                  y={-4}
                  width={8}
                  height={8}
                  fill={strokeSolid}
                  opacity={0.9}
                  transform="rotate(45)"
                />
              </g>
            )}

            {/* AbsAttenNet — revealed after benchmark passes */}
            {showAbsAttenNet &&
              (() => {
                const b = hullNormBounds(nvertices);
                const [mx, my] = normToScreen(
                  b.minX + (b.maxX - b.minX) * 0.37,
                  b.minY + (b.maxY - b.minY) * 0.58,
                  transform,
                );
                const modelColor = "#FFD700";
                const modelGlow = "rgba(255,215,0,0.3)";
                const dotR = 5;
                return (
                  <g style={{ pointerEvents: "none" }}>
                    <circle
                      cx={mx}
                      cy={my}
                      r={dotR + 8}
                      fill={modelGlow}
                      style={{
                        animation: "absAttenPulse 2s ease-in-out infinite",
                      }}
                    />
                    <circle cx={mx} cy={my} r={dotR + 5} fill={modelGlow} />
                    <circle cx={mx} cy={my} r={dotR} fill={modelColor} />
                    <circle
                      cx={mx}
                      cy={my}
                      r={1.8}
                      fill="rgba(255,255,255,0.9)"
                    />
                    <text
                      x={mx + dotR + 6}
                      y={my}
                      textAnchor="start"
                      dominantBaseline="central"
                      fontFamily="'JetBrains Mono', monospace"
                      fontSize={11}
                      fontWeight={700}
                      fill={modelColor}
                      stroke="rgba(0,0,0,0.55)"
                      strokeWidth={2.5}
                      strokeLinejoin="round"
                      paintOrder="stroke"
                      style={{ userSelect: "none" }}
                    >
                      AbsAttenNet
                    </text>
                  </g>
                );
              })()}

            {/* Label */}
            {showLabel && (
              <text
                x={cx}
                y={cy - 14}
                textAnchor="middle"
                dominantBaseline="auto"
                fontFamily="'JetBrains Mono', monospace"
                fontSize={isSelected ? 12 : 10}
                fontWeight={isSelected ? 700 : 500}
                style={{ pointerEvents: "none", userSelect: "none" }}
              >
                <tspan
                  fill={strokeSolid}
                  stroke={LABEL_SHADOW}
                  strokeWidth={isSelected ? 2 : 1.2}
                  strokeLinejoin="round"
                  paintOrder="stroke"
                >
                  {labelText}
                </tspan>
              </text>
            )}
          </g>
        );
      })
      .filter(Boolean);
  }, [
    voids,
    selectedVoidId,
    showVoidLabels,
    transform,
    width,
    height,
    onVoidClick,
    colorAssignment,
    researchDoneVoidIds,
    crossAnim,
    pairCycleIdx,
  ]);

  return (
    <svg
      className="absolute inset-0 pointer-events-none"
      width={width}
      height={height}
      style={{ zIndex: 11 }}
    >
      <defs>
        <style>{ANIM_CSS}</style>
      </defs>
      <g style={{ pointerEvents: "all" }}>{rendered}</g>
    </svg>
  );
};
