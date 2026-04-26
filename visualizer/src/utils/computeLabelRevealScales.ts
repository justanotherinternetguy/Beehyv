/**
 * For each cluster, computes the minimum zoom scale at which its label can be
 * shown without overlapping any larger cluster's label.
 *
 * This is called once after data loads (in useParquetData) and stored on the
 * cluster object. ClusterRings then just checks transform.scale >= revealScale,
 * giving perfectly monotonic visibility: labels appear as you zoom in and
 * disappear as you zoom out, with no flickering.
 *
 * Algorithm:
 *   Simulate the label layout at a range of zoom levels. At each level, run
 *   the same hide-smaller logic used in ClusterRings. The reveal scale for a
 *   cluster is the lowest scale at which it survives the collision pass.
 *
 * All positions are in normalised [0,1] space. Label sizes are divided by the
 * canvas dimensions to convert px → normalised, then further divided by scale
 * to get the size of a label in normalised units at that zoom level.
 */

import type { Cluster } from '../types';
import type { LabelSizePx } from './measureLabels';

/** Minimum scale before any label is shown (matches ClusterRings threshold). */
export const MIN_LABEL_SCALE = 400;

/**
 * Zoom levels to probe. Dense near the threshold, sparser at high zoom.
 * Labels that only appear at very high zoom get a high revealScale.
 */
const PROBE_SCALES = [
  400, 500, 600, 700, 800, 900, 1000,
  1200, 1500, 2000, 2500, 3000,
  4000, 6000, 10000, 20000, 80000,
];

/** Gap between labels in normalised space (mirrors LABEL_GAP_PX / scale). */
const LABEL_GAP_PX = 64;

function boxesOverlapNorm(
  ax: number, ay: number, aw: number, ah: number,
  bx: number, by: number, bw: number, bh: number,
  gapNorm: number,
): boolean {
  return (
    Math.abs(ax - bx) < (aw + bw) / 2 + gapNorm &&
    Math.abs(ay - by) < (ah + bh) / 2 + gapNorm
  );
}

export function computeLabelRevealScales(
  clusters:     Map<number, Cluster>,
  labelSizes:   Map<number, LabelSizePx>,
  canvasWidth:  number,
  canvasHeight: number,
): Map<number, number> {
  const revealScales = new Map<number, number>();

  const sorted = Array.from(clusters.values())
    .sort((a, b) => b.size - a.size);

  for (const scale of PROBE_SCALES) {
    // Fix 1: Set of ids (numbers), not arrays
    const visible = new Set<number>(sorted.map(c => c.id));

    for (let i = 0; i < sorted.length; i++) {
      const a = sorted[i];
      if (!visible.has(a.id)) continue;

      const aSz = labelSizes.get(a.id) ?? { width: 80, height: 20 };

      // Fix 3: label size in normalised space shrinks as scale increases
      const awNorm  = aSz.width  / canvasWidth  / (scale / MIN_LABEL_SCALE);
      const ahNorm  = aSz.height / canvasHeight / (scale / MIN_LABEL_SCALE);
      const gapNorm = LABEL_GAP_PX / Math.min(canvasWidth, canvasHeight) / (scale / MIN_LABEL_SCALE);

      for (let j = i + 1; j < sorted.length; j++) {
        const b = sorted[j];
        if (!visible.has(b.id)) continue;

        const bSz    = labelSizes.get(b.id) ?? { width: 80, height: 20 };
        const bwNorm = bSz.width  / canvasWidth  / (scale / MIN_LABEL_SCALE);
        const bhNorm = bSz.height / canvasHeight / (scale / MIN_LABEL_SCALE);

        if (!boxesOverlapNorm(
          a.centroid[0], a.centroid[1], awNorm, ahNorm,
          b.centroid[0], b.centroid[1], bwNorm, bhNorm,
          gapNorm,
        )) continue;

        visible.delete(b.id); // Fix 2: delete the id, not an array
      }
    }

    for (const id of visible) {
      if (!revealScales.has(id)) {
        revealScales.set(id, scale);
      }
    }

    if (revealScales.size === clusters.size) break;
  }

  for (const cluster of clusters.values()) {
    if (!revealScales.has(cluster.id)) {
      revealScales.set(cluster.id, Infinity);
    }
  }

  return revealScales;
}