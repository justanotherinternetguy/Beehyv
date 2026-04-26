import { useState, useEffect, useCallback, useRef } from 'react';
import * as arrow from 'apache-arrow';
import type { ProcessedPaper, Cluster, AtlasState } from '../types';
import { computeDensity, type Point2D } from '../utils/dbscan';
import { convexHull, expandHull, centroid, type Vec2 } from '../utils/convexHull';
import { clusterColor } from '../utils/colors';
import { measureLabels, type LabelSizePx } from '../utils/measureLabels';
import { computeLabelRevealScales } from '../utils/computeLabelRevealScales';

const PARQUET_URL        = '/public/umap_cv.parquet';
const CLUSTER_LABELS_URL = '/public/cluster_labels_cv.json';

const DENSITY_RADIUS = 0.015;
const LABEL_FONT     = "500 12px 'JetBrains Mono', monospace";

// Raw loaded data before reveal scales are stamped — never changes after load
interface RawAtlasData {
  papers:     ProcessedPaper[];
  clusters:   Map<number, Cluster>;
  labelSizes: Map<number, LabelSizePx>;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------
export function useParquetData(
  canvasWidth:  number,
  canvasHeight: number,
): AtlasState & { reload: () => void } {
  const [loadState, setLoadState] = useState<{
    loading:         boolean;
    loadingProgress: string;
    error:           string | null;
    raw:             RawAtlasData | null;
  }>({
    loading:         true,
    loadingProgress: 'Initializing…',
    error:           null,
    raw:             null,
  });

  // Stable ref to raw data so the reveal-scale effect doesn't re-run the fetch
  const rawRef = useRef<RawAtlasData | null>(null);

  // Final state with reveal scales stamped in
  const [atlasState, setAtlasState] = useState<AtlasState>({
    papers:          [],
    clusters:        new Map(),
    loading:         true,
    loadingProgress: 'Initializing…',
    error:           null,
  });

  // ── Load parquet once ──────────────────────────────────────────────────────
  const load = useCallback(async () => {
    setLoadState(s => ({ ...s, loading: true, error: null, loadingProgress: 'Fetching parquet file…' }));

    try {
      const [res, labelsRes] = await Promise.all([
        fetch(PARQUET_URL),
        fetch(CLUSTER_LABELS_URL),
      ]);

      if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);

      const clusterLabelMap: Record<string, string> = labelsRes.ok
        ? await labelsRes.json()
        : {};

      setLoadState(s => ({ ...s, loadingProgress: 'Reading parquet bytes…' }));
      const buffer = await res.arrayBuffer();

      setLoadState(s => ({ ...s, loadingProgress: 'Parsing parquet columns…' }));
      const table = await arrow.tableFromIPC(new Uint8Array(buffer));

      setLoadState(s => ({ ...s, loadingProgress: 'Extracting columns…' }));

      const n = table.numRows;
      const ids: (string | number)[] = [];
      const titles: string[]         = [];
      const dois: string[]           = [];
      const pdfUrls: string[]        = [];
      const xs: number[]             = [];
      const ys: number[]             = [];
      const clusterIds: number[]     = [];

      const colNames = table.schema.fields.map(f => f.name.toLowerCase());
      const getCol = (names: string[]) => {
        for (const name of names) {
          const idx = colNames.indexOf(name);
          if (idx >= 0) return table.getChildAt(idx);
        }
        return null;
      };

      const idCol      = getCol(['id']);
      const titleCol   = getCol(['title']);
      const doiCol     = getCol(['doi']);
      const pdfUrlCol  = getCol(['pdf_url', 'pdfurl', 'pdf']);
      const xCol       = getCol(['x', 'umap_x', 'umap1', 'dim1']);
      const yCol       = getCol(['y', 'umap_y', 'umap2', 'dim2']);
      const clusterCol = getCol(['cluster', 'cluster_id', 'label']);

      if (!xCol || !yCol) {
        throw new Error('Could not find x/y columns in parquet. Expected: x, y (or umap_x, umap_y)');
      }
      if (!clusterCol) {
        throw new Error(
          'No cluster column found in parquet. Expected: cluster, cluster_id, or label. ' +
          'Re-run 02_umap.py to generate cluster assignments from high-dimensional vectors.'
        );
      }

      for (let i = 0; i < n; i++) {
        ids.push(idCol ? String(idCol.get(i)) : String(i));
        titles.push(titleCol ? String(titleCol.get(i) ?? '') : '');
        dois.push(doiCol ? String(doiCol.get(i) ?? '') : '');
        pdfUrls.push(pdfUrlCol ? String(pdfUrlCol.get(i) ?? '') : '');
        xs.push(Number(xCol.get(i)));
        ys.push(Number(yCol.get(i)));
        clusterIds.push(Number(clusterCol.get(i)));
      }

      setLoadState(s => ({ ...s, loadingProgress: 'Normalizing coordinates…' }));

      let minX = Infinity, maxX = -Infinity;
      let minY = Infinity, maxY = -Infinity;
      for (let i = 0; i < n; i++) {
        if (xs[i] < minX) minX = xs[i];
        if (xs[i] > maxX) maxX = xs[i];
        if (ys[i] < minY) minY = ys[i];
        if (ys[i] > maxY) maxY = ys[i];
      }
      const rangeX = maxX - minX || 1;
      const rangeY = maxY - minY || 1;

      const nxs = xs.map(v => (v - minX) / rangeX);
      const nys = ys.map(v => (v - minY) / rangeY);

      setLoadState(s => ({ ...s, loadingProgress: 'Computing density…' }));

      const pts: Point2D[] = nxs.map((x, i) => ({ x, y: nys[i], index: i }));
      const density = computeDensity(pts, DENSITY_RADIUS);

      setLoadState(s => ({ ...s, loadingProgress: 'Building cluster hulls…' }));

      const clusterPoints = new Map<number, Vec2[]>();
      for (let i = 0; i < n; i++) {
        const cid = clusterIds[i];
        if (cid < 0) continue;
        if (!clusterPoints.has(cid)) clusterPoints.set(cid, []);
        clusterPoints.get(cid)!.push([nxs[i], nys[i]]);
      }

      const weightedSums = new Map<number, { wx: number; wy: number; w: number }>();
      for (const id of clusterPoints.keys()) {
        weightedSums.set(id, { wx: 0, wy: 0, w: 0 });
      }
      for (let i = 0; i < n; i++) {
        const id = clusterIds[i];
        if (id < 0) continue;
        const s = weightedSums.get(id);
        if (!s) continue;
        const w = density[i];
        s.wx += nxs[i] * w;
        s.wy += nys[i] * w;
        s.w  += w;
      }

      // Build clusters without revealScale — that's stamped separately
      const clusters = new Map<number, Cluster>();
      for (const [id, cpts] of clusterPoints) {
        const hull     = convexHull([...cpts]);
        const expanded = expandHull(hull, 0.008);
        const sums     = weightedSums.get(id)!;
        const c: Vec2  = sums.w > 0
          ? [sums.wx / sums.w, sums.wy / sums.w]
          : centroid(hull);

        clusters.set(id, {
          id,
          hull:        expanded,
          hullNorm:    hull,
          color:       clusterColor(id),
          centroid:    c,
          size:        cpts.length,
          label:       clusterLabelMap[String(id)] ?? `Cluster ${id}`,
          revealScale: Infinity,
        });
      }

      setLoadState(s => ({ ...s, loadingProgress: 'Measuring labels…' }));

      const ids2     = Array.from(clusters.keys());
      const strings  = ids2.map(id => clusters.get(id)!.label);
      const measured = measureLabels(strings, {
        font:     LABEL_FONT,
        maxChars: 16,
        paddingX: 6,
        paddingY: 4,
        fontSize: 12,
      });
      const labelSizes = new Map(ids2.map((id, i) => [id, measured[i]]));

      const papers: ProcessedPaper[] = [];
      for (let i = 0; i < n; i++) {
        papers.push({
          id:        ids[i],
          title:     titles[i],
          doi:       dois[i],
          pdfUrl:    pdfUrls[i],
          x:         xs[i],
          y:         ys[i],
          nx:        nxs[i],
          ny:        nys[i],
          density:   density[i],
          clusterId: clusterIds[i],
        });
      }

      const raw: RawAtlasData = { papers, clusters, labelSizes };
      rawRef.current = raw;

      setLoadState({
        loading:         false,
        loadingProgress: '',
        error:           null,
        raw,
      });
    } catch (err) {
      setLoadState(s => ({
        ...s,
        loading: false,
        error: err instanceof Error ? err.message : String(err),
        raw: null,
      }));
    }
  }, []); // no canvas size deps — load never re-runs due to resize

  useEffect(() => { load(); }, [load]);

  // ── Re-stamp reveal scales when canvas size changes ────────────────────────
  // This is cheap (pure CPU, no fetch) so it's fine to re-run on resize
  useEffect(() => {
    const raw = rawRef.current;
    if (!raw || loadState.loading || loadState.error) return;

    const { papers, clusters, labelSizes } = raw;

    // Clone clusters so we don't mutate the ref's data
    const updated = new Map<number, Cluster>();
    for (const [id, cluster] of clusters) {
      updated.set(id, { ...cluster });
    }

    const revealScales = computeLabelRevealScales(
      updated,
      labelSizes,
      canvasWidth,
      canvasHeight,
    );

    for (const [id, scale] of revealScales) {
      updated.get(id)!.revealScale = scale;
    }

    setAtlasState({
      papers,
      clusters: updated,
      loading:         false,
      loadingProgress: '',
      error:           null,
    });
  }, [loadState.loading, loadState.error, loadState.raw, canvasWidth, canvasHeight]);

  // Pass through loading/error states
  if (loadState.loading || loadState.error) {
    return {
      papers:          [],
      clusters:        new Map(),
      loading:         loadState.loading,
      loadingProgress: loadState.loadingProgress,
      error:           loadState.error,
      reload:          load,
    };
  }

  return { ...atlasState, reload: load };
}
