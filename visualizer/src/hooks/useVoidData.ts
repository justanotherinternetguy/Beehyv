import { useState, useEffect, useCallback } from 'react';

const VOIDS_URL = '/public/voids_ranked_cv.json';

export interface BorderPaper {
  title:          string;
  DOI:            string;
  x:              number;
  y:              number;
  nx:             number;
  ny:             number;
  cluster:        number;
  citation_count: number | null;
  year:           number | null;
  abstract:       string | null;
  enriched_via:   string | null;
}

export interface SelectedPaper {
  rank:           number;
  title:          string;
  DOI:            string;
  x:              number;
  y:              number;
  nx:             number;
  ny:             number;
  cluster:        number;
  citation_count: number | null;
  year:           number | null;
  abstract:       string | null;
  enriched_via:   string | null;
  scores: {
    combined:  number;
    citation:  number;
    recency:   number;
    sector:    number;
    angle_deg: number;
  };
}

export interface VoidShape {
  type:      'convex_hull';
  vertices:  [number, number][];
  nvertices: [number, number][];
}

export interface Void {
  void_id:         number;
  void_rank:       number;
  centroid:        [number, number];
  ncx:             number;
  ncy:             number;
  empty_radius:    number;
  name:            string;
  name_reasoning:  string;
  shape:           VoidShape;
  shape_area:      number;
  border_papers:   BorderPaper[];
  selected_papers: SelectedPaper[];
}

interface VoidDataState {
  voids:   Void[];
  loading: boolean;
  error:   string | null;
}

export function useVoidData(
  minX:  number,
  maxX:  number,
  minY:  number,
  maxY:  number,
  ready: boolean = true,
): VoidDataState {
  const [state, setState] = useState<VoidDataState>({
    voids:   [],
    loading: true,
    error:   null,
  });

  const load = useCallback(async () => {
    if (!ready) return;
    setState(s => ({ ...s, loading: true, error: null }));
    try {
      const res = await fetch(VOIDS_URL);
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      const raw: any[] = await res.json();

      const rangeX = maxX - minX || 1;
      const rangeY = maxY - minY || 1;
      const normXY = (x: number, y: number): [number, number] => [
        (x - minX) / rangeX,
        (y - minY) / rangeY,
      ];

      const voids: Void[] = raw.map(v => {
        const [ncx, ncy] = normXY(v.centroid[0], v.centroid[1]);

        const rawVerts: [number, number][] = v.shape?.vertices ?? [];
        const nvertices: [number, number][] = rawVerts.map(
          ([vx, vy]) => normXY(vx, vy),
        );

        const border_papers: BorderPaper[] = (v.border_papers ?? []).map((p: any) => {
          const [nx, ny] = normXY(p.x, p.y);
          return { ...p, nx, ny };
        });

        const selected_papers: SelectedPaper[] = (v.selected_papers ?? []).map((p: any) => {
          const [nx, ny] = normXY(p.x, p.y);
          return { ...p, nx, ny };
        });

        return {
          void_id:         v.void_id,
          void_rank:       v.void_rank,
          centroid:        v.centroid as [number, number],
          ncx,
          ncy,
          empty_radius:    v.empty_radius ?? 0,
          name:            v.name            ?? `Void ${v.void_id}`,
          name_reasoning:  v.name_reasoning  ?? '',
          shape: {
            type:      'convex_hull' as const,
            vertices:  rawVerts,
            nvertices,
          },
          shape_area:      v.shape_area ?? 0,
          border_papers,
          selected_papers,
        };
      });

      voids.sort((a, b) => b.empty_radius - a.empty_radius);
      setState({ voids, loading: false, error: null });
    } catch (err) {
      setState({
        voids:   [],
        loading: false,
        error:   err instanceof Error ? err.message : String(err),
      });
    }
  }, [ready, minX, maxX, minY, maxY]);

  useEffect(() => { load(); }, [load]);

  return state;
}