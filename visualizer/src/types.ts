export interface Paper {
  id: string | number;
  title: string;
  doi: string;
  pdfUrl?: string;
  x: number;
  y: number;
}

export interface ProcessedPaper extends Paper {
  nx: number; // normalized [0,1]
  ny: number;
  density: number; // 0-1
  clusterId: number; // -1 = noise
}

export interface Cluster {
  id: number;
  hull: [number, number][]; // screen-space convex hull points
  hullNorm: [number, number][]; // normalized-space hull points
  color: string;
  centroid: [number, number];
  size: number;
  label: string;
  revealScale: number; 
}

export interface ViewTransform {
  scale: number;
  offsetX: number;
  offsetY: number;
}

export interface AtlasState {
  papers: ProcessedPaper[];
  clusters: Map<number, Cluster>;
  loading: boolean;
  loadingProgress: string;
  error: string | null;
}

export interface PaperRef {
  doi: string;
  title: string;
  year?: number | null;
  citation_count?: number | null;
  abstract?: string | null;
}

export interface ResearchJobInfo {
  voidId: number;
  voidName: string;
  status: 'running' | 'done' | 'error';
  papers: PaperRef[];
}

export interface ResearchEvent {
  time: string;
  event: string;
  payload: Record<string, unknown>;
}

export interface MetricPoint {
  iteration: number;
  value: number;
  label: string;
}
