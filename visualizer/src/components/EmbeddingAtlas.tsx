/**
 * EmbeddingAtlas.tsx  — with void detection overlay + dark mode
 */

import React, {
  useRef,
  useState,
  useEffect,
  useCallback,
  useMemo,
} from "react";
import { quadtree as d3Quadtree } from "d3-quadtree";
import type { ProcessedPaper, ViewTransform } from "../types";
import { useWebGLRenderer } from "../hooks/useWebGLRenderer";
import { ClusterRings } from "./ClusterRings";
import { Tooltip } from "./Tooltip";
import { SearchBar } from "./SearchBar";
import { useParquetData } from "../hooks/useParquetData";
import { useVoidData } from "../hooks/useVoidData";
import { VoidOverlay } from "./VoidOverlay";
import { VoidPanel } from "./VoidPanel";
import { TabBar } from "./TabBar";
import { ResearchTab } from "./ResearchTab";
import { DeepResearchTab } from "./DeepResearchTab";
import { JobsDropdown } from "./JobsDropdown";
import type { ResearchJobInfo } from "../types";

const MIN_SCALE = 100;
const MAX_SCALE = 80000;
const HOVER_RADIUS_PX = 12;
const DRAG_THRESHOLD_PX = 4;

function buildQuadtree(papers: ProcessedPaper[]) {
  return d3Quadtree<ProcessedPaper>()
    .x((d) => d.nx)
    .y((d) => d.ny)
    .addAll(papers);
}

function screenToNorm(
  sx: number,
  sy: number,
  transform: ViewTransform,
): [number, number] {
  return [
    (sx - transform.offsetX) / transform.scale,
    (sy - transform.offsetY) / transform.scale,
  ];
}

export const EmbeddingAtlas: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ width: 800, height: 600 });

  // ── Dark mode ──────────────────────────────────────────────────────────────
  const [darkMode, setDarkMode] = useState(() => {
    return window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
  });

  const dm = darkMode; // shorthand

  const { papers, clusters, loading, loadingProgress, error, reload } =
    useParquetData(size.width, size.height);

  const bounds = useMemo(() => {
    if (papers.length === 0) return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
    let minX = Infinity,
      maxX = -Infinity,
      minY = Infinity,
      maxY = -Infinity;
    for (const p of papers) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
    return { minX, maxX, minY, maxY };
  }, [papers]);

  const { voids, loading: voidsLoading } = useVoidData(
    bounds.minX,
    bounds.maxX,
    bounds.minY,
    bounds.maxY,
    papers.length > 0,
  );

  const [selectedVoidId, setSelectedVoidId] = useState<number | null>(null);
  const [voidsVisible, setVoidsVisible] = useState(true);
  const [showVoidLabels, setShowVoidLabels] = useState(false);

  // ── Research tabs ──────────────────────────────────────────────────────────
  const [activeTab, setActiveTab] = useState<string>("viz");
  const [researchJobs, setResearchJobs] = useState<
    Map<string, ResearchJobInfo>
  >(new Map());
  const voidJobMap = useRef<Map<number, string>>(new Map());

  const [transform, setTransform] = useState<ViewTransform>({
    scale: 600,
    offsetX: 50,
    offsetY: 50,
  });
  const [hovered, setHovered] = useState<ProcessedPaper | null>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [selectedClusterId, setSelectedClusterId] = useState<number | null>(
    null,
  );
  const [hoveredClusterId, setHoveredClusterId] = useState<number | null>(null);
  const [searchResultIds, setSearchResultIds] = useState<Set<
    string | number
  > | null>(null);

  const qtRef = useRef<ReturnType<typeof buildQuadtree> | null>(null);
  const isDragging = useRef(false);
  const dragMoved = useRef(false);
  const lastMouse = useRef({ x: 0, y: 0 });
  const transformRef = useRef(transform);
  transformRef.current = transform;

  const voidBorderIds = useMemo<Set<string | number> | null>(() => {
    if (selectedVoidId === null) return null;
    const v = voids.find((v) => v.void_id === selectedVoidId);
    if (!v) return null;
    return new Set<string | number>(v.border_papers.map((p) => p.DOI));
  }, [selectedVoidId, voids]);

  const activeSearchResultIds = useMemo(() => {
    if (voidBorderIds !== null) return voidBorderIds;
    return searchResultIds;
  }, [voidBorderIds, searchResultIds]);

  const activeSelectedClusterId = useMemo(() => {
    if (selectedVoidId !== null) return null;
    return selectedClusterId;
  }, [selectedVoidId, selectedClusterId]);

  // ── Resize observer ───────────────────────────────────────────────────────
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry!.contentRect;
      setSize({ width, height });
    });
    ro.observe(el);
    setSize({ width: el.clientWidth, height: el.clientHeight });
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (papers.length > 0) qtRef.current = buildQuadtree(papers);
  }, [papers]);

  useEffect(() => {
    if (papers.length === 0) return;
    setTransform({
      scale: Math.min(size.width, size.height) * 0.9,
      offsetX: size.width * 0.05,
      offsetY: size.height * 0.05,
    });
  }, [size.width, size.height, papers.length > 0]);

  useWebGLRenderer(canvasRef, {
    papers,
    width: size.width,
    height: size.height,
    transform,
    hoveredId: hovered?.id ?? null,
    selectedClusterId: activeSelectedClusterId,
    searchResultIds: activeSearchResultIds,
    darkMode,
  });

  // ── Pointer events ────────────────────────────────────────────────────────
  const rafRef = useRef<number | null>(null);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    setMousePos({ x: sx, y: sy });

    if (isDragging.current) {
      const dx = sx - lastMouse.current.x;
      const dy = sy - lastMouse.current.y;
      if (
        Math.abs(e.clientX - lastMouse.current.x) > DRAG_THRESHOLD_PX ||
        Math.abs(e.clientY - lastMouse.current.y) > DRAG_THRESHOLD_PX
      )
        dragMoved.current = true;
      setTransform((t) => ({
        ...t,
        offsetX: t.offsetX + dx,
        offsetY: t.offsetY + dy,
      }));
      lastMouse.current = { x: sx, y: sy };
      return;
    }

    if (rafRef.current) return;
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      if (!qtRef.current) return;
      const t = transformRef.current;
      const [nx, ny] = screenToNorm(sx, sy, t);
      const radiusNorm = HOVER_RADIUS_PX / t.scale;
      const found = qtRef.current.find(nx, ny, radiusNorm);
      setHovered(found ?? null);
    });
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return;
    isDragging.current = true;
    dragMoved.current = false;
    lastMouse.current = { x: e.clientX, y: e.clientY };
  }, []);

  const handleMouseUp = useCallback(() => {
    isDragging.current = false;
  }, []);

  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      if (dragMoved.current) return;
      if (!hovered) {
        if (selectedClusterId !== null) setSelectedClusterId(null);
        if (selectedVoidId !== null) setSelectedVoidId(null);
        return;
      }
      const pdfUrl = hovered.pdfUrl?.trim();
      if (pdfUrl) {
        window.open(pdfUrl, "_blank", "noopener,noreferrer");
        return;
      }
      const doi = hovered.doi;
      if (doi && doi !== "null" && doi.trim()) {
        const url = `https://arxiv.org/abs/${doi}`;
        window.open(url, "_blank", "noopener,noreferrer");
      }
    },
    [hovered, selectedClusterId, selectedVoidId],
  );

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    setTransform((t) => {
      const newScale = Math.max(
        MIN_SCALE,
        Math.min(MAX_SCALE, t.scale * factor),
      );
      const scaleDelta = newScale / t.scale;
      return {
        scale: newScale,
        offsetX: mx - scaleDelta * (mx - t.offsetX),
        offsetY: my - scaleDelta * (my - t.offsetY),
      };
    });
  }, []);

  const lastTouches = useRef<React.TouchList | null>(null);
  const handleTouchStart = useCallback((e: React.TouchEvent) => {
    lastTouches.current = e.touches;
  }, []);
  const handleTouchMove = useCallback((e: React.TouchEvent) => {
    e.preventDefault();
    if (!lastTouches.current) return;
    if (e.touches.length === 1 && lastTouches.current.length === 1) {
      const dx = e.touches[0].clientX - lastTouches.current[0].clientX;
      const dy = e.touches[0].clientY - lastTouches.current[0].clientY;
      setTransform((t) => ({
        ...t,
        offsetX: t.offsetX + dx,
        offsetY: t.offsetY + dy,
      }));
    } else if (e.touches.length === 2 && lastTouches.current.length === 2) {
      const d0 = Math.hypot(
        lastTouches.current[0].clientX - lastTouches.current[1].clientX,
        lastTouches.current[0].clientY - lastTouches.current[1].clientY,
      );
      const d1 = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY,
      );
      const factor = d1 / (d0 || 1);
      const mx = (e.touches[0].clientX + e.touches[1].clientX) / 2;
      const my = (e.touches[0].clientY + e.touches[1].clientY) / 2;
      setTransform((t) => {
        const newScale = Math.max(
          MIN_SCALE,
          Math.min(MAX_SCALE, t.scale * factor),
        );
        const sf = newScale / t.scale;
        return {
          scale: newScale,
          offsetX: mx - sf * (mx - t.offsetX),
          offsetY: my - sf * (my - t.offsetY),
        };
      });
    }
    lastTouches.current = e.touches;
  }, []);

  const handleSearchResults = useCallback(
    (ids: Set<string | number> | null, focusPaper?: ProcessedPaper) => {
      setSearchResultIds(ids);
      setSelectedVoidId(null);
      if (focusPaper) {
        setTransform((t) => ({
          ...t,
          offsetX: size.width / 2 - focusPaper.nx * t.scale,
          offsetY: size.height / 2 - focusPaper.ny * t.scale,
        }));
      }
    },
    [size],
  );

  const handleClusterClick = useCallback(
    (id: number) => {
      setSelectedClusterId((prev) => (prev === id ? null : id));
      setSelectedVoidId(null);
      setSearchResultIds(null);
      const cluster = clusters.get(id);
      if (cluster) {
        setTransform((t) => ({
          ...t,
          offsetX: size.width / 2 - cluster.centroid[0] * t.scale,
          offsetY: size.height / 2 - cluster.centroid[1] * t.scale,
        }));
      }
    },
    [clusters, size],
  );

  const handleVoidSelect = useCallback(
    (id: number | null) => {
      setSelectedVoidId(id);
      setSelectedClusterId(null);
      setSearchResultIds(null);
      if (id === null) return;
      const v = voids.find((v) => v.void_id === id);
      if (!v) return;
      setTransform((t) => ({
        ...t,
        offsetX: size.width / 2 - v.ncx * t.scale,
        offsetY: size.height / 2 - v.ncy * t.scale,
      }));
    },
    [voids, size],
  );

  const IMAGENET_GAP = "Transformer-Augmented Vision Adaptation Gap";

  const handleInvestigate = useCallback(async () => {
    if (selectedVoidId === null) return;
    const v = voids.find((v) => v.void_id === selectedVoidId);
    if (!v) return;

    // Switch to existing job if already running
    const existingJobId = voidJobMap.current.get(v.void_id);
    if (existingJobId) {
      const isImagenet = v.name === IMAGENET_GAP;
      // setActiveTab(
      //   isImagenet ? `${existingJobId}:deep` : `${existingJobId}:logs`,
      // );
      setActiveTab(`${existingJobId}:deep`); // For now always open logs on click; users can switch to deep research tab from there if it's an imagenet gap job
      return;
    }

    const papers = v.selected_papers.map((p) => ({
      doi: p.DOI,
      title: p.title,
      year: p.year,
      citation_count: p.citation_count,
      abstract: p.abstract,
    }));

    try {
      const resp = await fetch("/api/investigate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ voidId: v.void_id, voidName: v.name, papers }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const { jobId } = (await resp.json()) as { jobId: string };

      voidJobMap.current.set(v.void_id, jobId);
      const isImagenet = v.name === IMAGENET_GAP;
      setResearchJobs((prev) => {
        const m = new Map(prev);
        m.set(jobId, {
          voidId: v.void_id,
          voidName: v.name,
          status: "running",
          papers,
        });
        return m;
      });
      // For imagenet gap open deep research tab; for others open full logs
      setActiveTab(isImagenet ? `${jobId}:deep` : `${jobId}:logs`);
    } catch (e) {
      console.error("[Investigate] Failed to start investigation:", e);
    }
  }, [selectedVoidId, voids]);

  const researchDoneVoidIds = useMemo<Set<number>>(() => {
    const s = new Set<number>();
    for (const [, job] of researchJobs) {
      if (job.status === 'done') s.add(job.voidId);
    }
    return s;
  }, [researchJobs]);

  const stats = useMemo(
    () => ({
      total: papers.length,
      clusterCount: clusters.size,
    }),
    [papers.length, clusters.size],
  );

  // ── Theme tokens ──────────────────────────────────────────────────────────
  const theme = {
    bg: dm
      ? "linear-gradient(135deg, #0d0f14 0%, #111318 40%, #120f18 100%)"
      : "linear-gradient(135deg, #f0f2f8 0%, #e8ecf5 40%, #f2eef8 100%)",
    dot: dm ? "rgba(120,130,180,0.07)" : "rgba(120,130,180,0.12)",
    glass: dm ? "rgba(20,22,30,0.88)" : "rgba(255,255,255,0.82)",
    glassBorder: dm ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.07)",
    glassShadow: dm
      ? "0 4px 16px rgba(0,0,0,0.4)"
      : "0 4px 16px rgba(0,0,0,0.07)",
    titleColor: dm ? "#e2e8f0" : "#1e293b",
    subColor: dm ? "#f59e0b" : "#94a3b8",
    btnBg: dm ? "rgba(20,22,30,0.9)" : "rgba(255,255,255,0.9)",
    btnBorder: dm ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)",
    btnColor: dm ? "#94a3b8" : "#475569",
    clusterBg: dm ? "rgba(20,22,30,0.92)" : "rgba(255,255,255,0.92)",
  };

  // ── Render ────────────────────────────────────────────────────────────────
  const shortVoidName = (name: string) =>
    name.length > 24 ? name.slice(0, 22) + "…" : name;

  const tabList = [
    { id: "viz", label: "Visualization" },
    ...Array.from(researchJobs.entries()).flatMap(([jid, job]) => {
      const short = shortVoidName(job.voidName);
      const isImagenet = job.voidName === IMAGENET_GAP;
      const logsTab = {
        id: `${jid}:logs`,
        label: `${short} — Full Logs`,
        closeable: true as const,
        status: job.status,
      };
      if (!isImagenet) return [logsTab];
      return [
        logsTab,
        {
          id: `${jid}:deep`,
          label: `${short} — Deep Research`,
          closeable: true as const,
          status: job.status,
        },
      ];
    }),
  ];

  const handleCloseTab = (tabId: string) => {
    const jobId = tabId.split(":")[0];
    // Remove the whole job if both tabs are being closed (we close per-tab but same jobId)
    const jobTabs = tabList.filter(
      (t) => t.id.startsWith(jobId) && t.id !== "viz",
    );
    if (jobTabs.length <= 1) {
      // Last tab for this job — clean up the job entirely
      setResearchJobs((prev) => {
        const m = new Map(prev);
        m.delete(jobId);
        return m;
      });
      for (const [vid, jid] of voidJobMap.current)
        if (jid === jobId) {
          voidJobMap.current.delete(vid);
          break;
        }
    }
    if (activeTab === tabId) setActiveTab("viz");
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
      {/* Content area — visualization always mounted; research overlay on top */}
      <div style={{ flex: 1, position: "relative", overflow: "hidden" }}>
        {/* Visualization pane */}
        <div
          className="absolute inset-0 overflow-hidden"
          style={{ background: theme.bg, transition: "background 0.4s" }}
        >
          {/* Grid texture */}
          <div
            className="absolute inset-0 pointer-events-none"
            style={{
              backgroundImage: `radial-gradient(circle, ${theme.dot} 1px, transparent 1px)`,
              backgroundSize: "28px 28px",
              zIndex: 0,
            }}
          />

          {/* Void panel */}
          <VoidPanel
            voids={voids}
            selectedVoidId={selectedVoidId}
            showVoidLabels={showVoidLabels}
            voidsVisible={voidsVisible}
            loading={voidsLoading}
            onSelectVoid={handleVoidSelect}
            onToggleLabels={() => setShowVoidLabels((v) => !v)}
            onToggleVoids={() => setVoidsVisible((v) => !v)}
            darkMode={dm}
          />

          {/* Main canvas + overlay container */}
          <div
            ref={containerRef}
            className="absolute inset-0"
            style={{
              cursor: isDragging.current
                ? "grabbing"
                : hovered
                  ? "pointer"
                  : "grab",
              zIndex: 1,
            }}
            onMouseMove={handleMouseMove}
            onMouseDown={handleMouseDown}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            onClick={handleClick}
            onWheel={handleWheel}
            onTouchStart={handleTouchStart}
            onTouchMove={handleTouchMove}
            onTouchEnd={() => {
              lastTouches.current = null;
            }}
          >
            <canvas
              ref={canvasRef}
              width={size.width}
              height={size.height}
              className="absolute inset-0"
              style={{ zIndex: 1 }}
            />

            {!loading && (
              <ClusterRings
                clusters={clusters}
                selectedClusterId={selectedClusterId}
                hoveredClusterId={hoveredClusterId}
                transform={transform}
                width={size.width}
                height={size.height}
                onClusterClick={handleClusterClick}
                onClusterHover={setHoveredClusterId}
                darkMode={dm}
              />
            )}

            {!loading && voidsVisible && voids.length > 0 && (
              <VoidOverlay
                voids={voids}
                selectedVoidId={selectedVoidId}
                showVoidLabels={showVoidLabels}
                transform={transform}
                width={size.width}
                height={size.height}
                onVoidClick={(id) =>
                  handleVoidSelect(id === selectedVoidId ? null : id)
                }
                researchDoneVoidIds={researchDoneVoidIds}
              />
            )}
          </div>

          {/* Tooltip */}
          <Tooltip
            paper={hovered}
            x={mousePos.x}
            y={mousePos.y}
            containerWidth={size.width}
            containerHeight={size.height}
            darkMode={dm}
          />

          {/* Search bar */}
          <div
            className="absolute top-4 left-1/2 -translate-x-1/2 flex items-center gap-3"
            style={{ zIndex: 50 }}
          >
            <SearchBar
              papers={papers}
              onResults={handleSearchResults}
              disabled={loading}
              darkMode={dm}
            />
          </div>

          {/* Top-left title */}
          <div className="absolute top-4 left-4" style={{ zIndex: 50 }}>
            <div
              style={{
                background: theme.glass,
                backdropFilter: "blur(16px)",
                WebkitBackdropFilter: "blur(16px)",
                border: `1px solid ${theme.glassBorder}`,
                borderRadius: 10,
                padding: "8px 14px",
                boxShadow: theme.glassShadow,
                transition: "background 0.4s, border-color 0.4s",
              }}
            >
              <h1
                style={{
                  fontFamily: "'Crimson Pro', Georgia, serif",
                  fontSize: 17,
                  lineHeight: 1,
                  fontWeight: 600,
                  color: theme.titleColor,
                  margin: 0,
                  transition: "color 0.4s",
                }}
              >
                arXiv Atlas
              </h1>
              <p
                style={{
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: 9,
                  letterSpacing: "0.04em",
                  marginTop: 4,
                  marginBottom: 0,
                  color: theme.subColor,
                  transition: "color 0.4s",
                }}
              >
                {stats.total > 0
                  ? `${stats.total.toLocaleString()} papers · ${stats.clusterCount} clusters · ${voids.length} voids`
                  : "Loading…"}
              </p>
            </div>
          </div>

          {/* Bottom-right controls: zoom + dark mode toggle */}
          <div
            className="absolute bottom-4 right-4 flex flex-col gap-2"
            style={{ zIndex: 50 }}
          >
            {[
              { label: "+", delta: 1.5, title: "Zoom in" },
              { label: "−", delta: 1 / 1.5, title: "Zoom out" },
              { label: "⊙", delta: null, title: "Reset view" },
            ].map(({ label, delta, title }) => (
              <button
                key={label}
                title={title}
                onClick={() => {
                  if (delta === null) {
                    setTransform({
                      scale: Math.min(size.width, size.height) * 0.9,
                      offsetX: size.width * 0.05,
                      offsetY: size.height * 0.05,
                    });
                    setSelectedClusterId(null);
                    setSelectedVoidId(null);
                    setSearchResultIds(null);
                  } else {
                    setTransform((t) => {
                      const newScale = Math.max(
                        MIN_SCALE,
                        Math.min(MAX_SCALE, t.scale * delta),
                      );
                      const sf = newScale / t.scale;
                      return {
                        scale: newScale,
                        offsetX:
                          size.width / 2 - sf * (size.width / 2 - t.offsetX),
                        offsetY:
                          size.height / 2 - sf * (size.height / 2 - t.offsetY),
                      };
                    });
                  }
                }}
                style={{
                  width: 36,
                  height: 36,
                  background: theme.btnBg,
                  backdropFilter: "blur(12px)",
                  WebkitBackdropFilter: "blur(12px)",
                  border: `1px solid ${theme.btnBorder}`,
                  borderRadius: 8,
                  boxShadow: dm
                    ? "0 2px 8px rgba(0,0,0,0.4)"
                    : "0 2px 8px rgba(0,0,0,0.07)",
                  fontSize: label === "⊙" ? 16 : 20,
                  color: theme.btnColor,
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  transition: "background 0.15s",
                }}
              >
                {label}
              </button>
            ))}

            {/* Dark mode toggle */}
            <button
              title={dm ? "Switch to light mode" : "Switch to dark mode"}
              onClick={() => setDarkMode((v) => !v)}
              style={{
                width: 36,
                height: 36,
                background: theme.btnBg,
                backdropFilter: "blur(12px)",
                WebkitBackdropFilter: "blur(12px)",
                border: `1px solid ${theme.btnBorder}`,
                borderRadius: 8,
                boxShadow: dm
                  ? "0 2px 8px rgba(0,0,0,0.4)"
                  : "0 2px 8px rgba(0,0,0,0.07)",
                fontSize: 16,
                color: theme.btnColor,
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                transition: "background 0.15s",
              }}
            >
              {dm ? "☀︎" : "☽"}
            </button>
          </div>

          {/* Bottom-left cluster info */}
          {selectedClusterId !== null && selectedVoidId === null && (
            <div className="absolute bottom-4 left-4" style={{ zIndex: 50 }}>
              <div
                style={{
                  background: theme.clusterBg,
                  backdropFilter: "blur(16px)",
                  WebkitBackdropFilter: "blur(16px)",
                  border: `1px solid ${clusters.get(selectedClusterId)?.color ?? "#ccc"}40`,
                  borderRadius: 10,
                  padding: "10px 14px",
                  boxShadow: dm
                    ? "0 4px 20px rgba(0,0,0,0.5)"
                    : "0 4px 20px rgba(0,0,0,0.08)",
                  maxWidth: 260,
                  transition: "background 0.4s",
                }}
              >
                <div className="flex items-center gap-2 mb-1">
                  <div
                    style={{
                      width: 10,
                      height: 10,
                      borderRadius: "50%",
                      background:
                        clusters.get(selectedClusterId)?.color ?? "#ccc",
                    }}
                  />
                  <span
                    style={{
                      fontFamily: "'JetBrains Mono', monospace",
                      fontSize: 11,
                      fontWeight: 500,
                      color: dm ? "#e2e8f0" : "#374151",
                    }}
                  >
                    {clusters.get(selectedClusterId)?.label ??
                      `Cluster ${selectedClusterId}`}
                  </span>
                  <button
                    onClick={() => setSelectedClusterId(null)}
                    style={{
                      marginLeft: "auto",
                      fontSize: 14,
                      background: "none",
                      border: "none",
                      cursor: "pointer",
                      color: dm ? "#4b5563" : "#d1d5db",
                    }}
                  >
                    ×
                  </button>
                </div>
                <p
                  style={{
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: 10,
                    color: dm ? "#6b7280" : "#6b7280",
                    margin: 0,
                  }}
                >
                  {clusters.get(selectedClusterId)?.size.toLocaleString() ?? 0}{" "}
                  papers
                </p>
              </div>
            </div>
          )}

          {/* Investigate button — shown at bottom centre when a void is selected */}
          {selectedVoidId !== null && (
            <div
              style={{
                position: "absolute",
                bottom: 20,
                left: "50%",
                transform: "translateX(-50%)",
                zIndex: 60,
              }}
            >
              <button
                onClick={handleInvestigate}
                style={{
                  background:
                    "linear-gradient(135deg, #f59e0b 0%, #d97706 100%)",
                  color: "white",
                  border: "none",
                  borderRadius: 10,
                  padding: "10px 28px",
                  fontSize: 12,
                  fontFamily: "'JetBrains Mono', monospace",
                  fontWeight: 700,
                  cursor: "pointer",
                  boxShadow:
                    "0 4px 24px rgba(245,158,11,0.55), 0 2px 8px rgba(0,0,0,0.35)",
                  letterSpacing: "0.08em",
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  textTransform: "uppercase",
                  whiteSpace: "nowrap",
                }}
              >
                <span style={{ fontSize: 14 }}>⚗</span>
                Investigate
              </button>
            </div>
          )}

          {/* Loading overlay */}
          {loading && (
            <div
              className="absolute inset-0 flex flex-col items-center justify-center"
              style={{
                background: dm
                  ? "rgba(13,15,20,0.92)"
                  : "rgba(240,242,248,0.9)",
                zIndex: 100,
                backdropFilter: "blur(4px)",
              }}
            >
              <div className="flex flex-col items-center gap-4">
                <div
                  style={{
                    width: 44,
                    height: 44,
                    border: `3px solid ${dm ? "rgba(245,158,11,0.15)" : "rgba(217,119,6,0.15)"}`,
                    borderTop: `3px solid ${dm ? "#f59e0b" : "#d97706"}`,
                    borderRadius: "50%",
                    animation: "spin 0.8s linear infinite",
                  }}
                />
                <div className="text-center">
                  <p
                    style={{
                      fontFamily: "'Crimson Pro', Georgia, serif",
                      fontSize: 18,
                      fontWeight: 500,
                      color: dm ? "#e2e8f0" : "#374151",
                      margin: 0,
                    }}
                  >
                    arXiv Atlas
                  </p>
                  <p
                    style={{
                      fontFamily: "'JetBrains Mono', monospace",
                      fontSize: 11,
                      color: dm ? "#6b7280" : "#9ca3af",
                      marginTop: 4,
                    }}
                  >
                    {loadingProgress}
                  </p>
                </div>
              </div>
              <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
            </div>
          )}

          {/* Error overlay */}
          {error && (
            <div
              className="absolute inset-0 flex items-center justify-center"
              style={{
                background: dm
                  ? "rgba(13,15,20,0.96)"
                  : "rgba(240,242,248,0.95)",
                zIndex: 100,
              }}
            >
              <div
                style={{
                  background: dm ? "#161820" : "white",
                  border: "1px solid rgba(232,86,74,0.3)",
                  borderRadius: 12,
                  padding: "24px 28px",
                  maxWidth: 440,
                  boxShadow: "0 8px 32px rgba(0,0,0,0.2)",
                }}
              >
                <p
                  style={{
                    fontFamily: "'Crimson Pro', Georgia, serif",
                    fontSize: 17,
                    fontWeight: 600,
                    color: dm ? "#e2e8f0" : "#1f2937",
                    margin: "0 0 8px",
                  }}
                >
                  Failed to load data
                </p>
                <p
                  style={{
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: 11,
                    color: dm ? "#6b7280" : "#6b7280",
                    marginBottom: 16,
                    wordBreak: "break-all",
                  }}
                >
                  {error}
                </p>
                <p
                  style={{
                    color: dm ? "#6b7280" : "#9ca3af",
                    fontSize: 12,
                    marginBottom: 16,
                  }}
                >
                  Make sure{" "}
                  <code
                    style={{
                      background: dm ? "#1f2228" : "#f3f4f6",
                      padding: "1px 4px",
                      borderRadius: 3,
                    }}
                  >
                    umap_200k.parquet
                  </code>{" "}
                  and{" "}
                  <code
                    style={{
                      background: dm ? "#1f2228" : "#f3f4f6",
                      padding: "1px 4px",
                      borderRadius: 3,
                    }}
                  >
                    voids.json
                  </code>{" "}
                  are in your{" "}
                  <code
                    style={{
                      background: dm ? "#1f2228" : "#f3f4f6",
                      padding: "1px 4px",
                      borderRadius: 3,
                    }}
                  >
                    public/
                  </code>{" "}
                  folder.
                </p>
                <button
                  onClick={reload}
                  style={{
                    background: dm ? "#f59e0b" : "#d97706",
                    color: "white",
                    border: "none",
                    borderRadius: 7,
                    padding: "8px 16px",
                    fontSize: 13,
                    cursor: "pointer",
                    fontFamily: "'JetBrains Mono', monospace",
                  }}
                >
                  Retry
                </button>
              </div>
            </div>
          )}
        </div>
        {/* end visualization pane */}

        {/* Research tab overlay — absolute on top of visualization when active */}
        {(() => {
          const [jobId, tabType] = activeTab.split(":");
          const job = researchJobs.get(jobId);
          if (!job) return null;

          const onStatusChange = (status: "running" | "done" | "error") =>
            setResearchJobs((prev) => {
              const m = new Map(prev);
              const j = m.get(jobId);
              if (j) m.set(jobId, { ...j, status });
              return m;
            });

          return (
            <div className="absolute z-200 h-full flex w-[50vw] right-0 top-0">
              <button
                className="absolute bottom-0 right-0 z-300 cursor-pointer text-white w-6 h-6 m-4 border-slate-700 bg-slate-900 border rounded-full flex items-center justify-center"
                onClick={() => setActiveTab("viz")}
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                  className="size-4"
                >
                  <path d="M6.28 5.22a.75.75 0 0 0-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 1 0 1.06 1.06L10 11.06l3.72 3.72a.75.75 0 1 0 1.06-1.06L11.06 10l3.72-3.72a.75.75 0 0 0-1.06-1.06L10 8.94 6.28 5.22Z" />
                </svg>
              </button>
              {tabType === "deep" ? (
                <DeepResearchTab
                  jobId={jobId}
                  voidId={job.voidId}
                  voidName={job.voidName}
                  papers={job.papers}
                  darkMode={dm}
                  onStatusChange={onStatusChange}
                />
              ) : (
                <ResearchTab
                  jobId={jobId}
                  voidName={job.voidName}
                  darkMode={dm}
                  onStatusChange={onStatusChange}
                />
              )}
            </div>
          );
        })()}
      </div>
      {/* end content area */}
    </div>
  );
};
