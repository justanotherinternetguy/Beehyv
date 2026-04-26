import { useEffect, useRef, useCallback } from "react";
import type { ProcessedPaper, ViewTransform } from "../types";
import { densityToRGB } from "../utils/colors";

interface RendererOptions {
  papers: ProcessedPaper[];
  width: number;
  height: number;
  transform: ViewTransform;
  hoveredId: string | number | null;
  selectedClusterId: number | null;
  searchResultIds: Set<string | number> | null;
  darkMode: boolean;
  crossPollinationActive?: boolean;
  goldDois?: Set<string | number>;
}

interface GLState {
  regl: any;
  drawPoints: any;
  positions: Float32Array;
  colors: Float32Array;
  pointCount: number;
}

export function useWebGLRenderer(
  canvasRef: React.RefObject<HTMLCanvasElement>,
  options: RendererOptions,
) {
  const glStateRef = useRef<GLState | null>(null);
  const animFrameRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || options.papers.length === 0) return;

    import("regl").then(({ default: createREGL }) => {
      const regl = createREGL({
        canvas,
        attributes: { antialias: true, alpha: true },
      });

      const n = options.papers.length;
      const positions = new Float32Array(n * 2);
      const colors = new Float32Array(n * 4);

      for (let i = 0; i < n; i++) {
        const p = options.papers[i];
        positions[i * 2] = p.nx;
        positions[i * 2 + 1] = p.ny;
        const [r, g, b] = densityToRGB(
          p.density,
          p.clusterId,
          options.darkMode,
        );
        colors[i * 4] = r;
        colors[i * 4 + 1] = g;
        colors[i * 4 + 2] = b;
        colors[i * 4 + 3] = options.darkMode ? 0.75 : 0.85;
      }

      const posBuffer = regl.buffer(positions);
      const colorBuffer = regl.buffer(colors);

      const drawPoints = regl({
        vert: `
          precision highp float;
          attribute vec2 position;
          attribute vec4 color;
          uniform vec2  offset;
          uniform float scale;
          uniform vec2  viewport;
          uniform float pointSize;
          varying vec4 vColor;
          void main() {
            vec2 pos  = (position + offset) * scale;
            vec2 clip = (pos / viewport) * 2.0 - 1.0;
            clip.y = -clip.y;
            gl_Position = vec4(clip, 0.0, 1.0);
            gl_PointSize = pointSize;
            vColor = color;
          }
        `,
        frag: `
          precision mediump float;
          varying vec4 vColor;
          uniform float dimFactor;
          void main() {
            vec2 cxy = 2.0 * gl_PointCoord - 1.0;
            float r = dot(cxy, cxy);
            if (r > 1.0) discard;
            float alpha = vColor.a * (1.0 - smoothstep(0.6, 1.0, r)) * dimFactor;
            gl_FragColor = vec4(vColor.rgb, alpha);
          }
        `,
        attributes: {
          position: posBuffer,
          color: colorBuffer,
        },
        uniforms: {
          offset: regl.prop<any, "offset">("offset"),
          scale: regl.prop<any, "scale">("scale"),
          viewport: regl.prop<any, "viewport">("viewport"),
          pointSize: regl.prop<any, "pointSize">("pointSize"),
          dimFactor: regl.prop<any, "dimFactor">("dimFactor"),
        },
        count: n,
        primitive: "points",
        blend: {
          enable: true,
          func: { src: "src alpha", dst: "one minus src alpha" },
        },
        depth: { enable: false },
      });

      glStateRef.current = {
        regl,
        drawPoints,
        positions,
        colors,
        pointCount: n,
      };
    });

    return () => {
      cancelAnimationFrame(animFrameRef.current);
      if (glStateRef.current) {
        glStateRef.current.regl.destroy();
        glStateRef.current = null;
      }
    };
  }, [options.papers, options.darkMode]);

  const render = useCallback(() => {
    const gl = glStateRef.current;
    const canvas = canvasRef.current;
    if (!gl || !canvas) return;

    const { regl, drawPoints } = gl;
    const {
      transform,
      width,
      height,
      selectedClusterId,
      searchResultIds,
      darkMode,
      crossPollinationActive,
      goldDois,
    } = options;

    const baseSize = Math.max(1.5, Math.min(8, transform.scale * 0.004));

    regl.clear({ color: [0, 0, 0, 0], depth: 1 });

    const uniformBase = {
      offset: [
        transform.offsetX / transform.scale,
        transform.offsetY / transform.scale,
      ],
      scale: transform.scale,
      viewport: [width, height],
      pointSize: baseSize,
      dimFactor: 1.0,
    };

    // Cross-pollination mode: dim all non-gold papers, render gold papers in gold
    if (crossPollinationActive && goldDois && goldDois.size > 0) {
      const n = options.papers.length;
      const newColors = new Float32Array(n * 4);
      for (let i = 0; i < n; i++) {
        const p = options.papers[i];
        if (goldDois.has(p.id)) {
          newColors[i * 4]     = 1.0;   // gold #FFD700
          newColors[i * 4 + 1] = 0.84;
          newColors[i * 4 + 2] = 0.0;
          newColors[i * 4 + 3] = 0.95;
        } else {
          newColors[i * 4]     = 0.22;  // desaturated grey
          newColors[i * 4 + 1] = 0.22;
          newColors[i * 4 + 2] = 0.24;
          newColors[i * 4 + 3] = 0.11;
        }
      }
      const tempBuf = regl.buffer(newColors);
      drawPoints({
        ...uniformBase,
        attributes: {
          position: { buffer: gl.positions, divisor: 0 },
          color: { buffer: tempBuf, divisor: 0 },
        },
      } as any);
      tempBuf.destroy();
      return;
    }

    if (selectedClusterId !== null || searchResultIds !== null) {
      const n = options.papers.length;
      const newColors = new Float32Array(n * 4);

      for (let i = 0; i < n; i++) {
        const p = options.papers[i];
        const isHighlighted =
          (selectedClusterId !== null && p.clusterId === selectedClusterId) ||
          (searchResultIds !== null && searchResultIds.has(p.id));

        const [r, g, b] = densityToRGB(p.density, p.clusterId, darkMode);
        newColors[i * 4] = r;
        newColors[i * 4 + 1] = g;
        newColors[i * 4 + 2] = b;
        // Dimmed points go nearly invisible in dark mode so highlights pop
        newColors[i * 4 + 3] = isHighlighted ? 0.9 : darkMode ? 0.0 : 0.08;
      }

      const tempBuf = regl.buffer(newColors);
      drawPoints({
        ...uniformBase,
        attributes: {
          position: { buffer: gl.positions, divisor: 0 },
          color: { buffer: tempBuf, divisor: 0 },
        },
      } as any);
      tempBuf.destroy();
    } else {
      drawPoints(uniformBase);
    }
  }, [options, canvasRef]);

  useEffect(() => {
    const loop = () => {
      render();
      animFrameRef.current = requestAnimationFrame(loop);
    };
    animFrameRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animFrameRef.current);
  }, [render]);

  return { render };
}
