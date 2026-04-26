import React, { useMemo } from 'react';
import { type Cluster, type ViewTransform } from '../types';
import { hullToSmoothPath, expandHull, type Vec2 } from '../utils/convexHull';
import { wrapLabel } from '../utils/measureLabels';
import { MIN_LABEL_SCALE } from '../utils/computeLabelRevealScales';

interface ClusterRingsProps {
  clusters:          Map<number, Cluster>;
  selectedClusterId: number | null;
  hoveredClusterId:  number | null;
  transform:         ViewTransform;
  width:             number;
  height:            number;
  onClusterClick:    (id: number) => void;
  onClusterHover:    (id: number | null) => void;
  darkMode:          boolean;
}

const LABEL_MAX_CHARS     = 16;
const LABEL_MAX_CHARS_SEL = 20;

function normToScreen(
  nx: number,
  ny: number,
  transform: ViewTransform,
): [number, number] {
  return [
    nx * transform.scale + transform.offsetX,
    ny * transform.scale + transform.offsetY,
  ];
}

export const ClusterRings: React.FC<ClusterRingsProps> = ({
  clusters,
  selectedClusterId,
  hoveredClusterId,
  transform,
  width,
  height,
  onClusterClick,
  onClusterHover,
  darkMode,
}) => {
  const staticData = useMemo(() => {
    return Array.from(clusters.values()).map(cluster => {
      const lines    = wrapLabel(cluster.label ?? `Cluster ${cluster.id}`, LABEL_MAX_CHARS);
      const linesSel = wrapLabel(cluster.label ?? `Cluster ${cluster.id}`, LABEL_MAX_CHARS_SEL);
      return { cluster, lines, linesSel };
    });
  }, [clusters]);

  const rings = useMemo(() => {
    return staticData.map(({ cluster, lines, linesSel }) => {
      const screenHull: Vec2[] = cluster.hull.map(([nx, ny]) =>
        normToScreen(nx, ny, transform),
      );

      const isSelected  = cluster.id === selectedClusterId;
      const isHovered   = cluster.id === hoveredClusterId;
      const displayHull = isSelected
        ? expandHull(screenHull, 6)
        : isHovered
          ? expandHull(screenHull, 3)
          : screenHull;

      const path           = hullToSmoothPath(displayHull);
      const centroidScreen = normToScreen(cluster.centroid[0], cluster.centroid[1], transform);

      const showLabel =
        isSelected ||
        isHovered  ||
        (transform.scale >= MIN_LABEL_SCALE && transform.scale >= (cluster.revealScale ?? Infinity));

      const activeLines = isSelected ? linesSel : lines;

      return { cluster, path, centroidScreen, isSelected, isHovered, showLabel, activeLines };
    });
  }, [staticData, transform, selectedClusterId, hoveredClusterId]);

  // Dark mode: outline text with a dark halo; light mode: white fill + dark stroke (original)
  const textFill   = darkMode ? 'rgba(230,235,255,0.95)' : 'white';
  const haloColor  = darkMode ? 'rgba(0,0,5,0.92)'       : 'rgba(0,0,0,0.85)';

  return (
    <svg
      className="absolute inset-0 pointer-events-none"
      width={width}
      height={height}
      style={{ zIndex: 10 }}
    >
      <defs>
        {rings.map(({ cluster }) => (
          <filter
            key={`glow-${cluster.id}`}
            id={`glow-${cluster.id}`}
            x="-20%"
            y="-20%"
            width="140%"
            height="140%"
          >
            <feGaussianBlur
              stdDeviation={cluster.id === selectedClusterId ? '4' : '2'}
              result="blur"
            />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        ))}
      </defs>

      {rings.map(({ cluster, path, centroidScreen, isSelected, isHovered, showLabel, activeLines }) => {
        const opacity =
          selectedClusterId !== null && !isSelected
            ? 0.12
            : isSelected
              ? 1
              : isHovered
                ? 0.75
                : darkMode ? 0.45 : 0.35;

        const strokeWidth = isSelected ? 2 : isHovered ? 1.5 : 1;
        const color       = cluster.color;

        const inView =
          centroidScreen[0] > -200 && centroidScreen[0] < width  + 200 &&
          centroidScreen[1] > -200 && centroidScreen[1] < height + 200;

        if (!inView && !isSelected) return null;

        const fontSize    = isSelected ? 14 : 12;
        const lineHeight  = fontSize * 1.3;
        const blockHeight = activeLines.length * lineHeight;
        const startY      = centroidScreen[1] - blockHeight / 2 + lineHeight / 2;

        return (
          <g
            key={cluster.id}
            className="pointer-events-auto cursor-pointer"
            onClick={() => onClusterClick(cluster.id)}
            onMouseEnter={() => onClusterHover(cluster.id)}
            onMouseLeave={() => onClusterHover(null)}
          >
            {/* Hull path */}
            {/* {path && (
              <path
                d={path}
                fill={`${color}${darkMode ? '18' : '0d'}`}
                stroke={color}
                strokeWidth={strokeWidth}
                strokeOpacity={opacity}
                fillOpacity={opacity * 0.5}
                style={{ pointerEvents: 'none' }}
              />
            )} */}

            {showLabel &&
              activeLines.map((line, i) => (
                <text
                  key={i}
                  x={centroidScreen[0]}
                  y={startY + i * lineHeight}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fontFamily="'JetBrains Mono', monospace"
                  fontSize={fontSize}
                  fontWeight={isSelected ? 600 : 500}
                  style={{ pointerEvents: 'none', userSelect: 'none' }}
                >
                  <tspan
                    fill={textFill}
                    stroke={haloColor}
                    strokeWidth={isSelected ? 3.5 : 2.5}
                    strokeLinejoin="round"
                    paintOrder="stroke"
                    fillOpacity={1}
                  >
                    {line}
                  </tspan>
                </text>
              ))}
          </g>
        );
      })}
    </svg>
  );
};