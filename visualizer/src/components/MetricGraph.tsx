import React, { useMemo } from 'react';
import type { MetricPoint } from '../types';

interface MetricGraphProps {
  points:    MetricPoint[];
  width?:    number;
  height?:   number;
  metricKey: string;
}

export const MetricGraph: React.FC<MetricGraphProps> = ({
  points,
  width  = 220,
  height = 110,
  metricKey,
}) => {
  const pad = { t: 12, r: 12, b: 24, l: 36 };
  const iw  = width  - pad.l - pad.r;
  const ih  = height - pad.t - pad.b;

  const { minV, maxV, pts } = useMemo(() => {
    if (!points.length) return { minV: 0, maxV: 1, pts: [] };
    const vals = points.map(p => p.value);
    const lo   = Math.max(0, Math.min(...vals) - 0.05);
    const hi   = Math.min(1, Math.max(...vals) + 0.05);
    const spread = hi - lo || 0.1;
    const xs = points.map((_, i) => pad.l + (i / Math.max(points.length - 1, 1)) * iw);
    const ys = points.map(p => pad.t + ih - ((p.value - lo) / spread) * ih);
    return { minV: lo, maxV: hi, pts: points.map((p, i) => ({ ...p, sx: xs[i], sy: ys[i] })) };
  }, [points, iw, ih, pad.l, pad.t]);

  const polyline = pts.map(p => `${p.sx},${p.sy}`).join(' ');
  const areaPath = pts.length
    ? `M${pts[0].sx},${pad.t + ih} ` +
      pts.map(p => `L${p.sx},${p.sy}`).join(' ') +
      ` L${pts[pts.length - 1].sx},${pad.t + ih} Z`
    : '';

  const yTicks = [0, 0.25, 0.5, 0.75, 1].filter(v => v >= minV - 0.05 && v <= maxV + 0.05);

  const latest = pts[pts.length - 1];

  return (
    <div
      style={{
        background:   'rgba(13,15,20,0.92)',
        border:       '1px solid rgba(99,102,241,0.25)',
        borderRadius: 10,
        padding:      '8px 10px 4px',
        backdropFilter: 'blur(12px)',
      }}
    >
      <div
        style={{
          fontFamily:    "'JetBrains Mono', monospace",
          fontSize:      9,
          color:         '#6b7280',
          letterSpacing: '0.08em',
          marginBottom:  4,
          textTransform: 'uppercase',
          display:       'flex',
          justifyContent: 'space-between',
          alignItems:    'center',
        }}
      >
        <span>{metricKey}</span>
        {latest && (
          <span style={{ color: '#4ade80', fontWeight: 700 }}>
            {latest.value.toFixed(4)}
          </span>
        )}
      </div>

      <svg width={width} height={height} style={{ display: 'block', overflow: 'visible' }}>
        {/* Y-axis ticks */}
        {yTicks.map(v => {
          const y = pad.t + ih - ((v - minV) / ((maxV - minV) || 0.1)) * ih;
          return (
            <g key={v}>
              <line x1={pad.l} y1={y} x2={pad.l + iw} y2={y}
                stroke="rgba(255,255,255,0.04)" strokeWidth={1} />
              <text x={pad.l - 4} y={y + 3} textAnchor="end"
                fill="#4b5563" fontSize={7} fontFamily="'JetBrains Mono', monospace">
                {v.toFixed(2)}
              </text>
            </g>
          );
        })}

        {/* X-axis */}
        <line x1={pad.l} y1={pad.t + ih} x2={pad.l + iw} y2={pad.t + ih}
          stroke="rgba(255,255,255,0.1)" strokeWidth={1} />

        {/* Area fill */}
        {pts.length > 1 && (
          <path d={areaPath} fill="url(#metricGrad)" opacity={0.3} />
        )}

        {/* Line */}
        {pts.length > 1 && (
          <polyline points={polyline}
            fill="none" stroke="#6366f1" strokeWidth={1.5}
            strokeLinejoin="round" strokeLinecap="round" />
        )}

        {/* Points */}
        {pts.map((p, i) => (
          <circle key={i} cx={p.sx} cy={p.sy} r={i === pts.length - 1 ? 4 : 2.5}
            fill={i === pts.length - 1 ? '#4ade80' : '#6366f1'}
            stroke={i === pts.length - 1 ? 'rgba(74,222,128,0.3)' : 'none'}
            strokeWidth={i === pts.length - 1 ? 6 : 0} />
        ))}

        {/* X labels */}
        {pts.map((p, i) => (
          <text key={i} x={p.sx} y={pad.t + ih + 10} textAnchor="middle"
            fill="#4b5563" fontSize={7} fontFamily="'JetBrains Mono', monospace">
            {p.label}
          </text>
        ))}

        <defs>
          <linearGradient id="metricGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor="#6366f1" stopOpacity={0.6} />
            <stop offset="100%" stopColor="#6366f1" stopOpacity={0}   />
          </linearGradient>
        </defs>
      </svg>
    </div>
  );
};
