import React from 'react';
import type { ProcessedPaper } from '../types';

interface TooltipProps {
  paper:            ProcessedPaper | null;
  x:                number;
  y:                number;
  containerWidth:   number;
  containerHeight:  number;
  darkMode:         boolean;
}

export const Tooltip: React.FC<TooltipProps> = ({
  paper,
  x,
  y,
  containerWidth,
  containerHeight,
  darkMode,
}) => {
  if (!paper) return null;

  const PAD       = 16;
  const TOOLTIP_W = 320;
  const TOOLTIP_H = 90;

  const left = x + PAD + TOOLTIP_W > containerWidth  ? x - TOOLTIP_W - PAD : x + PAD;
  const top  = y + PAD + TOOLTIP_H > containerHeight ? y - TOOLTIP_H - PAD : y + PAD;

  const hasDoi =
    paper.doi &&
    paper.doi !== 'null' &&
    paper.doi !== 'undefined' &&
    paper.doi.trim() !== '';
  const hasPdf = Boolean(paper.pdfUrl?.trim());

  const dm = darkMode;

  return (
    <div
      className="pointer-events-none absolute z-50 select-none"
      style={{ left, top, maxWidth: TOOLTIP_W }}
    >
      <div
        style={{
          background:          dm ? 'rgba(18,20,28,0.95)'    : 'rgba(255,255,255,0.93)',
          backdropFilter:      'blur(20px)',
          WebkitBackdropFilter:'blur(20px)',
          border:              `1px solid ${dm ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)'}`,
          borderRadius:        12,
          padding:             '12px 14px',
          boxShadow:           dm
            ? '0 8px 32px rgba(0,0,0,0.6), 0 1px 4px rgba(0,0,0,0.4)'
            : '0 8px 32px rgba(0,0,0,0.12), 0 1px 4px rgba(0,0,0,0.06)',
          transition: 'background 0.3s, border-color 0.3s',
        }}
      >
        <p
          style={{
            fontFamily: "'Crimson Pro', 'Georgia', serif",
            fontSize:   13,
            lineHeight: '1.4',
            fontWeight: 500,
            color:      dm ? '#e2e8f0' : '#1f2937',
            margin:     '0 0 4px',
          }}
        >
          {paper.title || '(No title)'}
        </p>

        {hasDoi && (
          <p
            style={{
              fontSize:   11,
              color:      dm ? '#818cf8' : '#3b82f6',
              margin:     '0 0 6px',
              overflow:   'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {paper.doi}
          </p>
        )}

        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 6 }}>
          <span
            style={{
              fontSize:   10,
              fontFamily: 'monospace',
              color:      dm ? '#4b5563' : '#9ca3af',
            }}
          >
            id: {paper.id}
          </span>
          <span style={{ color: dm ? '#374151' : '#e5e7eb' }}>·</span>
          <span
            style={{
              fontSize: 10,
              color:    dm ? '#4b5563' : '#9ca3af',
            }}
          >
            cluster {paper.clusterId < 0 ? 'noise' : paper.clusterId}
          </span>
        </div>

        {(hasDoi || hasPdf) && (
          <p
            style={{
              fontSize: 10,
              color:    dm ? '#374151' : '#94a3b8',
              margin:   '4px 0 0',
            }}
          >
            Click to open {hasPdf ? 'PDF' : 'paper'} ↗
          </p>
        )}
      </div>
    </div>
  );
};
