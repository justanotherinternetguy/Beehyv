import React, { useCallback, useEffect, useRef, useState } from 'react';
import type { ResearchJobInfo } from '../types';

interface Props {
  jobs:          Map<string, ResearchJobInfo>;
  activeTab:     string;
  onNavigate:    (tabId: string) => void;
  darkMode:      boolean;
}

const STATUS_COLOR: Record<string, string> = {
  running: '#22d3ee',
  done:    '#4ade80',
  error:   '#f87171',
};

const IMAGENET_GAP = 'Transformer-Augmented Vision Adaptation Gap';

export const JobsDropdown: React.FC<Props> = ({ jobs, activeTab, onNavigate, darkMode: dm }) => {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const running = [...jobs.values()].filter(j => j.status === 'running').length;
  const total   = jobs.size;

  const close = useCallback(() => setOpen(false), []);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) close();
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open, close]);

  if (!total) return null;

  return (
    <div ref={ref} style={{ position: 'relative', flexShrink: 0 }}>
      <button
        onClick={() => setOpen(v => !v)}
        style={{
          display:       'flex',
          alignItems:    'center',
          gap:           6,
          height:        38,
          padding:       '0 14px',
          background:    open ? (dm ? 'rgba(99,102,241,0.15)' : 'rgba(79,110,247,0.1)') : 'transparent',
          border:        'none',
          borderLeft:    `1px solid ${dm ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.09)'}`,
          cursor:        'pointer',
          fontFamily:    "'JetBrains Mono', monospace",
          fontSize:      11,
          color:         dm ? '#94a3b8' : '#64748b',
          userSelect:    'none',
        }}
        title="Active investigation jobs"
      >
        {running > 0 && (
          <span style={{
            width:        7,
            height:       7,
            borderRadius: '50%',
            background:   '#22d3ee',
            display:      'inline-block',
            animation:    'jdPulse 1.5s ease-in-out infinite',
            flexShrink:   0,
          }} />
        )}
        <span>Jobs {total}</span>
        <span style={{ fontSize: 9 }}>{open ? '▲' : '▼'}</span>
      </button>

      {open && (
        <div style={{
          position:    'absolute',
          top:         '100%',
          right:       0,
          minWidth:    280,
          background:  dm ? '#161820' : '#ffffff',
          border:      `1px solid ${dm ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.12)'}`,
          borderRadius: 10,
          boxShadow:   '0 8px 32px rgba(0,0,0,0.4)',
          zIndex:      2000,
          overflow:    'hidden',
        }}>
          <div style={{
            padding:     '8px 12px',
            borderBottom: `1px solid ${dm ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.07)'}`,
            fontFamily:  "'JetBrains Mono', monospace",
            fontSize:    9,
            color:       dm ? '#4b5563' : '#9ca3af',
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
          }}>
            Active Investigations
          </div>

          {[...jobs.entries()].map(([jobId, job]) => {
            const isImagenet    = job.voidName === IMAGENET_GAP;
            const logsTabId     = `${jobId}:logs`;
            const deepTabId     = `${jobId}:deep`;
            const activeIsThis  = activeTab.startsWith(jobId);
            const dotColor      = STATUS_COLOR[job.status] ?? '#6b7280';

            return (
              <div key={jobId} style={{
                borderBottom: `1px solid ${dm ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.05)'}`,
              }}>
                {/* Job header */}
                <div style={{
                  padding:    '8px 12px 5px',
                  display:    'flex',
                  alignItems: 'center',
                  gap:        7,
                  background: activeIsThis ? (dm ? 'rgba(99,102,241,0.07)' : 'rgba(79,110,247,0.05)') : 'transparent',
                }}>
                  <span style={{
                    width: 7, height: 7, borderRadius: '50%',
                    background: dotColor, flexShrink: 0,
                    boxShadow: `0 0 6px ${dotColor}`,
                    animation: job.status === 'running' ? 'jdPulse 1.5s ease-in-out infinite' : undefined,
                  }} />
                  <span style={{
                    fontFamily:   "'Crimson Pro', Georgia, serif",
                    fontSize:     13,
                    color:        dm ? '#c9d1e0' : '#1e293b',
                    flex:         1,
                    overflow:     'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace:   'nowrap',
                  }}>
                    {job.voidName}
                  </span>
                  <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 8, color: dotColor }}>
                    {job.status}
                  </span>
                </div>

                {/* Tab buttons */}
                <div style={{ display: 'flex', gap: 6, padding: '4px 12px 8px' }}>
                  <TabButton
                    label="Full Logs"
                    active={activeTab === logsTabId}
                    onClick={() => { onNavigate(logsTabId); close(); }}
                    dm={dm}
                  />
                  {isImagenet && (
                    <TabButton
                      label="Deep Research"
                      active={activeTab === deepTabId}
                      onClick={() => { onNavigate(deepTabId); close(); }}
                      dm={dm}
                      accent
                    />
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      <style>{`
        @keyframes jdPulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.25; } }
      `}</style>
    </div>
  );
};

const TabButton: React.FC<{
  label: string; active: boolean; onClick: () => void; dm: boolean; accent?: boolean;
}> = ({ label, active, onClick, dm, accent }) => (
  <button
    onClick={onClick}
    style={{
      background:   active
        ? (accent ? 'rgba(99,102,241,0.2)' : (dm ? 'rgba(255,255,255,0.09)' : 'rgba(0,0,0,0.07)'))
        : 'transparent',
      border:       `1px solid ${active ? (accent ? '#6366f1' : (dm ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.15)')) : 'rgba(255,255,255,0.06)'}`,
      borderRadius: 5,
      padding:      '3px 9px',
      fontFamily:   "'JetBrains Mono', monospace",
      fontSize:     9,
      color:        active ? (accent ? '#818cf8' : (dm ? '#e2e8f0' : '#1e293b')) : (dm ? '#4b5563' : '#9ca3af'),
      cursor:       'pointer',
      letterSpacing: '0.04em',
    }}
  >
    {label}
  </button>
);
