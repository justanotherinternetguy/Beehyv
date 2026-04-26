import React, { useCallback, useEffect, useRef, useState } from 'react';
import { SwarmViewer } from './SwarmViewer';

const SWARM_VOID = 'Transformer-Augmented Vision Adaptation Gap';

interface ResearchTabProps {
  jobId:          string;
  voidName:       string;
  darkMode:       boolean;
  onStatusChange: (status: 'running' | 'done' | 'error') => void;
}

interface LogLine {
  id:      number;
  message: string;
}

const LINE_COLORS = {
  phase:    '#818cf8',
  success:  '#4ade80',
  warning:  '#fbbf24',
  error:    '#f87171',
  stderr:   '#f59e0b',
  default:  '#a8b4c8',
};

function lineColor(msg: string): string {
  if (/PHASE|={4,}/.test(msg))                              return LINE_COLORS.phase;
  if (/COMPLETE|Done\.|SUCCESS|-> /.test(msg))              return LINE_COLORS.success;
  if (/\[WARN\]|WARNING/.test(msg))                         return LINE_COLORS.warning;
  if (/\[ERR\]|ERROR|Failed|error/.test(msg))               return LINE_COLORS.error;
  if (msg.startsWith('[ERR]') || msg.startsWith('[STDERR]')) return LINE_COLORS.stderr;
  return LINE_COLORS.default;
}

const STATUS_COLOR: Record<string, string> = {
  connecting: '#94a3b8',
  running:    '#22d3ee',
  done:       '#4ade80',
  error:      '#f87171',
};

const STATUS_LABEL: Record<string, string> = {
  connecting: 'Connecting…',
  running:    'Running',
  done:       'Complete',
  error:      'Error',
};

let _lineId = 0;

export const ResearchTab: React.FC<ResearchTabProps> = ({
  jobId,
  voidName,
  darkMode: dm,
  onStatusChange,
}) => {
  const [lines,       setLines]       = useState<LogLine[]>([]);
  const [status,      setStatus]      = useState<'connecting' | 'running' | 'done' | 'error'>('connecting');
  const [displayName, setDisplayName] = useState(voidName);
  const [jobStage,    setJobStage]    = useState<'ingesting' | 'ready' | 'researching' | 'complete'>('ingesting');
  const [launching,   setLaunching]   = useState(false);
  const bottomRef    = useRef<HTMLDivElement>(null);
  const autoScroll   = useRef(true);
  const containerRef = useRef<HTMLDivElement>(null);

  const appendLine = useCallback((msg: string) => {
    setLines(prev => [...prev, { id: _lineId++, message: msg }]);
  }, []);

  const scrollToBottom = useCallback(() => {
    if (autoScroll.current) {
      bottomRef.current?.scrollIntoView({ behavior: 'auto' });
    }
  }, []);

  useEffect(() => {
    const es = new EventSource(`/api/investigate/${encodeURIComponent(jobId)}/stream`);

    es.onmessage = (ev) => {
      let data: { type: string; message?: string; voidName?: string };
      try { data = JSON.parse(ev.data); } catch { return; }

      if (data.type === 'info') {
        if (data.voidName) setDisplayName(data.voidName);
        return;
      }

      if (data.type === 'log' && data.message) {
        appendLine(data.message);
        setTimeout(scrollToBottom, 20);
        // Detect ingest complete → show Research button
        if (data.message.includes('[INGEST_COMPLETE]') || data.message.includes('INGEST COMPLETE')) {
          setJobStage('ready');
        }
        if (data.message.includes('RESEARCH PHASE START')) {
          setJobStage('researching');
        }
        if (data.message.includes('RESEARCH COMPLETE')) {
          setJobStage('complete');
        }
        return;
      }

      if (data.type === 'done') {
        setStatus('done');
        onStatusChange('done');
        setJobStage('complete');
        es.close(); // prevent auto-reconnect loop
        return;
      }
      if (data.type === 'error') {
        setStatus('error');
        onStatusChange('error');
        es.close();
        return;
      }
    };

    es.onerror = () => {
      setStatus('error');
      onStatusChange('error');
      es.close();
    };

    es.onopen = () => setStatus('running');

    return () => es.close();
  }, [jobId, appendLine, scrollToBottom, onStatusChange]);

  const handleStartResearch = useCallback(async () => {
    setLaunching(true);
    try {
      await fetch(`/api/investigate/${jobId}/start-research`, { method: 'POST' });
      setJobStage('researching');
    } catch (e) {
      console.error('start-research error', e);
    } finally {
      setLaunching(false);
    }
  }, [jobId]);

  // Poll stage when 'ready' (in case the SSE stream missed the marker)
  useEffect(() => {
    if (jobStage !== 'ingesting') return;
    let cancelled = false;
    const t = setInterval(async () => {
      if (cancelled) return;
      try {
        const r = await fetch(`/api/investigate/${jobId}/status`);
        if (r.ok) {
          const d = await r.json() as { stage: typeof jobStage };
          if (d.stage !== 'ingesting') setJobStage(d.stage);
        }
      } catch { /* ignore */ }
    }, 2500);
    return () => { cancelled = true; clearInterval(t); };
  }, [jobId, jobStage]);

  // Detect when user scrolls up to pause auto-scroll
  const handleScroll = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
    autoScroll.current = atBottom;
  }, []);

  const dotColor   = STATUS_COLOR[status];
  const isSwarm    = displayName === SWARM_VOID;
  const swarmLines = isSwarm
    ? lines.filter(l => l.message.startsWith('[SWARM] ')).map(l => l.message)
    : [];
  const termLines  = isSwarm
    ? lines.filter(l => !l.message.startsWith('[SWARM] '))
    : lines;

  return (
    <div
      style={{
        flex:           1,
        display:        'flex',
        flexDirection:  'column',
        background:     '#0a0c10',
        overflow:       'hidden',
        height:         '100%',
      }}
    >
      {/* ── Header ──────────────────────────────────────────────── */}
      <div
        style={{
          padding:      '10px 18px',
          borderBottom: '1px solid rgba(255,255,255,0.07)',
          display:      'flex',
          alignItems:   'center',
          gap:          12,
          flexShrink:   0,
          background:   '#0d0f14',
        }}
      >
        <div
          style={{
            width:        9,
            height:       9,
            borderRadius: '50%',
            background:   dotColor,
            flexShrink:   0,
            boxShadow:    `0 0 10px ${dotColor}`,
            animation:    status === 'running' ? 'rtPulse 1.5s ease-in-out infinite' : undefined,
          }}
        />

        <div style={{ flex: 1, minWidth: 0 }}>
          <div
            style={{
              fontFamily: "'Crimson Pro', Georgia, serif",
              fontSize:   15,
              fontWeight: 600,
              color:      '#e2e8f0',
              overflow:   'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {displayName}
          </div>
          <div
            style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize:   10,
              color:      dotColor,
              marginTop:  2,
            }}
          >
            {STATUS_LABEL[status]} &middot; job {jobId.slice(-8)} &middot; {lines.length} lines
          </div>
        </div>

        {/* scroll-to-bottom button */}
        <button
          onClick={() => { autoScroll.current = true; scrollToBottom(); }}
          title="Scroll to bottom"
          style={{
            background: 'rgba(255,255,255,0.07)',
            border:     '1px solid rgba(255,255,255,0.1)',
            borderRadius: 6,
            color:      '#6b7280',
            fontSize:   14,
            cursor:     'pointer',
            padding:    '3px 8px',
            fontFamily: "'JetBrains Mono', monospace",
          }}
        >
          ↓
        </button>
      </div>

      {/* ── Main content area ───────────────────────────────────── */}
      <div style={{ flex: 1, display: 'flex', flexDirection: isSwarm ? 'row' : 'column', overflow: 'hidden', minHeight: 0 }}>

        {/* Terminal log (left panel or full width) */}
        <div
          ref={containerRef}
          onScroll={handleScroll}
          style={{
            flex:       isSwarm ? '0 0 38%' : 1,
            overflowY:  'auto',
            padding:    '10px 16px 20px',
            fontFamily: "'JetBrains Mono', monospace",
            fontSize:   12,
            lineHeight: 1.75,
            color:      '#a8b4c8',
            borderRight: isSwarm ? '1px solid rgba(255,255,255,0.06)' : 'none',
          }}
        >
          {lines.length === 0 && status === 'connecting' && (
            <div style={{ color: '#374151', fontStyle: 'italic' }}>
              Waiting for stream…
            </div>
          )}

          {termLines.map(({ id, message }) => (
            <div
              key={id}
              style={{
                color:      lineColor(message),
                whiteSpace: 'pre-wrap',
                wordBreak:  'break-all',
              }}
            >
              {message}
            </div>
          ))}

        {status === 'running' && (
          <div
            style={{
              color:         '#22d3ee',
              marginTop:     4,
              display:       'inline-block',
              animation:     'rtBlink 1s step-end infinite',
            }}
          >
            ▊
          </div>
        )}

        {status === 'done' && (
          <div
            style={{
              marginTop:  12,
              padding:    '8px 12px',
              border:     '1px solid rgba(74,222,128,0.25)',
              borderRadius: 6,
              color:      '#4ade80',
              background: 'rgba(74,222,128,0.05)',
              fontSize:   11,
            }}
          >
            Investigation complete. Check <code>outputs/investigations/</code> for generated code.
          </div>
        )}

        {status === 'error' && (
          <div
            style={{
              marginTop:    12,
              padding:      '10px 12px',
              border:       '1px solid rgba(248,113,113,0.35)',
              borderRadius: 6,
              color:        '#f87171',
              background:   'rgba(248,113,113,0.07)',
              fontSize:     11,
            }}
          >
            <div style={{ fontWeight: 700, marginBottom: 6 }}>
              Process exited with an error — last output:
            </div>
            <div style={{
              fontFamily:  "'JetBrains Mono', monospace",
              fontSize:    10,
              color:       '#fca5a5',
              lineHeight:  1.7,
              whiteSpace:  'pre-wrap',
              wordBreak:   'break-all',
            }}>
              {lines
                .slice(-20)
                .map(l => l.message)
                .filter(Boolean)
                .join('\n') || '(no output captured)'}
            </div>
          </div>
        )}

          <div ref={bottomRef} />
        </div>

        {/* SwarmViewer — right panel, only for the MNIST swarm void */}
        {isSwarm && (
          <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden', minHeight: 0 }}>
            <div style={{
              padding:      '6px 12px',
              borderBottom: '1px solid rgba(255,255,255,0.06)',
              background:   '#0d0f14',
              fontFamily:   "'JetBrains Mono', monospace",
              fontSize:     9,
              color:        '#a78bfa',
              letterSpacing: '0.06em',
              flexShrink:   0,
            }}>
              AGENT SWARM · {swarmLines.length} log entries
            </div>
            <SwarmViewer lines={swarmLines} />
          </div>
        )}
      </div>{/* end main content flex */}

      {/* ── Research button ─────────────────────────────────────────── */}
      {jobStage !== 'ingesting' && (
        <div style={{
          flexShrink:   0,
          borderTop:    '1px solid rgba(255,255,255,0.07)',
          padding:      '12px 16px',
          display:      'flex',
          justifyContent: 'center',
          background:   '#0d0f14',
        }}>
          {jobStage === 'ready' && (
            <button
              onClick={handleStartResearch}
              disabled={launching}
              style={{
                background:    'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)',
                color:         'white',
                border:        'none',
                borderRadius:  10,
                padding:       '10px 36px',
                fontSize:      13,
                fontFamily:    "'JetBrains Mono', monospace",
                fontWeight:    700,
                cursor:        launching ? 'wait' : 'pointer',
                boxShadow:     '0 4px 24px rgba(124,58,237,0.5)',
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                display:       'flex',
                alignItems:    'center',
                gap:           10,
              }}
            >
              <span style={{ fontSize: 16 }}>🔬</span>
              {launching ? 'Launching…' : 'Research'}
            </button>
          )}

          {jobStage === 'researching' && (
            <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: '#a78bfa', display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ width: 7, height: 7, borderRadius: '50%', background: '#a78bfa', display: 'inline-block', animation: 'rtPulse 1.5s ease-in-out infinite' }} />
              Agent swarm running…
            </div>
          )}

          {jobStage === 'complete' && (
            <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: '#4ade80', display: 'flex', alignItems: 'center', gap: 8 }}>
              <span>✓</span> Research complete
            </div>
          )}
        </div>
      )}

      <style>{`
        @keyframes rtPulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.2; } }
        @keyframes rtBlink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
      `}</style>
    </div>
  );
};
