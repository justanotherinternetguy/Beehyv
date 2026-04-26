/**
 * DeepResearchTab — live dashboard for the agentic research swarm.
 *
 * Layout:
 *   ┌─[Paper boxes + Generate Code buttons]───────────────────────────────┐
 *   │ Left: Orchestrator goal + Judge feedback                             │
 *   │ Middle (wide): Paper agent messages + cross-pollination animation    │
 *   │ Right: Ideas / MCP memory                                            │
 *   │                                    [Bottom-right: Metric graph]      │
 *   └──────────────────────────────────────────────────────────────────────┘
 */

import React, {
  useCallback, useEffect, useMemo, useRef, useState,
} from 'react';
import type { MetricPoint, PaperRef, ResearchEvent } from '../types';
import { MetricGraph } from './MetricGraph';

const IMAGENET_GAP = 'Transformer-Augmented Vision Adaptation Gap';

const mono   = "'JetBrains Mono', monospace";
const serif  = "'Crimson Pro', Georgia, serif";
const DIM    = 'rgba(255,255,255,0.06)';
const BORDER = 'rgba(255,255,255,0.08)';

// ── helpers ───────────────────────────────────────────────────────────────────

function shortId(id: string) {
  return id.replace(/^expert:/, '').replace(/_/g, ' ');
}

function truncate(s: string, n: number) {
  return s.length > n ? s.slice(0, n - 1) + '…' : s;
}

/** Safely convert an unknown event payload value to a display string. */
function toStr(val: unknown): string {
  if (val === null || val === undefined) return '';
  if (typeof val === 'string') return val;
  if (typeof val === 'number' || typeof val === 'boolean') return String(val);
  try { return JSON.stringify(val, null, 2); } catch { return String(val); }
}

// ── sub-components ─────────────────────────────────────────────────────────────

interface PaperBoxProps {
  paper:            PaperRef;
  index:            number;
  codegenJobId?:    string;
  codegenStatus?:   'idle' | 'running' | 'done' | 'error';
  onGenerate:       (paper: PaperRef) => void;
}

const PaperBox: React.FC<PaperBoxProps> = ({ paper, index, codegenStatus = 'idle', onGenerate }) => {
  const rank  = index + 1;
  const color = codegenStatus === 'done'    ? '#4ade80'
              : codegenStatus === 'running'  ? '#22d3ee'
              : codegenStatus === 'error'    ? '#f87171'
              : '#6366f1';

  return (
    <div style={{
      flex:          '0 0 calc(12.5% - 10px)',
      minWidth:      110,
      background:    'rgba(22,24,32,0.85)',
      border:        `1px solid ${BORDER}`,
      borderRadius:  8,
      padding:       '8px 10px',
      display:       'flex',
      flexDirection: 'column',
      gap:           6,
      cursor:        'default',
    }}>
      {/* Rank diamond */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
        <svg width={12} height={12} style={{ flexShrink: 0 }}>
          <rect x={2} y={2} width={8} height={8} transform="rotate(45 6 6)"
            fill={color} opacity={0.85} />
        </svg>
        <span style={{ fontFamily: mono, fontSize: 8, color: '#6b7280' }}>#{rank}</span>
        {paper.year && (
          <span style={{ fontFamily: mono, fontSize: 8, color: '#4b5563', marginLeft: 'auto' }}>
            {paper.year}
          </span>
        )}
      </div>

      {/* Title */}
      <p style={{
        fontFamily: serif,
        fontSize:   11,
        color:      '#c9d1e0',
        lineHeight: 1.4,
        margin:     0,
        flex:       1,
      }}>
        {truncate(paper.title.replace(/\n/g, ' ').trim(), 80)}
      </p>

      {/* DOI chip */}
      {paper.doi && (
        <a
          href={`https://arxiv.org/abs/${paper.doi}`}
          target="_blank"
          rel="noopener noreferrer"
          style={{ fontFamily: mono, fontSize: 8, color: '#6366f1', textDecoration: 'none' }}
        >
          {paper.doi}
        </a>
      )}

      {/* Generate code button */}
      <button
        onClick={() => onGenerate(paper)}
        disabled={codegenStatus === 'running'}
        style={{
          background:   codegenStatus === 'done' ? 'rgba(74,222,128,0.12)'
                      : codegenStatus === 'error' ? 'rgba(248,113,113,0.12)'
                      : 'rgba(99,102,241,0.15)',
          border:       `1px solid ${color}40`,
          borderRadius: 5,
          color,
          fontFamily:   mono,
          fontSize:     9,
          fontWeight:   700,
          padding:      '4px 0',
          cursor:       codegenStatus === 'running' ? 'wait' : 'pointer',
          letterSpacing: '0.05em',
          textTransform: 'uppercase',
          width:        '100%',
          animation:    codegenStatus === 'running' ? 'drPulse 1.5s ease-in-out infinite' : undefined,
        }}
      >
        {codegenStatus === 'running' ? 'Generating…'
        : codegenStatus === 'done'    ? '✓ Generated'
        : codegenStatus === 'error'   ? '✗ Retry'
        : 'Generate Code'}
      </button>
    </div>
  );
};

// ── Left panel ────────────────────────────────────────────────────────────────

const OrchestratorPanel: React.FC<{
  phase:       string;
  plan:        string;
  diagnosis:   string;
  judgeDecision: string;
  judgeReason:   string;
  iteration:   number;
}> = ({ phase, plan, diagnosis, judgeDecision, judgeReason, iteration }) => (
  <div style={{
    width:         220,
    flexShrink:    0,
    display:       'flex',
    flexDirection: 'column',
    gap:           10,
    padding:       '12px 14px',
    borderRight:   `1px solid ${DIM}`,
    overflowY:     'auto',
  }}>
    <div style={{ fontFamily: mono, fontSize: 9, color: '#6366f1', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
      Orchestrator
    </div>

    {/* Current phase */}
    <Section title="Phase">
      <div style={{ fontFamily: mono, fontSize: 11, color: '#22d3ee' }}>{phase || 'Waiting…'}</div>
      {iteration > 0 && (
        <div style={{ fontFamily: mono, fontSize: 9, color: '#4b5563', marginTop: 3 }}>
          Iteration {iteration}
        </div>
      )}
    </Section>

    {/* Diagnosis */}
    {diagnosis && (
      <Section title="Diagnosis">
        <p style={{ fontFamily: mono, fontSize: 10, color: '#94a3b8', lineHeight: 1.6, margin: 0, whiteSpace: 'pre-wrap' }}>
          {truncate(diagnosis, 320)}
        </p>
      </Section>
    )}

    {/* Current plan */}
    {plan && (
      <Section title="Current Plan">
        <p style={{ fontFamily: serif, fontSize: 12, color: '#c9d1e0', lineHeight: 1.6, margin: 0 }}>
          {truncate(plan, 400)}
        </p>
      </Section>
    )}

    {/* Judge feedback */}
    {judgeDecision && (
      <Section title="Judge Verdict">
        <div style={{
          display:    'flex',
          alignItems: 'center',
          gap:        6,
          marginBottom: 5,
        }}>
          <div style={{
            background: judgeDecision === 'keep' ? 'rgba(74,222,128,0.15)' : 'rgba(248,113,113,0.15)',
            border:     `1px solid ${judgeDecision === 'keep' ? '#4ade8040' : '#f8717140'}`,
            borderRadius: 4,
            padding:    '2px 7px',
            fontFamily: mono,
            fontSize:   10,
            fontWeight: 700,
            color:      judgeDecision === 'keep' ? '#4ade80' : '#f87171',
            textTransform: 'uppercase',
          }}>
            {judgeDecision}
          </div>
        </div>
        {judgeReason && (
          <p style={{ fontFamily: mono, fontSize: 10, color: '#6b7280', lineHeight: 1.5, margin: 0 }}>
            {truncate(judgeReason, 280)}
          </p>
        )}
      </Section>
    )}
  </div>
);

const Section: React.FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => (
  <div>
    <div style={{ fontFamily: mono, fontSize: 8, color: '#4b5563', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 5 }}>
      {title}
    </div>
    {children}
  </div>
);

// ── Agent message parser ──────────────────────────────────────────────────────

interface AgentMessage {
  agentId:   string;
  stage:     string;
  lines:     string[];
  thinking:  string[];
  isDone:    boolean;
}

function parseAgentMessages(logs: string[]): AgentMessage[] {
  const messages: AgentMessage[] = [];
  let current: AgentMessage | null = null;

  for (const raw of logs) {
    // Strip [ERR] / [OUT] prefix the backend adds
    const line = raw.replace(/^\[(?:ERR|OUT)\]\s*/, '').replace(/\x1b\[[0-9;]*m/g, '');

    // Agent start: "  ◆ label (stage)"
    const startMatch = line.match(/◆\s+(.+?)\s+\((.+?)\)/);
    if (startMatch) {
      if (current) messages.push(current);
      current = { agentId: startMatch[1].trim(), stage: startMatch[2].trim(), lines: [], thinking: [], isDone: false };
      continue;
    }

    if (!current) continue;

    // Agent done: "[done in X.Xs]"
    if (/\[done in [\d.]+s\]/.test(line)) {
      current.isDone = true;
      messages.push(current);
      current = null;
      continue;
    }

    // Thinking token (wrapped in <think>...</think> by our llm.py)
    const thinkMatch = line.match(/<think>([\s\S]*?)<\/think>/);
    if (thinkMatch) {
      current.thinking.push(thinkMatch[1]);
    } else if (line.trim() && !line.includes('[LLM]') && !line.includes('backend=')) {
      current.lines.push(line.trim());
    }
  }

  if (current) messages.push(current);
  return messages;
}

// ── Middle panel — agent messages + cross-pollination ─────────────────────────

interface AgentBubblePos { id: string; x: number; y: number; }

const AgentPanel: React.FC<{
  activeAgents:  string[];
  seedIdeas:     SeedIdea[];
  crossIdeas:    CrossIdea[];
  currentEvents: string[];
  allLogs:       string[];
}> = ({ activeAgents, seedIdeas, crossIdeas, currentEvents, allLogs }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dims, setDims] = useState({ w: 400, h: 300 });
  const logBoxRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver(([e]) => setDims({ w: e.contentRect.width, h: e.contentRect.height }));
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  // Auto-scroll log box
  useEffect(() => {
    logBoxRef.current?.scrollTo({ top: logBoxRef.current.scrollHeight, behavior: 'smooth' });
  }, [currentEvents.length]);

  const agentMessages = useMemo(() => parseAgentMessages(allLogs), [allLogs]);

  const bubbles: AgentBubblePos[] = useMemo(() => {
    const n = activeAgents.length;
    if (!n) return [];
    const cx = dims.w / 2, cy = 72, r = Math.min(cx - 55, 65);
    return activeAgents.map((id, i) => {
      const angle = (i / n) * 2 * Math.PI - Math.PI / 2;
      return { id, x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) };
    });
  }, [activeAgents, dims.w]);

  const crossPairs = useMemo(() =>
    crossIdeas.map(ci => ({ fromId: ci.agent_id, toId: ci.seed_agent_id })), [crossIdeas]);

  const bubbleMap = useMemo(() => new Map(bubbles.map(b => [b.id, b])), [bubbles]);

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div style={{ fontFamily: mono, fontSize: 9, color: '#6b7280', letterSpacing: '0.1em', textTransform: 'uppercase', padding: '12px 14px 6px', flexShrink: 0 }}>
        Agent Activity
      </div>

      {/* ── Agent bubble arc + cross-pollination arrows ── */}
      <div ref={containerRef} style={{ height: 155, flexShrink: 0, position: 'relative' }}>
        {crossPairs.length > 0 && bubbles.length > 0 && (
          <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 2 }}>
            <defs>
              <marker id="drArrow" markerWidth={8} markerHeight={6} refX={6} refY={3} orient="auto">
                <polygon points="0 0, 8 3, 0 6" fill="#a78bfa" opacity={0.7} />
              </marker>
            </defs>
            {crossPairs.map((pair, i) => {
              const from = bubbleMap.get(pair.fromId);
              const to   = bubbleMap.get(pair.toId);
              if (!from || !to) return null;
              const mx = (from.x + to.x) / 2;
              const my = (from.y + to.y) / 2 - 28;
              return (
                <path key={i}
                  d={`M${from.x},${from.y} Q${mx},${my} ${to.x},${to.y}`}
                  fill="none" stroke="#a78bfa" strokeWidth={1.5}
                  strokeDasharray="6 3" markerEnd="url(#drArrow)"
                  opacity={0.65}
                  style={{ animation: 'drDash 1.2s linear infinite' }}
                  strokeDashoffset={12 * (i % 3)}
                />
              );
            })}
          </svg>
        )}
        {bubbles.map(b => {
          const isActive = agentMessages.some(m => m.agentId === b.id && !m.isDone);
          return (
            <div key={b.id} style={{
              position: 'absolute', left: b.x - 44, top: b.y - 18, width: 88, zIndex: 3,
              background: isActive ? 'rgba(99,102,241,0.22)' : 'rgba(99,102,241,0.1)',
              border: `1px solid ${isActive ? 'rgba(99,102,241,0.7)' : 'rgba(99,102,241,0.25)'}`,
              borderRadius: 8, padding: '4px 8px', textAlign: 'center',
              boxShadow: isActive ? '0 0 12px rgba(99,102,241,0.4)' : 'none',
              transition: 'all 0.3s',
            }}>
              {isActive && <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#22d3ee', margin: '0 auto 3px', animation: 'drPulse 1s ease-in-out infinite' }} />}
              <div style={{ fontFamily: mono, fontSize: 9, color: isActive ? '#a5b4fc' : '#818cf8', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {truncate(b.id, 14)}
              </div>
            </div>
          );
        })}
        {activeAgents.length === 0 && (
          <div style={{ fontFamily: mono, fontSize: 10, color: '#374151', padding: '32px 0 0', textAlign: 'center' }}>
            Waiting for agents…
          </div>
        )}
      </div>

      <div style={{ height: 1, background: 'rgba(255,255,255,0.05)', flexShrink: 0 }} />

      {/* ── Scrollable message area ── */}
      <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 8, padding: '10px 12px 8px' }}>

        {/* Ideas from events.jsonl */}
        {crossIdeas.map((ci, i) => (
          <IdeaCard key={`cross-${i}`} tag="Cross-pollination" tagColor="#a78bfa"
            from={`${shortId(ci.agent_id)} × ${shortId(ci.seed_agent_id)}`}
            text={ci.text} connection={ci.connection} />
        ))}
        {seedIdeas.map((si, i) => (
          <IdeaCard key={`seed-${i}`} tag="Seed" tagColor="#22d3ee"
            from={shortId(si.agent_id)} text={si.text} />
        ))}

        {/* Live agent messages parsed from log stream */}
        {agentMessages.map((msg, i) => (
          <div key={i} style={{
            background: msg.isDone ? 'rgba(22,24,32,0.7)' : 'rgba(99,102,241,0.07)',
            border: `1px solid ${msg.isDone ? 'rgba(255,255,255,0.06)' : 'rgba(99,102,241,0.3)'}`,
            borderLeft: `2px solid ${msg.isDone ? '#6366f1' : '#22d3ee'}`,
            borderRadius: 6, padding: '7px 10px', flexShrink: 0,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
              {!msg.isDone && <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#22d3ee', animation: 'drPulse 1s ease-in-out infinite', flexShrink: 0 }} />}
              <span style={{ fontFamily: mono, fontSize: 9, color: msg.isDone ? '#6366f1' : '#22d3ee', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                {msg.stage}
              </span>
              <span style={{ fontFamily: mono, fontSize: 9, color: '#4b5563' }}>{msg.agentId}</span>
              {msg.isDone && <span style={{ fontFamily: mono, fontSize: 8, color: '#374151', marginLeft: 'auto' }}>✓ done</span>}
            </div>

            {msg.thinking.length > 0 && (
              <details style={{ marginBottom: 4 }}>
                <summary style={{ fontFamily: mono, fontSize: 8, color: '#4b5563', cursor: 'pointer', userSelect: 'none' }}>
                  thinking ({msg.thinking.join('').length} chars)
                </summary>
                <div style={{ fontFamily: mono, fontSize: 9, color: '#374151', lineHeight: 1.6, marginTop: 4, whiteSpace: 'pre-wrap', maxHeight: 120, overflowY: 'auto' }}>
                  {msg.thinking.join('')}
                </div>
              </details>
            )}

            {msg.lines.length > 0 && (
              <p style={{ fontFamily: serif, fontSize: 12, color: '#9ca3af', lineHeight: 1.55, margin: 0, whiteSpace: 'pre-wrap' }}>
                {truncate(msg.lines.join('\n'), 400)}
              </p>
            )}
          </div>
        ))}
      </div>

      {/* ── Raw log box pinned to bottom ── */}
      <div style={{ flexShrink: 0, borderTop: '1px solid rgba(255,255,255,0.05)', maxHeight: 90 }}>
        <div ref={logBoxRef} style={{
          overflowY:  'auto',
          maxHeight:  90,
          padding:    '5px 10px',
          fontFamily: mono,
          fontSize:   9,
          color:      '#374151',
          lineHeight: 1.7,
        }}>
          {currentEvents.slice(-20).map((l, i) => (
            <div key={i} style={{ color: l.includes('[ERR]') ? '#78350f' : l.includes('◆') ? '#6366f1' : '#374151' }}>
              {l.replace(/\x1b\[[0-9;]*m/g, '')}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const IdeaCard: React.FC<{
  tag: string; tagColor: string;
  from: string; text: string; connection?: string;
}> = ({ tag, tagColor, from, text, connection }) => (
  <div style={{
    background:   'rgba(22,24,32,0.7)',
    border:       `1px solid rgba(255,255,255,0.07)`,
    borderLeft:   `2px solid ${tagColor}`,
    borderRadius: 6,
    padding:      '7px 10px',
    flexShrink:   0,
  }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
      <span style={{
        fontFamily: mono, fontSize: 8, color: tagColor,
        background: `${tagColor}18`, borderRadius: 3, padding: '1px 5px',
        textTransform: 'uppercase', letterSpacing: '0.06em',
      }}>
        {tag}
      </span>
      <span style={{ fontFamily: mono, fontSize: 9, color: '#4b5563' }}>{from}</span>
    </div>
    <p style={{ fontFamily: serif, fontSize: 12, color: '#94a3b8', lineHeight: 1.55, margin: 0 }}>
      {truncate(text, 220)}
    </p>
    {connection && (
      <p style={{ fontFamily: mono, fontSize: 9, color: '#6b7280', lineHeight: 1.5, margin: '4px 0 0', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: 4 }}>
        {truncate(connection, 160)}
      </p>
    )}
  </div>
);

// ── Right panel — ideas / MCP ─────────────────────────────────────────────────

const IdeasPanel: React.FC<{
  seedIdeas:  SeedIdea[];
  crossIdeas: CrossIdea[];
  plan:       string;
}> = ({ seedIdeas, crossIdeas, plan }) => {
  const allIdeas = [...crossIdeas.slice(0, 6), ...seedIdeas.slice(0, 6)];

  return (
    <div style={{
      width:         240,
      flexShrink:    0,
      display:       'flex',
      flexDirection: 'column',
      borderLeft:    `1px solid ${DIM}`,
      overflow:      'hidden',
    }}>
      <div style={{ fontFamily: mono, fontSize: 9, color: '#6b7280', letterSpacing: '0.1em', textTransform: 'uppercase', padding: '12px 14px 6px' }}>
        Ideas &amp; Memory
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: '0 12px 12px', display: 'flex', flexDirection: 'column', gap: 8 }}>
        {allIdeas.length === 0 && (
          <div style={{ fontFamily: mono, fontSize: 10, color: '#374151', padding: 4 }}>
            No ideas yet…
          </div>
        )}

        {allIdeas.map((idea, i) => (
          <div key={i} style={{
            background:   'rgba(22,24,32,0.6)',
            border:       '1px solid rgba(255,255,255,0.06)',
            borderRadius: 6,
            padding:      '7px 9px',
          }}>
            <div style={{ fontFamily: mono, fontSize: 8, color: '#4b5563', marginBottom: 4, display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'paper_title' in idea ? '#a78bfa' : '#22d3ee' }}>
                {'seed_paper_title' in idea ? 'cross-pollinate' : 'seed'}
              </span>
              <span>{truncate(shortId(idea.agent_id), 16)}</span>
            </div>
            <p style={{ fontFamily: serif, fontSize: 11, color: '#9ca3af', lineHeight: 1.5, margin: 0 }}>
              {truncate(idea.text, 160)}
            </p>
            {('expected_effect' in idea) && idea.expected_effect && (
              <p style={{ fontFamily: mono, fontSize: 9, color: '#6366f1', margin: '4px 0 0', lineHeight: 1.4 }}>
                ↝ {truncate((idea as SeedIdea).expected_effect, 120)}
              </p>
            )}
          </div>
        ))}

        {/* Current plan */}
        {plan && (
          <div style={{ background: 'rgba(99,102,241,0.07)', border: '1px solid rgba(99,102,241,0.15)', borderRadius: 6, padding: '7px 9px' }}>
            <div style={{ fontFamily: mono, fontSize: 8, color: '#6366f1', letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: 5 }}>
              Current Plan
            </div>
            <p style={{ fontFamily: mono, fontSize: 10, color: '#94a3b8', lineHeight: 1.6, margin: 0 }}>
              {truncate(plan, 300)}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

// ── Event-derived types ───────────────────────────────────────────────────────

interface SeedIdea {
  idea_id:         string;
  agent_id:        string;
  paper_id:        string;
  paper_title:     string;
  text:            string;
  rationale:       string;
  expected_effect: string;
  changes:         string;
}

interface CrossIdea {
  idea_id:          string;
  agent_id:         string;
  paper_id:         string;
  seed_idea_id:     string;
  seed_agent_id:    string;
  seed_paper_title: string;
  text:             string;
  connection:       string;
  changes:          string;
}

// ── Main DeepResearchTab ──────────────────────────────────────────────────────

interface Props {
  jobId:          string;
  voidId:         number;
  voidName:       string;
  papers:         PaperRef[];
  darkMode:       boolean;
  onStatusChange: (s: 'running' | 'done' | 'error') => void;
}

export const DeepResearchTab: React.FC<Props> = ({
  jobId, voidId, voidName, papers, onStatusChange,
}) => {
  const [events,       setEvents]       = useState<ResearchEvent[]>([]);
  const [metricPoints, setMetricPoints] = useState<MetricPoint[]>([]);
  const [codegenMap,   setCodegenMap]   = useState<Map<string, { jobId: string; status: 'idle' | 'running' | 'done' | 'error' }>>(new Map());
  const [recentLogs,   setRecentLogs]   = useState<string[]>([]);
  const [jobStage,     setJobStage]     = useState<'ingesting' | 'ready' | 'researching' | 'complete'>('ingesting');
  const [launching,    setLaunching]    = useState(false);

  const metricKey = voidName === IMAGENET_GAP ? 'test_accuracy' : 'predicted_accuracy';

  // Poll job stage every 2 s
  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      if (cancelled) return;
      try {
        const r = await fetch(`/api/investigate/${jobId}/status`);
        if (r.ok) {
          const d = await r.json() as { stage: typeof jobStage };
          setJobStage(d.stage);
        }
      } catch { /* ignore */ }
      if (!cancelled) setTimeout(poll, 2000);
    };
    poll();
    return () => { cancelled = true; };
  }, [jobId]);

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

  // Poll research events every 2 s
  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      if (cancelled) return;
      try {
        const [evtResp, metResp] = await Promise.all([
          fetch(`/api/investigate/${jobId}/research-events`),
          fetch(`/api/investigate/${jobId}/latest-metrics`),
        ]);
        if (evtResp.ok) {
          const data: ResearchEvent[] = await evtResp.json();
          setEvents(data);
          // Extract metric points from experiment_done events
          const pts = data
            .filter(e => e.event === 'experiment_done')
            .map((e, i) => {
              const m = e.payload.metrics as Record<string, unknown> | undefined;
              const v = m ? (Number(m[metricKey] ?? m.test_accuracy ?? m.predicted_accuracy ?? 0)) : 0;
              return { iteration: (e.payload.iteration as number) ?? i, value: v, label: `i${(e.payload.iteration as number) ?? i}` };
            });
          if (pts.length) setMetricPoints(pts);
        }
        if (metResp.ok) {
          const met = await metResp.json() as Record<string, unknown> | null;
          if (met) {
            const v = Number(met[metricKey] ?? met.test_accuracy ?? met.predicted_accuracy ?? 0);
            if (v > 0) {
              setMetricPoints(prev => {
                const last = prev[prev.length - 1];
                if (last?.value === v) return prev;
                const n = prev.length;
                return [...prev, { iteration: n, value: v, label: `i${n}` }];
              });
            }
          }
        }
      } catch { /* network error — ignore */ }
      if (!cancelled) setTimeout(poll, 2000);
    };
    poll();
    return () => { cancelled = true; };
  }, [jobId, metricKey]);

  // Track recent log lines from the investigation SSE stream
  useEffect(() => {
    const es = new EventSource(`/api/investigate/${encodeURIComponent(jobId)}/stream`);
    const lines: string[] = [];
    es.onmessage = ev => {
      try {
        const d = JSON.parse(ev.data) as { type: string; message?: string };
        if (d.type === 'log' && d.message) {
          lines.push(d.message);
          if (lines.length > 60) lines.splice(0, lines.length - 60);
          setRecentLogs([...lines]);
        }
        if (d.type === 'done') { onStatusChange('done'); es.close(); }
        if (d.type === 'error') { onStatusChange('error'); es.close(); }
      } catch { /* ignore */ }
    };
    return () => es.close();
  }, [jobId, onStatusChange]);

  // Poll codegen job statuses
  useEffect(() => {
    if (!codegenMap.size) return;
    const running = [...codegenMap.values()].filter(v => v.status === 'running');
    if (!running.length) return;
    const t = setInterval(async () => {
      for (const entry of running) {
        try {
          const r = await fetch(`/api/paper2code/${entry.jobId}`);
          if (r.ok) {
            const d = await r.json() as { status: 'running' | 'done' | 'error' };
            if (d.status !== 'running') {
              setCodegenMap(prev => {
                const m = new Map(prev);
                for (const [doi, v] of m) {
                  if (v.jobId === entry.jobId) m.set(doi, { ...v, status: d.status });
                }
                return m;
              });
            }
          }
        } catch { /* ignore */ }
      }
    }, 3000);
    return () => clearInterval(t);
  }, [codegenMap]);

  const handleGenerate = useCallback(async (paper: PaperRef) => {
    try {
      const r = await fetch('/api/paper2code', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ doi: paper.doi, title: paper.title }),
      });
      if (!r.ok) throw new Error('failed');
      const { jobId: cjId } = await r.json() as { jobId: string };
      setCodegenMap(prev => {
        const m = new Map(prev);
        m.set(paper.doi, { jobId: cjId, status: 'running' });
        return m;
      });
    } catch (e) {
      console.error('paper2code error', e);
    }
  }, []);

  // Derive structured state from events
  const latestIteration = useMemo(() => {
    const exps = events.filter(e => e.event === 'experiment_done');
    return exps.length ? (exps[exps.length - 1].payload.iteration as number ?? 0) : 0;
  }, [events]);

  const currentPhase = useMemo(() => {
    const phases = recentLogs.filter(l => l.includes('PHASE') || l.includes('▶')).slice(-1);
    return phases[0]?.replace(/\[.*?s\]/, '').replace('▶', '').trim() ?? '';
  }, [recentLogs]);

  const seedIdeas = useMemo<SeedIdea[]>(() => {
    const e = events.filter(ev => ev.event === 'seed_ideas_done').slice(-1)[0];
    const raw = (e?.payload.ideas ?? []) as Record<string, unknown>[];
    return raw.map(r => ({
      idea_id:         toStr(r.idea_id),
      agent_id:        toStr(r.agent_id),
      paper_id:        toStr(r.paper_id),
      paper_title:     toStr(r.paper_title),
      text:            toStr(r.text),
      rationale:       toStr(r.rationale),
      expected_effect: toStr(r.expected_effect),
      changes:         toStr(r.changes),
    }));
  }, [events]);

  const crossIdeas = useMemo<CrossIdea[]>(() => {
    const e = events.filter(ev => ev.event === 'cross_ideas_done').slice(-1)[0];
    const raw = (e?.payload.ideas ?? []) as Record<string, unknown>[];
    return raw.map(r => ({
      idea_id:          toStr(r.idea_id),
      agent_id:         toStr(r.agent_id),
      paper_id:         toStr(r.paper_id),
      seed_idea_id:     toStr(r.seed_idea_id),
      seed_agent_id:    toStr(r.seed_agent_id ?? r.agent_id),
      seed_paper_title: toStr(r.seed_paper_title),
      text:             toStr(r.text),
      connection:       toStr(r.connection),
      changes:          toStr(r.changes),
    }));
  }, [events]);

  const currentPlan = useMemo(() => {
    const e = events.filter(ev => ev.event === 'plan_done').slice(-1)[0];
    return toStr(e?.payload.plan);
  }, [events]);

  const diagnosis = useMemo(() => {
    const e = events.filter(ev => ev.event === 'orchestration_diagnosis_done').slice(-1)[0];
    return toStr(e?.payload.diagnosis);
  }, [events]);

  const judgeEvent = useMemo(() => {
    const e = events.filter(ev => ev.event === 'judge_done').slice(-1)[0];
    const raw = e?.payload.judge as Record<string, unknown> | string | undefined;
    if (!raw) return undefined;
    if (typeof raw === 'string') return { decision: raw, reason: '' };
    return {
      decision: toStr(raw.decision),
      reason:   toStr(raw.reason ?? raw.summary ?? raw.raw ?? ''),
    };
  }, [events]);

  const activeAgents = useMemo<string[]>(() => {
    const e = events.filter(ev => ev.event === 'agents_selected').slice(-1)[0];
    return (e?.payload.agents as string[]) ?? [];
  }, [events]);

  return (
    <div style={{
      flex:          1,
      display:       'flex',
      flexDirection: 'column',
      background:    '#0a0c10',
      overflow:      'hidden',
      position:      'relative',
    }}>
      {/* ── Top: paper boxes ─────────────────────────────────────────── */}
      <div style={{
        display:      'flex',
        gap:          10,
        padding:      '12px 16px',
        borderBottom: `1px solid ${DIM}`,
        flexShrink:   0,
        overflowX:    'auto',
        background:   '#0d0f14',
      }}>
        {papers.map((p, i) => (
          <PaperBox
            key={p.doi || i}
            paper={p}
            index={i}
            codegenStatus={codegenMap.get(p.doi)?.status ?? 'idle'}
            onGenerate={handleGenerate}
          />
        ))}
      </div>

      {/* ── Main 3-column area ────────────────────────────────────────── */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>

        {/* Left: orchestrator */}
        <OrchestratorPanel
          phase={currentPhase}
          plan={currentPlan}
          diagnosis={diagnosis}
          judgeDecision={judgeEvent?.decision ?? ''}
          judgeReason={judgeEvent?.reason ?? ''}
          iteration={latestIteration}
        />

        {/* Middle: agents */}
        <AgentPanel
          activeAgents={activeAgents}
          seedIdeas={seedIdeas}
          crossIdeas={crossIdeas}
          currentEvents={recentLogs}
          allLogs={recentLogs}
        />

        {/* Right: ideas */}
        <IdeasPanel
          seedIdeas={seedIdeas}
          crossIdeas={crossIdeas}
          plan={currentPlan}
        />
      </div>

      {/* ── Bottom-centre: Research button ───────────────────────────── */}
      <div style={{
        position:   'absolute',
        bottom:     20,
        left:       '50%',
        transform:  'translateX(-50%)',
        zIndex:     20,
        display:    'flex',
        alignItems: 'center',
        gap:        12,
      }}>
        {jobStage === 'ingesting' && (
          <div style={{
            background:   'rgba(13,15,20,0.85)',
            border:       '1px solid rgba(255,255,255,0.1)',
            borderRadius: 10,
            padding:      '9px 22px',
            fontFamily:   mono,
            fontSize:     12,
            color:        '#22d3ee',
            display:      'flex',
            alignItems:   'center',
            gap:          8,
            animation:    'drPulse 1.5s ease-in-out infinite',
          }}>
            <span style={{ fontSize: 14 }}>⏳</span>
            Downloading &amp; ingesting papers…
          </div>
        )}

        {jobStage === 'ready' && (
          <button
            onClick={handleStartResearch}
            disabled={launching}
            style={{
              background:    'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)',
              color:         'white',
              border:        'none',
              borderRadius:  12,
              padding:       '12px 40px',
              fontSize:      14,
              fontFamily:    mono,
              fontWeight:    700,
              cursor:        launching ? 'wait' : 'pointer',
              boxShadow:     '0 4px 28px rgba(124,58,237,0.55), 0 2px 8px rgba(0,0,0,0.4)',
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              display:       'flex',
              alignItems:    'center',
              gap:           10,
              transition:    'transform 0.12s, box-shadow 0.12s',
            }}
            onMouseEnter={e => { (e.currentTarget as HTMLElement).style.transform = 'scale(1.04)'; }}
            onMouseLeave={e => { (e.currentTarget as HTMLElement).style.transform = 'scale(1)'; }}
          >
            <span style={{ fontSize: 18 }}>🔬</span>
            {launching ? 'Launching…' : 'Research'}
          </button>
        )}

        {jobStage === 'researching' && (
          <div style={{
            background:   'rgba(124,58,237,0.12)',
            border:       '1px solid rgba(124,58,237,0.35)',
            borderRadius: 10,
            padding:      '9px 22px',
            fontFamily:   mono,
            fontSize:     12,
            color:        '#a78bfa',
            display:      'flex',
            alignItems:   'center',
            gap:          8,
          }}>
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#a78bfa', display: 'inline-block', animation: 'drPulse 1.5s ease-in-out infinite' }} />
            Agent swarm running…
          </div>
        )}

        {jobStage === 'complete' && (
          <div style={{
            background:   'rgba(74,222,128,0.1)',
            border:       '1px solid rgba(74,222,128,0.3)',
            borderRadius: 10,
            padding:      '9px 22px',
            fontFamily:   mono,
            fontSize:     12,
            color:        '#4ade80',
            display:      'flex',
            alignItems:   'center',
            gap:          8,
          }}>
            <span>✓</span>
            Research complete
          </div>
        )}
      </div>

      {/* ── Bottom-right: metric graph ────────────────────────────────── */}
      <div style={{
        position: 'absolute',
        bottom:   16,
        right:    16,
        zIndex:   10,
      }}>
        <MetricGraph
          points={metricPoints}
          width={240}
          height={120}
          metricKey={metricKey}
        />
      </div>

      <style>{`
        @keyframes drPulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
        @keyframes drDash   { to { stroke-dashoffset: -18; } }
      `}</style>
    </div>
  );
};
