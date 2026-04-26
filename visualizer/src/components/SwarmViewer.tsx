/**
 * SwarmViewer — structured live display for the mnist_fcnn research swarm.
 */

import React, { useEffect, useRef, useState } from 'react';

const mono  = "'JetBrains Mono', monospace";
const serif = "'Crimson Pro', Georgia, serif";

// ── Log parsing ───────────────────────────────────────────────────────────────

const LOG_RE = /^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s+(\w+)\s+(.+)$/s;

interface ParsedLine {
  kind:       'phase' | 'phase_done' | 'agent_start' | 'agent_done'
            | 'selected' | 'token' | 'info' | 'other';
  body:       string;
  agentId?:   string;
  stage?:     string;
  elapsed?:   string;
  agents?:    string[];
  tokenText?: string;
}

function parseLine(raw: string): ParsedLine {
  const s = raw.startsWith('[SWARM] ') ? raw.slice(8) : raw;
  const m = LOG_RE.exec(s);
  if (!m) return { kind: 'other', body: s };
  const rest = m[3];

  if (rest.startsWith('PHASE_DONE '))
    return { kind: 'phase_done', body: rest.slice(11) };
  if (rest.startsWith('PHASE '))
    return { kind: 'phase', body: rest.slice(6) };
  if (rest.startsWith('AGENT_START ')) {
    const kv = parseKV(rest.slice(12));
    return { kind: 'agent_start', body: rest, agentId: kv.agent, stage: kv.stage };
  }
  if (rest.startsWith('AGENT_DONE ')) {
    const kv = parseKV(rest.slice(11));
    return { kind: 'agent_done', body: rest, agentId: kv.agent, elapsed: kv.elapsed };
  }
  if (rest.startsWith('SELECTED ')) {
    const kv = parseKV(rest.slice(9));
    return { kind: 'selected', body: rest, agents: kv.agents?.split(/,\s*/) ?? [] };
  }
  if (rest.startsWith('TOKEN '))
    return { kind: 'token', body: rest, tokenText: extractToken(rest.slice(6)) };
  if (rest.startsWith('INFO '))
    return { kind: 'info', body: rest.slice(5) };
  return { kind: 'other', body: rest };
}

function parseKV(s: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const part of s.split(/\s+(?=\w+=)/)) {
    const eq = part.indexOf('=');
    if (eq !== -1) out[part.slice(0, eq)] = part.slice(eq + 1);
  }
  return out;
}

function extractToken(s: string): string {
  const q = s[0];
  if (q === "'" || q === '"') {
    return s.slice(1, -1)
      .replace(/\\n/g, '\n').replace(/\\t/g, '\t')
      .replace(/\\'/g, "'").replace(/\\"/g, '"').replace(/\\\\/g, '\\');
  }
  return s;
}

// ── Idea parsing ──────────────────────────────────────────────────────────────

interface Idea {
  kind:        'research' | 'cross';
  text:        string;
  rationale:   string;
  expected:    string;
  changes:     string;
  connection?: string;
}

function parseIdeas(token: string): Idea[] {
  const out: Idea[] = [];
  const field = (block: string, key: string) => {
    const m = new RegExp(`${key}:\\s*([\\s\\S]*?)(?=\\n[A-Z_]+:|$)`, 'i').exec(block);
    return m ? m[1].trim() : '';
  };
  let m: RegExpExecArray | null;
  const re1 = /---MODEL_IDEA---([\s\S]*?)---END---/g;
  const re2 = /---MODEL_CROSS_IDEA---([\s\S]*?)---END---/g;
  while ((m = re1.exec(token)) !== null)
    out.push({ kind: 'research', text: field(m[1], 'TEXT'), rationale: field(m[1], 'RATIONALE'),
               expected: field(m[1], 'EXPECTED_EFFECT'), changes: field(m[1], 'CHANGES') });
  while ((m = re2.exec(token)) !== null)
    out.push({ kind: 'cross', text: field(m[1], 'TEXT'), rationale: field(m[1], 'RATIONALE'),
               expected: field(m[1], 'EXPECTED_EFFECT'), changes: field(m[1], 'CHANGES'),
               connection: field(m[1], 'CONNECTION') });
  return out;
}

// ── Data model ────────────────────────────────────────────────────────────────

interface AgentEntry {
  agentId:   string;
  label:     string;
  stage:     string;          // 'research proposal' | 'cross-pollinating'
  status:    'active' | 'done';
  ideas:     Idea[];
  elapsed?:  string;
  examining: string[];        // other selected agents (populated for cross stage)
}

interface Iteration {
  num:      number;
  selected: string[];         // full ids e.g. 'expert:introcnn'
  agents:   AgentEntry[];     // in arrival order; one entry per (agentId, stage)
  result?:  string;           // from PHASE_DONE Iteration N ...
}

interface SwarmState {
  phases:     { text: string; done: boolean }[];
  iterations: Iteration[];
}

function buildState(lines: string[]): SwarmState {
  const phases: { text: string; done: boolean }[] = [];
  const iterations: Iteration[] = [];
  let   cur: Iteration | null = null;
  let   curEntry: AgentEntry | null = null;
  let   pendingTokens: string[] = [];

  const flush = () => {
    if (!curEntry || !pendingTokens.length) { pendingTokens = []; return; }
    const ideas = parseIdeas(pendingTokens.join(''));
    curEntry.ideas.push(...ideas);
    pendingTokens = [];
  };

  for (const raw of lines) {
    const p = parseLine(raw);
    switch (p.kind) {

      case 'phase': {
        // Close any open phase
        if (phases.length && !phases[phases.length - 1].done)
          phases[phases.length - 1].done = false; // still in progress
        // Detect "Research iteration N"
        const itMatch = /^Research iteration (\d+)$/.exec(p.body);
        if (itMatch) {
          flush();
          curEntry = null;
          cur = { num: parseInt(itMatch[1]), selected: [], agents: [], result: undefined };
          iterations.push(cur);
        }
        phases.push({ text: p.body, done: false });
        break;
      }

      case 'phase_done': {
        if (phases.length) phases[phases.length - 1].done = true;
        phases.push({ text: p.body, done: true });
        // Detect "Iteration N decision=..." to close out the iteration
        const itDone = /^Iteration (\d+)/.exec(p.body);
        if (itDone && cur && cur.num === parseInt(itDone[1])) {
          cur.result = p.body;
        }
        break;
      }

      case 'selected': {
        if (cur) cur.selected = p.agents ?? [];
        break;
      }

      case 'agent_start': {
        flush();
        if (!p.agentId) break;
        const label = p.agentId.replace(/^expert:/, '').replace(/_/g, ' ');
        const stage = p.stage ?? '';
        // examining = all other selected agents (for cross-pollination stage)
        const selectedIds = cur?.selected ?? [];
        const examining   = stage.includes('cross')
          ? selectedIds.filter(s => s !== p.agentId).map(s => s.replace(/^expert:/, '').replace(/_/g, ' '))
          : [];
        curEntry = { agentId: p.agentId!, label, stage, status: 'active', ideas: [], examining };
        cur?.agents.push(curEntry);
        break;
      }

      case 'agent_done': {
        flush();
        const id = p.agentId ?? curEntry?.agentId;
        if (id && cur) {
          // Mark the most recent entry for this agent as done
          for (let i = cur.agents.length - 1; i >= 0; i--) {
            if (cur.agents[i].agentId === id && cur.agents[i].status === 'active') {
              cur.agents[i].status  = 'done';
              cur.agents[i].elapsed = p.elapsed;
              break;
            }
          }
        }
        curEntry = null;
        break;
      }

      case 'token': {
        if (curEntry && p.tokenText) pendingTokens.push(p.tokenText);
        break;
      }
    }
  }
  flush();

  return { phases, iterations };
}

// ── Sub-components ────────────────────────────────────────────────────────────

const IdeaCard: React.FC<{ idea: Idea; index: number }> = ({ idea, index }) => {
  const [open, setOpen] = useState(false);
  const isResearch = idea.kind === 'research';
  const accent     = isResearch ? '#38bdf8' : '#a78bfa';
  const label      = isResearch ? 'IDEA' : 'CROSS';

  return (
    <div onClick={() => setOpen(o => !o)} style={{
      background:   isResearch ? 'rgba(56,189,248,0.04)' : 'rgba(167,139,250,0.04)',
      border:       `1px solid ${accent}2a`,
      borderLeft:   `3px solid ${accent}`,
      borderRadius: 5,
      padding:      '7px 10px',
      cursor:       'pointer',
      marginBottom: 5,
    }}>
      {/* Label + chevron */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
        <span style={{
          fontFamily: mono, fontSize: 8, letterSpacing: '0.05em',
          color: accent, background: `${accent}1a`, borderRadius: 3, padding: '1px 5px',
        }}>
          {label} {index + 1}
        </span>
        <span style={{ fontFamily: mono, fontSize: 9, color: '#374151', marginLeft: 'auto' }}>
          {open ? '▲' : '▼'}
        </span>
      </div>

      {/* Summary text — always visible */}
      <div style={{ fontFamily: serif, fontSize: 12, color: '#d1d5db', lineHeight: 1.45 }}>
        {idea.text || '(no summary)'}
      </div>

      {/* Expanded detail */}
      {open && (
        <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 8, borderTop: `1px solid ${accent}18`, paddingTop: 8 }}>
          {idea.connection && (
            <div style={{
              background:   'rgba(167,139,250,0.08)',
              border:       '1px solid rgba(167,139,250,0.2)',
              borderRadius: 4, padding: '6px 8px',
            }}>
              <div style={{ fontFamily: mono, fontSize: 8, color: '#a78bfa', marginBottom: 3, letterSpacing: '0.06em' }}>
                BRIDGE
              </div>
              <div style={{ fontFamily: serif, fontSize: 11, color: '#c4b5fd', lineHeight: 1.5 }}>
                {idea.connection}
              </div>
            </div>
          )}
          <DetailField label="Rationale"       value={idea.rationale} color="#fbbf24" />
          <DetailField label="Expected Effect" value={idea.expected}  color="#4ade80" />
          <DetailField label="Code Changes"    value={idea.changes}   color="#f59e0b" mono />
        </div>
      )}
    </div>
  );
};

const DetailField: React.FC<{ label: string; value: string; color: string; mono?: boolean }> =
  ({ label, value, color, mono: isMono }) => (
    <div>
      <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 8, color, marginBottom: 2, letterSpacing: '0.06em', textTransform: 'uppercase' }}>
        {label}
      </div>
      <div style={{
        fontFamily:  isMono ? "'JetBrains Mono', monospace" : serif,
        fontSize:    isMono ? 10 : 11,
        color:       '#9ca3af',
        lineHeight:  1.55,
        whiteSpace:  'pre-wrap',
        wordBreak:   'break-word',
      }}>
        {value || '—'}
      </div>
    </div>
  );

// ── Agent entry card ──────────────────────────────────────────────────────────

const AgentEntryCard: React.FC<{ entry: AgentEntry }> = ({ entry }) => {
  const isCross   = entry.stage.includes('cross');
  const isActive  = entry.status === 'active';
  const stageColor = isCross   ? '#a78bfa'
                   : isActive  ? '#38bdf8'
                   : '#4ade80';
  const dotColor  = isActive ? stageColor : '#4ade80';

  const researchIdeas = entry.ideas.filter(i => i.kind === 'research');
  const crossIdeas    = entry.ideas.filter(i => i.kind === 'cross');

  return (
    <div style={{
      background:   'rgba(14,16,22,0.95)',
      border:       `1px solid ${stageColor}30`,
      borderRadius: 8,
      overflow:     'hidden',
      marginBottom: 8,
    }}>
      {/* ── Header ── */}
      <div style={{
        display:    'flex', alignItems: 'center', gap: 8,
        padding:    '8px 12px',
        background: `${stageColor}0a`,
        borderBottom: (entry.ideas.length > 0 || entry.examining.length > 0)
          ? `1px solid ${stageColor}1a` : 'none',
      }}>
        {/* Status dot */}
        <div style={{
          width: 7, height: 7, borderRadius: '50%', flexShrink: 0,
          background: dotColor,
          boxShadow:  isActive ? `0 0 7px ${dotColor}` : 'none',
          animation:  isActive ? 'swPulse 1.4s ease-in-out infinite' : undefined,
        }} />

        {/* Agent name */}
        <span style={{ fontFamily: mono, fontSize: 11, color: '#e2e8f0', fontWeight: 700, flexShrink: 0 }}>
          {entry.label}
        </span>

        {/* Stage badge */}
        <span style={{
          fontFamily:  mono, fontSize: 8, letterSpacing: '0.05em',
          color:       stageColor,
          background:  `${stageColor}18`,
          border:      `1px solid ${stageColor}30`,
          borderRadius: 4, padding: '1px 7px',
          textTransform: 'uppercase', flexShrink: 0,
        }}>
          {isCross ? '⟳ cross-pollinating' : '◆ research proposal'}
        </span>

        {/* Elapsed */}
        {entry.elapsed && (
          <span style={{ fontFamily: mono, fontSize: 9, color: '#374151', marginLeft: 'auto' }}>
            {entry.elapsed}
          </span>
        )}
        {isActive && !entry.elapsed && (
          <span style={{
            fontFamily: mono, fontSize: 9, color: stageColor,
            marginLeft: 'auto', animation: 'swPulse 1.4s ease-in-out infinite',
          }}>
            running…
          </span>
        )}
      </div>

      {/* ── Cross-pollination: show which agents are being examined ── */}
      {isCross && entry.examining.length > 0 && (
        <div style={{
          display:    'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap',
          padding:    '5px 12px',
          background: 'rgba(167,139,250,0.04)',
          borderBottom: '1px solid rgba(167,139,250,0.1)',
        }}>
          <span style={{ fontFamily: mono, fontSize: 9, color: '#6b7280', flexShrink: 0 }}>
            {entry.label}
          </span>
          <span style={{ fontFamily: mono, fontSize: 10, color: '#a78bfa' }}>×</span>
          {entry.examining.map((other, i) => (
            <React.Fragment key={other}>
              <span style={{
                fontFamily: mono, fontSize: 9,
                color:      '#c4b5fd',
                background: 'rgba(167,139,250,0.12)',
                border:     '1px solid rgba(167,139,250,0.25)',
                borderRadius: 4, padding: '1px 7px',
              }}>
                {other}
              </span>
              {i < entry.examining.length - 1 && (
                <span style={{ fontFamily: mono, fontSize: 9, color: '#374151' }}>+</span>
              )}
            </React.Fragment>
          ))}
        </div>
      )}

      {/* ── Ideas ── */}
      {(researchIdeas.length > 0 || crossIdeas.length > 0) && (
        <div style={{ padding: '8px 10px' }}>
          {researchIdeas.length > 0 && (
            <>
              <div style={{
                fontFamily: mono, fontSize: 8, color: '#38bdf8',
                letterSpacing: '0.07em', marginBottom: 6, textTransform: 'uppercase',
              }}>
                Research Ideas ({researchIdeas.length})
              </div>
              {researchIdeas.map((idea, i) => <IdeaCard key={i} idea={idea} index={i} />)}
            </>
          )}
          {crossIdeas.length > 0 && (
            <div style={{ marginTop: researchIdeas.length > 0 ? 8 : 0 }}>
              <div style={{
                fontFamily: mono, fontSize: 8, color: '#a78bfa',
                letterSpacing: '0.07em', marginBottom: 6, textTransform: 'uppercase',
              }}>
                Cross-Pollination Ideas ({crossIdeas.length})
              </div>
              {crossIdeas.map((idea, i) => <IdeaCard key={i} idea={idea} index={i} />)}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// ── Iteration block ───────────────────────────────────────────────────────────

const IterationBlock: React.FC<{ iter: Iteration }> = ({ iter }) => {
  const isDone    = !!iter.result;
  const accuracy  = iter.result ? /test_accuracy=([\d.]+)/.exec(iter.result)?.[1] : null;
  const decision  = iter.result ? /decision=(\w+)/.exec(iter.result)?.[1] : null;
  const decColor  = decision === 'keep' ? '#4ade80' : decision === 'revert' ? '#f87171' : '#fbbf24';

  return (
    <div style={{ marginBottom: 14 }}>
      {/* Iteration header */}
      <div style={{
        display:      'flex', alignItems: 'center', gap: 8,
        padding:      '6px 10px',
        background:   'rgba(255,255,255,0.03)',
        border:       '1px solid rgba(255,255,255,0.07)',
        borderRadius: '6px 6px 0 0',
        borderBottom: 'none',
      }}>
        <span style={{ fontFamily: mono, fontSize: 10, color: '#6b7280', fontWeight: 700, letterSpacing: '0.04em' }}>
          ITERATION {iter.num}
        </span>

        {/* Selected agent chips */}
        {iter.selected.length > 0 && (
          <>
            <span style={{ fontFamily: mono, fontSize: 9, color: '#374151' }}>·</span>
            {iter.selected.map(id => {
              const lbl = id.replace(/^expert:/, '').replace(/_/g, ' ');
              return (
                <span key={id} style={{
                  fontFamily: mono, fontSize: 8,
                  color:      '#94a3b8',
                  background: 'rgba(148,163,184,0.08)',
                  border:     '1px solid rgba(148,163,184,0.15)',
                  borderRadius: 3, padding: '1px 6px',
                }}>
                  {lbl}
                </span>
              );
            })}
          </>
        )}

        {/* Result badges */}
        {isDone && (
          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6 }}>
            {decision && (
              <span style={{
                fontFamily: mono, fontSize: 8, color: decColor,
                background: `${decColor}18`, border: `1px solid ${decColor}30`,
                borderRadius: 3, padding: '1px 7px', textTransform: 'uppercase',
              }}>
                {decision}
              </span>
            )}
            {accuracy && (
              <span style={{ fontFamily: mono, fontSize: 9, color: '#4ade80', fontWeight: 700 }}>
                {(parseFloat(accuracy) * 100).toFixed(1)}% accuracy
              </span>
            )}
          </div>
        )}
        {!isDone && (
          <span style={{
            marginLeft: 'auto', fontFamily: mono, fontSize: 8, color: '#38bdf8',
            animation: 'swPulse 1.4s ease-in-out infinite',
          }}>
            in progress
          </span>
        )}
      </div>

      {/* Agent entries */}
      <div style={{
        border:       '1px solid rgba(255,255,255,0.07)',
        borderTop:    'none',
        borderRadius: '0 0 6px 6px',
        padding:      '8px 8px 4px',
        background:   'rgba(10,12,16,0.5)',
      }}>
        {iter.agents.length === 0 ? (
          <div style={{ fontFamily: mono, fontSize: 10, color: '#374151', padding: '4px 2px' }}>
            Waiting for agents…
          </div>
        ) : (
          iter.agents.map((entry, i) => (
            <AgentEntryCard key={entry.agentId + '|' + entry.stage + '|' + i} entry={entry} />
          ))
        )}
      </div>
    </div>
  );
};

// ── Main component ────────────────────────────────────────────────────────────

export const SwarmViewer: React.FC<{ lines: string[] }> = ({ lines }) => {
  const [state, setState] = useState<SwarmState>({ phases: [], iterations: [] });
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setState(buildState(lines));
  }, [lines]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'auto' });
  }, [state.iterations.length, state.phases.length]);

  const { phases, iterations } = state;

  return (
    <div style={{
      flex: 1, minHeight: 0,
      display: 'flex', flexDirection: 'column',
      background: '#090b0f',
    }}>
      <style>{`@keyframes swPulse{0%,100%{opacity:1}50%{opacity:0.25}}`}</style>

      {/* ── Phase log — compact scrollable box ── */}
      <div style={{
        flexShrink:   0,
        maxHeight:    96,
        overflowY:    'auto',
        padding:      '7px 14px',
        borderBottom: '1px solid rgba(255,255,255,0.07)',
        background:   'rgba(8,10,14,0.98)',
        display:      'flex', flexDirection: 'column', gap: 2,
      }}>
        {phases.length === 0 ? (
          <span style={{ fontFamily: mono, fontSize: 10, color: '#374151', fontStyle: 'italic' }}>
            Waiting for phases…
          </span>
        ) : phases.map((ph, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'baseline', gap: 7 }}>
            <span style={{ fontFamily: mono, fontSize: 10, color: ph.done ? '#4ade80' : '#818cf8', flexShrink: 0 }}>
              {ph.done ? '✓' : '▶'}
            </span>
            <span style={{ fontFamily: mono, fontSize: 10, color: ph.done ? '#4ade8099' : '#818cf8cc', lineHeight: 1.4 }}>
              {ph.text}
            </span>
          </div>
        ))}
      </div>

      {/* ── Iteration blocks — scrollable main area ── */}
      <div style={{
        flex: 1, minHeight: 0,
        overflowY: 'auto',
        padding:   '10px 10px',
      }}>
        {iterations.length === 0 && (
          <div style={{ fontFamily: mono, fontSize: 11, color: '#374151', fontStyle: 'italic', padding: '4px 2px' }}>
            Waiting for research iterations to begin…
          </div>
        )}

        {iterations.map(iter => (
          <IterationBlock key={iter.num} iter={iter} />
        ))}

        <div ref={bottomRef} />
      </div>
    </div>
  );
};
