import React, { useState, useCallback } from 'react';
import type { Void, SelectedPaper } from '../hooks/useVoidData';

interface VoidPanelProps {
  voids:          Void[];
  selectedVoidId: number | null;
  showVoidLabels: boolean;
  voidsVisible:   boolean;
  loading:        boolean;
  onSelectVoid:   (id: number | null) => void;
  onToggleLabels: () => void;
  onToggleVoids:  () => void;
  darkMode:       boolean;
}

const PANEL_W = 310;
const mono  = "'JetBrains Mono', monospace";
const serif = "'Crimson Pro', Georgia, serif";

function emptinessOf(v: Void): number {
  return Math.min(1, Math.max(0, v.empty_radius / 0.35));
}

// ── Compass ───────────────────────────────────────────────────────────────────

const SectorCompass: React.FC<{
  angleDeg: number;
  sector:   number;
  dim?:     boolean;
}> = ({ angleDeg, sector, dim = false }) => {
  const rad = (angleDeg * Math.PI) / 180;
  const nx  = Math.sin(rad) * 7;
  const ny  = -Math.cos(rad) * 7;
  const op  = dim ? 0.45 : 1.0;

  return (
    <svg width={24} height={24} viewBox="-12 -12 24 24" style={{ flexShrink: 0, display: 'block' }}>
      <circle r={10} fill="none" stroke="#fbbf24" strokeWidth={0.8} opacity={dim ? 0.3 : 0.55} />
      {Array.from({ length: 8 }, (_, i) => {
        const a  = (i * Math.PI) / 4;
        const x1 = Math.sin(a) * 7.5;
        const y1 = -Math.cos(a) * 7.5;
        const x2 = Math.sin(a) * 10;
        const y2 = -Math.cos(a) * 10;
        return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#fbbf24" strokeWidth={0.6} opacity={0.35} />;
      })}
      <line x1={0} y1={0} x2={nx} y2={ny} stroke="#d97706" strokeWidth={1.6} strokeLinecap="round" opacity={op} />
      <circle r={2} fill="#fbbf24" opacity={op} />
      <text x={0} y={14} textAnchor="middle" fontFamily={mono} fontSize={6} fill="#a16207" opacity={op}>
        S{sector}
      </text>
    </svg>
  );
};

// ── Component ─────────────────────────────────────────────────────────────────

export const VoidPanel: React.FC<VoidPanelProps> = ({
  voids,
  selectedVoidId,
  showVoidLabels,
  voidsVisible,
  loading,
  onSelectVoid,
  onToggleLabels,
  onToggleVoids,
  darkMode,
}) => {
  const [panelOpen, setPanelOpen] = useState(false);

  const dm           = darkMode;
  const selectedVoid = voids.find(v => v.void_id === selectedVoidId) ?? null;

  // ── Derived theme tokens ─────────────────────────────────────────────────
  const panelBg     = dm ? 'rgba(13,15,20,0.95)'    : 'rgba(255,255,255,0.94)';
  const panelBorder = dm ? 'rgba(255,255,255,0.06)'  : 'rgba(0,0,0,0.07)';
  const panelShadow = dm ? '4px 0 24px rgba(0,0,0,0.6)' : '4px 0 24px rgba(0,0,0,0.07)';
  const tabBg       = dm ? 'rgba(13,15,20,0.92)'    : 'rgba(255,255,255,0.92)';
  const tabBorder   = dm ? 'rgba(255,255,255,0.06)'  : 'rgba(0,0,0,0.08)';
  const headingC    = dm ? '#e2e8f0' : '#1c1917';
  const subC        = dm ? '#a16207' : '#a16207';  // kept amber in both modes
  const divider     = dm ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)';
  const controlBorder = dm ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)';
  const labelCtrlC  = dm ? '#a8a29e' : '#78716c';
  const rankBadgeC  = dm ? '#57534e' : '#a8a29e';
  const voidNameSel = dm ? '#fbbf24' : '#92400e';
  const voidNameDef = dm ? '#d4cfc9' : '#292524';
  const emptyBarBg  = dm ? 'rgba(251,191,36,0.12)' : '#fef3c7';
  const selHeaderBg = dm ? 'rgba(20,18,10,0.97)'   : 'rgba(255,251,235,0.97)';
  const selHeaderBC = dm ? 'rgba(251,191,36,0.15)'  : 'rgba(251,191,36,0.2)';
  const selHeaderC  = dm ? '#b45309' : '#a16207';
  const selHeaderSC = dm ? '#c4924a' : '#c4924a';
  const metaBlockC  = dm ? '#e2e8f0' : '#92400e';
  const reasoningC  = dm ? '#a8a29e' : '#57534e';
  const metaC       = dm ? '#6b7280' : '#a8a29e';
  const cardBg0     = dm ? 'rgba(25,22,12,0.85)'   : 'rgba(255,251,235,0.85)';
  const cardBg1     = dm ? 'rgba(18,20,26,0.60)'   : 'rgba(255,255,255,0.6)';
  const cardBorder0 = dm ? 'rgba(251,191,36,0.5)'  : 'rgba(251,191,36,0.6)';
  const cardBorder1 = dm ? 'rgba(255,255,255,0.06)': 'rgba(0,0,0,0.07)';
  const cardTitleC  = dm ? '#e2e8f0' : '#1c1917';
  const rankPillC0  = dm ? '#fbbf24' : '#92400e';
  const rankPillBg0 = dm ? 'rgba(251,191,36,0.15)': '#fef3c7';
  const rankPillC1  = dm ? '#6b7280' : '#a8a29e';
  const rankPillBg1 = dm ? 'rgba(255,255,255,0.06)': 'rgba(0,0,0,0.05)';
  const chipHiC     = dm ? '#fbbf24' : '#92400e';
  const chipHiBg    = dm ? 'rgba(251,191,36,0.15)' : 'rgba(251,191,36,0.18)';
  const chipDimC    = dm ? '#4b5563' : '#a8a29e';
  const chipDimBg   = dm ? 'rgba(255,255,255,0.05)': 'rgba(0,0,0,0.05)';
  const doiC        = dm ? '#d97706' : '#a16207';
  const scoreTrackBg= dm ? 'rgba(255,255,255,0.08)': 'rgba(0,0,0,0.08)';
  const scoreValC   = dm ? '#fbbf24' : '#d97706';
  const scoreLblC   = dm ? '#4b5563' : '#a8a29e';
  const borderLblC  = dm ? '#d97706' : '#a16207';
  const borderRowBC = dm ? 'rgba(255,255,255,0.04)': 'rgba(0,0,0,0.04)';
  const borderTitleC= dm ? '#d4cfc9' : '#44403c';
  const borderDoiC  = dm ? '#b45309' : '#a16207';

  const handleItemClick = useCallback(
    (id: number) => { onSelectVoid(id === selectedVoidId ? null : id); },
    [selectedVoidId, onSelectVoid],
  );

  const openDOI = useCallback((doi: string) => {
    window.open(`https://arxiv.org/abs/${doi}`, '_blank', 'noopener,noreferrer');
  }, []);

  return (
    <>
      {/* ── Toggle tab ── */}
      <div
        onClick={() => setPanelOpen(o => !o)}
        title={panelOpen ? 'Close void panel' : 'Open void panel'}
        style={{
          position: 'absolute',
          top: '50%',
          left: panelOpen ? PANEL_W : 0,
          transform: 'translateY(-50%)',
          zIndex: 60,
          background: tabBg,
          backdropFilter: 'blur(12px)',
          WebkitBackdropFilter: 'blur(12px)',
          border: `1px solid ${tabBorder}`,
          borderLeft: 'none',
          borderRadius: '0 8px 8px 0',
          padding: '12px 6px',
          cursor: 'pointer',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 4,
          boxShadow: '2px 0 12px rgba(0,0,0,0.12)',
          transition: 'left 0.28s cubic-bezier(.4,0,.2,1), background 0.3s',
        }}
      >
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
          <polygon
            points="8,2 14,8 8,14 2,8"
            stroke="#d97706"
            strokeWidth="1.4"
            strokeDasharray="3 2"
            fill="rgba(217,119,6,0.08)"
          />
          <circle cx="8" cy="8" r="1.5" fill="rgba(217,119,6,0.5)" />
        </svg>
        <span
          style={{
            writingMode: 'vertical-rl',
            textOrientation: 'mixed',
            fontFamily: mono,
            fontSize: 10,
            letterSpacing: '0.08em',
            color: '#92400e',
            userSelect: 'none',
          }}
        >
          {panelOpen ? 'CLOSE' : 'VOIDS'}
        </span>
      </div>

      {/* ── Panel ── */}
      <div
        style={{
          position: 'absolute',
          top: 0, left: 0,
          width: PANEL_W,
          height: '100%',
          zIndex: 55,
          background: panelBg,
          backdropFilter: 'blur(20px)',
          WebkitBackdropFilter: 'blur(20px)',
          borderRight: `1px solid ${panelBorder}`,
          boxShadow: panelShadow,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          transform: panelOpen ? 'translateX(0)' : `translateX(-${PANEL_W}px)`,
          pointerEvents: panelOpen ? 'all' : 'none',
          transition: 'transform 0.28s cubic-bezier(.4,0,.2,1), background 0.3s',
        }}
      >
        {/* Header */}
        <div style={{ padding: '16px 16px 10px', borderBottom: `1px solid ${divider}`, flexShrink: 0 }}>
          <p style={{ fontFamily: serif, fontSize: 17, fontWeight: 600, color: headingC, margin: 0, lineHeight: 1 }}>
            Knowledge Voids
          </p>
          <p style={{ fontFamily: mono, fontSize: 9, color: subC, marginTop: 4, letterSpacing: '0.04em', marginBottom: 0 }}>
            {loading ? 'Loading…' : `${voids.length} sparse regions · ranked by emptiness`}
          </p>
        </div>

        {/* Controls */}
        <div
          style={{
            padding: '8px 16px',
            borderBottom: `1px solid ${controlBorder}`,
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            flexShrink: 0,
          }}
        >
          {[
            { label: 'Show shapes', checked: voidsVisible,    onChange: onToggleVoids  },
            { label: 'Show labels', checked: showVoidLabels,  onChange: onToggleLabels },
          ].map(({ label, checked, onChange }) => (
            <label
              key={label}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                cursor: 'pointer',
                fontFamily: mono,
                fontSize: 10,
                color: labelCtrlC,
                userSelect: 'none',
              }}
            >
              <input
                type="checkbox"
                checked={checked}
                onChange={onChange}
                style={{ accentColor: '#d97706' }}
              />
              {label}
            </label>
          ))}
          {selectedVoidId !== null && (
            <button
              onClick={() => onSelectVoid(null)}
              style={{
                marginLeft: 'auto',
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                fontFamily: mono,
                fontSize: 9,
                color: dm ? '#4b5563' : '#a8a29e',
                padding: '2px 4px',
              }}
            >
              clear ×
            </button>
          )}
        </div>

        {/* ── Void list ── */}
        <div style={{ overflowY: 'auto', flex: '1 1 0' as any, minHeight: 0, padding: '4px 0' }}>
          {loading && (
            <p style={{ fontFamily: mono, fontSize: 10, color: dm ? '#4b5563' : '#a8a29e', padding: 16, textAlign: 'center' }}>
              Loading voids…
            </p>
          )}

          {!loading && voids.map(v => {
            const selected = v.void_id === selectedVoidId;
            const fill     = emptinessOf(v);

            return (
              <div
                key={v.void_id}
                onClick={() => handleItemClick(v.void_id)}
                style={{
                  padding: '10px 14px',
                  cursor: 'pointer',
                  background: selected ? (dm ? 'rgba(251,191,36,0.07)' : 'rgba(251,191,36,0.10)') : 'transparent',
                  borderLeft: selected ? '3px solid #fbbf24' : '3px solid transparent',
                  transition: 'background 0.12s',
                }}
              >
                <div style={{ fontFamily: mono, fontSize: 9, color: rankBadgeC, marginBottom: 3, letterSpacing: '0.04em' }}>
                  #{v.void_rank}{' · '}r={v.empty_radius.toFixed(3)}{' · '}{v.border_papers.length} border papers
                  {v.shape_area > 0 && ` · area ${v.shape_area.toFixed(3)}`}
                </div>
                <div
                  style={{
                    fontFamily: serif,
                    fontSize: 13,
                    fontWeight: selected ? 700 : 500,
                    color: selected ? voidNameSel : voidNameDef,
                    lineHeight: 1.3,
                    marginBottom: 3,
                  }}
                >
                  {v.name ?? `Void ${v.void_id}`}
                </div>
                <div
                  style={{
                    height: 3, borderRadius: 2,
                    background: emptyBarBg,
                    marginTop: 5,
                    position: 'relative',
                    overflow: 'hidden',
                  }}
                >
                  <div
                    style={{
                      position: 'absolute', top: 0, left: 0, height: '100%',
                      width: `${fill * 100}%`,
                      background: '#fbbf24',
                      borderRadius: 2,
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>

        {/* ── Detail pane ── */}
        {selectedVoid && (
          <div
            style={{
              flexShrink: 0,
              height: '50%',
              minHeight: 0,
              overflowY: 'auto',
              borderTop: `1px solid ${divider}`,
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            {/* Sticky header */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                padding: '9px 14px 7px',
                position: 'sticky',
                top: 0,
                zIndex: 2,
                background: selHeaderBg,
                backdropFilter: 'blur(8px)',
                WebkitBackdropFilter: 'blur(8px)',
                borderBottom: `1px solid ${selHeaderBC}`,
                flexShrink: 0,
              }}
            >
              <svg width={11} height={11} viewBox="-6 -6 12 12" style={{ flexShrink: 0 }}>
                <polygon points="0,-5 5,0 0,5 -5,0" fill="#fbbf24" />
              </svg>
              <span style={{ fontFamily: mono, fontSize: 9, color: selHeaderC, letterSpacing: '0.07em', textTransform: 'uppercase', fontWeight: 600 }}>
                Selected · {selectedVoid.selected_papers.length} picks
              </span>
              <span style={{ fontFamily: mono, fontSize: 8, color: selHeaderSC, marginLeft: 'auto', fontStyle: 'italic' }}>
                cross-pollination
              </span>
            </div>

            {/* Metadata */}
            <div style={{ padding: '10px 14px 8px', flexShrink: 0 }}>
              <div style={{ fontFamily: serif, fontSize: 14, fontWeight: 700, color: metaBlockC, lineHeight: 1.3, marginBottom: 4 }}>
                {selectedVoid.name ?? `Void ${selectedVoid.void_rank}`}
              </div>
              {selectedVoid.name_reasoning && (
                <div style={{ fontFamily: serif, fontSize: 12, color: reasoningC, lineHeight: 1.55, fontStyle: 'italic', marginBottom: 8 }}>
                  {selectedVoid.name_reasoning}
                </div>
              )}
              <div style={{ fontFamily: mono, fontSize: 9, color: metaC, letterSpacing: '0.03em', marginBottom: 10 }}>
                rank #{selectedVoid.void_rank}{' · '}empty_r={selectedVoid.empty_radius.toFixed(4)}
                {selectedVoid.shape_area > 0 && ` · hull ${selectedVoid.shape_area.toFixed(3)}`}
                {' · '}{selectedVoid.shape?.vertices?.length ?? 0} hull verts
              </div>
            </div>
            <div style={{ height: 1, background: divider, margin: '6px 0 0', flexShrink: 0 }} />

            {/* Selected papers */}
            {selectedVoid.selected_papers.length > 0 && (
              <>
                {selectedVoid.selected_papers.map(p => (
                  <div
                    key={p.rank}
                    onClick={() => openDOI(p.DOI)}
                    style={{
                      margin: p.rank === 0 ? '7px 10px 5px' : '4px 10px',
                      borderRadius: 7,
                      border: `1px solid ${p.rank === 0 ? cardBorder0 : cardBorder1}`,
                      background: p.rank === 0 ? cardBg0 : cardBg1,
                      overflow: 'hidden',
                      cursor: 'pointer',
                      position: 'relative',
                      transition: 'background 0.1s',
                      flexShrink: 0,
                    }}
                  >
                    {/* Accent bar */}
                    <div
                      style={{
                        position: 'absolute', left: 0, top: 0, bottom: 0, width: 3,
                        background: 'linear-gradient(to bottom, #fbbf24, #f59e0b)',
                        opacity: 0.25 + p.scores.combined * 0.75,
                        borderRadius: '7px 0 0 7px',
                      }}
                    />
                    <div style={{ padding: '8px 10px 9px 14px' }}>
                      {/* Compass + rank + angle */}
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 5 }}>
                        <SectorCompass angleDeg={p.scores.angle_deg} sector={p.scores.sector} dim={p.rank > 0} />
                        <span
                          style={{
                            fontFamily: mono, fontSize: 8,
                            color: p.rank === 0 ? rankPillC0 : rankPillC1,
                            background: p.rank === 0 ? rankPillBg0 : rankPillBg1,
                            borderRadius: 3, padding: '2px 6px', letterSpacing: '0.03em', flexShrink: 0,
                          }}
                        >
                          #{p.rank + 1}
                        </span>
                        <span style={{ fontFamily: mono, fontSize: 8, color: '#c4a264', marginLeft: 'auto' }}>
                          {p.scores.angle_deg.toFixed(0)}° · S{p.scores.sector}
                        </span>
                      </div>

                      {/* Title */}
                      <div style={{ fontFamily: serif, fontSize: 12, color: cardTitleC, lineHeight: 1.4, marginBottom: 6 }}>
                        {p.title.replace(/\n/g, ' ').trim()}
                      </div>

                      {/* Chips */}
                      <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 6, flexWrap: 'wrap' }}>
                        {p.year != null && (
                          <span style={{ fontFamily: mono, fontSize: 8, color: chipHiC, background: chipHiBg, borderRadius: 3, padding: '2px 5px', flexShrink: 0 }}>
                            {p.year}
                          </span>
                        )}
                        {p.citation_count != null && (
                          <span style={{ fontFamily: mono, fontSize: 8, color: p.citation_count > 50 ? chipHiC : chipDimC, background: p.citation_count > 50 ? chipHiBg : chipDimBg, borderRadius: 3, padding: '2px 5px', flexShrink: 0 }}>
                            {p.citation_count.toLocaleString()} ✦
                          </span>
                        )}
                        {p.DOI && p.DOI !== 'null' && (
                          <span style={{ fontFamily: mono, fontSize: 8, color: doiC, background: 'transparent', borderRadius: 3, padding: '2px 5px', flexShrink: 0 }}>
                            {p.DOI.length > 15 ? p.DOI.slice(0, 13) + '…' : p.DOI}
                          </span>
                        )}
                      </div>

                      {/* Score bars */}
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                          {[
                            { lbl: 'cite', val: p.scores.citation, color: '#fbbf24' },
                            { lbl: 'rec',  val: p.scores.recency,  color: '#f59e0b' },
                          ].map(({ lbl, val, color }) => (
                            <React.Fragment key={lbl}>
                              <span style={{ fontFamily: mono, fontSize: 8, color: scoreLblC, width: 22, flexShrink: 0 }}>{lbl}</span>
                              <div style={{ flex: 1, height: 4, borderRadius: 2, background: scoreTrackBg, overflow: 'hidden', position: 'relative' }}>
                                <div style={{ position: 'absolute', top: 0, left: 0, bottom: 0, width: `${Math.min(1, Math.max(0, val)) * 100}%`, background: color, borderRadius: 2 }} />
                              </div>
                            </React.Fragment>
                          ))}
                          <span style={{ fontFamily: mono, fontSize: 8, color: scoreValC, width: 32, textAlign: 'right', flexShrink: 0 }}>
                            {p.scores.combined.toFixed(3)}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
                <div style={{ height: 1, background: divider, margin: '6px 0 0', flexShrink: 0 }} />
              </>
            )}

            {/* Border papers */}
            <div style={{ fontFamily: mono, fontSize: 9, color: borderLblC, letterSpacing: '0.06em', textTransform: 'uppercase', padding: '4px 14px 5px', flexShrink: 0 }}>
              Border papers ({selectedVoid.border_papers.length})
            </div>
            {selectedVoid.border_papers.map((p, i) => (
              <div
                key={i}
                onClick={() => openDOI(p.DOI)}
                style={{
                  padding: '5px 14px',
                  borderBottom: `1px solid ${borderRowBC}`,
                  cursor: 'pointer',
                }}
              >
                <p style={{ fontFamily: serif, fontSize: 12, color: borderTitleC, lineHeight: 1.35, margin: 0 }}>
                  {p.title.replace(/\n/g, ' ').trim()}
                </p>
                {p.DOI && p.DOI !== 'null' && (
                  <p style={{ fontFamily: mono, fontSize: 9, color: borderDoiC, margin: '2px 0 0' }}>
                    {p.DOI}
                  </p>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  );
};