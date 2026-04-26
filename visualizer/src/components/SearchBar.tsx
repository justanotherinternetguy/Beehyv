import React, { useState, useCallback, useRef } from 'react';
import type { ProcessedPaper } from '../types';

interface SearchBarProps {
  papers:    ProcessedPaper[];
  onResults: (ids: Set<string | number> | null, focusPaper?: ProcessedPaper) => void;
  disabled?: boolean;
  darkMode:  boolean;
}

export const SearchBar: React.FC<SearchBarProps> = ({
  papers,
  onResults,
  disabled,
  darkMode,
}) => {
  const [query,     setQuery]     = useState('');
  const [results,   setResults]   = useState<ProcessedPaper[]>([]);
  const [open,      setOpen]      = useState(false);
  const [activeIdx, setActiveIdx] = useState(0);
  const inputRef    = useRef<HTMLInputElement>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout>>();

  const dm = darkMode;

  const search = useCallback((q: string) => {
    if (!q.trim()) {
      setResults([]);
      setOpen(false);
      onResults(null);
      return;
    }

    const lower = q.toLowerCase();
    const terms = lower.split(/\s+/).filter(Boolean);

    const scored: { paper: ProcessedPaper; score: number }[] = [];
    for (const paper of papers) {
      const title = paper.title.toLowerCase();
      let score = 0;
      for (const term of terms) {
        const idx = title.indexOf(term);
        if (idx === -1) { score = -1; break; }
        score += (idx === 0 ? 3 : 1) + (term.length / title.length) * 2;
      }
      if (score > 0) scored.push({ paper, score });
    }

    scored.sort((a, b) => b.score - a.score);
    const top    = scored.slice(0, 8).map(s => s.paper);
    const allIds = new Set(scored.map(s => s.paper.id));

    setResults(top);
    setOpen(top.length > 0);
    setActiveIdx(0);
    onResults(allIds.size > 0 ? allIds : null, top[0]);
  }, [papers, onResults]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setQuery(val);
    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => search(val), 120);
  };

  const handleSelect = (paper: ProcessedPaper) => {
    setQuery(paper.title);
    setOpen(false);
    onResults(new Set([paper.id]), paper);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!open) return;
    if (e.key === 'ArrowDown') { e.preventDefault(); setActiveIdx(i => Math.min(i + 1, results.length - 1)); }
    if (e.key === 'ArrowUp')   { e.preventDefault(); setActiveIdx(i => Math.max(i - 1, 0)); }
    if (e.key === 'Enter' && results[activeIdx]) handleSelect(results[activeIdx]);
    if (e.key === 'Escape') { setOpen(false); setQuery(''); onResults(null); }
  };

  const handleClear = () => {
    setQuery('');
    setResults([]);
    setOpen(false);
    onResults(null);
    inputRef.current?.focus();
  };

  const bg     = dm ? 'rgba(18,20,28,0.92)'    : 'rgba(255,255,255,0.92)';
  const border = dm ? 'rgba(255,255,255,0.10)'  : 'rgba(0,0,0,0.1)';
  const shadow = dm ? '0 4px 20px rgba(0,0,0,0.5)' : '0 4px 20px rgba(0,0,0,0.08)';
  const iconC  = dm ? '#4b5563' : '#94a3b8';
  const textC  = dm ? '#e2e8f0' : '#374151';
  const phC    = dm ? '#374151' : '#9ca3af';

  return (
    <div className="relative" style={{ width: 340 }}>
      {/* Input row */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '0 12px',
          background: bg,
          backdropFilter: 'blur(20px)',
          WebkitBackdropFilter: 'blur(20px)',
          border: `1px solid ${border}`,
          borderRadius: open ? '10px 10px 0 0' : 10,
          height: 42,
          boxShadow: shadow,
          transition: 'border-radius 0.1s, background 0.3s',
        }}
      >
        <svg width="16" height="16" fill="none" stroke={iconC} strokeWidth="2" viewBox="0 0 24 24" style={{ flexShrink: 0 }}>
          <circle cx="11" cy="11" r="8"/>
          <path d="m21 21-4.35-4.35"/>
        </svg>
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          onFocus={() => results.length > 0 && setOpen(true)}
          placeholder="Search papers by title…"
          disabled={disabled}
          style={{
            flex: 1,
            outline: 'none',
            background: 'transparent',
            border: 'none',
            fontSize: 14,
            fontFamily: "'Crimson Pro', Georgia, serif",
            color: textC,
            caretColor: dm ? '#6366f1' : '#4F6EF7',
          }}
        />
        {/* Placeholder colour via CSS custom property trick isn't possible inline,
            but we add a global rule via a <style> tag below */}
        {query && (
          <button
            onClick={handleClear}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              color: dm ? '#374151' : '#d1d5db',
              flexShrink: 0,
              padding: 0,
              display: 'flex',
              transition: 'color 0.15s',
            }}
          >
            <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path d="M18 6 6 18M6 6l12 12"/>
            </svg>
          </button>
        )}
      </div>

      {/* Dropdown */}
      {open && (
        <div
          style={{
            position: 'absolute',
            top: '100%',
            left: 0,
            right: 0,
            background: dm ? 'rgba(18,20,28,0.97)' : 'rgba(255,255,255,0.97)',
            backdropFilter: 'blur(20px)',
            WebkitBackdropFilter: 'blur(20px)',
            border: `1px solid ${border}`,
            borderTop: `1px solid ${dm ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.06)'}`,
            borderRadius: '0 0 10px 10px',
            boxShadow: dm ? '0 8px 32px rgba(0,0,0,0.6)' : '0 8px 32px rgba(0,0,0,0.1)',
            zIndex: 100,
            overflow: 'hidden',
          }}
        >
          {results.map((p, i) => (
            <button
              key={p.id}
              onClick={() => handleSelect(p)}
              style={{
                display: 'block',
                width: '100%',
                textAlign: 'left',
                padding: '10px 12px',
                background: i === activeIdx
                  ? dm ? 'rgba(99,102,241,0.12)' : 'rgba(79,110,247,0.06)'
                  : 'transparent',
                border: 'none',
                borderBottom: i < results.length - 1
                  ? `1px solid ${dm ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.04)'}`
                  : 'none',
                cursor: 'pointer',
                transition: 'background 0.1s',
              }}
            >
              <p
                style={{
                  fontFamily: "'Crimson Pro', Georgia, serif",
                  fontSize: 13,
                  color: dm ? '#e2e8f0' : '#374151',
                  margin: 0,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                {p.title}
              </p>
              {p.doi && p.doi !== 'null' && (
                <p
                  style={{
                    fontSize: 10,
                    color: dm ? '#4b5563' : '#9ca3af',
                    margin: '2px 0 0',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {p.doi}
                </p>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};