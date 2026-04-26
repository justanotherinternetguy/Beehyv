import React from 'react';

export interface TabDef {
  id: string;
  label: string;
  closeable?: boolean;
  status?: 'running' | 'done' | 'error';
}

interface TabBarProps {
  tabs: TabDef[];
  activeTabId: string;
  onTabChange: (id: string) => void;
  onTabClose: (id: string) => void;
  darkMode: boolean;
}

const STATUS_COLOR: Record<string, string> = {
  running: '#22d3ee',
  done:    '#4ade80',
  error:   '#f87171',
};

export const TabBar: React.FC<TabBarProps> = ({
  tabs,
  activeTabId,
  onTabChange,
  onTabClose,
  darkMode: dm,
}) => (
  <div
    style={{
      display:         'flex',
      alignItems:      'stretch',
      height:          38,
      flexShrink:      0,
      background:      dm ? '#0a0c10' : '#e8ecf5',
      borderBottom:    `1px solid ${dm ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.12)'}`,
      overflowX:       'auto',
      overflowY:       'hidden',
      zIndex:          1000,
      userSelect:      'none',
    }}
  >
    {tabs.map(tab => {
      const active = tab.id === activeTabId;
      const dotColor = tab.status ? STATUS_COLOR[tab.status] : undefined;

      return (
        <div
          key={tab.id}
          onClick={() => onTabChange(tab.id)}
          style={{
            display:       'flex',
            alignItems:    'center',
            gap:           6,
            padding:       '0 14px 0 12px',
            cursor:        'pointer',
            flexShrink:    0,
            maxWidth:      240,
            minWidth:      100,
            background:    active
              ? (dm ? '#161820' : '#ffffff')
              : 'transparent',
            borderRight:   `1px solid ${dm ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.09)'}`,
            borderBottom:  active
              ? `2px solid ${dm ? '#6366f1' : '#4F6EF7'}`
              : '2px solid transparent',
            transition:    'background 0.12s',
          }}
        >
          {dotColor && (
            <div
              style={{
                width:        6,
                height:       6,
                borderRadius: '50%',
                background:   dotColor,
                flexShrink:   0,
                boxShadow:    `0 0 6px ${dotColor}`,
                animation:    tab.status === 'running' ? 'tbPulse 1.5s ease-in-out infinite' : undefined,
              }}
            />
          )}

          <span
            style={{
              fontFamily:    "'JetBrains Mono', monospace",
              fontSize:      11,
              color:         active
                ? (dm ? '#e2e8f0' : '#1e293b')
                : (dm ? '#6b7280' : '#9ca3af'),
              whiteSpace:    'nowrap',
              overflow:      'hidden',
              textOverflow:  'ellipsis',
              flex:          1,
            }}
          >
            {tab.label}
          </span>

          {tab.closeable && (
            <button
              onClick={e => { e.stopPropagation(); onTabClose(tab.id); }}
              style={{
                fontSize:   13,
                lineHeight: 1,
                color:      dm ? '#4b5563' : '#94a3b8',
                background: 'none',
                border:     'none',
                cursor:     'pointer',
                padding:    '0 2px',
                flexShrink: 0,
              }}
              title="Close tab"
            >
              ×
            </button>
          )}
        </div>
      );
    })}

    <style>{`
      @keyframes tbPulse {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.3; }
      }
    `}</style>
  </div>
);
