// src/components/controls/MultiDocToggle.jsx
// Toggle for enabling/disabling query decomposition (multi-doc parallel retrieval)

export default function MultiDocToggle({ enabled, onChange, disabled }) {
  return (
    <button
      id="multi-doc-toggle"
      onClick={() => !disabled && onChange(!enabled)}
      disabled={disabled}
      title={
        enabled
          ? "Query Decomposition ON — multi-topic questions are split into parallel searches"
          : "Query Decomposition OFF — all questions use a single global search"
      }
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '6px',
        padding: '6px 13px',
        borderRadius: 'var(--radius-full)',
        border: `1px solid ${enabled ? 'var(--accent)' : 'var(--border)'}`,
        background: enabled ? 'var(--accent-dim)' : 'transparent',
        color: enabled ? 'var(--accent)' : 'var(--text-muted)',
        fontSize: '12px',
        fontWeight: 500,
        cursor: disabled ? 'not-allowed' : 'pointer',
        opacity: disabled ? 0.5 : 1,
        transition: 'all 0.2s ease',
        whiteSpace: 'nowrap',
        letterSpacing: '0.01em',
      }}
    >
      {/* Layers / multi-doc icon */}
      <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/>
        <path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/>
        <path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>
      </svg>
      Multi-Doc {enabled ? 'ON' : 'OFF'}
    </button>
  );
}
