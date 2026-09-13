import React from 'react';

// Action -> tier-3 badge color. HOLD reads as neutral/muted (no action taken),
// HARVEST as blue (charging), PARTIAL_DEPLOY as amber (a measured spend),
// FULL_DEPLOY as red (the aggressive, highest-risk call).
const ACTION_BADGE_CLASS = {
  HARVEST: 'badge-blue',
  PARTIAL_DEPLOY: 'badge-amber',
  FULL_DEPLOY: 'badge-red',
  HOLD: 'badge-muted',
};

export function SignalTag({ action, label }) {
  const cls = ACTION_BADGE_CLASS[action] || 'badge-muted';
  return (
    <span className={`badge ${cls}`}>
      <span className="dot" />
      {label || action}
    </span>
  );
}

export function HBar({ value, colorVar = '--blue', max = 1 }) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100));
  return (
    <div className="hbar-track">
      <div className="hbar-fill" style={{ width: `${pct}%`, background: `var(${colorVar})` }} />
    </div>
  );
}

export function Stat({ label, value, unit }) {
  return (
    <div className="stat">
      <div className="label">{label}</div>
      <div className="value num">
        {value}
        {unit ? <span style={{ fontSize: '13px', color: 'var(--muted)', fontWeight: 500 }}> {unit}</span> : null}
      </div>
    </div>
  );
}

export function Panel({ title, right, children }) {
  return (
    <div className="glass panel">
      {title && (
        <div className="panel-head">
          <h3>{title}</h3>
          {right}
        </div>
      )}
      <div className="panel-body">{children}</div>
    </div>
  );
}

export function DemoBadge() {
  return (
    <span className="badge badge-outline" title="No backend reachable — showing seeded demo data">
      <span className="dot" />
      demo data
    </span>
  );
}

export function ErrorBanner({ error }) {
  if (!error) return null;
  return <div className="error-banner">Request failed — {error}</div>;
}

export function Loading({ label = 'Contacting ApexGuard API…' }) {
  return (
    <div>
      <div className="loader-line" />
      <div style={{ fontSize: '11px', color: 'var(--muted)', marginTop: '8px', fontFamily: "'JetBrains Mono', monospace" }}>{label}</div>
    </div>
  );
}
