import React from 'react';
import { Link } from 'react-router-dom';
import { useSession } from '../store';
import { SignalTag } from '../components/Atoms';
import Pipeline from '../components/Pipeline';

export default function Overview() {
  const { lastDecision, sessionStats } = useSession();

  return (
    <div>
      <div className="page-head">
        <div className="page-title">ApexGuard</div>
        <div className="page-desc">
          Real-time verification-gated energy strategy: propose an overtake action, verify it against
          the rules envelope, act — and prove the reasoning before it's trusted.
        </div>
      </div>

      <div className="panel" style={{ marginBottom: 16 }}>
        <div className="panel-head">
          <span>Live pipeline</span>
          {lastDecision && <SignalTag action={lastDecision.response.recommended_action} />}
        </div>
        <div className="panel-body">
          {lastDecision ? (
            <>
              <Pipeline latencyMs={lastDecision.response.latency_ms} active={4} />
              <div className="explanation" style={{ marginTop: 0 }}>{lastDecision.response.explanation}</div>
            </>
          ) : (
            <div className="empty-state">
              No decisions run yet this session — <Link to="/decision" style={{ color: 'var(--signal-cyan)' }}>try a scenario</Link>.
            </div>
          )}
        </div>
      </div>

      <div className="kpi-grid">
        <div className="panel panel-body">
          <div className="stat">
            <div className="label">Decisions served this session</div>
            <div className="value num">{sessionStats.decisionsServed}</div>
          </div>
        </div>
        <div className="panel panel-body">
          <div className="stat">
            <div className="label">Mean verified latency</div>
            <div className="value num">
              {sessionStats.meanLatencyMs != null ? `${sessionStats.meanLatencyMs.toFixed(2)} ms` : '—'}
            </div>
          </div>
        </div>
        <div className="panel panel-body">
          <div className="stat">
            <div className="label">Current phase</div>
            <div className="value num">{sessionStats.lastPhase || '—'}</div>
          </div>
        </div>
      </div>
    </div>
  );
}
