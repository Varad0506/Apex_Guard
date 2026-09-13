import React, { useEffect, useState } from 'react';
import api from '../api';
import { Panel, ErrorBanner, Loading, DemoBadge } from '../components/Atoms';

function StateBadge({ state }) {
  const isActive = state !== 'not started' && state !== false;
  if (!isActive) return <span className="badge badge-outline">not started</span>;
  const cls = state === 'trained-v1' || state === true ? 'badge-green' : 'badge-amber';
  return <span className={`badge ${cls}`}><span className="dot" />{String(state)}</span>;
}

export default function SystemHealth() {
  const [ready, setReady] = useState(null);
  const [live, setLive] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  function refresh() {
    setLoading(true);
    Promise.all([api.health(), api.healthReady()])
      .then(([h, r]) => { setLive(h); setReady(r); })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }

  useEffect(() => { refresh(); }, []);

  return (
    <div>
      <div className="page-head">
        <div className="page-title">System Health</div>
        <div className="page-desc">Real component states from /v1/health and /v1/health/ready — three states only.</div>
      </div>

      <ErrorBanner error={error} />
      {loading && <Loading label="Checking readiness…" />}

      {ready && (
        <Panel title="Components" right={<div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>{(ready?._mock || live?._mock) && <DemoBadge />}<button className="btn" onClick={refresh}>Refresh</button></div>}>
          <table>
            <thead><tr><th>Component</th><th>State</th></tr></thead>
            <tbody>
              <tr><td>Process liveness</td><td><StateBadge state={live?.status === 'ok' ? 'ok' : 'not started'} /></td></tr>
              <tr><td>Overtake probability model</td><td><StateBadge state={ready.overtake_model} /></td></tr>
              <tr><td>PPO policy</td><td><StateBadge state={ready.policy_available} /></td></tr>
              <tr><td>Traffic engine</td><td><StateBadge state={ready.traffic_engine} /></td></tr>
            </tbody>
          </table>
          <div style={{ marginTop: 14, fontSize: 11, color: 'var(--paper-faint)' }}>
            readiness status: {ready.status}
          </div>
        </Panel>
      )}
    </div>
  );
}
