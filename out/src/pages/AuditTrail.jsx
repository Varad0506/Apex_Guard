import React, { useState } from 'react';
import { useSession } from '../store';
import { Panel, SignalTag, ErrorBanner } from '../components/Atoms';

export default function AuditTrail() {
  const { decisions } = useSession();
  const [expanded, setExpanded] = useState(null);
  const [filter, setFilter] = useState('ALL');
  const [error] = useState(null);

  const actions = ['ALL', 'HARVEST', 'HOLD', 'PARTIAL_DEPLOY', 'FULL_DEPLOY'];
  const rows = decisions.filter((d) => filter === 'ALL' || d.response.recommended_action === filter);

  return (
    <div>
      <div className="page-head">
        <div className="page-title">Audit Trail</div>
        <div className="page-desc">
          Every decision this session, most recent first. Click a row to see the exact request/response
          pair — the backend has no list endpoint, so this trail is the session's own client-tracked
          history; audit_id lookups still hit the real <code>GET /v1/audits/&#123;id&#125;</code>.
        </div>
      </div>

      <ErrorBanner error={error} />

      <Panel
        title="Decisions"
        right={
          <select value={filter} onChange={(e) => setFilter(e.target.value)} style={{ width: 160 }}>
            {actions.map((a) => <option key={a} value={a}>{a}</option>)}
          </select>
        }
      >
        {rows.length === 0 && <div className="empty-state">No decisions logged yet this session — run one from the Decision Engine.</div>}
        <table>
          <thead>
            <tr><th>time</th><th>action</th><th>confidence</th><th>audit_id</th></tr>
          </thead>
          <tbody>
            {rows.map((d, i) => (
              <React.Fragment key={d.response.audit_id + i}>
                <tr onClick={() => setExpanded(expanded === i ? null : i)}>
                  <td className="num">{new Date(d.at).toLocaleTimeString()}</td>
                  <td><SignalTag action={d.response.recommended_action} /></td>
                  <td className="num">{(d.response.recommendation_confidence * 100).toFixed(0)}%</td>
                  <td className="mono" style={{ color: 'var(--paper-faint)' }}>{d.response.audit_id}</td>
                </tr>
                {expanded === i && (
                  <tr>
                    <td colSpan={4} style={{ cursor: 'default', background: 'var(--graphite-base)' }}>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, padding: '10px 0' }}>
                        <div>
                          <div style={{ fontSize: 11, color: 'var(--paper-faint)', marginBottom: 6 }}>request</div>
                          <pre style={{ fontSize: 11, whiteSpace: 'pre-wrap', margin: 0, color: 'var(--paper-dim)' }}>
                            {JSON.stringify(d.request, null, 2)}
                          </pre>
                        </div>
                        <div>
                          <div style={{ fontSize: 11, color: 'var(--paper-faint)', marginBottom: 6 }}>response</div>
                          <pre style={{ fontSize: 11, whiteSpace: 'pre-wrap', margin: 0, color: 'var(--paper-dim)' }}>
                            {JSON.stringify(d.response, null, 2)}
                          </pre>
                        </div>
                      </div>
                    </td>
                  </tr>
                )}
              </React.Fragment>
            ))}
          </tbody>
        </table>
      </Panel>
    </div>
  );
}
