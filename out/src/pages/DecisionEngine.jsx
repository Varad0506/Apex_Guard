import React, { useState } from 'react';
import api from '../api';
import { useSession } from '../store';
import { defaultDecisionRequest, makeRequestId } from '../defaults';
import TelemetryForm from '../components/TelemetryForm';
import Pipeline from '../components/Pipeline';
import { SignalTag, HBar, Panel, ErrorBanner, Loading } from '../components/Atoms';

export default function DecisionEngine() {
  const [request, setRequest] = useState(defaultDecisionRequest);
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [pipelineStage, setPipelineStage] = useState(0);
  const { recordDecision } = useSession();

  async function runDecision() {
    setLoading(true);
    setError(null);
    setPipelineStage(0);
    const payload = { ...request, request_id: makeRequestId(), timestamp_ms: Date.now() };
    try {
      const res = await api.decide(payload);
      setResponse(res);
      recordDecision(payload, res);
      // Sequence the pipeline lights across the real latency window, then
      // land in the final lit state.
      const total = Math.max(120, res.latency_ms * 8); // stretch for visibility
      const step = total / 4;
      for (let i = 1; i <= 4; i++) {
        await new Promise((r) => setTimeout(r, step));
        setPipelineStage(i);
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const rejected = response && !response.rule_compliant;

  return (
    <div>
      <div className="page-head">
        <div className="page-title">Decision Engine</div>
        <div className="page-desc">
          Propose an action, verify it against the reserve/rules envelope, then act — one telemetry
          snapshot at a time. Every field below maps directly to a <code>DecisionRequest</code> field.
        </div>
      </div>

      <ErrorBanner error={error} />

      <div className="grid-3">
        <Panel title="Inputs">
          <TelemetryForm request={request} onChange={setRequest} />
        </Panel>

        <div>
          <Panel title="Pipeline" right={loading ? <span className="num" style={{ fontSize: 11 }}>running…</span> : null}>
            <Pipeline latencyMs={response?.latency_ms} active={pipelineStage} />
            <button className="btn btn-primary btn-block" onClick={runDecision} disabled={loading}>
              {loading ? 'Deciding…' : 'Run decision'}
            </button>
          </Panel>

          <div style={{ height: 14 }} />

          {response ? (
            <Panel title="Candidates considered">
              {response.candidates.map((c) => (
                <div className="candidate-row" key={c.action}>
                  <SignalTag action={c.action} />
                  <span className="num" style={{ color: 'var(--paper-dim)' }}>
                    {(c.pass_probability * 100).toFixed(0)}%
                  </span>
                  <span style={{ color: 'var(--paper-dim)', fontSize: 12 }}>{c.reason}</span>
                  <span className="badge" style={{ justifySelf: 'end' }}>{c.status}</span>
                </div>
              ))}
            </Panel>
          ) : (
            <Panel title="Candidates considered">
              <div className="empty-state">Run a decision to see every candidate action the verifier ranked.</div>
            </Panel>
          )}
        </div>

        <Panel title="Result">
          {loading && !response && <Loading label="Running propose → verify → act…" />}
          {!loading && !response && (
            <div className="empty-state">No decision run yet this session — adjust inputs and run one.</div>
          )}
          {response && (
            <div>
              <div style={{ fontSize: 29, marginBottom: 8 }}>
                <SignalTag action={response.recommended_action} />
              </div>
              <div className={`explanation${rejected ? ' rejected' : ''}`}>
                {response.explanation}
              </div>

              <div className="field">
                <label><span>Recommendation confidence</span><span className="val">{(response.recommendation_confidence * 100).toFixed(0)}%</span></label>
                <HBar value={response.recommendation_confidence} colorVar="--signal-cyan" />
              </div>
              <div className="field">
                <label><span>Overtake success probability</span><span className="val">{(response.overtake_success_probability * 100).toFixed(0)}%</span></label>
                <HBar value={response.overtake_success_probability} colorVar="--signal-cyan" />
              </div>
              <div className="field">
                <label><span>Counterattack risk</span><span className="val">{(response.counterattack_risk * 100).toFixed(0)}%</span></label>
                <HBar value={response.counterattack_risk} colorVar="--signal-amber" />
              </div>

              <div className="stat-row" style={{ marginTop: 14 }}>
                <div className="stat"><div className="label">Risk level</div><div className="value num">{response.risk_level}</div></div>
                <div className="stat"><div className="label">Rule compliant</div><div className="value num">{response.rule_compliant ? 'YES' : 'NO'}</div></div>
                <div className="stat"><div className="label">Fallback used</div><div className="value num">{response.fallback_used ? 'YES' : 'NO'}</div></div>
                <div className="stat"><div className="label">Latency</div><div className="value num">{response.latency_ms.toFixed(2)} ms</div></div>
              </div>

              {response.policy_proposal && response.policy_proposal !== response.recommended_action && (
                <div style={{ marginTop: 12, fontSize: 12, color: 'var(--paper-dim)' }}>
                  PPO proposed <strong style={{ color: 'var(--paper)' }}>{response.policy_proposal}</strong>, verifier
                  overrode to <strong style={{ color: 'var(--paper)' }}>{response.recommended_action}</strong>.
                </div>
              )}

              <div style={{ marginTop: 14, fontSize: 11, color: 'var(--paper-faint)' }}>
                audit_id: {response.audit_id} · status: {response.status} · traffic model: {response.traffic_model}
              </div>
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}
