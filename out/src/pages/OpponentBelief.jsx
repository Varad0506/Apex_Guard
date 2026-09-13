import React, { useState } from 'react';
import api from '../api';
import { defaultDecisionRequest, makeRequestId } from '../defaults';
import TelemetryForm from '../components/TelemetryForm';
import { Panel, ErrorBanner, Loading, HBar } from '../components/Atoms';

const TACTICAL_LABELS = { ATTACKING: 'Attacking', DEFENDING: 'Defending', HARVESTING: 'Harvesting', CONSERVING: 'Conserving' };

function cyanAlpha(p) {
  return `rgba(61,191,201,${0.08 + p * 0.82})`;
}

export default function OpponentBelief() {
  const [request, setRequest] = useState(() => ({ ...defaultDecisionRequest(), battle_id: 'belief-session-1' }));
  const [belief, setBelief] = useState(null);
  const [history, setHistory] = useState([]); // real tactical_belief per tick, this battle
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function tick() {
    setLoading(true);
    setError(null);
    try {
      const payload = { ...request, request_id: makeRequestId(), timestamp_ms: Date.now() };
      const res = await api.decide(payload);
      setBelief(res.opponent_belief);
      setHistory((h) => [...h, res.opponent_belief.tactical_belief].slice(-16));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  function resetBattle() {
    setHistory([]);
    setBelief(null);
    setRequest((r) => ({ ...r, battle_id: `belief-session-${Date.now()}` }));
  }

  const states = Object.keys(TACTICAL_LABELS);

  return (
    <div>
      <div className="page-head">
        <div className="page-title">Opponent Belief</div>
        <div className="page-desc">
          A hidden-state HMM over the rival's tactical intent, updated one telemetry tick at a time within
          a battle. Adjust the target's gap and sector delta, then advance the tick to watch the posterior move.
        </div>
      </div>

      <ErrorBanner error={error} />

      <div className="grid-3">
        <Panel title="Inputs (tick source)">
          <TelemetryForm request={request} onChange={setRequest} />
        </Panel>

        <div>
          <Panel title="Tactical belief — 4 states" right={<button className="btn" onClick={resetBattle}>Reset battle</button>}>
            {!belief && !loading && <div className="empty-state">Advance a tick to infer the rival's current tactical state.</div>}
            {loading && <Loading label="Updating HMM belief…" />}
            {belief && states.map((k) => (
              <div className="field" key={k}>
                <label><span>{TACTICAL_LABELS[k]}</span><span className="val">{(belief.tactical_belief[k] * 100).toFixed(1)}%</span></label>
                <HBar value={belief.tactical_belief[k]} colorVar="--signal-cyan" />
              </div>
            ))}
            <button className="btn btn-primary btn-block" onClick={tick} disabled={loading} style={{ marginTop: 10 }}>
              Advance tick
            </button>
          </Panel>

          <div style={{ height: 14 }} />

          {belief && (
            <Panel title="Other hidden-state beliefs">
              <div className="stat-row">
                <div className="stat"><div className="label">Override available</div><div className="value num">{(belief.override_belief.AVAILABLE * 100).toFixed(0)}%</div></div>
                <div className="stat"><div className="label">Low-drag aero</div><div className="value num">{(belief.aero_belief.LOW_DRAG * 100).toFixed(0)}%</div></div>
                <div className="stat"><div className="label">Counter-harvest trap</div><div className="value num">{(belief.counter_harvest_trap_probability * 100).toFixed(0)}%</div></div>
                <div className="stat"><div className="label">Filter confidence</div><div className="value num">{(belief.confidence * 100).toFixed(0)}%</div></div>
              </div>
              <div style={{ marginTop: 10, fontSize: 11, color: 'var(--paper-faint)' }}>
                temporal: {String(belief.temporal)} · updates: {belief.update_count} · model: {belief.racer_pattern?.model}
              </div>
            </Panel>
          )}
        </div>

        <Panel title="Belief over time (this battle)">
          {history.length === 0 && <div className="empty-state">No ticks yet — the heatmap fills in as the belief updates.</div>}
          {history.length > 0 && (
            <div style={{ overflowX: 'auto' }}>
              <table>
                <thead>
                  <tr>
                    <th>State</th>
                    {history.map((_, i) => <th key={i} className="num">t{i + 1}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {states.map((k) => (
                    <tr key={k}>
                      <td>{TACTICAL_LABELS[k]}</td>
                      {history.map((h, i) => (
                        <td key={i} className="num" style={{ background: cyanAlpha(h[k]), textAlign: 'center' }}>
                          {(h[k] * 100).toFixed(0)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}
