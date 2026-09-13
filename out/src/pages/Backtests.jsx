import React, { useEffect, useState } from 'react';
import api from '../api';
import { Panel, ErrorBanner, Loading, DemoBadge } from '../components/Atoms';

export default function Backtests() {
  const [backtests, setBacktests] = useState([]);
  const [replays, setReplays] = useState([]);
  const [selected, setSelected] = useState(null);
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState(null);
  const [runReplayId, setRunReplayId] = useState('');
  const [running, setRunning] = useState(false);
  const [maxFrames, setMaxFrames] = useState(60);

  function refresh() {
    api.listBacktests().then((r) => setBacktests(r.backtests)).catch((e) => setError(e.message));
  }

  useEffect(() => {
    refresh();
    api.listReplays().then((r) => {
      setReplays(r.replays);
      if (r.replays.length) setRunReplayId(r.replays[0].id);
    }).catch((e) => setError(e.message));
  }, []);

  async function open(id) {
    setSelected(id);
    setDetail(null);
    try {
      const d = await api.getBacktest(id);
      setDetail(d);
    } catch (e) {
      setError(e.message);
    }
  }

  async function runNew() {
    setRunning(true);
    setError(null);
    try {
      const result = await api.runBacktest(runReplayId, maxFrames);
      refresh();
      setDetail(result);
      setSelected(`${runReplayId}_backtest (just run)`);
    } catch (e) {
      setError(e.message);
    } finally {
      setRunning(false);
    }
  }

  return (
    <div>
      <div className="page-head">
        <div className="page-title">Backtests</div>
        <div className="page-desc">Stored backtest runs against real OpenF1 replays, sortable, with a clearly separate action for running a new one.</div>
      </div>

      <ErrorBanner error={error} />

      <div className="glass panel" style={{ marginBottom: 14 }}>
        <div className="panel-head">Run new backtest</div>
        <div className="panel-body" style={{ display: 'flex', gap: 10, alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <div className="field" style={{ marginBottom: 0, minWidth: 260 }}>
            <label><span>Replay</span></label>
            <select value={runReplayId} onChange={(e) => setRunReplayId(e.target.value)}>
              {replays.map((r) => <option key={r.id} value={r.id}>{r.id}</option>)}
            </select>
          </div>
          <div className="field" style={{ marginBottom: 0, width: 140 }}>
            <label><span>Max frames</span><span className="val">{maxFrames}</span></label>
            <input type="range" min={10} max={200} step={10} value={maxFrames} onChange={(e) => setMaxFrames(parseInt(e.target.value, 10))} />
          </div>
          <button className="btn btn-primary" onClick={runNew} disabled={running || !runReplayId}>
            {running ? 'Running…' : 'Run new backtest'}
          </button>
        </div>
        {running && <div style={{ padding: '0 14px 14px' }}><Loading label="POST /v1/backtests/run in flight…" /></div>}
      </div>

      <div className="grid-3" style={{ gridTemplateColumns: '1.1fr 1fr' }}>
        <Panel title="Stored backtests">
          <table>
            <thead><tr><th>id</th><th>file</th></tr></thead>
            <tbody>
              {backtests.map((b) => (
                <tr key={b.id} onClick={() => open(b.id)} style={selected === b.id ? { background: 'rgba(61,191,201,0.06)' } : undefined}>
                  <td className="mono">{b.id}</td>
                  <td style={{ color: 'var(--paper-faint)' }}>{b.file}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>

        <Panel title={selected ? `Result — ${selected}` : 'Result'} right={detail?._mock ? <DemoBadge /> : undefined}>
          {!detail && <div className="empty-state">Select a stored backtest or run a new one.</div>}
          {detail && (
            <div>
              <div className="stat-row">
                <div className="stat"><div className="label">Frames evaluated</div><div className="value num">{detail.frames_evaluated}</div></div>
                <div className="stat"><div className="label">Rule compliance</div><div className="value num">{detail.metrics?.rule_compliance_pct}%</div></div>
                <div className="stat"><div className="label">Mean confidence</div><div className="value num">{detail.metrics?.mean_decision_confidence_pct}%</div></div>
                <div className="stat"><div className="label">Mean latency</div><div className="value num">{detail.metrics?.mean_latency_ms} ms</div></div>
              </div>
              <div style={{ marginTop: 12, fontSize: 11, color: 'var(--paper-faint)' }}>
                {detail.event} {detail.session} · {detail.driver} vs {detail.target_driver}
              </div>
              {detail.action_counts && (
                <div style={{ marginTop: 14 }}>
                  <div style={{ fontSize: 11, color: 'var(--paper-faint)', marginBottom: 6 }}>Action counts</div>
                  {Object.entries(detail.action_counts).map(([k, v]) => (
                    <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, padding: '3px 0' }}>
                      <span>{k}</span><span className="num">{v}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}
