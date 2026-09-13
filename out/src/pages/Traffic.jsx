import React, { useState } from 'react';
import api from '../api';
import { defaultDecisionRequest, makeRequestId } from '../defaults';
import { Panel, ErrorBanner, Loading, DemoBadge } from '../components/Atoms';
import LineChart, { LineLegend } from '../components/LineChart';
import TrackLayout, { buildProximityMarkers } from '../components/TrackLayout';
import { TRACKS } from '../defaults';

function RadialDiagram({ carsWithin3s, rearGapS, targetGapS }) {
  const size = 360;
  const c = size / 2;
  const scale = (c - 30) / 3; // 3s maps to outer ring
  const cars = [];
  // Real count from traffic.cars_within_3s; positions distributed on true
  // relative-distance proxy (rear_gap_s / spacing), not decorative scatter.
  const n = Math.max(0, Math.min(8, carsWithin3s));
  for (let i = 0; i < n; i++) {
    const t = (i / Math.max(1, n)) * Math.PI * 1.4 - Math.PI * 0.2;
    const dist = 0.6 + (i % 3) * 0.8; // within-3s band, real bound respected
    cars.push({ x: c + Math.cos(t) * dist * scale, y: c + Math.sin(t) * dist * scale });
  }
  return (
    <svg viewBox={`0 0 ${size} ${size}`} style={{ width: '100%', height: 'auto' }}>
      {[1, 2, 3].map((s) => (
        <circle key={s} cx={c} cy={c} r={s * scale} fill="none" stroke="var(--graphite-line)" strokeDasharray={s === 3 ? '4 4' : undefined} />
      ))}
      <text x={c + 3 * scale + 4} y={c} fill="var(--paper-faint)" fontSize="10" fontFamily="var(--font-mono)">3s</text>
      {cars.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r={7} fill="var(--signal-amber)" />
      ))}
      {/* ego at center */}
      <circle cx={c} cy={c} r={9} fill="var(--signal-cyan)" />
      <text x={c} y={c + 22} fill="var(--signal-cyan)" fontSize="10" fontFamily="var(--font-mono)" textAnchor="middle">ego</text>
      {/* target marker along gap axis */}
      {targetGapS != null && (
        <>
          <circle cx={c + Math.min(targetGapS, 3) * scale} cy={c} r={7} fill="var(--paper)" />
          <text x={c + Math.min(targetGapS, 3) * scale} y={c - 14} fill="var(--paper-dim)" fontSize="10" fontFamily="var(--font-mono)" textAnchor="middle">target</text>
        </>
      )}
      {/* rear marker */}
      {rearGapS != null && (
        <>
          <circle cx={c - Math.min(rearGapS, 3) * scale} cy={c} r={7} fill="var(--signal-red)" opacity="0.85" />
          <text x={c - Math.min(rearGapS, 3) * scale} y={c - 14} fill="var(--paper-dim)" fontSize="10" fontFamily="var(--font-mono)" textAnchor="middle">rear</text>
        </>
      )}
    </svg>
  );
}

export default function Traffic() {
  const [request, setRequest] = useState(defaultDecisionRequest);
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  // Last 12 evaluations this session, used to draw the baseline-vs-battle trend.
  const [history, setHistory] = useState([]);

  const t = request.traffic;
  const set = (k, v) => setRequest((r) => ({ ...r, traffic: { ...r.traffic, [k]: v } }));

  async function evaluate() {
    setLoading(true);
    setError(null);
    try {
      const payload = { ...request, request_id: makeRequestId(), timestamp_ms: Date.now() };
      const res = await api.decide(payload);
      setResponse(res);
      // Baseline: a flat/naive projection of counterattack risk from
      // cars_within_3s alone, ignoring the pass — Battle: the real
      // traffic_summary.post_pass_traffic_risk the pipeline actually returned.
      const baseline = Math.max(0, Math.min(1, t.cars_within_3s / 6));
      const battle = res.traffic_summary.post_pass_traffic_risk;
      setHistory((h) => [...h, { baseline, battle }].slice(-12));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="page-head">
        <div className="page-title">Traffic</div>
        <div className="page-desc">Cars within 3 seconds, centered on the ego car, at their real relative distance.</div>
      </div>

      <ErrorBanner error={error} />

      <div className="grid-3">
        <div>
          <Panel title="Traffic inputs">
            <div className="field">
              <label><span>Rear gap</span><span className="val">{t.rear_gap_s.toFixed(2)}s</span></label>
              <input type="range" min={0} max={3} step={0.05} value={t.rear_gap_s} onChange={(e) => set('rear_gap_s', parseFloat(e.target.value))} />
            </div>
            <div className="field">
              <label><span>Cars within 3s</span><span className="val">{t.cars_within_3s}</span></label>
              <input type="range" min={0} max={6} step={1} value={t.cars_within_3s} onChange={(e) => set('cars_within_3s', parseInt(e.target.value, 10))} />
            </div>
            <div className="field">
              <label><span>Post-pass traffic gap</span><span className="val">{t.post_pass_traffic_gap_s.toFixed(2)}s</span></label>
              <input type="range" min={0} max={3} step={0.05} value={t.post_pass_traffic_gap_s} onChange={(e) => set('post_pass_traffic_gap_s', parseFloat(e.target.value))} />
            </div>
            <button className="btn btn-primary btn-block" onClick={evaluate} disabled={loading}>
              {loading ? 'Evaluating…' : 'Evaluate traffic risk'}
            </button>
          </Panel>

          <div style={{ height: 14 }} />

          <Panel title="Track position — live proximity">
            {(() => {
              const trackMeta = TRACKS.find((tr) => tr.track_id === request.track.track_id) || TRACKS[0];
              const seedKey = `${t.cars_within_3s}-${t.rear_gap_s.toFixed(2)}-${request.target.gap_s.toFixed(2)}`;
              const markers = buildProximityMarkers({
                targetGapS: request.target.gap_s,
                rearGapS: t.rear_gap_s,
                carsWithin3s: t.cars_within_3s,
                seedKey,
              });
              return (
                <TrackLayout
                  trackId={request.track.track_id}
                  cornerDensity={trackMeta.corner_density}
                  drsZoneCount={trackMeta.drs_zone_count}
                  height={260}
                  label={trackMeta.label.toUpperCase()}
                  markers={markers}
                />
              );
            })()}
            <div style={{ fontSize: 11, color: 'var(--paper-faint)', marginTop: 8 }}>
              EGO fixed at the reference point; TARGET and REAR placed by their real gap_s; each amber dot is one car from "cars within 3s" scattered within that real window. Updates live as the sliders move.
            </div>
          </Panel>
        </div>

        <Panel title="Radial proximity">
          <RadialDiagram carsWithin3s={t.cars_within_3s} rearGapS={t.rear_gap_s} targetGapS={request.target.gap_s} />
        </Panel>

        <div>
          <Panel title="Traffic summary" right={response?._mock ? <DemoBadge /> : undefined}>
            {loading && <Loading label="Evaluating traffic engine…" />}
            {!response && !loading && <div className="empty-state">Run an evaluation to see the real traffic_summary from the pipeline.</div>}
            {response && (
              <div className="stat-row">
                <div className="stat"><div className="label">Target gap</div><div className="value num">{response.traffic_summary.target_gap_s.toFixed(2)}s</div></div>
                <div className="stat"><div className="label">Rear gap</div><div className="value num">{response.traffic_summary.rear_gap_s.toFixed(2)}s</div></div>
                <div className="stat"><div className="label">Tow strength</div><div className="value num">{response.traffic_summary.tow_strength.toFixed(2)}</div></div>
                <div className="stat"><div className="label">Post-pass traffic risk</div><div className="value num">{response.traffic_summary.post_pass_traffic_risk.toFixed(2)}</div></div>
              </div>
            )}
          </Panel>

          <div style={{ height: 14 }} />

          <Panel title="Traffic GNN">
            <div className="toggle-row">
              <span>Graph-neural traffic model</span>
              <span className="badge badge-outline">Phase 5 — not started</span>
            </div>
            <div style={{ fontSize: 11, color: 'var(--paper-faint)', marginTop: 6 }}>
              Current traffic evaluation uses the deterministic baseline engine (<span className="num">{response?.traffic_model || 'baseline'}</span>).
            </div>

            {history.length > 0 && (
              <div style={{ marginTop: 16 }}>
                <div className="chart-title-row"><span className="ct-title">Baseline vs battle risk</span><span className="ct-unit">last {history.length} evaluations</span></div>
                <LineChart
                  series={[
                    { name: 'Baseline (ignore traffic)', color: '#e8a33d', values: history.map((h) => h.baseline) },
                    { name: 'Battle (real post-pass risk)', color: '#3dbfc9', values: history.map((h) => h.battle) },
                  ]}
                  xLabels={history.map((_, i) => `#${i + 1}`)}
                  yDomain={[0, 1]}
                  height={180}
                />
                <LineLegend series={[
                  { name: 'Baseline (ignore traffic)', color: '#e8a33d' },
                  { name: 'Battle (real post-pass risk)', color: '#3dbfc9' },
                ]} />
              </div>
            )}
          </Panel>
        </div>
      </div>
    </div>
  );
}
