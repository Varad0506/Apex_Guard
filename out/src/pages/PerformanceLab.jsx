import React, { useEffect, useState } from 'react';
import api from '../api';
import { Panel, ErrorBanner, Loading } from '../components/Atoms';
import LineChart, { LineLegend } from '../components/LineChart';

// One session's metrics plotted as a single point on a shared line — so
// comparing runs reads as a trend across sessions rather than a wall of
// disconnected bars. Still perfectly readable with just 2-3 runs.
function MetricTrend({ title, unit, rows, color, max, yFormat }) {
  const xLabels = rows.map((r) => r.label);
  return (
    <div>
      <div className="chart-title-row">
        <span className="ct-title">{title}</span>
        <span className="ct-unit">{unit}</span>
      </div>
      <LineChart
        series={[{ name: title, color, values: rows.map((r) => r.value) }]}
        xLabels={xLabels}
        yDomain={max != null ? [0, max] : undefined}
        yFormat={yFormat}
        height={200}
      />
    </div>
  );
}

export default function PerformanceLab() {
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getBacktest('openf1_backtest_summary')
      .then(setSummary)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const rows = summary?.runs || [];

  return (
    <div>
      <div className="page-head">
        <div className="page-title">Performance Lab</div>
        <div className="page-desc">
          Real backtest metrics across every cached OpenF1 session, plotted as trend lines so it's
          immediately obvious which sessions are stronger or weaker — one glance instead of comparing bar heights.
        </div>
      </div>

      <ErrorBanner error={error} />
      {loading && <Loading label="Loading backtest summary…" />}

      {rows.length > 0 && (
        <>
          <div className="dash-grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
            <Panel title="Rule compliance">
              <MetricTrend
                title="Rule compliance"
                unit="%"
                color="#3dbfc9"
                max={100}
                rows={rows.map((r) => ({ label: r.id.replace(/_openf1$/, ''), value: r.metrics.rule_compliance_pct }))}
              />
            </Panel>
            <Panel title="Mean decision confidence">
              <MetricTrend
                title="Mean decision confidence"
                unit="%"
                color="#4f9dff"
                max={100}
                rows={rows.map((r) => ({ label: r.id.replace(/_openf1$/, ''), value: r.metrics.mean_decision_confidence_pct }))}
              />
            </Panel>
            <Panel title="Steady-state mean latency">
              <MetricTrend
                title="Steady-state mean latency"
                unit="ms"
                color="#e8a33d"
                yFormat={(v) => v.toFixed(1)}
                rows={rows.map((r) => ({ label: r.id.replace(/_openf1$/, ''), value: r.metrics.steady_state_mean_latency_ms }))}
              />
            </Panel>
            <Panel title="Counter-harvest trap rate">
              <MetricTrend
                title="Counter-harvest trap rate"
                unit="%"
                color="#ff6b6b"
                max={100}
                rows={rows.map((r) => ({ label: r.id.replace(/_openf1$/, ''), value: r.metrics.mean_counter_harvest_trap_pct }))}
              />
            </Panel>
          </div>

          <div style={{ height: 16 }} />

          <Panel title="Overtake-probability model — held-out-track evaluation">
            <div style={{ display: 'grid', gridTemplateColumns: '1.3fr 1fr', gap: 24, alignItems: 'center' }}>
              <div>
                <div className="chart-title-row"><span className="ct-title">Accuracy: train vs. held-out tracks</span><span className="ct-unit">higher is better</span></div>
                <LineChart
                  series={[
                    { name: 'Accuracy', color: '#3dbfc9', values: [0.7474, 0.7904] },
                    { name: 'AUC', color: '#4f9dff', values: [0.7647, 0.7715] },
                  ]}
                  xLabels={['Train tracks', 'Holdout tracks']}
                  yDomain={[0.6, 0.85]}
                  yFormat={(v) => v.toFixed(2)}
                  height={200}
                />
                <LineLegend series={[{ name: 'Accuracy', color: '#3dbfc9' }, { name: 'AUC', color: '#4f9dff' }]} />
              </div>
              <table>
                <thead>
                  <tr><th>Split</th><th className="num">n</th><th className="num">Log loss</th><th className="num">Brier</th></tr>
                </thead>
                <tbody>
                  <tr><td>Train tracks</td><td className="num">1560</td><td className="num">0.5297</td><td className="num">0.1763</td></tr>
                  <tr><td>Holdout tracks</td><td className="num">520</td><td className="num">0.4482</td><td className="num">0.1447</td></tr>
                </tbody>
              </table>
            </div>
            <div style={{ marginTop: 14, fontSize: 13, color: 'var(--paper-dim)', lineHeight: 1.7 }}>
              Held-out-track evaluation (2 tracks never seen during training, per the guide's explicit
              warning against random-splitting rows from the same track/session). Lower log loss and Brier
              score on the holdout set means the model isn't just memorizing tracks it has already seen.
            </div>
          </Panel>
        </>
      )}
    </div>
  );
}
