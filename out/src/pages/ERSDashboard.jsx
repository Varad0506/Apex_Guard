import React, { useMemo } from 'react';
import { useSession } from '../store';
import { Panel, Stat, SignalTag } from '../components/Atoms';
import LineChart, { LineLegend } from '../components/LineChart';
import { mulberry32, seedFromString } from '../mock';

const N = 20;

// Deterministic demo trace shown before any real decisions have been made
// this session, so the dashboard is never a blank page on first load.
function buildDemoTrace() {
  const rand = mulberry32(seedFromString('ers-dashboard-demo'));
  const soc = [];
  const budget = [];
  let socV = 78;
  let budgetV = 2000;
  for (let i = 0; i < N; i++) {
    socV = Math.max(16, socV - (rand() * 2.4 - 0.4));
    budgetV = Math.max(100, budgetV - rand() * 90);
    soc.push(+socV.toFixed(1));
    budget.push(Math.round(budgetV));
  }
  return { soc, budget };
}

export default function ERSDashboard() {
  const { decisions, lastDecision, sessionStats } = useSession();

  const chrono = useMemo(() => [...decisions].reverse(), [decisions]);
  const hasLive = chrono.length > 0;
  const demo = useMemo(() => buildDemoTrace(), []);

  const soc = hasLive ? chrono.map((d) => d.request?.ego?.soc_pct ?? 0) : demo.soc;
  const budget = hasLive
    ? chrono.map((d) => d.request?.rules?.deployment_budget_remaining_kj ?? 0)
    : demo.budget;
  const xLabels = hasLive
    ? chrono.map((_, i) => `#${i + 1}`)
    : Array.from({ length: N }, (_, i) => `${i + 1}`);

  const actionCounts = { HARVEST: 0, PARTIAL_DEPLOY: 0, FULL_DEPLOY: 0, HOLD: 0 };
  chrono.forEach((d) => {
    const a = d.response?.recommended_action;
    if (a && actionCounts[a] != null) actionCounts[a] += 1;
  });
  const totalActions = Object.values(actionCounts).reduce((a, b) => a + b, 0);

  const latestSoc = soc[soc.length - 1];
  const latestBudget = budget[budget.length - 1];

  return (
    <div>
      <div className="page-head">
        <div className="page-title">ERS Dashboard</div>
        <div className="page-desc">
          Energy Recovery System — battery state of charge, deployment budget, and harvest/deploy
          decisions, isolated from powertrain and chassis metrics so the energy story reads on its own.
        </div>
      </div>

      <div className="dash-kpis">
        <div className="panel panel-body">
          <Stat label="Current SOC" value={latestSoc != null ? latestSoc.toFixed(1) : '—'} unit="%" />
        </div>
        <div className="panel panel-body">
          <Stat label="Deployment budget left" value={latestBudget != null ? Math.round(latestBudget) : '—'} unit="kJ" />
        </div>
        <div className="panel panel-body">
          <Stat label="Decisions this session" value={sessionStats.decisionsServed} />
        </div>
        <div className="panel panel-body">
          <div className="stat">
            <div className="label">Last recommended action</div>
            <div style={{ marginTop: 8 }}>
              {lastDecision ? <SignalTag action={lastDecision.response.recommended_action} /> : <span className="value num">—</span>}
            </div>
          </div>
        </div>
      </div>

      <div className="dash-grid">
        <Panel title="State of charge over time">
          <div className="chart-title-row"><span className="ct-title">Battery SOC</span><span className="ct-unit">%</span></div>
          <LineChart series={[{ name: 'SOC %', color: '#3dbfc9', values: soc }]} xLabels={xLabels} yDomain={[10, 100]} height={260} />
          {!hasLive && <div style={{ fontSize: 12.5, color: 'var(--paper-faint)', marginTop: 8 }}>Demo trace — run a decision to see live session data.</div>}
        </Panel>

        <div className="dash-side">
          <Panel title="Deployment budget remaining">
            <LineChart series={[{ name: 'Budget (kJ)', color: '#e8a33d', values: budget }]} xLabels={xLabels} height={200} />
          </Panel>
          <Panel title="Action mix this session">
            {totalActions === 0 ? (
              <div className="empty-state">No decisions yet — action mix will populate as you run scenarios.</div>
            ) : (
              Object.entries(actionCounts).map(([action, count]) => (
                <div key={action} style={{ marginBottom: 10 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, marginBottom: 4 }}>
                    <span style={{ color: 'var(--paper-dim)' }}>{action.replace('_', ' ')}</span>
                    <span className="num">{count}</span>
                  </div>
                  <div className="hbar-track">
                    <div className="hbar-fill" style={{ width: `${(count / totalActions) * 100}%`, background: 'var(--signal-cyan)' }} />
                  </div>
                </div>
              ))
            )}
          </Panel>
        </div>
      </div>
    </div>
  );
}
