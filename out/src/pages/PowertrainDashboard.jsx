import React, { useMemo } from 'react';
import { useSession } from '../store';
import { Panel, Stat } from '../components/Atoms';
import LineChart from '../components/LineChart';
import { mulberry32, seedFromString } from '../mock';

const N = 20;

function buildDemoTrace() {
  const rand = mulberry32(seedFromString('powertrain-dashboard-demo'));
  const speed = []; const throttle = []; const grip = [];
  let speedV = 260, throttleV = 92, gripV = 0.86;
  for (let i = 0; i < N; i++) {
    speedV = Math.max(120, Math.min(330, speedV + (rand() - 0.5) * 40));
    throttleV = Math.max(0, Math.min(100, throttleV + (rand() - 0.5) * 18));
    gripV = Math.max(0.5, Math.min(1, gripV - rand() * 0.01));
    speed.push(Math.round(speedV));
    throttle.push(Math.round(throttleV));
    grip.push(+gripV.toFixed(3));
  }
  return { speed, throttle, grip };
}

export default function PowertrainDashboard() {
  const { decisions } = useSession();
  const chrono = useMemo(() => [...decisions].reverse(), [decisions]);
  const hasLive = chrono.length > 0;
  const demo = useMemo(() => buildDemoTrace(), []);

  const speed = hasLive ? chrono.map((d) => d.request?.ego?.speed_kph ?? 0) : demo.speed;
  const throttle = hasLive ? chrono.map((d) => d.request?.ego?.throttle_pct ?? 0) : demo.throttle;
  const grip = hasLive ? chrono.map((d) => d.request?.ego?.tyre_grip_estimate ?? 0) : demo.grip;
  const xLabels = hasLive ? chrono.map((_, i) => `#${i + 1}`) : Array.from({ length: N }, (_, i) => `${i + 1}`);

  const latestSpeed = speed[speed.length - 1];
  const latestThrottle = throttle[throttle.length - 1];
  const latestGrip = grip[grip.length - 1];
  const avgSpeed = speed.length ? speed.reduce((a, b) => a + b, 0) / speed.length : 0;

  return (
    <div>
      <div className="page-head">
        <div className="page-title">Powertrain Dashboard</div>
        <div className="page-desc">
          Engine and chassis performance — speed, throttle application, and tyre grip — kept separate
          from the ERS view so mechanical performance and energy strategy can each be read at a glance.
        </div>
      </div>

      <div className="dash-kpis">
        <div className="panel panel-body"><Stat label="Current speed" value={latestSpeed != null ? Math.round(latestSpeed) : '—'} unit="kph" /></div>
        <div className="panel panel-body"><Stat label="Throttle application" value={latestThrottle != null ? Math.round(latestThrottle) : '—'} unit="%" /></div>
        <div className="panel panel-body"><Stat label="Tyre grip estimate" value={latestGrip != null ? latestGrip.toFixed(2) : '—'} /></div>
        <div className="panel panel-body"><Stat label="Average speed this session" value={Math.round(avgSpeed)} unit="kph" /></div>
      </div>

      <div className="dash-grid">
        <Panel title="Speed over time">
          <div className="chart-title-row"><span className="ct-title">Speed</span><span className="ct-unit">kph</span></div>
          <LineChart series={[{ name: 'Speed', color: '#4f9dff', values: speed }]} xLabels={xLabels} height={260} />
          {!hasLive && <div style={{ fontSize: 12.5, color: 'var(--paper-faint)', marginTop: 8 }}>Demo trace — run a decision to see live session data.</div>}
        </Panel>

        <div className="dash-side">
          <Panel title="Throttle application">
            <LineChart series={[{ name: 'Throttle %', color: '#3ddc84', values: throttle }]} xLabels={xLabels} yDomain={[0, 100]} height={200} />
          </Panel>
          <Panel title="Tyre grip estimate">
            <LineChart series={[{ name: 'Tyre grip', color: '#e8a33d', values: grip }]} xLabels={xLabels} yDomain={[0.4, 1]} height={200} />
          </Panel>
        </div>
      </div>
    </div>
  );
}
