import React, { useEffect, useState } from 'react';
import { NavLink, Routes, Route, Navigate, useNavigate, useLocation } from 'react-router-dom';

import Overview from './pages/Overview';
import DecisionEngine from './pages/DecisionEngine';
import RaceReplay from './pages/RaceReplay';
import Simulate from './pages/Simulate';
import OpponentBelief from './pages/OpponentBelief';
import Traffic from './pages/Traffic';
import ScenarioLibrary from './pages/ScenarioLibrary';
import TrackLibrary from './pages/TrackLibrary';
import PerformanceLab from './pages/PerformanceLab';
import ERSDashboard from './pages/ERSDashboard';
import PowertrainDashboard from './pages/PowertrainDashboard';
import Backtests from './pages/Backtests';
import AuditTrail from './pages/AuditTrail';
import SystemHealth from './pages/SystemHealth';
import { useAppChrome, ShortcutOverlay } from './chrome';

// Nav groups mirror the tier-3 static cockpit design: Live / Analytics /
// Library / System. Each item's `key` doubles as its keyboard shortcut.
const NAV_GROUPS = [
  {
    label: 'Live',
    items: [
      { key: '1', to: '/', label: 'Overview', end: true, icon: 'grid' },
      { key: '2', to: '/decision', label: 'Decision Engine', icon: 'bolt' },
      { key: '3', to: '/replay', label: 'Race Replay', icon: 'play' },
      { key: '4', to: '/simulate', label: 'Simulate', icon: 'wave' },
      { key: '5', to: '/opponent', label: 'Opponent Belief', icon: 'target' },
      { key: '6', to: '/traffic', label: 'Traffic', icon: 'cars' },
    ],
  },
  {
    label: 'Analytics',
    items: [
      { key: '7', to: '/performance', label: 'Performance Lab', icon: 'bars' },
      { key: '8', to: '/backtests', label: 'Backtests', icon: 'flask' },
      { key: 'e', to: '/ers', label: 'ERS Dashboard', icon: 'bolt' },
      { key: 'p', to: '/powertrain', label: 'Powertrain Dashboard', icon: 'wave' },
    ],
  },
  {
    label: 'Library',
    items: [
      { key: '9', to: '/scenarios', label: 'Scenario Library', icon: 'book' },
      { key: '0', to: '/tracks', label: 'Track Library', icon: 'circuit' },
    ],
  },
  {
    label: 'System',
    items: [
      { key: 'a', to: '/audit', label: 'Audit Trail', icon: 'clock' },
      { key: 'h', to: '/health', label: 'System Health', icon: 'pulse' },
    ],
  },
];

const NAV = NAV_GROUPS.flatMap((g) => g.items);

function NavIcon({ name }) {
  const common = { viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: '1.8' };
  switch (name) {
    case 'grid':
      return <svg {...common}><rect x="3" y="3" width="7" height="7" rx="1.5" /><rect x="14" y="3" width="7" height="7" rx="1.5" /><rect x="3" y="14" width="7" height="7" rx="1.5" /><rect x="14" y="14" width="7" height="7" rx="1.5" /></svg>;
    case 'bolt':
      return <svg {...common}><path d="M13 2 4 14h6l-1 8 9-12h-6l1-8z" /></svg>;
    case 'play':
      return <svg {...common}><circle cx="12" cy="12" r="9" /><path d="M10 8.5v7l6-3.5-6-3.5z" /></svg>;
    case 'wave':
      return <svg {...common}><path d="M3 12h4l2-7 4 14 2-7h6" /></svg>;
    case 'target':
      return <svg {...common}><circle cx="12" cy="12" r="3.2" /><path d="M12 3v3.5M12 17.5V21M3 12h3.5M17.5 12H21M5.6 5.6l2.5 2.5M15.9 15.9l2.5 2.5M18.4 5.6l-2.5 2.5M8.1 15.9l-2.5 2.5" /></svg>;
    case 'cars':
      return <svg {...common}><circle cx="7" cy="17" r="2" /><circle cx="17" cy="17" r="2" /><path d="M3 17V9l3-4h9l4 5v7M3 12h18" /></svg>;
    case 'bars':
      return <svg {...common}><path d="M4 20V10M11 20V4M18 20v-7" /></svg>;
    case 'flask':
      return <svg {...common}><path d="M9 3h6M10 3v6l-5 9a2 2 0 0 0 2 3h10a2 2 0 0 0 2-3l-5-9V3" /></svg>;
    case 'book':
      return <svg {...common}><path d="M5 3v18M5 4h11l-2 4 2 4H5" /></svg>;
    case 'circuit':
      return <svg {...common}><circle cx="12" cy="12" r="9" /><path d="M12 3c2 3 2 15 0 18M3 12c3-2 15-2 18 0" /></svg>;
    case 'clock':
      return <svg {...common}><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 3" /></svg>;
    case 'pulse':
      return <svg {...common}><path d="M3 12h4l2 6 4-14 2 8h6" /></svg>;
    default:
      return <svg {...common}><circle cx="12" cy="12" r="3" fill="currentColor" /></svg>;
  }
}

export default function App() {
  const navigate = useNavigate();
  const location = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useAppChrome(NAV, navigate);

  useEffect(() => {
    document.body.classList.toggle('sidebar-open', sidebarOpen);
  }, [sidebarOpen]);

  const current = NAV.find((n) => (n.end ? location.pathname === n.to : location.pathname.startsWith(n.to) && n.to !== '/'))
    || NAV.find((n) => n.to === '/');

  return (
    <>
      <div className="ambient"><div className="grid-lines" /><div className="scanline" /></div>

      <div className="app-shell">
        <div className="sidebar-scrim" onClick={() => setSidebarOpen(false)} />
        <aside className="sidebar">
          <div className="brand">
            <span className="mark">
              <img src="/assets/logo-mark-white.png" alt="ApexGuard logo" className="mark-logo" />
            </span>
            ApexGuard
          </div>

          {NAV_GROUPS.map((group) => (
            <React.Fragment key={group.label}>
              <div className="nav-label">{group.label}</div>
              {group.items.map((n) => (
                <NavLink
                  key={n.to}
                  to={n.to}
                  end={n.end}
                  className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
                  onClick={() => setSidebarOpen(false)}
                >
                  <NavIcon name={n.icon} />
                  {n.label}
                </NavLink>
              ))}
            </React.Fragment>
          ))}

          <div className="sidebar-footer">
            <span className="status-pip"><span className="dot" /> LIVE</span>
            <span>v1 · /v1/health/ready</span>
          </div>
        </aside>

        <main className="main">
          <div className="topbar">
            <div>
              <button
                type="button"
                className="btn btn-icon menu-toggle"
                aria-label="Toggle navigation"
                onClick={() => setSidebarOpen((v) => !v)}
                style={{ marginRight: 12 }}
              >
                ☰
              </button>
              <span className="route-crumb">ApexGuard / <b>{current?.label || 'Cockpit'}</b></span>
            </div>
            <div className="topbar-right">
              <button type="button" className="btn btn-icon" title="Toggle data density (D)" onClick={() => window.toggleDensity && window.toggleDensity()}>⇕</button>
              <button
                type="button"
                className="btn btn-icon"
                title="Keyboard shortcuts (?)"
                onClick={() => document.querySelector('.shortcut-overlay')?.classList.add('show')}
              >
                ?
              </button>
              <span className="badge badge-green"><span className="dot" /> ENGINE NOMINAL</span>
            </div>
          </div>

          <div className="content">
            <Routes>
              <Route path="/" element={<Overview />} />
              <Route path="/decision" element={<DecisionEngine />} />
              <Route path="/replay" element={<RaceReplay />} />
              <Route path="/simulate" element={<Simulate />} />
              <Route path="/opponent" element={<OpponentBelief />} />
              <Route path="/traffic" element={<Traffic />} />
              <Route path="/scenarios" element={<ScenarioLibrary />} />
              <Route path="/tracks" element={<TrackLibrary />} />
              <Route path="/performance" element={<PerformanceLab />} />
              <Route path="/ers" element={<ERSDashboard />} />
              <Route path="/powertrain" element={<PowertrainDashboard />} />
              <Route path="/backtests" element={<Backtests />} />
              <Route path="/audit" element={<AuditTrail />} />
              <Route path="/health" element={<SystemHealth />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </div>
        </main>
      </div>

      <ShortcutOverlay navItems={NAV} />
    </>
  );
}
