import React, { useMemo } from 'react';
import { mulberry32, seedFromString } from '../mock';

// Generates a smooth closed-loop track shape (SVG path) seeded per track_id,
// plus zone segments (DRS / braking / overtake) and deploy-opportunity
// markers, derived from that track's own drs_zone_count / corner_density.
// There's no real track geometry in the API, so this is an honest schematic
// layout — a plausible aerial-view racetrack, distinct and reproducible per
// track, not a fabricated map of the real circuit.

const VIEW = 500;

function catmullRomPath(points) {
  const p = points, n = p.length;
  let d = `M ${p[0].x.toFixed(1)} ${p[0].y.toFixed(1)} `;
  for (let i = 0; i < n; i++) {
    const p0 = p[(i - 1 + n) % n];
    const p1 = p[i];
    const p2 = p[(i + 1) % n];
    const p3 = p[(i + 2) % n];
    const c1x = p1.x + (p2.x - p0.x) / 6;
    const c1y = p1.y + (p2.y - p0.y) / 6;
    const c2x = p2.x - (p3.x - p1.x) / 6;
    const c2y = p2.y - (p3.y - p1.y) / 6;
    d += `C ${c1x.toFixed(1)} ${c1y.toFixed(1)}, ${c2x.toFixed(1)} ${c2y.toFixed(1)}, ${p2.x.toFixed(1)} ${p2.y.toFixed(1)} `;
  }
  return d + 'Z';
}

// Finely sample the same Catmull-Rom curve used for the path, so zone
// segments and position markers can be placed at arbitrary progress along
// the loop (not just at the original control points).
function sampleLoop(points, samplesPerSeg = 14) {
  const n = points.length;
  const out = [];
  for (let i = 0; i < n; i++) {
    const p0 = points[(i - 1 + n) % n];
    const p1 = points[i];
    const p2 = points[(i + 1) % n];
    const p3 = points[(i + 2) % n];
    for (let s = 0; s < samplesPerSeg; s++) {
      const t = s / samplesPerSeg;
      const t2 = t * t, t3 = t2 * t;
      const x = 0.5 * ((2 * p1.x) + (-p0.x + p2.x) * t + (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2 + (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t3);
      const y = 0.5 * ((2 * p1.y) + (-p0.y + p2.y) * t + (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2 + (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3);
      out.push({ x, y });
    }
  }
  return out;
}

function buildTrackGeometry(trackId, cornerDensity, drsZoneCount) {
  const rand = mulberry32(seedFromString(trackId || 'default'));
  const n = 7 + Math.round(cornerDensity * 6); // denser corners -> more vertices
  const cx = VIEW / 2, cy = VIEW / 2;
  const baseR = VIEW * 0.34;
  const points = [];
  for (let i = 0; i < n; i++) {
    const angle = (i / n) * Math.PI * 2;
    const jitter = 0.62 + rand() * 0.58; // irregular radius -> natural-looking corners
    const rx = baseR * jitter;
    const ry = baseR * jitter * 0.72;
    points.push({ x: cx + Math.cos(angle) * rx, y: cy + Math.sin(angle) * ry });
  }
  const path = catmullRomPath(points);
  const samples = sampleLoop(points);
  const total = samples.length;

  const drsCount = Math.max(1, drsZoneCount || 1);
  const drsZones = [];
  for (let i = 0; i < drsCount; i++) {
    const start = Math.floor(((i + rand() * 0.3) / drsCount) * total) % total;
    const len = Math.floor(total * (0.07 + rand() * 0.05));
    drsZones.push({ start, len });
  }

  const brakingCount = 1 + Math.round(cornerDensity * 2);
  const brakingZones = [];
  for (let i = 0; i < brakingCount; i++) {
    const start = Math.floor(rand() * total);
    const len = Math.floor(total * (0.025 + rand() * 0.03));
    brakingZones.push({ start, len });
  }

  const overtakeIdx = Math.floor(rand() * total);
  const deployCount = 2 + Math.round((1 - cornerDensity) * 3);
  const deployMarkers = Array.from({ length: deployCount }, () => Math.floor(rand() * total));

  return { points, path, samples, drsZones, brakingZones, overtakeIdx, deployMarkers };
}

function segToPoints(samples, seg) {
  const out = [];
  for (let i = 0; i < seg.len; i++) out.push(samples[(seg.start + i) % samples.length]);
  return out;
}

function polylinePoints(pts) {
  return pts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');
}

// Builds a set of position markers straight from real traffic/target
// telemetry — EGO fixed at the mid-loop reference point, TARGET and REAR
// placed relative to it by their real gap_s values, and one unlabeled dot
// per car reported in cars_within_3s, scattered within that real 3s window
// (seeded so it doesn't reshuffle on every re-render, only when the inputs
// change). This is how the same real numbers shown in the radial diagram
// get placed onto the track loop instead of an abstract radial plot.
export function buildProximityMarkers({ targetGapS = 1, rearGapS = 2, carsWithin3s = 0, seedKey = 'proximity', scale = 20 }) {
  const rand = mulberry32(seedFromString(seedKey));
  const egoProgress = 0.5;
  const markers = [
    { label: 'EGO', progress: egoProgress, color: 'var(--signal-cyan)' },
  ];
  if (targetGapS != null) {
    markers.push({ label: 'TARGET', progress: egoProgress + targetGapS / scale, color: '#e9edf1' });
  }
  if (rearGapS != null) {
    markers.push({ label: 'REAR', progress: egoProgress - rearGapS / scale, color: 'var(--signal-red)' });
  }
  const n = Math.max(0, Math.min(6, Math.round(carsWithin3s)));
  for (let i = 0; i < n; i++) {
    const side = rand() > 0.5 ? 1 : -1;
    const offset = side * (0.15 + rand() * 0.85) * (3 / scale);
    markers.push({ label: '', progress: egoProgress + offset, color: 'var(--signal-amber)' });
  }
  return markers;
}

// markers: [{ label, progress (0-1, wraps), color }]
export default function TrackLayout({ trackId, cornerDensity = 0.4, drsZoneCount = 2, markers = [], height = 300, showLegend = true, label }) {
  const geo = useMemo(
    () => buildTrackGeometry(trackId, cornerDensity, drsZoneCount),
    [trackId, cornerDensity, drsZoneCount],
  );

  return (
    <div>
      <svg viewBox={`0 0 ${VIEW} ${VIEW}`} style={{ width: '100%', height, display: 'block' }} xmlns="http://www.w3.org/2000/svg">
        <path d={geo.path} fill="none" stroke="var(--graphite-line)" strokeWidth="10" strokeLinejoin="round" strokeLinecap="round" />
        {geo.brakingZones.map((z, i) => (
          <polyline key={`b${i}`} points={polylinePoints(segToPoints(geo.samples, z))} fill="none" stroke="var(--signal-red)" strokeWidth="10" strokeLinecap="round" opacity="0.85" />
        ))}
        {geo.drsZones.map((z, i) => (
          <polyline key={`d${i}`} points={polylinePoints(segToPoints(geo.samples, z))} fill="none" stroke="var(--signal-cyan)" strokeWidth="10" strokeLinecap="round" opacity="0.85" />
        ))}
        <circle cx={geo.samples[geo.overtakeIdx].x} cy={geo.samples[geo.overtakeIdx].y} r="7" fill="#5b8def" stroke="rgba(0,0,0,0.35)" strokeWidth="1" />
        {geo.deployMarkers.map((idx, i) => {
          const p = geo.samples[idx];
          return <text key={i} x={p.x} y={p.y} fontSize="15" textAnchor="middle" dominantBaseline="middle" fill="var(--signal-amber)">⚡</text>;
        })}
        {markers.map((m, i) => {
          const wrapped = ((m.progress % 1) + 1) % 1;
          const idx = Math.floor(wrapped * geo.samples.length) % geo.samples.length;
          const p = geo.samples[idx];
          const isCar = !m.label;
          return (
            <g key={i}>
              <circle cx={p.x} cy={p.y} r={isCar ? 5.5 : 8.5} fill={m.color} stroke="rgba(0,0,0,0.45)" strokeWidth="1.5" />
              {!isCar && <text x={p.x} y={p.y - 13} fontSize="10.5" textAnchor="middle" fill={m.color} fontFamily="var(--font-mono)">{m.label}</text>}
            </g>
          );
        })}
        {label && <text x={VIEW - 12} y={24} fontSize="11" textAnchor="end" fill="var(--paper-faint)" fontFamily="var(--font-mono)">{label}</text>}
      </svg>
      {showLegend && (
        <div className="mc-legend" style={{ marginTop: 2 }}>
          <span><i style={{ background: 'var(--signal-cyan)' }} /> DRS zone</span>
          <span><i style={{ background: 'var(--signal-red)' }} /> Braking zone</span>
          <span><i style={{ background: '#5b8def' }} /> Overtake zone</span>
          <span>⚡ Deploy opportunity</span>
        </div>
      )}
    </div>
  );
}
