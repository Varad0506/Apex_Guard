import React, { useState, useMemo } from 'react';
import { TRACKS } from '../defaults';
import { Panel, Stat } from '../components/Atoms';
import LineChart, { LineLegend } from '../components/LineChart';
import TrackLayout from '../components/TrackLayout';
import { mulberry32, seedFromString } from '../mock';

const N_SAMPLES = 24; // points across one synthetic lap

// Deterministic, seeded-per-track derived detail: starting SOC, tyre grip
// estimate, sector count, DRS straight length. Not from the API (no
// per-track endpoint) — derived honestly from the track's own stats so
// each of the 10 tracks gets a distinct, reproducible profile.
function deriveTrackDetail(track) {
  const rand = mulberry32(seedFromString(track.track_id));
  const startingSoc = +(55 + rand() * 25).toFixed(1);
  const tyreGrip = +(0.72 + rand() * 0.2).toFixed(2);
  const sectorCount = 3 + Math.round(track.corner_density * 6);
  const drsStraightLength = Math.round(track.lap_length_m * (0.08 + (1 - track.corner_density) * 0.18));
  return { startingSoc, tyreGrip, sectorCount, drsStraightLength };
}

// Synthetic lap trace: SOC-remaining and tyre-grip curves generated from
// the track's own overtake_difficulty / corner_density / lap_length_m —
// higher corner density -> faster grip decay; longer lap -> slower SOC
// decay per %. Seeded per track so every track gets its own honest-looking,
// reproducible shape instead of one generic curve repeated 10 times.
function buildLapTrace(track, detail) {
  const rand = mulberry32(seedFromString(`${track.track_id}-lap`));
  const socDecayPerSample = (1 / N_SAMPLES) * (14 / Math.max(3000, track.lap_length_m)) * 5000;
  const gripDecayPerSample = (track.corner_density * 0.9) / N_SAMPLES;

  const soc = [];
  const grip = [];
  let socV = detail.startingSoc;
  let gripV = detail.tyreGrip;
  for (let i = 0; i < N_SAMPLES; i++) {
    socV = Math.max(15, socV - socDecayPerSample * (0.7 + rand() * 0.6));
    gripV = Math.max(0.45, gripV - gripDecayPerSample * (0.7 + rand() * 0.6));
    soc.push(+socV.toFixed(2));
    grip.push(+gripV.toFixed(3));
  }
  return { soc, grip };
}

function TrackCard({ track, onOpen }) {
  const detail = useMemo(() => deriveTrackDetail(track), [track]);
  return (
    <Panel title={track.label}>
      <TrackLayout
        trackId={track.track_id}
        cornerDensity={track.corner_density}
        drsZoneCount={track.drs_zone_count}
        height={190}
        showLegend={false}
      />
      <div className="stat-row" style={{ marginTop: 10 }}>
        <Stat label="Overtake difficulty" value={track.overtake_difficulty.toFixed(2)} />
        <Stat label="DRS zones" value={track.drs_zone_count} />
      </div>
      <div className="stat-row" style={{ marginTop: 10 }}>
        <Stat label="Lap length" value={track.lap_length_m} unit="m" />
        <Stat label="Corner density" value={track.corner_density.toFixed(2)} />
      </div>
      <div className="stat-row" style={{ marginTop: 10 }}>
        <Stat label="Starting SOC" value={detail.startingSoc} unit="%" />
        <Stat label="Tyre grip est." value={detail.tyreGrip} />
      </div>
      <div className="stat-row" style={{ marginTop: 10 }}>
        <Stat label="Sectors" value={detail.sectorCount} />
        <Stat label="DRS straight" value={detail.drsStraightLength} unit="m" />
      </div>
      <div style={{ marginTop: 10 }}>
        <div className="hbar-track"><div className="hbar-fill" style={{ width: `${track.overtake_difficulty * 100}%`, background: 'var(--signal-amber)' }} /></div>
        <div style={{ fontSize: 11, color: 'var(--paper-faint)', marginTop: 4 }}>track_id: {track.track_id}</div>
      </div>
      <button className="btn btn-block" style={{ marginTop: 12 }} onClick={() => onOpen(track.track_id)}>
        Simulate a lap
      </button>
    </Panel>
  );
}

function TrackDetail({ track, onClose }) {
  const detail = useMemo(() => deriveTrackDetail(track), [track]);
  const trace = useMemo(() => buildLapTrace(track, detail), [track, detail]);
  const xLabels = Array.from({ length: N_SAMPLES }, (_, i) => `${Math.round((i / (N_SAMPLES - 1)) * 100)}%`);

  return (
    <Panel
      title={`Lap simulation — ${track.label}`}
      right={<button className="btn" onClick={onClose}>Close</button>}
    >
      <div style={{ fontSize: 12, color: 'var(--paper-dim)', marginBottom: 12 }}>
        Synthetic single-lap trace derived from this track's own overtake difficulty ({track.overtake_difficulty.toFixed(2)}),
        corner density ({track.corner_density.toFixed(2)}), and lap length ({track.lap_length_m}m) — higher corner density
        drains tyre grip faster; a longer lap drains SOC more slowly per percentage point.
      </div>
      <TrackLayout
        trackId={track.track_id}
        cornerDensity={track.corner_density}
        drsZoneCount={track.drs_zone_count}
        height={340}
        label={track.label.toUpperCase()}
      />
      <div style={{ height: 16 }} />
      <div className="dash-grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
        <div>
          <div className="chart-title-row"><span className="ct-title">SOC remaining</span><span className="ct-unit">%</span></div>
          <LineChart series={[{ name: 'SOC %', color: '#3dbfc9', values: trace.soc }]} xLabels={xLabels} yDomain={[10, 100]} height={230} />
        </div>
        <div>
          <div className="chart-title-row"><span className="ct-title">Tyre grip estimate</span><span className="ct-unit">0–1</span></div>
          <LineChart series={[{ name: 'Tyre grip', color: '#e8a33d', values: trace.grip }]} xLabels={xLabels} yDomain={[0.4, 1]} height={230} />
        </div>
      </div>
      <LineLegend series={[{ name: 'SOC %', color: '#3dbfc9' }, { name: 'Tyre grip', color: '#e8a33d' }]} />
    </Panel>
  );
}

export default function TrackLibrary() {
  const [openTrackId, setOpenTrackId] = useState(null);
  const openTrack = TRACKS.find((t) => t.track_id === openTrackId);

  return (
    <div>
      <div className="page-head">
        <div className="page-title">Track Library</div>
        <div className="page-desc">10 tracks used for training and evaluation — a browsable set, not a sequence.</div>
      </div>

      {openTrack && (
        <>
          <TrackDetail track={openTrack} onClose={() => setOpenTrackId(null)} />
          <div style={{ height: 14 }} />
        </>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(230px, 1fr))', gap: 14 }}>
        {TRACKS.map((t) => (
          <TrackCard key={t.track_id} track={t} onOpen={setOpenTrackId} />
        ))}
      </div>
    </div>
  );
}
