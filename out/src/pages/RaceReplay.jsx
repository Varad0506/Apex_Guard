import React, { useEffect, useState, useRef, useMemo } from 'react';
import api from '../api';
import { useSession } from '../store';
import { Panel, ErrorBanner, Loading, SignalTag, DemoBadge } from '../components/Atoms';
import MonteCarloChart, { McLegend } from '../components/MonteCarloChart';
import TrackLayout from '../components/TrackLayout';
import { mulberry32, seedFromString } from '../mock';
import { TRACKS, defaultDecisionRequest, makeRequestId } from '../defaults';

const LAP_SCENARIOS = {
  isolated_pass: { label: 'Isolated pass', startGap: 0.9, drift: -0.03 },
  low_soc_guard: { label: 'Low SOC guard', startGap: 1.6, drift: 0.01 },
  rear_drs_threat: { label: 'Rear DRS threat', startGap: 0.5, drift: 0.02 },
};

function trackMetaFor(trackId) {
  return TRACKS.find((t) => t.track_id === trackId) || TRACKS[0];
}

// Client-side Monte Carlo rollout of gap-to-target across `laps` laps, seeded
// per track+scenario so it's reproducible rather than reshuffled every
// render. Overtake difficulty widens the spread (harder track -> more
// variance in how the gap evolves lap to lap).
function buildLapRollout(track, scenarioId, laps, nPaths) {
  const scenario = LAP_SCENARIOS[scenarioId];
  const rand = mulberry32(seedFromString(`${track.track_id}-${scenarioId}-${laps}`));
  const spread = 0.15 + track.overtake_difficulty * 0.5;
  const paths = [];
  for (let p = 0; p < nPaths; p++) {
    const path = [];
    let gap = Math.max(0.05, scenario.startGap + (rand() - 0.5) * 0.6);
    for (let l = 0; l < laps; l++) {
      gap = Math.max(0.03, gap + scenario.drift + (rand() - 0.5) * spread);
      path.push(gap);
    }
    paths.push(path);
  }
  return paths;
}

export default function RaceReplay() {
  const [replays, setReplays] = useState([]);
  const [replayId, setReplayId] = useState(null);
  const [replay, setReplay] = useState(null);
  const [frameIdx, setFrameIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [decideRes, setDecideRes] = useState(null);
  const timerRef = useRef(null);
  const { recordDecision } = useSession();

  // --- Lap Simulation panel state ---
  const [simTrackId, setSimTrackId] = useState(TRACKS[0].track_id);
  const [simScenarioId, setSimScenarioId] = useState('isolated_pass');
  const [simLaps, setSimLaps] = useState(8);
  const [simTick, setSimTick] = useState(1);
  const [simDecideRes, setSimDecideRes] = useState(null);
  const [simDeciding, setSimDeciding] = useState(false);

  const simTrack = TRACKS.find((t) => t.track_id === simTrackId) || TRACKS[0];
  const simPaths = useMemo(
    () => buildLapRollout(simTrack, simScenarioId, simLaps, 50),
    [simTrack, simScenarioId, simLaps],
  );
  const simRevealed = Math.min(simTick, simLaps);
  const simVisiblePaths = useMemo(
    () => simPaths.map((p) => p.slice(0, simRevealed)),
    [simPaths, simRevealed],
  );
  const simLapLabels = Array.from({ length: simLaps }, (_, i) => `L${i + 1}`);

  function resetSim() {
    setSimTick(1);
    setSimDecideRes(null);
  }

  async function decideOnSimTick() {
    setSimDeciding(true);
    setError(null);
    try {
      const meanGap = simVisiblePaths.reduce((s, p) => s + p[p.length - 1], 0) / simVisiblePaths.length;
      const frame = {
        ...defaultDecisionRequest(),
        request_id: makeRequestId(),
        battle_id: `lapsim-${simTrackId}-${simScenarioId}`,
        target: { ...defaultDecisionRequest().target, gap_s: meanGap },
        track: { ...defaultDecisionRequest().track, track_id: simTrackId, overtake_difficulty: simTrack.overtake_difficulty },
      };
      const res = await api.decide(frame);
      setSimDecideRes(res);
      recordDecision(frame, res);
    } catch (e) {
      setError(e.message);
    } finally {
      setSimDeciding(false);
    }
  }

  useEffect(() => {
    api.listReplays().then((r) => setReplays(r.replays)).catch((e) => setError(e.message));
  }, []);

  async function loadReplay(id) {
    setLoading(true);
    setError(null);
    setDecideRes(null);
    try {
      const r = await api.getReplay(id);
      setReplay(r);
      setReplayId(id);
      setFrameIdx(0);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (playing && replay) {
      timerRef.current = setInterval(() => {
        setFrameIdx((i) => (i + 1 >= replay.frames.length ? 0 : i + 1));
      }, 180);
    }
    return () => clearInterval(timerRef.current);
  }, [playing, replay]);

  const frame = replay?.frames?.[frameIdx];

  async function decideOnFrame() {
    if (!frame) return;
    try {
      const res = await api.decide(frame);
      setDecideRes(res);
      recordDecision(frame, res);
    } catch (e) {
      setError(e.message);
    }
  }

  return (
    <div>
      <div className="page-head">
        <div className="page-title">Race Replay</div>
        <div className="page-desc">Real cached OpenF1 sessions, frame by frame, with the same DecisionRequest schema the live pipeline consumes.</div>
      </div>

      <ErrorBanner error={error} />

      <Panel title="Cached OpenF1 sessions">
        <div className="scroll-strip">
          {replays.map((r) => (
            <div key={r.id} className={`card-select${replayId === r.id ? ' active' : ''}`} onClick={() => loadReplay(r.id)}>
              <div className="mono" style={{ fontSize: 13 }}>{r.id.replace(/_openf1$/, '').replace(/_/g, ' ')}</div>
              <div style={{ fontSize: 11, color: 'var(--paper-faint)', marginTop: 4 }}>{r.file}</div>
            </div>
          ))}
        </div>
      </Panel>

      <div style={{ height: 14 }} />

      {loading && <Loading label="Loading real OpenF1 replay…" />}

      {replay && (
        <div className="grid-3">
          <Panel title="Session" right={replay._mock ? <DemoBadge /> : undefined}>
            <div style={{ fontSize: 12, color: 'var(--paper-dim)', lineHeight: 1.8 }}>
              <div>{replay.year} {replay.event} · {replay.session}</div>
              <div>{replay.driver} vs {replay.target_driver}</div>
              <div>{replay.frames.length} frames · {replay.duration_s?.toFixed(1)}s span</div>
              <div style={{ marginTop: 8, color: 'var(--paper-faint)' }}>source: {replay.source} ({replay.provider})</div>
            </div>
          </Panel>

          <Panel title={`Frame ${frameIdx + 1} / ${replay.frames.length}`}>
            {frame && (() => {
              const trackMeta = trackMetaFor(frame.track?.track_id);
              // No real (x,y) position in the API — target's progress advances
              // steadily through the replay's frame count (looping ~1.5 laps
              // across the session for visual motion), ego and P3 sit behind
              // it proportional to their real gap_s values, scaled onto the
              // loop. Honest placement of real gap data on a schematic loop,
              // not a fabricated GPS trace.
              const targetProgress = (frameIdx / Math.max(1, replay.frames.length - 1)) * 1.5;
              const egoProgress = targetProgress - (frame.target.gap_s / 20);
              const p3Progress = egoProgress - (frame.traffic.rear_gap_s / 20);
              return (
                <TrackLayout
                  trackId={frame.track?.track_id}
                  cornerDensity={trackMeta.corner_density}
                  drsZoneCount={trackMeta.drs_zone_count}
                  label={`${trackMeta.label.toUpperCase()} · ${frame.track.segment_type.replace(/_/g, ' ')}`}
                  markers={[
                    { label: 'TARGET', progress: targetProgress, color: '#e9edf1' },
                    { label: 'EGO', progress: egoProgress, color: 'var(--signal-cyan)' },
                    { label: 'P3', progress: p3Progress, color: 'var(--paper-faint)' },
                  ]}
                />
              );
            })()}
            <input
              type="range" min={0} max={replay.frames.length - 1} value={frameIdx}
              onChange={(e) => setFrameIdx(parseInt(e.target.value, 10))}
              style={{ marginTop: 10 }}
            />
            <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
              <button className="btn" onClick={() => setPlaying((p) => !p)}>{playing ? 'Pause' : 'Play'}</button>
              <button className="btn btn-primary" onClick={decideOnFrame}>Decide on this frame</button>
            </div>
            {frame && (
              <div className="stat-row" style={{ marginTop: 14 }}>
                <div className="stat"><div className="label">Gap to target</div><div className="value num">{frame.target.gap_s.toFixed(2)}s</div></div>
                <div className="stat"><div className="label">Ego SOC</div><div className="value num">{frame.ego.soc_pct.toFixed(1)}%</div></div>
                <div className="stat"><div className="label">Segment</div><div className="value num" style={{ fontSize: 15 }}>{frame.track.segment_type}</div></div>
              </div>
            )}
          </Panel>

          <Panel title="Decision on frame">
            {!decideRes && <div className="empty-state">Step to a frame and run the decision pipeline on it.</div>}
            {decideRes && (
              <div>
                <SignalTag action={decideRes.recommended_action} />
                <div className="explanation" style={{ marginTop: 10 }}>{decideRes.explanation}</div>
                <div style={{ fontSize: 11, color: 'var(--paper-faint)' }}>latency {decideRes.latency_ms.toFixed(2)} ms · audit {decideRes.audit_id}</div>
              </div>
            )}
          </Panel>
        </div>
      )}

      <div style={{ height: 14 }} />

      <div className="grid-3">
        <Panel title="Lap Simulation">
          <div className="field">
            <label><span>Track</span></label>
            <select value={simTrackId} onChange={(e) => { setSimTrackId(e.target.value); resetSim(); }}>
              {TRACKS.map((t) => <option key={t.track_id} value={t.track_id}>{t.label}</option>)}
            </select>
          </div>
          <div className="field">
            <label><span>Scenario preset</span></label>
            <select value={simScenarioId} onChange={(e) => { setSimScenarioId(e.target.value); resetSim(); }}>
              {Object.entries(LAP_SCENARIOS).map(([id, s]) => <option key={id} value={id}>{s.label}</option>)}
            </select>
          </div>
          <div className="field">
            <label><span>Lap count</span><span className="val">{simLaps}</span></label>
            <input type="range" min={3} max={20} step={1} value={simLaps} onChange={(e) => { setSimLaps(parseInt(e.target.value, 10)); resetSim(); }} />
          </div>
          <div style={{ display: 'flex', gap: 8, marginBottom: 14 }}>
            <button className="btn" onClick={() => setSimTick((t) => Math.min(simLaps, t + 1))} disabled={simTick >= simLaps}>Advance tick</button>
            <button className="btn" onClick={resetSim}>Reset</button>
          </div>
          <TrackLayout
            trackId={simTrackId}
            cornerDensity={simTrack.corner_density}
            drsZoneCount={simTrack.drs_zone_count}
            height={240}
            label={`${simTrack.label.toUpperCase()} · LAP ${simRevealed}/${simLaps}`}
            markers={[
              { label: 'TARGET', progress: simRevealed / simLaps, color: '#e9edf1' },
              { label: 'EGO', progress: (simRevealed / simLaps) - (simVisiblePaths[0]?.[simVisiblePaths[0].length - 1] ?? 1) / 20, color: 'var(--signal-cyan)' },
            ]}
          />
        </Panel>

        <Panel title={`Path belief evolution — lap ${simRevealed} / ${simLaps}`}>
          <MonteCarloChart paths={simVisiblePaths} xLabels={simLapLabels.slice(0, simRevealed)} color="#3dbfc9" yDomain={undefined} />
          <McLegend items={[{ label: 'Sample rollouts', color: 'rgba(61,191,201,0.3)' }, { label: 'Mean gap-to-target', color: '#3dbfc9' }, { label: 'Min/max band', band: true }]} />
          <div style={{ fontSize: 11, color: 'var(--paper-faint)', marginTop: 10 }}>
            {simPaths.length} seeded rollouts of gap-to-target (s) for {simTrack.label} · {LAP_SCENARIOS[simScenarioId].label}. The fan narrows/shifts as more laps are revealed.
          </div>
        </Panel>

        <Panel title="Decide on this tick" right={simDecideRes?._mock ? <DemoBadge /> : undefined}>
          {!simDecideRes && <div className="empty-state">Advance the tick, then decide using the current synthesized frame.</div>}
          {simDecideRes && (
            <div>
              <SignalTag action={simDecideRes.recommended_action} />
              <div className="explanation" style={{ marginTop: 10 }}>{simDecideRes.explanation}</div>
              <div style={{ fontSize: 11, color: 'var(--paper-faint)' }}>latency {simDecideRes.latency_ms?.toFixed?.(2)} ms · audit {simDecideRes.audit_id}</div>
            </div>
          )}
          <button className="btn btn-primary btn-block" onClick={decideOnSimTick} disabled={simDeciding} style={{ marginTop: 12 }}>
            {simDeciding ? 'Deciding…' : 'Decide on this tick'}
          </button>
        </Panel>
      </div>
    </div>
  );
}
