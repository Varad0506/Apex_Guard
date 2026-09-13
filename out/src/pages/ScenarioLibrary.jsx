import React, { useEffect, useState } from 'react';
import api from '../api';
import { useSession } from '../store';
import { Panel, ErrorBanner, Loading, SignalTag, DemoBadge } from '../components/Atoms';
import TrackLayout, { buildProximityMarkers } from '../components/TrackLayout';
import { TRACKS } from '../defaults';

const EXPECTED = {
  'isolated_pass': { label: 'Isolated pass', expect: 'Clear overtake attempt, minimal traffic — expect a deploy action.' },
  'low_soc_guard': { label: 'Low SOC guard', expect: 'Reserve near the rules floor — expect HOLD or HARVEST, never a reserve breach.' },
  'rear_drs_threat': { label: 'Rear DRS threat', expect: 'A car close behind with DRS — expect the verifier to weigh counterattack risk.' },
};

export default function ScenarioLibrary() {
  const [list, setList] = useState([]);
  const [results, setResults] = useState({}); // id -> {response, passed}
  const [running, setRunning] = useState(null);
  const [error, setError] = useState(null);
  const { recordDecision } = useSession();

  useEffect(() => {
    api.listScenarios().then((r) => setList(r.scenarios)).catch((e) => setError(e.message));
  }, []);

  async function runScenario(id) {
    setRunning(id);
    setError(null);
    try {
      const req = await api.getScenario(id);
      const res = await api.decide(req);
      recordDecision(req, res);
      const passed = res.rule_compliant && res.status !== 'DEGRADED';
      setResults((r) => ({ ...r, [id]: { request: req, response: res, passed } }));
    } catch (e) {
      setError(e.message);
    } finally {
      setRunning(null);
    }
  }

  return (
    <div>
      <div className="page-head">
        <div className="page-title">Scenario Library</div>
        <div className="page-desc">Three independent cached scenarios, each stating its expected outcome before it runs against the real pipeline.</div>
      </div>

      <ErrorBanner error={error} />

      <div className="grid-3">
        {list.map((s) => {
          const meta = EXPECTED[s.id] || { label: s.id, expect: '—' };
          const result = results[s.id];
          return (
            <Panel key={s.id} title={meta.label}>
              <div style={{ fontSize: 12, color: 'var(--paper-dim)', marginBottom: 12, minHeight: 36 }}>{meta.expect}</div>
              <button className="btn btn-primary btn-block" onClick={() => runScenario(s.id)} disabled={running === s.id}>
                {running === s.id ? 'Running…' : 'Run scenario'}
              </button>
              {running === s.id && <div style={{ marginTop: 10 }}><Loading /></div>}
              {result && (
                <div style={{ marginTop: 14 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                    <span style={{ color: result.passed ? 'var(--signal-cyan)' : 'var(--signal-red)', fontWeight: 700 }}>
                      {result.passed ? '✓' : '✕'}
                    </span>
                    <span style={{ fontSize: 12, color: 'var(--paper-dim)' }}>{result.passed ? 'Pass' : 'Fail'}</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <SignalTag action={result.response.recommended_action} />
                    {result.response._mock && <DemoBadge />}
                  </div>
                  <div className="explanation" style={{ marginTop: 10 }}>{result.response.explanation}</div>
                  {result.request && (() => {
                    const trackMeta = TRACKS.find((tr) => tr.track_id === result.request.track?.track_id) || TRACKS[0];
                    const markers = buildProximityMarkers({
                      targetGapS: result.request.target?.gap_s,
                      rearGapS: result.request.traffic?.rear_gap_s,
                      carsWithin3s: result.request.traffic?.cars_within_3s,
                      seedKey: `${s.id}-${result.request.request_id}`,
                    });
                    return (
                      <div style={{ marginTop: 12 }}>
                        <TrackLayout
                          trackId={result.request.track?.track_id}
                          cornerDensity={trackMeta.corner_density}
                          drsZoneCount={trackMeta.drs_zone_count}
                          height={180}
                          showLegend={false}
                          markers={markers}
                        />
                      </div>
                    );
                  })()}
                </div>
              )}
            </Panel>
          );
        })}
      </div>
    </div>
  );
}
