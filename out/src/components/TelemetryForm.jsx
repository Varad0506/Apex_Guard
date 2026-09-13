import React from 'react';
import { SEGMENT_TYPES, MODES, TRACKS } from '../defaults';

// Every slider maps 1:1 to a DecisionRequest field. Labeled the way an
// engineer would say it out loud, not by internal field name.
function Slider({ label, value, min, max, step, unit, onChange }) {
  return (
    <div className="field">
      <label>
        <span>{label}</span>
        <span className="val">{value}{unit ? ` ${unit}` : ''}</span>
      </label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </div>
  );
}

function set(obj, path, value) {
  const next = JSON.parse(JSON.stringify(obj));
  const parts = path.split('.');
  let cur = next;
  for (let i = 0; i < parts.length - 1; i++) cur = cur[parts[i]];
  cur[parts[parts.length - 1]] = value;
  return next;
}

function get(obj, path) {
  return path.split('.').reduce((o, k) => (o == null ? o : o[k]), obj);
}

export default function TelemetryForm({ request, onChange, sections }) {
  const s = (path, value) => onChange(set(request, path, value));
  const g = (path) => get(request, path);
  const show = (name) => !sections || sections.includes(name);

  return (
    <div>
      {show('Ego') && (
        <>
          <div className="group-label">Ego</div>
          <Slider label="SOC" value={g('ego.soc_pct')} min={0} max={100} step={1} unit="%" onChange={(v) => s('ego.soc_pct', v)} />
          <Slider label="Speed" value={g('ego.speed_kph')} min={0} max={370} step={1} unit="kph" onChange={(v) => s('ego.speed_kph', v)} />
          <Slider label="Throttle" value={g('ego.throttle_pct')} min={0} max={100} step={1} unit="%" onChange={(v) => s('ego.throttle_pct', v)} />
          <Slider label="Brake" value={g('ego.brake_pct')} min={0} max={100} step={1} unit="%" onChange={(v) => s('ego.brake_pct', v)} />
          <Slider label="Tyre grip estimate" value={g('ego.tyre_grip_estimate')} min={0} max={1} step={0.01} onChange={(v) => s('ego.tyre_grip_estimate', v)} />
          <Slider label="Laps remaining" value={g('ego.laps_remaining')} min={0} max={70} step={1} onChange={(v) => s('ego.laps_remaining', v)} />
          <div className="field">
            <label><span>Current mode</span></label>
            <select value={g('ego.current_mode')} onChange={(e) => s('ego.current_mode', e.target.value)}>
              {MODES.map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
          </div>
        </>
      )}

      {show('Target car') && (
        <>
          <div className="group-label">Target car</div>
          <Slider label="Gap to target" value={g('target.gap_s')} min={0} max={5} step={0.05} unit="s" onChange={(v) => s('target.gap_s', v)} />
          <Slider label="Relative speed" value={g('target.relative_speed_kph')} min={-40} max={40} step={1} unit="kph" onChange={(v) => s('target.relative_speed_kph', v)} />
          <Slider label="Stint age" value={g('target.stint_age_laps')} min={0} max={45} step={1} unit="laps" onChange={(v) => s('target.stint_age_laps', v)} />
          <Slider label="Recent sector delta" value={g('target.recent_sector_delta_s')} min={-1} max={1} step={0.01} unit="s" onChange={(v) => s('target.recent_sector_delta_s', v)} />
        </>
      )}

      {show('Traffic') && (
        <>
          <div className="group-label">Traffic</div>
          <Slider label="Rear gap" value={g('traffic.rear_gap_s')} min={0} max={5} step={0.05} unit="s" onChange={(v) => s('traffic.rear_gap_s', v)} />
          <Slider label="Cars within 3s" value={g('traffic.cars_within_3s')} min={0} max={6} step={1} onChange={(v) => s('traffic.cars_within_3s', v)} />
          <Slider label="Post-pass traffic gap" value={g('traffic.post_pass_traffic_gap_s')} min={0} max={5} step={0.05} unit="s" onChange={(v) => s('traffic.post_pass_traffic_gap_s', v)} />
        </>
      )}

      {show('Track') && (
        <>
          <div className="group-label">Track</div>
          <div className="field">
            <label><span>Track</span></label>
            <select value={g('track.track_id')} onChange={(e) => s('track.track_id', e.target.value)}>
              {TRACKS.map((t) => <option key={t.track_id} value={t.track_id}>{t.label}</option>)}
            </select>
          </div>
          <div className="field">
            <label><span>Segment type</span></label>
            <select value={g('track.segment_type')} onChange={(e) => s('track.segment_type', e.target.value)}>
              {SEGMENT_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
          <div className="toggle-row">
            <span>DRS available</span>
            <input className="checkbox" type="checkbox" checked={g('track.drs_available')} onChange={(e) => s('track.drs_available', e.target.checked)} />
          </div>
          <Slider label="Straight remaining" value={g('track.straight_remaining_m')} min={0} max={1200} step={10} unit="m" onChange={(v) => s('track.straight_remaining_m', v)} />
          <Slider label="Braking zone" value={g('track.braking_zone_m')} min={0} max={300} step={5} unit="m" onChange={(v) => s('track.braking_zone_m', v)} />
          <Slider label="Overtake difficulty" value={g('track.overtake_difficulty')} min={0} max={1} step={0.01} onChange={(v) => s('track.overtake_difficulty', v)} />
          <div className="toggle-row">
            <span>Detection point active</span>
            <input className="checkbox" type="checkbox" checked={g('track.detection_point_active')} onChange={(e) => s('track.detection_point_active', e.target.checked)} />
          </div>
          <div className="toggle-row">
            <span>Overtake mode available</span>
            <input className="checkbox" type="checkbox" checked={g('track.overtake_mode_available')} onChange={(e) => s('track.overtake_mode_available', e.target.checked)} />
          </div>
          <div className="toggle-row">
            <span>ERS key acceleration zone</span>
            <input className="checkbox" type="checkbox" checked={g('track.ers_key_acceleration_zone')} onChange={(e) => s('track.ers_key_acceleration_zone', e.target.checked)} />
          </div>
        </>
      )}

      {show('Rules') && (
        <>
          <div className="group-label">Rules</div>
          <Slider label="Deployment budget remaining" value={g('rules.deployment_budget_remaining_kj')} min={0} max={4000} step={50} unit="kJ" onChange={(v) => s('rules.deployment_budget_remaining_kj', v)} />
          <Slider label="Minimum reserve SOC" value={g('rules.minimum_reserve_soc_pct')} min={0} max={40} step={1} unit="%" onChange={(v) => s('rules.minimum_reserve_soc_pct', v)} />
          <div className="toggle-row">
            <span>Full deploy allowed</span>
            <input className="checkbox" type="checkbox" checked={g('rules.full_deploy_allowed')} onChange={(e) => s('rules.full_deploy_allowed', e.target.checked)} />
          </div>
        </>
      )}

      {show('Decision mode') && (
        <>
          <div className="group-label">Decision mode</div>
          <div className="field">
            <select value={g('decision_mode')} onChange={(e) => s('decision_mode', e.target.value)}>
              <option value="VERIFIED">VERIFIED</option>
              <option value="FAST">FAST</option>
            </select>
          </div>
        </>
      )}
    </div>
  );
}
