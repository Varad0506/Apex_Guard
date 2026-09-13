// Deterministic mock data used as a fallback when the real ApexGuard API is
// unreachable (e.g. this static build with no backend running). Every shape
// here matches exactly what the FastAPI backend returns for the same
// endpoint, so pages don't need to branch on "is this mock" to render —
// they just get a populated, working screen. Responses are tagged `_mock:
// true` so the UI can show a small "demo data" badge instead of silently
// pretending it's live.

import { TRACKS } from './defaults';

// --- seeded PRNG -----------------------------------------------------------
// mulberry32: small, fast, deterministic. Same seed -> same sequence, so
// mock output doesn't reshuffle on every re-render, only when the key
// (track/scenario/replay id) changes.
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function rand() {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function seedFromString(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

const DRIVERS = ['VER', 'LEC', 'HAM', 'NOR', 'PIA', 'RUS', 'SAI', 'ALO'];
const EVENTS = ['Bahrain GP', 'Monaco GP', 'Silverstone GP', 'Monza GP', 'Suzuka GP'];

// --- health / readiness -----------------------------------------------------
export function mockHealth() {
  return { status: 'ok', _mock: true };
}

export function mockHealthReady() {
  return {
    status: 'ready',
    overtake_model: 'trained-v1',
    policy_available: true,
    traffic_engine: true,
    _mock: true,
  };
}

// --- scenarios ---------------------------------------------------------------
const SCENARIO_IDS = ['isolated_pass', 'low_soc_guard', 'rear_drs_threat'];

export function mockScenarios() {
  return { scenarios: SCENARIO_IDS.map((id) => ({ id })), _mock: true };
}

export function mockScenario(id) {
  const rand = mulberry32(seedFromString(id));
  const base = {
    request_id: `mock-${id}`,
    battle_id: `mock-battle-${id}`,
    timestamp_ms: Date.now(),
    telemetry_age_ms: 35,
    ego: {
      soc_pct: 62, speed_kph: 285, throttle_pct: 97, brake_pct: 0,
      tyre_grip_estimate: 0.8, laps_remaining: 10, current_mode: 'HOLD',
    },
    target: { gap_s: 0.5, relative_speed_kph: 14, stint_age_laps: 18, recent_sector_delta_s: -0.2 },
    traffic: { rear_gap_s: 2.3, cars_within_3s: 1, post_pass_traffic_gap_s: 2.0 },
    track: {
      track_id: 'synth_high_speed_01', segment_type: 'DRS_STRAIGHT', drs_available: true,
      straight_remaining_m: 600, braking_zone_m: 120, overtake_difficulty: 0.4,
      detection_point_active: false, overtake_mode_available: false, ers_key_acceleration_zone: false,
    },
    rules: { deployment_budget_remaining_kj: 1900, minimum_reserve_soc_pct: 15, full_deploy_allowed: true },
    decision_mode: 'VERIFIED',
  };
  if (id === 'isolated_pass') {
    base.target.gap_s = 0.3;
    base.traffic.cars_within_3s = 0;
  } else if (id === 'low_soc_guard') {
    base.ego.soc_pct = 16.5;
    base.rules.full_deploy_allowed = false;
  } else if (id === 'rear_drs_threat') {
    base.traffic.rear_gap_s = 0.6;
    base.track.detection_point_active = true;
  }
  void rand; // reserved for future jitter; kept deterministic for now
  return { ...base, _mock: true };
}

// --- decide ------------------------------------------------------------------
export function mockDecide(request) {
  const key = `${request?.request_id || ''}-${request?.battle_id || ''}-${JSON.stringify(request?.ego || {})}`;
  const rand = mulberry32(seedFromString(key));
  const soc = request?.ego?.soc_pct ?? 60;
  const gap = request?.target?.gap_s ?? 1;
  const reserveFloor = request?.rules?.minimum_reserve_soc_pct ?? 15;

  let action = 'HOLD';
  if (soc <= reserveFloor + 3) action = 'HARVEST';
  else if (gap < 0.6 && (request?.track?.drs_available)) action = rand() > 0.4 ? 'FULL_DEPLOY' : 'PARTIAL_DEPLOY';
  else if (gap < 1.2) action = 'PARTIAL_DEPLOY';

  const ruleCompliant = !(action === 'FULL_DEPLOY' && soc - 8 < reserveFloor);
  const finalAction = ruleCompliant ? action : 'HARVEST';

  return {
    recommended_action: finalAction,
    rule_compliant: ruleCompliant,
    status: ruleCompliant ? 'OK' : 'DEGRADED',
    explanation: `[demo data] With ${soc.toFixed(1)}% SOC and a ${gap.toFixed(2)}s gap, the mock verifier recommends ${finalAction} while respecting the ${reserveFloor}% reserve floor.`,
    latency_ms: 4 + rand() * 6,
    audit_id: `mock-audit-${Math.floor(rand() * 1e6)}`,
    traffic_summary: {
      target_gap_s: gap,
      rear_gap_s: request?.traffic?.rear_gap_s ?? 2.2,
      tow_strength: clamp(1 - gap / 3, 0, 1),
      post_pass_traffic_risk: clamp((request?.traffic?.cars_within_3s ?? 1) / 6 + rand() * 0.15, 0, 1),
    },
    traffic_model: 'baseline',
    opponent_belief: mockOpponentBelief(rand),
    _mock: true,
  };
}

function mockOpponentBelief(rand) {
  const raw = { ATTACKING: rand(), DEFENDING: rand(), HARVESTING: rand(), CONSERVING: rand() };
  const total = Object.values(raw).reduce((a, b) => a + b, 0) || 1;
  Object.keys(raw).forEach((k) => { raw[k] = raw[k] / total; });
  return {
    tactical_belief: raw,
    override_belief: { AVAILABLE: rand() },
    aero_belief: { LOW_DRAG: rand() },
    counter_harvest_trap_probability: rand() * 0.4,
    confidence: 0.5 + rand() * 0.45,
    temporal: true,
    update_count: 1,
    racer_pattern: { model: 'demo-hmm-v1' },
  };
}

// --- replays -------------------------------------------------------------------
const REPLAY_IDS = [
  'bahrain_ver_lec_openf1', 'monaco_ham_nor_openf1', 'silverstone_pia_rus_openf1',
  'monza_sai_alo_openf1', 'suzuka_ver_ham_openf1',
];

export function mockReplays() {
  return {
    replays: REPLAY_IDS.map((id) => ({ id, file: `${id}.json` })),
    _mock: true,
  };
}

export function mockReplay(id) {
  const rand = mulberry32(seedFromString(id));
  const idx = Math.abs(seedFromString(id)) % EVENTS.length;
  const driver = DRIVERS[Math.abs(seedFromString(id + 'a')) % DRIVERS.length];
  let targetDriver = DRIVERS[Math.abs(seedFromString(id + 'b')) % DRIVERS.length];
  if (targetDriver === driver) targetDriver = DRIVERS[(DRIVERS.indexOf(targetDriver) + 1) % DRIVERS.length];

  const nFrames = 40 + Math.floor(rand() * 40);
  let gap = 1.2 + rand() * 1.5;
  let soc = 70 + rand() * 15;
  const frames = [];
  for (let i = 0; i < nFrames; i++) {
    gap = clamp(gap + (rand() - 0.52) * 0.15, 0.05, 3.2);
    soc = clamp(soc - rand() * 0.25, 12, 100);
    const segType = ['STRAIGHT', 'DRS_STRAIGHT', 'CORNER', 'BRAKING_ZONE'][Math.floor(rand() * 4)];
    frames.push({
      request_id: `${id}-f${i}`,
      battle_id: id,
      timestamp_ms: Date.now() + i * 200,
      telemetry_age_ms: 35,
      ego: {
        soc_pct: soc, speed_kph: 250 + rand() * 60, throttle_pct: 90 + rand() * 10, brake_pct: rand() * 20,
        tyre_grip_estimate: 0.7 + rand() * 0.2, laps_remaining: Math.max(1, 20 - Math.floor(i / 4)), current_mode: 'HOLD',
      },
      target: { gap_s: gap, relative_speed_kph: (rand() - 0.5) * 30, stint_age_laps: 10 + Math.floor(i / 4), recent_sector_delta_s: (rand() - 0.5) * 0.6 },
      traffic: { rear_gap_s: 0.5 + rand() * 2.5, cars_within_3s: Math.floor(rand() * 3), post_pass_traffic_gap_s: 0.5 + rand() * 2.5 },
      track: {
        track_id: 'synth_high_speed_01', segment_type: segType, drs_available: segType === 'DRS_STRAIGHT',
        straight_remaining_m: rand() * 700, braking_zone_m: rand() * 150, overtake_difficulty: 0.3 + rand() * 0.4,
        detection_point_active: rand() > 0.7, overtake_mode_available: rand() > 0.6, ers_key_acceleration_zone: rand() > 0.7,
      },
      rules: { deployment_budget_remaining_kj: 1500 + rand() * 800, minimum_reserve_soc_pct: 15, full_deploy_allowed: soc > 25 },
      decision_mode: 'VERIFIED',
    });
  }

  return {
    id,
    year: 2023 + (Math.abs(seedFromString(id)) % 3),
    event: EVENTS[idx],
    session: 'Race',
    driver,
    target_driver: targetDriver,
    frames,
    duration_s: nFrames * 0.2,
    source: 'cached',
    provider: 'openf1 (demo)',
    _mock: true,
  };
}

// --- backtests -------------------------------------------------------------------
export function mockBacktests() {
  return {
    backtests: REPLAY_IDS.map((id) => ({ id: `${id}_backtest`, file: `${id}_backtest.json` })),
    _mock: true,
  };
}

function mockBacktestResult(id, framesEvaluated) {
  const rand = mulberry32(seedFromString(id));
  const replayId = id.replace(/_backtest$/, '');
  const idx = Math.abs(seedFromString(replayId)) % EVENTS.length;
  const driver = DRIVERS[Math.abs(seedFromString(replayId + 'a')) % DRIVERS.length];
  let targetDriver = DRIVERS[Math.abs(seedFromString(replayId + 'b')) % DRIVERS.length];
  if (targetDriver === driver) targetDriver = DRIVERS[(DRIVERS.indexOf(targetDriver) + 1) % DRIVERS.length];

  const actionCounts = { HOLD: 0, HARVEST: 0, PARTIAL_DEPLOY: 0, FULL_DEPLOY: 0 };
  const actions = Object.keys(actionCounts);
  for (let i = 0; i < framesEvaluated; i++) {
    actions[Math.floor(rand() * actions.length)] && (actionCounts[actions[Math.floor(rand() * actions.length)]] += 1);
  }

  return {
    id,
    event: EVENTS[idx],
    session: 'Race',
    driver,
    target_driver: targetDriver,
    frames_evaluated: framesEvaluated,
    metrics: {
      rule_compliance_pct: +(92 + rand() * 8).toFixed(1),
      mean_decision_confidence_pct: +(65 + rand() * 25).toFixed(1),
      mean_latency_ms: +(3 + rand() * 5).toFixed(2),
    },
    action_counts: actionCounts,
    _mock: true,
  };
}

export function mockBacktest(id) {
  return mockBacktestResult(id, 40 + Math.floor(mulberry32(seedFromString(id))() * 60));
}

export function mockRunBacktest(replayId, maxFrames) {
  return mockBacktestResult(`${replayId}_backtest`, maxFrames || 60);
}

// --- simulate / quantum-inspired -------------------------------------------------
export function mockSimulateQuantum(payload) {
  const key = JSON.stringify(payload);
  const rand = mulberry32(seedFromString(key));
  const nPaths = payload?.n_paths || 60;

  const passProb = clamp(0.35 + rand() * 0.45, 0, 1);
  const counterProb = clamp(0.1 + rand() * 0.3, 0, 1);
  const breachProb = clamp(rand() * 0.15, 0, 1);
  const expUtil = passProb - counterProb * 0.6 - breachProb * 0.9;

  // Amplitude-inspired encoding: same underlying probabilities, re-expressed
  // as amplitudes (sqrt of probability) plus a small deterministic jitter so
  // the two panels look related but distinct, same as the real endpoint.
  const toAmp = (p) => clamp(Math.sqrt(p) * (0.92 + rand() * 0.12), 0, 1);

  return {
    classical_monte_carlo: {
      sample_count: nPaths,
      pass_probability: passProb,
      counterattack_probability: counterProb,
      reserve_breach_probability: breachProb,
      expected_utility: expUtil,
    },
    quantum_inspired: {
      amplitude_success: toAmp(passProb),
      amplitude_counterattack: toAmp(counterProb),
      amplitude_reserve_breach: toAmp(breachProb),
      expected_utility: expUtil * (0.95 + rand() * 0.1),
    },
    note: '[demo data] No backend reachable — this is a seeded mock rollout, not a live quantum-inspired simulation.',
    _mock: true,
  };
}
