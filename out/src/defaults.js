export const ACTION_COLOR_CLASS = {
  HARVEST: 's-harvest',
  PARTIAL_DEPLOY: 's-partial_deploy',
  FULL_DEPLOY: 's-full_deploy',
  HOLD: 's-hold',
};

export const ACTION_HEX = {
  HARVEST: '#3dbfc9',
  PARTIAL_DEPLOY: '#e8a33d',
  FULL_DEPLOY: '#d14545',
  HOLD: '#d14545',
};

export const TRACKS = [
  { track_id: 'synth_balanced_01', label: 'Balanced Circuit 01', overtake_difficulty: 0.5, drs_zone_count: 2, lap_length_m: 5300, corner_density: 0.44 },
  { track_id: 'synth_balanced_02', label: 'Balanced Circuit 02', overtake_difficulty: 0.53, drs_zone_count: 2, lap_length_m: 5250, corner_density: 0.46 },
  { track_id: 'synth_extreme_high_speed_01', label: 'Extreme High-Speed 01', overtake_difficulty: 0.2, drs_zone_count: 2, lap_length_m: 7000, corner_density: 0.18 },
  { track_id: 'synth_high_speed_01', label: 'High-Speed 01', overtake_difficulty: 0.35, drs_zone_count: 2, lap_length_m: 5800, corner_density: 0.28 },
  { track_id: 'synth_high_speed_02', label: 'High-Speed 02', overtake_difficulty: 0.3, drs_zone_count: 3, lap_length_m: 6200, corner_density: 0.24 },
  { track_id: 'synth_low_speed_street_01', label: 'Low-Speed Street 01', overtake_difficulty: 0.88, drs_zone_count: 1, lap_length_m: 3300, corner_density: 0.85 },
  { track_id: 'synth_low_speed_street_02', label: 'Low-Speed Street 02', overtake_difficulty: 0.82, drs_zone_count: 1, lap_length_m: 3600, corner_density: 0.8 },
  { track_id: 'synth_power_circuit_01', label: 'Power Circuit 01', overtake_difficulty: 0.25, drs_zone_count: 3, lap_length_m: 5900, corner_density: 0.2 },
  { track_id: 'synth_technical_mixed_01', label: 'Technical Mixed 01', overtake_difficulty: 0.65, drs_zone_count: 2, lap_length_m: 4900, corner_density: 0.6 },
  { track_id: 'synth_technical_mixed_02', label: 'Technical Mixed 02', overtake_difficulty: 0.6, drs_zone_count: 2, lap_length_m: 5100, corner_density: 0.55 },
];

export function makeRequestId() {
  return `ui-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
}

export function defaultDecisionRequest() {
  return {
    request_id: makeRequestId(),
    battle_id: 'ui-session-battle',
    timestamp_ms: Date.now(),
    telemetry_age_ms: 40,
    ego: {
      soc_pct: 62,
      speed_kph: 288,
      throttle_pct: 98,
      brake_pct: 0,
      tyre_grip_estimate: 0.82,
      laps_remaining: 12,
      current_mode: 'HOLD',
    },
    target: {
      gap_s: 0.45,
      relative_speed_kph: 16,
      stint_age_laps: 20,
      recent_sector_delta_s: -0.25,
    },
    traffic: {
      rear_gap_s: 2.4,
      cars_within_3s: 1,
      post_pass_traffic_gap_s: 2.1,
    },
    track: {
      track_id: 'synth_high_speed_01',
      segment_type: 'DRS_STRAIGHT',
      drs_available: true,
      straight_remaining_m: 620,
      braking_zone_m: 130,
      overtake_difficulty: 0.4,
      detection_point_active: false,
      overtake_mode_available: false,
      ers_key_acceleration_zone: false,
    },
    rules: {
      deployment_budget_remaining_kj: 1900,
      minimum_reserve_soc_pct: 15,
      full_deploy_allowed: true,
    },
    decision_mode: 'VERIFIED',
  };
}

export const SEGMENT_TYPES = ['STRAIGHT', 'DRS_STRAIGHT', 'CORNER', 'BRAKING_ZONE'];
export const MODES = ['HOLD', 'HARVEST', 'PARTIAL_DEPLOY', 'FULL_DEPLOY'];
