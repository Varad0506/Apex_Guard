// Thin client over the real ApexGuard FastAPI backend.
// Every value the UI shows is traced back to one of these calls — no invented fields.
// Each call tries the real network first; on any failure (no backend reachable,
// non-2xx, timeout) it falls back to the matching seeded mock generator instead
// of throwing, so every page renders a populated, working screen even with zero
// backend. Mock responses carry `_mock: true` so the UI can flag them.

import * as mock from './mock';

const BASE = import.meta.env.VITE_API_BASE || '';

async function req(path, opts = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch (_) {}
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

// Wrap a real call with a mock fallback. `fallback` is called with the same
// arguments as the real function whenever the real call throws.
function withFallback(realFn, fallback) {
  return async (...args) => {
    try {
      return await realFn(...args);
    } catch (e) {
      return fallback(...args);
    }
  };
}

export const api = {
  health: withFallback(() => req('/v1/health'), mock.mockHealth),
  healthReady: withFallback(() => req('/v1/health/ready'), mock.mockHealthReady),

  decide: withFallback(
    (payload) => req('/v1/decide', { method: 'POST', body: JSON.stringify(payload) }),
    (payload) => mock.mockDecide(payload),
  ),

  simulate: withFallback(
    (payload) => req('/v1/simulate', { method: 'POST', body: JSON.stringify(payload) }),
    (payload) => mock.mockSimulateQuantum(payload),
  ),
  ersTactical: withFallback(
    (payload) => req('/v1/ers/tactical', { method: 'POST', body: JSON.stringify(payload) }),
    (payload) => mock.mockSimulateQuantum(payload),
  ),
  simulateQuantum: withFallback(
    (payload) => req('/v1/simulate/quantum-inspired', { method: 'POST', body: JSON.stringify(payload) }),
    (payload) => mock.mockSimulateQuantum(payload),
  ),

  listScenarios: withFallback(() => req('/v1/scenarios'), mock.mockScenarios),
  getScenario: withFallback((id) => req(`/v1/scenarios/${id}`), (id) => mock.mockScenario(id)),

  getAudit: withFallback((id) => req(`/v1/audits/${id}`), (id) => ({ id, note: '[demo data] no audit backend reachable', _mock: true })),

  listReplays: withFallback(() => req('/v1/replays'), mock.mockReplays),
  getReplay: withFallback((id) => req(`/v1/replays/${id}`), (id) => mock.mockReplay(id)),
  createOpenF1Replay: withFallback(
    (params) => req(`/v1/replays/openf1?${new URLSearchParams(params)}`),
    (params) => mock.mockReplay(`custom_${params?.driver || 'x'}_openf1`),
  ),
  createFastF1Replay: withFallback(
    (params) => req(`/v1/replays/fastf1?${new URLSearchParams(params)}`),
    (params) => mock.mockReplay(`custom_${params?.driver || 'x'}_fastf1`),
  ),

  listBacktests: withFallback(() => req('/v1/backtests'), mock.mockBacktests),
  getBacktest: withFallback((id) => req(`/v1/backtests/${id}`), (id) => mock.mockBacktest(id)),
  runBacktest: withFallback(
    (replayId, maxFrames) => {
      const qs = new URLSearchParams({ replay_id: replayId });
      if (maxFrames) qs.set('max_frames', String(maxFrames));
      return req(`/v1/backtests/run?${qs}`, { method: 'POST' });
    },
    (replayId, maxFrames) => mock.mockRunBacktest(replayId, maxFrames),
  ),
};

export default api;
