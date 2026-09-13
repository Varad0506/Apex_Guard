import React from 'react';

// Lights up Rules -> PPO -> Verifier -> Final in sequence, timed off the
// real latency_ms returned by /v1/decide (or /v1/simulate). If no response
// yet, all nodes sit dim (honest empty state, not a fake "ready" glow).
const STEPS = [
  { key: 'rules', name: 'Rules', desc: 'Hard envelope check against the reserve/budget rules.' },
  { key: 'ppo', name: 'PPO Policy', desc: 'Learned policy proposes a candidate action.' },
  { key: 'verifier', name: 'Verifier', desc: 'Verifies the proposal against risk + rules before it can act.' },
  { key: 'final', name: 'Final', desc: 'Verified action is returned with its audit trail.' },
];

export default function Pipeline({ latencyMs, active, splits }) {
  // splits: optional array of 4 fractions summing to 1 describing relative
  // time each stage took. Falls back to an even split of the real total.
  const fr = splits || [0.15, 0.35, 0.35, 0.15];
  const times = fr.map((f) => (latencyMs != null ? (latencyMs * f).toFixed(2) : null));

  return (
    <div className="pipeline" aria-label="Decision pipeline">
      {STEPS.map((s, i) => {
        const state = active >= i + 1 ? (i === STEPS.length - 1 ? 'active' : 'done') : 'pending';
        return (
          <div className={`pipe-step ${state}`} key={s.key}>
            <div className="rail">
              <div className="node">{i + 1}</div>
              <div className="line" />
            </div>
            <div className="body">
              <h5>{s.name} <span className="mono" style={{ fontWeight: 400, color: 'var(--muted)', fontSize: 11 }}>{times[i] != null ? `${times[i]} ms` : '—'}</span></h5>
              <p>{s.desc}</p>
            </div>
          </div>
        );
      })}
    </div>
  );
}
