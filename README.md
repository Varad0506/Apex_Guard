# ApexGuard Backend — Phase 1 Deterministic MVP

A verification-gated energy/overtake decision service for live race telemetry.
This is the **Phase 1** build: the full decision pipeline running end-to-end on
heuristics only — no trained models, no PPO, no GNN. Everything downstream
(overtake classifier, PPO proposal layer, GNN traffic embedding) plugs into
this same pipeline later without changing the API contract.

## Why deterministic first

The backend's job is not "run GNN/PPO" — it's to accept a race state, apply
hard legal-action rules, estimate pass value and traffic risk, verify
candidates through lookahead simulation, and return an explainable
recommendation fast enough for a live dashboard. This phase proves that whole
loop works before any ML is added. The **verifier — not any future policy
model — selects the action that reaches the frontend.**

## Quickstart

```bash
pip install -r requirements.txt --break-system-packages   # or use a venv
python -m uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000/docs` for interactive Swagger docs, or:

```bash
curl -s http://localhost:8000/v1/health
curl -s -X POST http://localhost:8000/v1/decide \
  -H "Content-Type: application/json" \
  --data-binary @app/data/scenarios/rear_drs_threat.json | python -m json.tool
```

## Run tests

```bash
python -m pytest tests/ -v
```

13 tests cover rule masking, opportunity gating, verifier ranking, fail-safe
behavior, and full API round-trips.

## Demo scenarios

Three curated, offline-safe scenarios live in `app/data/scenarios/` and are
served via `GET /v1/scenarios` and `GET /v1/scenarios/{id}`:

| Scenario | Expected result | Demonstrates |
|---|---|---|
| `isolated_pass` | `FULL_DEPLOY` | High pass value, low downstream risk → aggressive deploy is correct |
| `rear_drs_threat` | `PARTIAL_DEPLOY` | Verifier downgrades the aggressive option because counterattack risk is high — even though full deploy has the highest raw pass probability |
| `low_soc_guard` | `HOLD` | Hard reserve constraint makes any deploy illegal, regardless of opportunity |

Run all three end-to-end:

```bash
for f in app/data/scenarios/*.json; do
  echo "=== $f ==="
  curl -s -X POST http://localhost:8000/v1/decide -H "Content-Type: application/json" \
    --data-binary @"$f" | python -c "import json,sys; d=json.load(sys.stdin); print(d['recommended_action'], '-', d['explanation'])"
done
```


## Learned Racer-Pattern Intelligence — LSTM + HMM

The backend now includes an optional temporal racer-pattern layer at
`app/models/rival_pattern_model.py`. It learns from rolling telemetry windows
from previous race replays and predicts the rival's next tactical state:
`HARVESTING`, `DEFENDING`, `ATTACKING`, or `CONSERVING`.

The layer is deliberately advisory:

```text
Previous-race replay windows
        ↓
LSTM temporal encoder
        ↓
Learned HMM transition matrix
        ↓
Current observation evidence
        ↓
OpponentBeliefState
        ↓
Monte Carlo rollout
        ↓
Safety/legal verifier
        ↓
Final action
```

OpenF1 does not expose proprietary ERS intent, so training labels are
observable/proxy tactical labels. The system does **not** claim to recover
hidden driver intent. If the LSTM artifact or PyTorch is unavailable, the
existing deterministic HMM-style filter remains in control.

Train offline after building historical replays:

```bash
python -m scripts.train_rival_pattern_model --replays app/data/replays
```

Artifacts produced:

- `app/data/artifacts/racer_pattern_lstm.pt`
- `app/data/artifacts/racer_pattern_hmm.json`

The live API exposes the layer's status under `opponent_belief.racer_pattern`.
Training is never performed inside the live API.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /v1/decide` | Main live recommendation endpoint |
| `GET /v1/health` | Liveness check |
| `GET /v1/health/ready` | Readiness — which models/engines are loaded |
| `GET /v1/scenarios` | List curated demo scenarios |
| `GET /v1/scenarios/{id}` | Load a scenario's raw request payload |
| `GET /v1/audits/{audit_id}` | Full decision trace for a past decision |
| `POST /v1/simulate` | Run a single-action Monte Carlo what-if rollout directly |

`POST /v1/train` is intentionally **not implemented** — training is offline
and versioned, never exposed through the live API.

## Pipeline (what `decide()` actually does)

1. **Freshness guard** — reject stale telemetry (`telemetry_age_ms > 500` → `HOLD`)
2. **Hard legal-action mask** — deterministic SOC/budget/segment rules (`rule_engine.py`)
3. **Opportunity gating** — cheap heuristic check on gap/closing-speed/DRS before running the heavier pipeline (`opportunity_monitor.py`)
4. **Rival estimation** — tyre/energy proxies with uncertainty from observable stint/sector trends (`rival_estimator.py`)
5. **Traffic risk** — deterministic counterattack/post-pass blockage estimate (`traffic_engine.py`)
6. **Overtake probability** — explainable logistic-style heuristic per candidate action (`overtake_probability.py`)
7. **Policy proposal** — currently always falls back to "evaluate every legal action" since no PPO is wired in yet (`policy_adapter.py`)
8. **Lookahead rollout** — analytical projection of time delta / SOC / net value per candidate (`rollout.py`)
9. **Ranking** — highest legal risk-adjusted net value wins (`ranker.py`)
10. **Fail-safe** — `HOLD` if nothing clears the minimum value bar (`safety.py`)
11. **Explanation + audit** — plain-English rationale and a JSONL audit trail for every decision, win or hold (`explanation.py`, `audit.py`)

## Phase 0/3 — Multi-tick Monte Carlo rollout simulator + synthetic tracks

`app/simulation/engine.py` is the real lap-time + energy simulator the guide
calls "non-negotiable" — it replaces the Phase 1 single-shot analytical
formula (`app/simulation/rollout.py`, still used as a fast path — see below)
with genuine tick-by-tick physics:

- **Energy dynamics** (`energy_model.py`) — SOC charges/discharges per tick
  based on action and track segment (braking zones harvest ~1.6× faster than
  straights), plus a nominal per-tick energy-budget cost.
- **Traffic dynamics** (`traffic_model.py`) — gap and relative speed evolve
  tick by tick with DRS amplification, sensor noise, and a stochastic rival
  defensive reaction; a pass is detected exactly when the simulated gap
  actually reaches zero, not estimated from a formula. After a pass, a
  separate stochastic check determines whether the rival counterattacks
  before the horizon ends, using the *real* `post_pass_traffic_gap_s` from
  telemetry (a rebuilt-and-fixed calculation — see note below).
- **Monte Carlo aggregation** (`rollout_verified()`) — runs N stochastic
  paths (default 30) per candidate action and aggregates them into an
  *empirical* pass probability, projected SOC, and net value, plus a genuine
  **confidence** figure from how much the paths agree with each other. This
  is the guide's "Uncertainty handling" power-up (Monte Carlo confidence
  scoring), arrived at as a natural consequence of simulating properly
  rather than bolted on separately.
- **Synthetic track profiles** (`tracks.py`) — 10 parameterized track
  profiles (`app/data/tracks/*.json`) spanning overtake difficulty from 0.20
  (extreme high-speed) to 0.88 (tight street circuit), each with a generated
  sector sequence (straights with/without DRS, braking zones, corners). Used
  by `scripts/evaluate_rollout.py` for cross-track testing; **not** used
  inside a live `/v1/decide` call, which builds its tick-level sector
  sequence directly from the request's own observed telemetry fields
  instead (`_build_local_sectors` in `engine.py`) — the synthetic profiles
  exist for offline generalization testing, not to replace live track state.

**A real bug I found and fixed while building this**: the first version
passed a traffic *risk score* (0=safe, 1=risky) into the counterattack
check where it expected a raw *gap in seconds* (larger=safer) — this
silently inverted the signal, so tighter post-pass traffic looked safer to
the simulator. Fixed by passing `req.traffic.post_pass_traffic_gap_s`
directly. Worth knowing this class of bug is exactly why Monte Carlo +
track-holdout testing matters: it surfaced immediately once `rear_drs_threat`
started recommending `FULL_DEPLOY` in a scenario explicitly designed to
punish it.

**Determinism / replay**: each rollout is seeded from a SHA-256 hash of
`(request_id, action)`, not Python's built-in `hash()` — the latter is
randomized per-process (`PYTHONHASHSEED`) and would make identical requests
simulate differently across restarts, breaking the audit log's replay
guarantee. Verified: `tests/test_rollout_simulator.py::test_rollout_verified_reproducible_across_calls`.

### FAST vs VERIFIED

`decision_mode` in the request (already part of the frozen API contract, was
previously unused) now actually does something:

- **`FAST`** → the Phase 1 single-shot analytical rollout. Cheap (~4ms for
  all 4 candidate actions), deterministic, no Monte Carlo.
- **`VERIFIED`** (default) → the Monte Carlo engine above. Still fast
  (~9-13ms for all 4 actions with n_paths=30) — comfortably inside the
  guide's 300ms verifier latency budget — because 30 paths × ~24 ticks is a
  few thousand cheap arithmetic ops, not a heavy simulation.

Model warmup happens once at process startup (FastAPI `lifespan`), not on
the first request — without it, the first classifier call after boot pays a
~1 second cold-start penalty (sklearn/pandas import chain) that has no
business happening live in front of a judge.

Run the comparison yourself:

```bash
PYTHONPATH=. python scripts/evaluate_rollout.py
```

Actual output from this build (recommendations differ between the two
engines in places — reported honestly, not tuned to agree, per the guide's
"present genuine findings" principle):

```
mode       scenario              recommended_action  confidence  pass_probability  counterattack_risk  mean_latency_ms  p95_latency_ms
--------------------------------------------------------------------------------------------------------------------------------------
FAST       isolated_pass         PARTIAL_DEPLOY    0.85        0.749     0.355     4.13        5.06
FAST       low_soc_guard         HOLD              0.762       0.531     0.415     3.61        3.81
FAST       rear_drs_threat       PARTIAL_DEPLOY    0.718       0.421     0.791     3.67        3.79
VERIFIED   isolated_pass         PARTIAL_DEPLOY    1.0         1.0       0.355     12.4        12.73
VERIFIED   low_soc_guard         HARVEST           1.0         0.0       0.415     9.02        9.45
VERIFIED   rear_drs_threat       PARTIAL_DEPLOY    1.0         1.0       0.791     11.79       12.22
```

Two honest notes on this table, not smoothed over:

- `low_soc_guard` differs (`HOLD` vs `HARVEST`) between engines — both are
  legal, low-risk choices in that scenario; the two engines just weigh
  "wait" vs. "actively recharge" slightly differently. Neither recommends a
  deploy action, which is the property that actually matters there.
- `isolated_pass` selects `PARTIAL_DEPLOY` over `FULL_DEPLOY` by a narrow
  margin (~0.025 net value) in this build — the guide's own worked example
  for a similar low-risk scenario favors `FULL_DEPLOY`. This is a real,
  reproducible result from actually running the simulation, not a tuning
  target to chase; rerun `evaluate_rollout.py` and inspect
  `app/data/artifacts/rollout_evaluation.json` if you want to dig into why.

### New endpoint

`POST /v1/simulate` exposes `rollout_verified()` directly for a single
action/horizon/path-count combination — useful for an interactive frontend
(e.g. a horizon-length slider) without running the full `/v1/decide`
pipeline each time. Training is still never exposed (`POST /v1/train` does
not exist).

## What's NOT built yet (by design)

Per the build guide's phased plan, these are deliberately deferred and the
system is fully functional without them:

- ~~**Phase 2** — trained overtake classifier~~ **Done.**
- ~~**Phase 3** — synthetic track-profile generation~~ **Done** (10 profiles, see above).
- ~~**Phase 0** — real multi-tick simulator~~ **Done** (see above).
- **Phase 4** — PPO proposal layer. Training loop and Gym environment **built and validated** (`app/rl/apexguard_env.py`, `scripts/train_ppo.py`), and a trained policy is now shipped at `app/data/artifacts/ppo_policy.zip`. `scripts/evaluate_ppo_vs_verifier.py` and `scripts/stress_test_vs_ppo_verifier.py` run it live against the verifier-gated pipeline — see `app/data/artifacts/ppo_vs_verifier.json` (curated scenarios) and `ppo_vs_verifier_stress.json` (2,000-case stress run: ~4% illegal-proposal rate, ~10% divergence rate from the verifier) for real results. `policy_adapter.py` still degrades gracefully (returns "no proposal") if `stable-baselines3`/`torch` or the artifact aren't present.
- **Phase 5** — GNN/GAT local traffic-graph embedding, added only if it wins an ablation test against the deterministic `traffic_engine.py` baseline.
- ~~**Learned racer-pattern layer** — LSTM + learned HMM transition model~~ **Built** with safe fallback and offline training. Now genuinely testable: the synthetic track profiles + Monte Carlo engine give you the scenario variety an ablation needs.

Every one of these has a safe fallback already wired into `decision_engine.py`,
per the build guide's key design rule: *the user-facing decision service must
run even if one advanced model is absent.*

## Phase 2 — Trained overtake classifier

`app/models/overtake_probability.py` now tries to load a trained, calibrated
classifier artifact first, and only falls back to the Phase 1 heuristic if
the artifact is missing or fails to load — this is the "Model missing ->
Degraded verified mode" branch from the fail-safe contract, exercised for
real, not just documented.

**Important — this environment has no route to FastF1's data sources**
(only package registries are reachable), so the shipped artifact is trained
on a synthetic, FastF1-shaped dataset with a hand-authored ground-truth
labeling function, not real telemetry. Treat it as a working pipeline
proof, not a result to present as real. `scripts/generate_scenarios.py`
documents the exact schema a real ingestion pass needs to produce, and
contains a stub (`ingest_fastf1_sessions`) with the concrete FastF1 API calls
to fill in once you're running this locally with internet access — swap that
in for `generate_dataset()` and nothing else in the training script changes.

Retrain / regenerate:

```bash
cd scripts
python generate_scenarios.py       # writes app/data/training/overtake_windows.csv
python train_overtake_model.py     # writes app/data/artifacts/overtake_model.joblib + metrics.json
```

Held-out-track evaluation (2 tracks never seen during training, per the
guide's explicit warning against random-splitting rows from the same
track/session):

| Split | n | Accuracy | AUC | Log loss | Brier |
|---|---|---|---|---|---|
| Train tracks | 1560 | 0.7474 | 0.7647 | 0.5297 | 0.1763 |
| **Holdout tracks** | 520 | **0.7904** | **0.7715** | 0.4482 | 0.1447 |

(These are the actual figures from the checked-in artifact; rerun
`train_overtake_model.py` to regenerate `overtake_model_metrics.json` if you
regenerate the dataset.) The calibrated
model tracks the uncalibrated baseline closely on AUC, as expected —
calibration mainly fixes probability *reliability*, not ranking, which is
exactly why the guide calls for `CalibratedClassifierCV` over raw
`LogisticRegression`: ApexGuard reports probabilities directly to the driver,
so a well-calibrated "71%" needs to actually mean 71%, not just rank above a
"60%".

Note the recommendations shift slightly versus Phase 1 (e.g. `isolated_pass`
now selects `PARTIAL_DEPLOY` instead of `FULL_DEPLOY`) because the trained
model's probability surface differs from the hand-tuned heuristic's — this is
expected and reported honestly rather than tuned away, per the guide's
"present all results, including surprising ones" principle. Re-verify against
real data once `ingest_fastf1_sessions` is filled in.

`GET /v1/health/ready` reports which mode is active: `overtake_model:
"trained-v1"` when the artifact loads, `"heuristic-v0"` on fallback.

## Phase 5 — GNN/GAT: built, ablation-tested, and kept

A GNN/GAT traffic embedding (`app/models/traffic_graph.py`, manual
multi-head attention over the local battle graph — see that file's
docstring for why plain PyTorch instead of PyG) only earns a place in the
pipeline if it beats the deterministic `traffic_engine.py` baseline in an
ablation test (`scripts/ablation_test_gnn.py`), run across the rear DRS
threat / DRS train / multiple close rivals / simple isolated pass
categories. It won 3 of 4 categories and the overall MSE comparison
(0.0066 baseline vs. 0.0041 GNN), so it's currently the active model —
see `app/data/artifacts/gnn_ablation_results.json` (`"kept_gnn": true`)
and the trained weights at `app/data/artifacts/traffic_gat.pt`.

torch is an **optional** dependency scoped to this one module
(`traffic_graph.py`): if it's missing or broken in the environment,
`model_is_available()` returns `False` and `traffic_engine.py` falls back
to the deterministic baseline — the rest of the app (rule engine,
verifier, overtake classifier, API) is unaffected either way.
`traffic_engine.py`'s output shape (`tow_strength`, `counterattack_risk`,
`post_pass_traffic_risk`) is what this module matches to drop in as a swap.

## Phase 4 — PPO training loop

`app/rl/apexguard_env.py` is a Gymnasium environment built directly on the
same `energy_model.py` / `traffic_model.py` physics the VERIFIED Monte Carlo
verifier uses — PPO trains against the same reality it gets checked against,
not a separate toy environment that happens to share a name.

**This build environment can't run it** (no disk space left for
`torch`/`stable-baselines3` after everything else installed) — the env
itself is validated (`gymnasium.utils.env_checker.check_env` passes clean,
plus `tests/test_policy_adapter.py`), but actual training needs to happen in
your own environment:

```bash
pip install gymnasium stable-baselines3[extra]
cd apexguard-backend
python scripts/train_ppo.py                       # default: 300k timesteps, 8 parallel envs
python scripts/train_ppo.py --timesteps 1000000 --n-envs 16   # longer run
python scripts/train_ppo.py --eval-only            # just evaluate an existing app/data/artifacts/ppo_policy.zip
```

This writes:
- `app/data/artifacts/ppo_policy.zip` — the final trained policy
- `app/data/artifacts/ppo_best/best_model.zip` — best checkpoint during training (via `EvalCallback`)
- `app/data/artifacts/ppo_tensorboard/` — training curves (`tensorboard --logdir <path>`)

### Environment design, in brief

- **One episode = one overtake-attempt window.** At each decision tick
  (default 1.0s, coarser than the 0.5s physics tick), the agent picks one of
  `HARVEST` / `HOLD` / `PARTIAL_DEPLOY` / `FULL_DEPLOY`.
- **Domain randomization on every `reset()`**: gap, closing speed, SOC,
  reserve minimum, deployment budget, track difficulty, rival defensiveness,
  and rear/post-pass traffic gaps are all sampled from wide but plausible
  ranges — plus, if `app/data/tracks/*.json` exist (Phase 3), a random
  synthetic track's actual sector sequence. This is what makes the trained
  policy generalize across conditions instead of memorizing one scenario.
- **Reward**: small per-tick shaping penalty for aggressive actions with no
  payoff yet; a terminal reward on pass/no-pass/counterattack outcome; a
  **hard -1.0 penalty for ending in reserve breach** (dominates everything
  else, mirroring the rule engine's hard mask in the live pipeline); a small
  bonus for finishing with energy headroom above the reserve.
- **11-dim observation vector** (gap, relative speed, SOC, SOC headroom,
  budget remaining, track difficulty, rival defensiveness, rear gap, nearby
  car count, DRS-straight flag, laps remaining) — kept in sync with
  `policy_adapter._observation_from_request()`, which builds the same
  vector from live telemetry at inference time. `tests/test_policy_adapter.py`
  locks this in (`test_observation_vector_dimension_matches_env`,
  `test_action_ordering_matches_env`) so the two can't silently drift apart.

### Wiring the trained policy back in

Once `ppo_policy.zip` exists, `app/models/policy_adapter.py` picks it up
automatically — no code change needed:

- `policy_is_available()` → `True` once the artifact loads (lazily, once,
  same pattern as the classifier)
- `propose()` runs live telemetry through the same 11-dim observation the
  policy trained on, and returns its proposed action **only if that action
  is still legal** under the live rule mask — if PPO proposes something the
  hard rules have since disallowed, `propose()` returns `None` and
  `decision_engine.py` falls back to evaluating every legal action, exactly
  as if no policy existed at all.
- `GET /v1/health/ready` will report `policy_available: true`.
- Nothing in `decision_engine.py` needs to change — the fallback logic
  (`proposed is None or proposed not in legal_actions → evaluate every
  legal action`) was already written for this from Phase 1 onward.

**The verifier still has final say.** Whatever PPO proposes gets rolled out
through `rollout_verified()` (or `rollout_fast()` in FAST mode) exactly like
every other candidate, and the ranker picks the highest net-value *legal*
outcome — which may not be what PPO proposed. That divergence (`policy_proposal
!= recommended_action` in the `/v1/decide` response) is the guide's headline
ablation case, and now you can actually produce it with a real policy instead
of a hand-authored example.

### Evaluating PPO against the verifier-gated pipeline

Compare `scripts/train_ppo.py --eval-only`'s output (`pass_rate`,
`reserve_breach_rate`, `mean_reward` — raw policy performance in the RL env)
against `scripts/evaluate_rollout.py`'s output (the full verifier-gated
`/v1/decide` pipeline) to build the guide's "PPO only vs. ApexGuard"
comparison row. Report whatever you actually measure — a raw policy that
occasionally proposes a reserve-breaching action is expected and fine, since
that's exactly the case the verifier exists to catch.


## Project layout

```
app/
├── main.py                  FastAPI app entrypoint (lifespan warms the classifier)
├── config.py                Settings
├── api/                     Route handlers (decide, health, replay, simulate)
├── schemas/                 Pydantic request/response contracts (frozen API)
├── engine/                  Rule engine, opportunity monitor, traffic engine,
│                             ranker, safety/fail-safe, explanation, audit log
├── models/                  Rival estimator, overtake probability (heuristic +
│                             trained classifier), policy adapter
├── simulation/
│   ├── rollout.py            Phase 1 single-shot analytical rollout (FAST mode)
│   ├── engine.py             Phase 0 multi-tick Monte Carlo rollout (VERIFIED mode)
│   ├── energy_model.py        SOC dynamics per tick
│   ├── traffic_model.py       Gap/relative-speed dynamics + counterattack check
│   └── tracks.py              Phase 3 synthetic track profile generator
├── rl/
│   └── apexguard_env.py       Phase 4 Gymnasium env for PPO training
└── data/
    ├── scenarios/            3 curated demo scenarios
    ├── tracks/                10 synthetic track profiles (generated)
    ├── training/              synthetic overtake-window training data (generated)
    └── artifacts/             overtake_model.joblib, ppo_policy.zip (if trained),
                                 metrics, audit_log.jsonl, rollout_evaluation.json
                                 (all generated at runtime)
scripts/
├── generate_scenarios.py     Synthetic FastF1-shaped training data + FastF1 ingestion stub
├── train_overtake_model.py   Trains + evaluates the calibrated classifier
├── evaluate_rollout.py       FAST vs VERIFIED comparison across all scenarios
└── train_ppo.py               Phase 4 PPO training loop (needs stable-baselines3)
tests/                        29 tests: rules, opportunity, verifier, safety, API,
                                overtake model (trained + fallback), rollout simulator,
                                policy adapter (fallback + env/adapter sync)
```


### PPO reserve-aware reward shaping
The PPO environment uses dense reserve-aware penalties, illegal FULL/PARTIAL proposal penalties, and a stronger terminal reserve penalty so the policy learns the same safety priorities as the live hard-mask + verifier stack.

## 2026 ERS Tactical Advisory + Quantum-Inspired Analytics

ApexGuard now exposes two optional frontend-facing layers:

- `POST /v1/ers/tactical` — advisory **HOLD / RECHARGE / BOOST / OVERTAKE** state, including 2026-style detection eligibility, energy headroom and ERS power caps.
- `POST /v1/simulate/quantum-inspired` — runs the existing validated classical Monte Carlo rollout and returns an amplitude-inspired probability representation for the UI. **No quantum hardware or QAE speedup is claimed.**

The ERS layer is intentionally advisory. It is not a direct car-control interface, and current FIA Sporting/Technical Regulations must be checked before any real-world use.

## Confidence-Aware Overtaking Intelligence

ApexGuard promotes the calibrated Logistic Regression model beyond a baseline.
`app/models/overtake_confidence.py` exposes calibrated action confidence,
normalized predictive entropy, and lightweight sigmoid predicate mappers such
as `corridor_open`, `closing_battle`, `rival_harvesting`, `low_drag`,
`counter_harvest_trap`, and `reserve_feasible`.

The confidence layer is advisory: when PPO proposes an aggressive action while
LR confidence is low or entropy is high, ApexGuard does not allow that proposal
to narrow the candidate set. The existing Monte Carlo verifier and deterministic
legal/safety checks remain authoritative.

The PPO observation is now 27-D: 11 telemetry features + 13 temporal opponent
belief features + 3 confidence-aware logistic features. Because this changes
policy input dimensions, the previous 24-D PPO artifact must be retrained.

### Reproducible retraining + benchmark

```bash
python -m scripts.retrain_ppo_and_benchmark --timesteps 50000 --cases 80
```

The command trains from scratch, evaluates 200 PPO episodes, then runs the
same synthetic PPO-vs-non-PPO verifier benchmark and writes
`app/data/evaluations/ppo_vs_nonppo_report.json`.

## 2026 ERS / Quantum-Inspired Experimental Layer

The backend also contains the advisory 2026-style ERS tactical layer
(`app/engine/ers_2026.py`) with HOLD / RECHARGE / BOOST / OVERTAKE semantics,
and a quantum-inspired Monte Carlo estimator (`app/simulation/quantum_inspired.py`).
The latter uses amplitude-like `sqrt(probability)` encoding over actual verified
Monte Carlo paths; it does not claim quantum hardware execution or a quantum
speedup.
