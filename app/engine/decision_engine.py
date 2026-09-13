import time
from app.schemas.common import ActionType, RiskLevel, DecisionStatus, CandidateStatus, DecisionMode
from app.schemas.decision import DecisionResponse, CandidateOutcome, TrafficSummary
from app.schemas.telemetry import DecisionRequest
from app.engine import rule_engine, opportunity_monitor, safety, ranker, explanation, audit
from app.engine.traffic_engine import evaluate_with_source as evaluate_traffic
from app.simulation.rollout import rollout as rollout_fast
from app.simulation.engine import rollout_verified
from app.models import rival_estimator, overtake_probability, policy_adapter
from app.models import opponent_belief, overtake_confidence


def _risk_level(counterattack_risk: float) -> RiskLevel:
    if counterattack_risk >= 0.66:
        return RiskLevel.HIGH
    if counterattack_risk >= 0.33:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def decide(req: DecisionRequest) -> DecisionResponse:
    start_time = time.perf_counter()

    # 1. Freshness guard.
    if req.telemetry_age_ms > safety.TELEMETRY_STALE_MS:
        return safety.safe_hold(req, "telemetry is stale", start_time)

    # 2. Hard legal-action mask.
    legal_actions = rule_engine.allowed_actions(req)
    if not legal_actions:
        return safety.safe_hold(req, "no legal action is available", start_time)

    # 3. Opportunity gating - skip heavy pipeline if no credible window.
    opportunity_score = opportunity_monitor.score(req)
    if opportunity_score < safety.OPPORTUNITY_THRESHOLD:
        return safety.low_opportunity_hold(req, opportunity_score, start_time)

    # 4. Rival latent-state estimate with uncertainty.
    rival = rival_estimator.estimate(req)

    # 5. Hidden opponent-state belief. SOC, Override and active-aero state are
    # not directly observable, so downstream risk logic receives probabilities.
    belief = opponent_belief.update_temporal(req, rival, req.battle_id)

    # 6. Traffic / counterattack risk assessment.
    traffic, traffic_model = evaluate_traffic(req, rival, belief)

    # 7. Per-action overtake probability (heuristic or trained classifier).
    # Used directly by the FAST analytical rollout below; the VERIFIED
    # Monte Carlo simulator computes its own empirical pass probability
    # from actually simulating whether the gap closes to zero, so this
    # value isn't fed into it -- the simulation result is authoritative
    # when decision_mode == VERIFIED.
    pass_probs = overtake_probability.predict_for_actions(req, rival, traffic, legal_actions)

    # 8. Confidence-aware logistic intelligence. This turns the calibrated
    # classifier into an uncertainty + predicate layer for auditability.
    overtake_intelligence = overtake_confidence.assess(
        req, rival, traffic, legal_actions, belief
    )

    # 9. Policy proposal, with safe fallback to all legal actions.
    fallback_used = False
    proposed = policy_adapter.propose(req, legal_actions)
    intelligence_uncertain = (
        overtake_intelligence["normalized_entropy"] >= 0.85
        or overtake_intelligence["confidence"] < 0.50
    )
    aggressive_actions = {ActionType.PARTIAL_DEPLOY, ActionType.FULL_DEPLOY}
    if proposed in aggressive_actions and intelligence_uncertain:
        # Confidence-aware correction: do not let an uncertain PPO proposal
        # narrow the candidate set. The verifier still makes the authoritative
        # final selection after evaluating all legal actions.
        fallback_used = True
        candidate_actions = list(legal_actions)
    elif proposed is None or proposed not in legal_actions:
        fallback_used = True
        candidate_actions = list(legal_actions)
    else:
        candidate_actions = list({proposed, ActionType.HOLD} & set(legal_actions) | {proposed})
        candidate_actions = [a for a in candidate_actions if a in legal_actions]
        if ActionType.HOLD not in candidate_actions and ActionType.HOLD in legal_actions:
            candidate_actions.append(ActionType.HOLD)

    # 10. Roll out every candidate action. VERIFIED mode runs the Phase 0
    # multi-tick Monte Carlo simulator (real, stochastic, gives genuine
    # confidence). FAST mode uses the cheap single-shot analytical formula
    # for tighter latency budgets.
    if req.decision_mode == DecisionMode.FAST:
        outcomes = [
            rollout_fast(req, action, pass_probs[action], traffic)
            for action in candidate_actions
        ]
    else:
        outcomes = [
            rollout_verified(req, action, traffic, rival, opponent_belief_state=belief)
            for action in candidate_actions
        ]

    # 11. FAST mode has no path-level hidden-state sampling, so retain a small
    # explicit trap adjustment there. VERIFIED mode already propagates the
    # filtered hidden-state posterior through Monte Carlo and therefore does
    # not double-count the same risk with a post-hoc penalty.
    if req.decision_mode == DecisionMode.FAST:
        trap_penalty = belief.counter_harvest_trap_probability
        for outcome in outcomes:
            action_multiplier = {
                ActionType.HARVEST: 0.10,
                ActionType.HOLD: 0.20,
                ActionType.PARTIAL_DEPLOY: 0.70,
                ActionType.FULL_DEPLOY: 1.00,
            }[outcome.action]
            outcome.expected_net_value = round(
                outcome.expected_net_value - 0.18 * trap_penalty * action_multiplier,
                4,
            )

    # 12. Rank risk-adjusted outcomes.
    selected = ranker.select_highest_legal_net_value(outcomes)

    # 13. Apply fail-safe logic.
    if selected is None or selected.expected_net_value < safety.MINIMUM_VALUE:
        return safety.safe_hold(req, "no candidate met safety and value thresholds", start_time)

    # 14. Build explanation + audit record.
    explanation_text = explanation.build(selected, outcomes)

    candidates_payload = []
    for outcome in outcomes:
        is_selected = outcome.action == selected.action
        candidates_payload.append(
            CandidateOutcome(
                action=outcome.action,
                legal=outcome.legal,
                pass_probability=outcome.pass_probability,
                immediate_time_delta_s=outcome.immediate_time_delta_s,
                projected_soc_pct=outcome.projected_soc_pct,
                expected_net_value=outcome.expected_net_value,
                status=CandidateStatus.SELECTED if is_selected else CandidateStatus.REJECTED,
                reason=(
                    "Highest legal risk-adjusted value."
                    if is_selected
                    else f"Lower expected net value than {selected.action.value} "
                         f"({outcome.expected_net_value:+.3f} vs {selected.expected_net_value:+.3f})."
                ),
            )
        )
    # Include illegal actions in the payload for transparency.
    for action in ActionType:
        if action not in candidate_actions:
            candidates_payload.append(
                CandidateOutcome(
                    action=action,
                    legal=False,
                    pass_probability=0.0,
                    immediate_time_delta_s=0.0,
                    projected_soc_pct=req.ego.soc_pct,
                    expected_net_value=-1.0,
                    status=CandidateStatus.ILLEGAL,
                    reason="Blocked by hard rule mask (SOC/budget/segment/telemetry constraint).",
                )
            )

    latency_ms = (time.perf_counter() - start_time) * 1000
    audit_id = audit.new_audit_id(req.request_id)

    response = DecisionResponse(
        request_id=req.request_id,
        status=DecisionStatus.DEGRADED if fallback_used and policy_adapter.policy_is_available() else DecisionStatus.OK,
        policy_proposal=proposed,
        recommended_action=selected.action,
        recommendation_confidence=round(
            selected.confidence if req.decision_mode != DecisionMode.FAST
            else min(0.99, 0.55 + selected.pass_probability * 0.4),
            3,
        ),
        overtake_success_probability=selected.pass_probability,
        counterattack_risk=traffic.counterattack_risk,
        risk_level=_risk_level(traffic.counterattack_risk),
        rule_compliant=True,
        fallback_used=fallback_used,
        traffic_model=traffic_model,
        opponent_belief=belief.summary(),
        overtake_intelligence=overtake_intelligence,
        latency_ms=round(latency_ms, 2),
        explanation=explanation_text,
        traffic_summary=TrafficSummary(
            target_gap_s=req.target.gap_s,
            rear_gap_s=req.traffic.rear_gap_s,
            tow_strength=traffic.tow_strength,
            post_pass_traffic_risk=traffic.post_pass_traffic_risk,
        ),
        candidates=candidates_payload,
        audit_id=audit_id,
    )

    audit.log_decision(req, response)
    return response
