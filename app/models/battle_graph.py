"""Phase 5: builds the local battle graph the guide specifies -- up to 5
cars (ego, target ahead, car behind ego, car ahead of target, one nearby
DRS-train car), with the node/edge feature sets from the guide's spec.

Our DecisionRequest schema is an aggregate 1v1-plus-traffic-summary
representation (rear_gap_s, cars_within_3s, post_pass_traffic_gap_s), not
full per-car telemetry for every nearby car -- F1 doesn't expose that
publicly either (see the FastF1 ingestion note in generate_scenarios.py).
So the "car ahead of target" and "nearby DRS train car" nodes here are
*synthesized* from the aggregate signals we do have (count of nearby cars,
post-pass traffic gap), not read directly. This is a real modeling choice,
not a shortcut to hide: it's documented so anyone extending this with real
per-car telemetry (e.g. from full timing feed access) knows exactly which
nodes to replace with real data first.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from app.schemas.telemetry import DecisionRequest
from app.models.rival_estimator import RivalEstimate

# Fixed node ordering -- must stay in sync with any trained model's input layout.
NODE_NAMES = ["ego", "target_ahead", "car_behind_ego", "car_ahead_of_target", "nearby_drs_train"]
N_NODES = len(NODE_NAMES)
NODE_FEATURE_DIM = 8   # speed, position_progress, sector_delta, stint_age,
                        # tyre_grip, tyre_uncertainty, energy_state, energy_uncertainty
EDGE_FEATURE_DIM = 7   # gap, relative_speed, time_to_catch, tow_strength,
                        # same_drs_zone, counterattack_exposure, post_pass_blockage_risk


@dataclass
class BattleGraph:
    node_features: np.ndarray   # (N_NODES, NODE_FEATURE_DIM)
    edge_index: np.ndarray      # (2, n_edges) -- source/target node indices
    edge_features: np.ndarray   # (n_edges, EDGE_FEATURE_DIM)
    node_mask: np.ndarray       # (N_NODES,) -- 1.0 if node is present/real, 0.0 if padding


def _node_features(req: DecisionRequest, rival: RivalEstimate) -> np.ndarray:
    ego, target = req.ego, req.target
    nodes = np.zeros((N_NODES, NODE_FEATURE_DIM), dtype=np.float32)

    # ego
    nodes[0] = [
        ego.speed_kph / 320.0, 1.0, 0.0, 0.0,
        ego.tyre_grip_estimate, 0.05, ego.soc_pct / 100.0, 0.02,
    ]
    # target_ahead
    nodes[1] = [
        (ego.speed_kph + target.relative_speed_kph) / 320.0,
        1.0 - min(1.0, target.gap_s / 2.0),
        target.recent_sector_delta_s,
        target.stint_age_laps / 40.0,
        rival.tyre_grip_estimate, rival.tyre_uncertainty,
        0.5, 0.3,  # target's energy state isn't observable -- neutral prior with high uncertainty
    ]
    # car_behind_ego -- synthesized from rear_gap_s; speed inferred as
    # slightly faster than ego (that's what a tight rear gap implies).
    rear_gap = req.traffic.rear_gap_s
    nodes[2] = [
        min(1.0, (ego.speed_kph + 8.0) / 320.0),
        1.0 + min(1.0, 1.0 / max(rear_gap, 0.1)) * 0.1,
        0.0, 0.5,
        0.75, 0.25, 0.6, 0.3,
    ]
    # car_ahead_of_target -- synthesized from post_pass_traffic_gap_s; only
    # meaningfully "present" when that gap is tight (i.e. there's traffic to
    # block into). node_mask reflects this.
    ppg = req.traffic.post_pass_traffic_gap_s
    nodes[3] = [
        (ego.speed_kph + target.relative_speed_kph - 5.0) / 320.0,
        1.0 - min(1.0, target.gap_s / 2.0) + min(1.0, 1.0 / max(ppg, 0.1)) * 0.05,
        0.0, 0.4,
        0.7, 0.3, 0.55, 0.35,
    ]
    # nearby_drs_train -- synthesized from cars_within_3s; present only if
    # there's meaningfully more than one nearby car (i.e. an actual train).
    nodes[4] = [
        ego.speed_kph / 320.0, 0.9, 0.0, 0.3,
        0.65, 0.35, 0.5, 0.4,
    ]

    return nodes


def _node_mask(req: DecisionRequest) -> np.ndarray:
    mask = np.ones(N_NODES, dtype=np.float32)
    mask[0] = 1.0  # ego always present
    mask[1] = 1.0  # target always present (that's the whole point of the request)
    mask[2] = 1.0 if req.traffic.rear_gap_s < 3.0 else 0.0
    mask[3] = 1.0 if req.traffic.post_pass_traffic_gap_s < 2.0 else 0.0
    mask[4] = 1.0 if req.traffic.cars_within_3s >= 2 else 0.0
    return mask


def _edges(req: DecisionRequest, rival: RivalEstimate, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Builds edges only between present nodes. Returns (edge_index[2,E], edge_features[E,7])."""
    ego, target, traffic, track = req.ego, req.target, req.traffic, req.track
    same_drs = 1.0 if (track.drs_available and track.segment_type == "DRS_STRAIGHT") else 0.0

    candidate_edges = [
        # (src, dst, gap_s, relative_speed_kph, tow_strength_hint, counterattack_exposure, post_pass_blockage)
        (0, 1, target.gap_s, target.relative_speed_kph,
         max(0.0, min(1.0, (1.2 - target.gap_s) * 0.6)), 0.2, 0.1),
        (2, 0, traffic.rear_gap_s, 8.0,
         max(0.0, min(1.0, (1.0 - traffic.rear_gap_s / 1.5))), rival.defensive_likelihood, 0.0),
        (1, 3, traffic.post_pass_traffic_gap_s, -3.0,
         max(0.0, min(1.0, (1.0 - traffic.post_pass_traffic_gap_s / 1.5))), 0.1,
         max(0.0, min(1.0, 1.0 - traffic.post_pass_traffic_gap_s / 1.5))),
        (4, 0, 1.5, 3.0, 0.3, 0.15, 0.1),
    ]

    src_list, dst_list, feats = [], [], []
    for src, dst, gap, rel_speed, tow, counter_exp, blockage in candidate_edges:
        if mask[src] < 0.5 or mask[dst] < 0.5:
            continue
        time_to_catch = gap / max(0.1, abs(rel_speed) / 3.6) if rel_speed != 0 else 99.0
        src_list.append(src)
        dst_list.append(dst)
        feats.append([
            gap, rel_speed / 30.0, min(1.0, 10.0 / max(time_to_catch, 0.1)),
            tow, same_drs, counter_exp, blockage,
        ])

    if not src_list:
        # Guarantee at least the ego->target edge exists even in a degenerate case.
        src_list, dst_list = [0], [1]
        feats = [[target.gap_s, target.relative_speed_kph / 30.0, 0.0, 0.0, same_drs, 0.0, 0.0]]

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_features = np.array(feats, dtype=np.float32)
    return edge_index, edge_features


def build_battle_graph(req: DecisionRequest, rival: RivalEstimate) -> BattleGraph:
    mask = _node_mask(req)
    nodes = _node_features(req, rival)
    edge_index, edge_features = _edges(req, rival, mask)
    return BattleGraph(node_features=nodes, edge_index=edge_index, edge_features=edge_features, node_mask=mask)