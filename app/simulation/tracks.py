"""Phase 3: parameterized synthetic track profiles. 8-12 tracks, built from
distributions tuned to be plausible using real historical data as a rough
guide -- NOT a claim of exact reconstruction of any real circuit. Used for
PPO / policy and verifier generalization stress tests once those exist, and
right now for exercising the Phase 0 rollout simulator across a variety of
track geometries.
"""

import json
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List

TRACKS_DIR = Path(__file__).resolve().parents[1] / "data" / "tracks"


@dataclass
class Sector:
    kind: str  # "STRAIGHT" | "BRAKING_ZONE" | "CORNER"
    length_m: float
    has_drs: bool = False


@dataclass
class TrackProfile:
    track_id: str
    straight_length_distribution_m: List[float]
    drs_zone_count: int
    avg_braking_zone_m: float
    corner_density: float  # corners per km, normalized 0-1 vs field max
    energy_recovery_factor: float  # relative regen strength under braking
    overtake_difficulty: float
    lap_length_m: float
    sectors: List[Sector] = field(default_factory=list)

    def to_json(self) -> dict:
        d = asdict(self)
        return d


# 10 synthetic profiles spanning the realistic overtake-difficulty range,
# loosely inspired by real circuit archetypes without claiming to reproduce
# any specific one.
_PROFILE_SEEDS = {
    "synth_low_speed_street_01": dict(
        straight_length_distribution_m=[180, 260, 340],
        drs_zone_count=1, avg_braking_zone_m=95, corner_density=0.85,
        energy_recovery_factor=0.68, overtake_difficulty=0.88, lap_length_m=3300,
    ),
    "synth_low_speed_street_02": dict(
        straight_length_distribution_m=[150, 300, 400],
        drs_zone_count=1, avg_braking_zone_m=105, corner_density=0.80,
        energy_recovery_factor=0.62, overtake_difficulty=0.82, lap_length_m=3600,
    ),
    "synth_technical_mixed_01": dict(
        straight_length_distribution_m=[280, 420, 520],
        drs_zone_count=2, avg_braking_zone_m=120, corner_density=0.60,
        energy_recovery_factor=0.58, overtake_difficulty=0.65, lap_length_m=4900,
    ),
    "synth_technical_mixed_02": dict(
        straight_length_distribution_m=[260, 390, 480],
        drs_zone_count=2, avg_braking_zone_m=128, corner_density=0.55,
        energy_recovery_factor=0.55, overtake_difficulty=0.60, lap_length_m=5100,
    ),
    "synth_balanced_01": dict(
        straight_length_distribution_m=[350, 480, 600],
        drs_zone_count=2, avg_braking_zone_m=132, corner_density=0.44,
        energy_recovery_factor=0.57, overtake_difficulty=0.50, lap_length_m=5300,
    ),
    "synth_balanced_02": dict(
        straight_length_distribution_m=[320, 460, 590],
        drs_zone_count=2, avg_braking_zone_m=138, corner_density=0.46,
        energy_recovery_factor=0.54, overtake_difficulty=0.53, lap_length_m=5250,
    ),
    "synth_high_speed_01": dict(
        straight_length_distribution_m=[480, 650, 800],
        drs_zone_count=2, avg_braking_zone_m=140, corner_density=0.28,
        energy_recovery_factor=0.48, overtake_difficulty=0.35, lap_length_m=5800,
    ),
    "synth_high_speed_02": dict(
        straight_length_distribution_m=[500, 700, 900],
        drs_zone_count=3, avg_braking_zone_m=145, corner_density=0.24,
        energy_recovery_factor=0.45, overtake_difficulty=0.30, lap_length_m=6200,
    ),
    "synth_power_circuit_01": dict(
        straight_length_distribution_m=[600, 780, 1000],
        drs_zone_count=3, avg_braking_zone_m=150, corner_density=0.20,
        energy_recovery_factor=0.42, overtake_difficulty=0.25, lap_length_m=5900,
    ),
    "synth_extreme_high_speed_01": dict(
        straight_length_distribution_m=[700, 900, 1150],
        drs_zone_count=2, avg_braking_zone_m=130, corner_density=0.18,
        energy_recovery_factor=0.40, overtake_difficulty=0.20, lap_length_m=7000,
    ),
}


def _build_sectors(rng: random.Random, meta: dict) -> List[Sector]:
    """Lay out a plausible sector sequence for a track profile: alternating
    straight / braking-zone / corner segments, with DRS on a subset of the
    longer straights, until we roughly fill the lap length."""
    sectors: List[Sector] = []
    remaining = meta["lap_length_m"]
    drs_remaining = meta["drs_zone_count"]
    straight_options = meta["straight_length_distribution_m"]

    while remaining > 200:
        straight_len = min(remaining * 0.4, rng.choice(straight_options) * rng.uniform(0.7, 1.3))
        straight_len = max(80.0, round(straight_len, 1))
        has_drs = drs_remaining > 0 and rng.random() < 0.6
        if has_drs:
            drs_remaining -= 1
        sectors.append(Sector(kind="STRAIGHT", length_m=straight_len, has_drs=has_drs))
        remaining -= straight_len
        if remaining <= 200:
            break

        braking_len = round(max(60.0, rng.gauss(meta["avg_braking_zone_m"], 15)), 1)
        sectors.append(Sector(kind="BRAKING_ZONE", length_m=braking_len))
        remaining -= braking_len
        if remaining <= 200:
            break

        corner_len = round(max(80.0, rng.gauss(150 * (1 + meta["corner_density"]), 40)), 1)
        sectors.append(Sector(kind="CORNER", length_m=corner_len))
        remaining -= corner_len

    return sectors


def generate_all_profiles(seed: int = 7) -> List[TrackProfile]:
    rng = random.Random(seed)
    profiles = []
    for track_id, meta in _PROFILE_SEEDS.items():
        sectors = _build_sectors(rng, meta)
        profiles.append(TrackProfile(track_id=track_id, sectors=sectors, **meta))
    return profiles


def write_profiles(profiles: List[TrackProfile], directory: Path = TRACKS_DIR) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for profile in profiles:
        path = directory / f"{profile.track_id}.json"
        with path.open("w") as f:
            json.dump(profile.to_json(), f, indent=2)


_cache: dict = {}


def load_profile(track_id: str, directory: Path = TRACKS_DIR) -> TrackProfile:
    if track_id in _cache:
        return _cache[track_id]
    path = directory / f"{track_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Track profile '{track_id}' not found at {path}")
    with path.open() as f:
        data = json.load(f)
    sectors = [Sector(**s) for s in data.pop("sectors")]
    profile = TrackProfile(sectors=sectors, **data)
    _cache[track_id] = profile
    return profile


def list_profile_ids(directory: Path = TRACKS_DIR) -> List[str]:
    if not directory.exists():
        return []
    return sorted(p.stem for p in directory.glob("*.json"))


if __name__ == "__main__":
    profiles = generate_all_profiles()
    write_profiles(profiles)
    print(f"Wrote {len(profiles)} synthetic track profiles to {TRACKS_DIR}")
    for p in profiles:
        print(f"  {p.track_id}: {len(p.sectors)} sectors, "
              f"difficulty={p.overtake_difficulty}, drs_zones={p.drs_zone_count}")
