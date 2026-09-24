from __future__ import annotations

import numpy as np
import pytest

from modems import ModemsScenarioGenerator
from modems.core import DEFAULT_TIME_WINDOW, ModemsRequest, ScenarioTiming, ScenarioType
from modems.generator import (
    NORMAL_SOC_RANGE,
    STRESS_SOC_RANGE,
    ScenarioSocRange,
    SocRangeSpec,
)

# --------------------------------------------------------------------------------------
# generate_random_scenario
# --------------------------------------------------------------------------------------


def test_scenario_generation_is_seeded_reproducible() -> None:
    """Two generators with the same seed produce byte-identical scenarios"""
    gen1 = ModemsScenarioGenerator(seed=11)
    s1 = gen1.generate_random_scenario(nr_agents=2, nr_requests=5)
    gen2 = ModemsScenarioGenerator(seed=11)
    s2 = gen2.generate_random_scenario(nr_agents=2, nr_requests=5)
    assert s1.to_dict() == s2.to_dict()


def test_scenario_counts_match_request() -> None:
    """generate_random_scenario() produces exactly the requested agent/request counts"""
    gen = ModemsScenarioGenerator(seed=1)
    scenario = gen.generate_random_scenario(nr_agents=3, nr_requests=7)
    assert len(scenario.agents) == 3
    assert len(scenario.requests) == 7


def test_scenario_default_tw_length_and_earliest_pickup_floor() -> None:
    """Every generated request has the default TW length and a >=15min lead"""
    gen = ModemsScenarioGenerator(seed=1)
    scenario = gen.generate_random_scenario(nr_agents=1, nr_requests=10)
    for r in scenario.requests:
        assert r.tw_length == DEFAULT_TIME_WINDOW
        assert r.earliest_pickup >= 15.0


def test_scheduled_requests_count() -> None:
    """nr_scheduled controls exactly how many requests are marked scheduled"""
    gen = ModemsScenarioGenerator(seed=1)
    scenario = gen.generate_random_scenario(nr_agents=1, nr_requests=6, nr_scheduled=2)
    assert sum(1 for r in scenario.requests if r.is_scheduled()) == 2
    assert sum(1 for r in scenario.requests if r.is_new()) == 4


def test_clustered_scenario_is_more_geographically_concentrated_than_random() -> None:
    """Clustered pickups have a smaller mean distance to their centroid than Random"""

    def spread(scenario_type: ScenarioType, seed: int) -> float:
        gen = ModemsScenarioGenerator(seed=seed)
        scenario = gen.generate_random_scenario(
            nr_agents=1,
            nr_requests=20,
            scenario_type=scenario_type,
            nr_clusters=1,
            cluster_spread=2.0,
        )
        pickup_indices = [r.node_pickup_index for r in scenario.requests]
        locations = scenario.network.locations
        # station index -> array row is (nr_hubs + station_idx - 1) for 1-based indices
        coords = np.array(
            [locations[scenario.network.nr_hubs + idx - 1] for idx in pickup_indices]
        )
        centroid = coords.mean(axis=0)
        return np.mean(np.linalg.norm(coords - centroid, axis=1))

    random_spread = spread(ScenarioType.random, seed=2)
    clustered_spread = spread(ScenarioType.clustered, seed=2)
    assert clustered_spread < random_spread


def test_tight_timing_concentrates_earliest_pickups_more_than_loose() -> None:
    """Tight earliest-pickup times have a smaller std-dev than Loose ones"""

    def spread_in_time(timing: ScenarioTiming, seed: int) -> float:
        gen = ModemsScenarioGenerator(seed=seed)
        scenario = gen.generate_random_scenario(
            nr_agents=1,
            nr_requests=20,
            scenario_timing=timing,
            nr_surges=1,
            temporal_spread=1.0,
        )
        times = np.array([r.earliest_pickup for r in scenario.requests])
        return times.std()

    loose_spread = spread_in_time(ScenarioTiming.loose, seed=3)
    tight_spread = spread_in_time(ScenarioTiming.tight, seed=3)
    assert tight_spread < loose_spread


# --------------------------------------------------------------------------------------
# generate_workday_requests
# --------------------------------------------------------------------------------------


def test_workday_arrivals_sorted_and_within_horizon() -> None:
    """Arrivals come back sorted by time and confined to [0, workday_length)"""
    gen = ModemsScenarioGenerator(seed=4)
    arrivals = gen.generate_workday_requests(
        workday_length=120.0, base_rate_per_hour=6.0, nr_surges=1
    )
    times = [t for t, _ in arrivals]
    assert times == sorted(times)
    assert all(0.0 <= t < 120.0 for t in times)


def test_workday_arrivals_lead_times_within_bounds() -> None:
    """Each request (earliest_pickup - time_submit) lead follows [min_lead, max_lead]"""
    gen = ModemsScenarioGenerator(seed=4)
    arrivals = gen.generate_workday_requests(
        workday_length=120.0,
        base_rate_per_hour=6.0,
        nr_surges=1,
        min_lead=15.0,
        max_lead=45.0,
    )
    for t, r in arrivals:
        lead = r.earliest_pickup - t
        assert 15.0 - 1e-6 <= lead <= 45.0 + 1e-6


def test_workday_generation_is_seeded_reproducible() -> None:
    """Two generators with the same seed produce identical arrival streams"""
    gen1 = ModemsScenarioGenerator(seed=9)
    a1 = gen1.generate_workday_requests(
        workday_length=60.0, base_rate_per_hour=6.0, nr_surges=1
    )
    gen2 = ModemsScenarioGenerator(seed=9)
    a2 = gen2.generate_workday_requests(
        workday_length=60.0, base_rate_per_hour=6.0, nr_surges=1
    )
    assert [t for t, _ in a1] == [t for t, _ in a2]
    assert [r.to_dict() for _, r in a1] == [r.to_dict() for _, r in a2]


def test_zero_base_rate_only_produces_surge_arrivals() -> None:
    """With base_rate_per_hour=0, only the fixed-size surge bursts are generated"""
    gen = ModemsScenarioGenerator(seed=1)
    arrivals = gen.generate_workday_requests(
        workday_length=60.0,
        base_rate_per_hour=0.0,
        nr_surges=1,
        surge_size_range=(3, 3),
    )
    assert len(arrivals) == 3


# --------------------------------------------------------------------------------------
# request_id via the generator (seeded RNG stream)
# --------------------------------------------------------------------------------------


def test_generator_request_ids_are_deterministic_and_unique() -> None:
    """Same seed -> identical, and within-scenario unique, request_id sequences"""
    gen1 = ModemsScenarioGenerator(seed=42)
    s1 = gen1.generate_random_scenario(nr_agents=2, nr_requests=5)
    gen2 = ModemsScenarioGenerator(seed=42)
    s2 = gen2.generate_random_scenario(nr_agents=2, nr_requests=5)

    ids1: list[str] = [r.request_id for r in s1.requests]
    ids2: list[str] = [r.request_id for r in s2.requests]
    assert ids1 == ids2
    assert len(set(ids1)) == len(ids1)


def test_generator_request_ids_differ_across_seeds() -> None:
    """Different seeds produce disjoint request_id sets"""
    gen1 = ModemsScenarioGenerator(seed=1)
    s1 = gen1.generate_random_scenario(nr_agents=1, nr_requests=3)
    gen2 = ModemsScenarioGenerator(seed=2)
    s2 = gen2.generate_random_scenario(nr_agents=1, nr_requests=3)
    ids1: set[str] = {r.request_id for r in s1.requests}
    ids2: set[str] = {r.request_id for r in s2.requests}
    assert ids1.isdisjoint(ids2)


def test_workday_request_ids_are_unique_and_reproducible() -> None:
    """Workday-stream request_ids are seed-reproducible and within-stream unique"""
    gen1 = ModemsScenarioGenerator(seed=9)
    a1: list[tuple[float, ModemsRequest]] = gen1.generate_workday_requests(
        workday_length=60.0, base_rate_per_hour=6.0, nr_surges=1
    )
    gen2 = ModemsScenarioGenerator(seed=9)
    a2: list[tuple[float, ModemsRequest]] = gen2.generate_workday_requests(
        workday_length=60.0, base_rate_per_hour=6.0, nr_surges=1
    )
    ids1 = [r.request_id for _, r in a1]
    ids2 = [r.request_id for _, r in a2]
    assert ids1 == ids2
    assert len(set(ids1)) == len(ids1)


# --------------------------------------------------------------------------------------
# SocRangeSpec / ScenarioSocRange
# --------------------------------------------------------------------------------------


def test_soc_range_spec_normal_and_stress_match_module_constants() -> None:
    """normal()/stress() have the right kind and the right default bounds"""
    normal = SocRangeSpec.normal()
    stress = SocRangeSpec.stress()
    assert normal.kind == ScenarioSocRange.normal
    assert normal.bounds == NORMAL_SOC_RANGE
    assert stress.kind == ScenarioSocRange.stress
    assert stress.bounds == STRESS_SOC_RANGE


def test_soc_range_spec_custom_accepts_valid_bounds() -> None:
    """custom() accepts any 0 < lb <= ub <= 1.0, including lb == ub"""
    spec = SocRangeSpec.custom(0.35, 0.55)
    assert spec.kind == ScenarioSocRange.custom
    assert spec.bounds == (0.35, 0.55)
    assert SocRangeSpec.custom(0.5, 0.5).bounds == (0.5, 0.5)


@pytest.mark.parametrize(
    "lb, ub",
    [
        (0.0, 0.5),  # lb must be strictly positive
        (0.5, 0.3),  # lb must not exceed ub
        (0.5, 1.5),  # ub must not exceed 1.0
    ],
)
def test_soc_range_spec_custom_rejects_invalid_bounds(lb: float, ub: float) -> None:
    """custom() raises ValueError outside 0 < lb <= ub <= 1.0"""
    with pytest.raises(ValueError, match="0 < lb <= ub <= 1.0"):
        SocRangeSpec.custom(lb, ub)


def test_soc_range_spec_suffix_and_label() -> None:
    """suffix() names scenarios/points; label() is the plain kind string"""
    assert SocRangeSpec.normal().suffix == "_soc80-100"
    assert SocRangeSpec.stress().suffix == "_soc50-70"
    assert SocRangeSpec.custom(0.3, 0.5).suffix == "_soc30-50"
    assert SocRangeSpec.normal().label == "normal"
    assert SocRangeSpec.custom(0.3, 0.5).label == "custom"


def test_soc_range_spec_to_dict_is_compact_for_presets() -> None:
    """normal()/stress() serialize by kind alone; bounds are implied, not stored"""
    assert SocRangeSpec.normal().to_dict() == {"kind": "normal"}
    assert SocRangeSpec.stress().to_dict() == {"kind": "stress"}


def test_soc_range_spec_to_dict_carries_bounds_for_custom() -> None:
    """custom() bounds are not recoverable from kind alone, so they must be stored"""
    assert SocRangeSpec.custom(0.3, 0.5).to_dict() == {
        "kind": "custom",
        "lb": 0.3,
        "ub": 0.5,
    }


@pytest.mark.parametrize(
    "spec",
    [SocRangeSpec.normal(), SocRangeSpec.stress(), SocRangeSpec.custom(0.3, 0.5)],
)
def test_soc_range_spec_to_dict_from_dict_round_trip(spec: SocRangeSpec) -> None:
    """to_dict()/from_dict() serialization round-trips for every kind"""
    restored = SocRangeSpec.from_dict(spec.to_dict())
    assert restored.kind == spec.kind
    assert restored.bounds == spec.bounds
