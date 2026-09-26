"""Unit tests for modems.generator: static scenarios, workday streams, SoC ranges"""

from __future__ import annotations

import numpy as np
import pytest

from modems.core import DEFAULT_TIME_WINDOW, ScenarioTiming, ScenarioType
from modems.generator import (
    DEFAULT_MIN_LEAD,
    NORMAL_SOC_RANGE,
    STATIC_HORIZON,
    STRESS_SOC_RANGE,
    SURGE_DURATION,
    SURGE_GAP_BY_TIMING,
    ModemsScenarioGenerator,
    ScenarioSocRange,
    SocRangeSpec,
    _peaks_plateaus,
    max_surges,
    surge_spans,
    timing_profile,
)
from modems.network import NetworkNodeType

from .builders import generated, line_network

# --------------------------------------------------------------------------------------
# Network selection
# --------------------------------------------------------------------------------------


def test_default_network_is_fixed_campus_layout() -> None:
    a, b = ModemsScenarioGenerator(seed=1), ModemsScenarioGenerator(seed=2)
    assert (a.network.nr_hubs, a.network.nr_stations) == (3, 15)
    assert np.array_equal(a.network.locations, b.network.locations)


def test_random_network_uses_requested_size() -> None:
    gen = ModemsScenarioGenerator(seed=1, random_network=True, nr_hubs=2, nr_stations=6)
    assert (gen.network.nr_hubs, gen.network.nr_stations) == (2, 6)


def test_given_network_is_copied_and_exclusive_with_random_network() -> None:
    network = line_network([0.0, 1.0, 2.0, 3.0])
    gen = ModemsScenarioGenerator(seed=1, network=network)
    assert gen.network is not network
    assert np.array_equal(gen.network.travel_times, network.travel_times)
    with pytest.raises(ValueError, match="mutually exclusive"):
        ModemsScenarioGenerator(seed=1, network=network, random_network=True)


# --------------------------------------------------------------------------------------
# Static scenarios
# --------------------------------------------------------------------------------------


def test_scenario_is_seed_reproducible() -> None:
    kwargs = dict(nr_agents=2, nr_requests=5, scenario_type="mixed")
    assert generated(11, **kwargs).to_dict() == generated(11, **kwargs).to_dict()
    assert generated(11, **kwargs).to_dict() != generated(12, **kwargs).to_dict()


@pytest.mark.parametrize("scenario_type", list(ScenarioType))
@pytest.mark.parametrize("timing", list(ScenarioTiming))
def test_scenario_fields_lie_in_their_documented_domains(
    scenario_type: ScenarioType, timing: ScenarioTiming
) -> None:
    scenario = generated(
        3,
        nr_agents=4,
        nr_requests=30,
        nr_scheduled=5,
        scenario_type=scenario_type,
        scenario_timing=timing,
        soc_lb=0.6,
        soc_ub=0.7,
    )
    network = scenario.network
    assert (scenario.type, scenario.timing) == (scenario_type, timing)
    assert len(scenario.agents) == 4 and len(scenario.requests) == 30
    for agent in scenario.agents:
        limit = network.nr_hubs if agent.node_type == "h" else network.nr_stations
        assert 1 <= agent.node_index <= limit
        assert agent.load_max in (5, 6, 8)
        assert 0.0 <= agent.time_initial <= 10.0
        assert 0.6 <= agent.soc_initial <= 0.7
    for r in scenario.requests:
        assert r.node_pickup_index != r.node_delivery_index
        assert 1 <= r.load <= 4
        assert r.service_time == pytest.approx(min(1.0, max(0.2, r.load / 10)))
        assert r.tw_length == DEFAULT_TIME_WINDOW
        assert DEFAULT_MIN_LEAD <= r.earliest_pickup <= STATIC_HORIZON
    assert [r.is_scheduled() for r in scenario.requests] == [True] * 5 + [False] * 25
    ids = [r.request_id for r in scenario.requests] + [
        a.agent_id for a in scenario.agents
    ]
    assert len(set(ids)) == len(ids)


def test_agents_start_at_both_hubs_and_stations() -> None:
    scenario = generated(1, nr_agents=40, nr_requests=0)
    kinds = {agent.node_type for agent in scenario.agents}
    assert kinds == {NetworkNodeType.hub, NetworkNodeType.station}


def test_clustered_endpoints_concentrate_on_few_stations() -> None:
    """One tight cluster snaps to a handful of stations, and random spans the network"""

    def distinct_pickups(scenario_type: ScenarioType) -> int:
        scenario = generated(
            2,
            nr_agents=1,
            nr_requests=40,
            scenario_type=scenario_type,
            nr_clusters=1,
            cluster_spread=2.0,
        )
        return len({r.node_pickup_index for r in scenario.requests})

    assert distinct_pickups(ScenarioType.clustered) <= 5
    assert distinct_pickups(ScenarioType.random) >= 12


def test_peaks_timing_concentrates_pickups_in_the_plateaus() -> None:
    plateaus = _peaks_plateaus(DEFAULT_MIN_LEAD, STATIC_HORIZON, DEFAULT_MIN_LEAD)

    def plateau_share(timing: ScenarioTiming) -> float:
        times = [
            r.earliest_pickup
            for seed in range(20)
            for r in generated(
                seed, nr_agents=1, nr_requests=20, scenario_timing=timing
            ).requests
        ]
        return float(np.mean([any(a <= t <= b for a, b in plateaus) for t in times]))

    # plateaus cover 1/3 of [15, 60]: uniform ~1/3, peaks ~1/2 + 1/2 * 1/3 = 2/3
    assert plateau_share(ScenarioTiming.uniform) == pytest.approx(1 / 3, abs=0.07)
    assert plateau_share(ScenarioTiming.peaks) == pytest.approx(2 / 3, abs=0.07)


# --------------------------------------------------------------------------------------
# Timing profile and surge geometry
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("timing", list(ScenarioTiming))
@pytest.mark.parametrize("t_lb", [0.0, 15.0, 130.0])
def test_timing_profile_is_a_density_on_its_support(
    timing: ScenarioTiming, t_lb: float
) -> None:
    pieces = timing_profile(timing, 0.0, 480.0, t_lb)
    assert sum((b - a) * h for a, b, h in pieces) == pytest.approx(1.0)
    assert min(a for a, _, _ in pieces) == t_lb
    assert max(b for _, b, _ in pieces) == 480.0


def test_peaks_plateaus_are_placed_relative_to_the_full_horizon() -> None:
    assert _peaks_plateaus(0.0, 480.0, 15.0) == [(120.0, 200.0), (280.0, 360.0)]
    assert _peaks_plateaus(0.0, 480.0, 150.0) == [(150.0, 200.0), (280.0, 360.0)]
    static = _peaks_plateaus(DEFAULT_MIN_LEAD, STATIC_HORIZON, DEFAULT_MIN_LEAD)
    assert [(a + b) / 2 for a, b in static] == pytest.approx([30.0, 45.0])


@pytest.mark.parametrize("t_lb", [-1.0, 480.0])
def test_timing_profile_rejects_lower_bound_outside_horizon(t_lb: float) -> None:
    with pytest.raises(ValueError):
        timing_profile(ScenarioTiming.uniform, 0.0, 480.0, t_lb)


def test_surge_spans_keep_the_gap_from_plateaus() -> None:
    assert surge_spans(ScenarioTiming.uniform, 480.0) == [(15.0, 480.0)]
    gap = SURGE_GAP_BY_TIMING[ScenarioTiming.peaks]
    assert surge_spans(ScenarioTiming.peaks, 480.0) == [
        (15.0, 120.0 - gap),
        (200.0 + gap, 280.0 - gap),
        (360.0 + gap, 480.0),
    ]


@pytest.mark.parametrize(
    "timing, length, capacity",
    [
        (ScenarioTiming.uniform, 90.0, 1),
        (ScenarioTiming.uniform, 480.0, 8),
        (ScenarioTiming.peaks, 90.0, 0),
        (ScenarioTiming.peaks, 240.0, 2),
        (ScenarioTiming.peaks, 480.0, 5),
    ],
)
def test_max_surges_is_the_exact_generation_limit(
    timing: ScenarioTiming, length: float, capacity: int
) -> None:
    assert max_surges(timing, length) == capacity
    gen = ModemsScenarioGenerator(seed=1)
    gen.generate_workday_requests(
        workday_length=length, nr_surges=capacity, scenario_timing=timing
    )
    with pytest.raises(ValueError, match="exceed the maximum"):
        gen.generate_workday_requests(
            workday_length=length, nr_surges=capacity + 1, scenario_timing=timing
        )


@pytest.mark.parametrize("timing", list(ScenarioTiming))
def test_sampled_surges_never_overlap_and_avoid_plateaus(
    timing: ScenarioTiming,
) -> None:
    gap = SURGE_GAP_BY_TIMING[timing]
    plateaus = _peaks_plateaus(0.0, 480.0, 15.0) if timing == "peaks" else []
    spans = surge_spans(timing, 480.0)
    capacity = max_surges(timing, 480.0)
    gen = ModemsScenarioGenerator(seed=2)
    for nr_surges in (1, capacity):
        for _ in range(100):
            starts = gen._sample_surge_starts(nr_surges, spans, SURGE_DURATION, gap)
            assert len(starts) == nr_surges and starts == sorted(starts)
            for a, b in zip(starts, starts[1:]):
                assert b - (a + SURGE_DURATION) >= gap - 1e-9
            for t in starts:
                assert any(
                    lb - 1e-9 <= t and t + SURGE_DURATION <= ub + 1e-9
                    for lb, ub in spans
                )
                for pa, pb in plateaus:
                    assert t + SURGE_DURATION <= pa - gap + 1e-9 or t >= pb + gap - 1e-9


# --------------------------------------------------------------------------------------
# Workday request streams
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("timing", list(ScenarioTiming))
def test_workday_submissions_leads_and_pickups_respect_bounds(
    timing: ScenarioTiming,
) -> None:
    arrivals = ModemsScenarioGenerator(seed=4).generate_workday_requests(
        workday_length=480.0,
        base_rate_per_hour=8,
        nr_surges=3,
        scenario_timing=timing,
        scenario_type="clustered",
    )
    times = [t for t, _ in arrivals]
    assert arrivals and times == sorted(times)
    for t_submit, r in arrivals:
        assert 15.0 <= r.earliest_pickup <= 480.0
        assert (
            15.0 - 1e-9
            <= r.earliest_pickup - t_submit
            <= min(45.0, r.earliest_pickup) + 1e-9
        )
        assert r.is_new() and 1 <= r.load <= 5
    ids = [r.request_id for _, r in arrivals]
    assert len(set(ids)) == len(ids)


def test_workday_stream_is_seed_reproducible() -> None:
    kwargs = dict(
        workday_length=240.0, base_rate_per_hour=6, nr_surges=2, scenario_timing="peaks"
    )
    a1 = ModemsScenarioGenerator(seed=9).generate_workday_requests(**kwargs)
    a2 = ModemsScenarioGenerator(seed=9).generate_workday_requests(**kwargs)
    assert [(t, r.to_dict()) for t, r in a1] == [(t, r.to_dict()) for t, r in a2]


@pytest.mark.parametrize(
    "timing, surges", [("uniform", 0), ("uniform", 3), ("peaks", 3)]
)
def test_workday_mean_request_count(timing: str, surges: int) -> None:
    """E[count] = base rate x (L - min_lead)/60 + one hour of base rate per surge"""
    counts = [
        len(
            ModemsScenarioGenerator(seed=seed).generate_workday_requests(
                workday_length=480.0,
                base_rate_per_hour=5,
                nr_surges=surges,
                scenario_timing=timing,
            )
        )
        for seed in range(150)
    ]
    assert np.mean(counts) == pytest.approx(
        5 * (480.0 - 15.0) / 60.0 + 5 * surges, rel=0.05
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"base_rate_per_hour": 0}, "base_rate_per_hour"),
        ({"min_lead": 50.0, "max_lead": 40.0}, "min_lead"),
        ({"min_lead": -1.0}, "min_lead"),
        ({"workday_length": 15.0}, "workday_length"),
        ({"nr_surges": -1}, "nr_surges"),
    ],
)
def test_workday_rejects_invalid_arguments(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ModemsScenarioGenerator(seed=1).generate_workday_requests(**kwargs)


# --------------------------------------------------------------------------------------
# SocRangeSpec
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec, kind, bounds, suffix",
    [
        (SocRangeSpec.normal(), ScenarioSocRange.normal, NORMAL_SOC_RANGE, "80_100"),
        (SocRangeSpec.stress(), ScenarioSocRange.stress, STRESS_SOC_RANGE, "50_70"),
        (
            SocRangeSpec.custom(0.29, 0.57),
            ScenarioSocRange.custom,
            (0.29, 0.57),
            "29_57",
        ),
        (SocRangeSpec.custom(0.5, 0.5), ScenarioSocRange.custom, (0.5, 0.5), "50_50"),
    ],
)
def test_soc_range_spec_kind_bounds_suffix_and_round_trip(
    spec: SocRangeSpec, kind: ScenarioSocRange, bounds: tuple, suffix: str
) -> None:
    assert spec.kind is kind and spec.label == kind.value
    assert spec.bounds == bounds
    assert spec.suffix == suffix
    serialized = spec.to_dict()
    assert ("lb" in serialized) == (kind is ScenarioSocRange.custom)
    assert SocRangeSpec.from_dict(serialized) == spec


@pytest.mark.parametrize("lb, ub", [(0.0, 0.5), (0.5, 0.3), (0.5, 1.5)])
def test_soc_range_spec_custom_rejects_invalid_bounds(lb: float, ub: float) -> None:
    with pytest.raises(ValueError, match="0 < lb <= ub <= 1.0"):
        SocRangeSpec.custom(lb, ub)
