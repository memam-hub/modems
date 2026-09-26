"""
Unit tests for modems.algorithms: preprocessing (Alg.1), greedy completion (Alg.2),
and feasible-insertion enumeration (Alg.3): append-only (MILPs) and insertion (ALNS).
alns_feasible_insertions is validated against the brute-force oracle variant_v0_naive,
which tries every (pickup, delivery) position pair through ModemsJourney.insert_at():
candidate sets, propagated states, and delta_obj must match
"""

from __future__ import annotations

import math

import pytest

from modems.algorithms import (
    _sort_key,
    alns_feasible_insertions,
    greedy_complete,
    milp_feasible_insertions,
    preprocess,
)
from modems.core import ModemsAgent, ModemsRequest, ProblemContext
from modems.insertion_ablation import variant_v0_naive
from modems.solution import ModemsJourney, ModemsSolution

from .builders import (
    duration_limited_scenario,
    generated,
    greedy_solution,
    line_scenario,
    make_ctx,
)

STATE_FIELDS = ("t_arr", "t_wait", "t_start", "t_dep", "tau", "phi_arr", "phi_dep")


def partially_filled(
    ctx: ProblemContext, nr_held: int
) -> tuple[ModemsSolution, list[str]]:
    """Greedy-fill all but the last nr_held requests (in greedy order), return both"""
    base, unassigned = preprocess(ctx)
    assert base is not None
    ordered = sorted(unassigned, key=lambda r: _sort_key(ctx, r))
    solution, _, _ = greedy_complete(base, set(ordered[:-nr_held]))
    return solution, ordered[-nr_held:]


# --------------------------------------------------------------------------------------
# Algorithm 1: preprocess
# --------------------------------------------------------------------------------------

SCHEDULED = ModemsRequest(1, 4, earliest_pickup=5.0, status="scheduled", request_id="s")
NEW = ModemsRequest(2, 3, earliest_pickup=20.0, request_id="n")
INHERITED_ROUTE = ["a_1_h_1", "r_1_p_1", "r_1_d_4", "h_1"]


def inherited_plan(
    ctx: ProblemContext, route: list[str] = INHERITED_ROUTE, **kwargs
) -> ModemsSolution:
    plan = ModemsSolution(ctx)
    plan.journeys["agent_1"] = ModemsJourney.from_route(ctx, "agent_1", route, **kwargs)
    plan.accepted = set(plan.journeys["agent_1"].request_pickup)
    return plan


def test_preprocess_with_only_new_requests_returns_the_idle_fleet() -> None:
    ctx = make_ctx(line_scenario([NEW]))
    base, unassigned = preprocess(ctx)
    assert base is not None and unassigned == {"request_1"}
    assert base.accepted == set() and not base.journeys["agent_1"].is_active


def test_preprocess_adopts_the_inherited_route_with_its_exact_timing() -> None:
    ctx = make_ctx(line_scenario([SCHEDULED, NEW]), "milp3")
    starts = {"r_1_p_1": 7.0, "r_1_d_4": 11.0}
    plan = inherited_plan(ctx, t_start_of=starts, tau_of={"r_1_p_1": 0.0})

    base, unassigned = preprocess(ctx, partial_plan=plan)

    assert base is not None and unassigned == {"request_2"}
    assert base.accepted == {"request_1"}
    journey = base.journeys["agent_1"]
    assert journey.route == INHERITED_ROUTE
    assert [s.t_start for s in journey.states[1:3]] == [7.0, 11.0]
    completed, _, ok = greedy_complete(base, unassigned)
    assert ok and completed.accepted == {"request_1", "request_2"}


def _no_plan(ctx: ProblemContext) -> None:
    return None


def _plan_missing_agent(ctx: ProblemContext) -> ModemsSolution:
    plan = inherited_plan(ctx)
    plan.journeys.pop("agent_1")
    return plan


def _plan_with_other_request_id(ctx: ProblemContext) -> ModemsSolution:
    other = ModemsRequest(1, 4, earliest_pickup=5.0, status="scheduled", request_id="x")
    return inherited_plan(make_ctx(line_scenario([other, NEW])))


def _plan_routing_a_new_request(ctx: ProblemContext) -> ModemsSolution:
    return inherited_plan(
        ctx, ["a_1_h_1", "r_1_p_1", "r_2_p_2", "r_2_d_3", "r_1_d_4", "h_1"]
    )


def _plan_without_the_scheduled_request(ctx: ProblemContext) -> ModemsSolution:
    return ModemsSolution(ctx)


def _plan_with_impossible_timing(ctx: ProblemContext) -> ModemsSolution:
    plan = inherited_plan(ctx)
    plan.journeys["agent_1"].states[1].t_start = 0.5  # before arrival at t=1
    return plan


@pytest.mark.parametrize(
    "make_plan",
    [
        _no_plan,
        _plan_missing_agent,
        _plan_with_other_request_id,
        _plan_routing_a_new_request,
        _plan_without_the_scheduled_request,
        _plan_with_impossible_timing,
    ],
)
def test_preprocess_fails_on_missing_or_inconsistent_partial_plans(make_plan) -> None:
    ctx = make_ctx(line_scenario([SCHEDULED, NEW]))
    base, unassigned = preprocess(ctx, partial_plan=make_plan(ctx))
    assert base is None
    assert unassigned == {"request_2"}


def test_preprocess_fails_when_an_agent_cannot_reach_a_hub() -> None:
    agent = ModemsAgent("s", 6, soc_initial=0.26, soc_min_operational=0.25)
    base, _ = preprocess(make_ctx(line_scenario([NEW], agents=[agent])))
    assert base is None


# --------------------------------------------------------------------------------------
# Algorithm 2: greedy_complete
# --------------------------------------------------------------------------------------


def test_greedy_order_is_deadline_then_load_then_name() -> None:
    requests = [
        ModemsRequest(1, 2, load=1, earliest_pickup=20.0, request_id="a"),
        ModemsRequest(1, 2, load=1, earliest_pickup=10.0, request_id="b"),
        ModemsRequest(1, 2, load=3, earliest_pickup=20.0, request_id="c"),
        ModemsRequest(
            1, 2, load=1, earliest_pickup=20.0, tw_length=5.0, request_id="d"
        ),
    ]
    ctx = make_ctx(line_scenario(requests))
    ordered = sorted(ctx.request_names, key=lambda r: _sort_key(ctx, r))
    assert ordered == ["request_2", "request_3", "request_1", "request_4"]


@pytest.mark.parametrize("strategy", ["milp1", "milp2", "milp3", "alns"])
@pytest.mark.parametrize("objective", ["closed", "open"])
def test_greedy_produces_a_consistent_feasible_solution(
    strategy: str, objective: str
) -> None:
    ctx = make_ctx(generated(7, nr_agents=2, nr_requests=6), strategy, objective)
    base, unassigned = preprocess(ctx)
    solution, rejected, ok = greedy_complete(base, unassigned)
    assert ok
    assert all(journey._is_feasible() for journey in solution.journeys.values())
    routed = {r for j in solution.journeys.values() for r in j.request_pickup}
    assert routed == solution.accepted
    assert rejected == solution.rejected == solution.pending()
    assert solution.accepted  # generated demand is serviceable for every strategy
    assert base.accepted == set()  # input plan is not mutated


def test_greedy_milp2_serves_requests_that_require_waiting() -> None:
    """Arriving at t=2 for a window [10, 15] is feasible: the agent waits"""
    request = ModemsRequest(2, 5, earliest_pickup=10.0)
    solution = greedy_solution(make_ctx(line_scenario([request]), "milp2"))
    assert solution.accepted == {"request_1"}
    assert solution.journeys["agent_1"].states[1].t_start == 10.0


@pytest.mark.parametrize("strategy, accepted", [("milp2", False), ("milp3", True)])
def test_greedy_hard_time_window_rejects_late_arrivals(
    strategy: str, accepted: bool
) -> None:
    """Arriving at s6 at t=6 misses the window [0, 1]: only soft-TW strategies accept"""
    late = ModemsRequest(6, 1, earliest_pickup=0.0, tw_length=1.0)
    solution = greedy_solution(make_ctx(line_scenario([late]), strategy))
    assert (solution.accepted == {"request_1"}) == accepted


def test_greedy_non_selective_fails_on_the_first_unservable_request() -> None:
    ctx = make_ctx(duration_limited_scenario(), "milp1")
    base, unassigned = preprocess(ctx)
    _, rejected, ok = greedy_complete(base, unassigned)
    assert not ok and rejected == {"request_2"}


def test_greedy_adopts_the_minimum_delta_candidate() -> None:
    ctx = make_ctx(generated(3, nr_agents=2, nr_requests=4))
    base, unassigned = preprocess(ctx)
    first = sorted(unassigned, key=lambda r: _sort_key(ctx, r))[0]
    best = min(alns_feasible_insertions(base, first), key=lambda c: c.delta_obj)
    solution, _, _ = greedy_complete(base, {first})
    assert solution.journeys[best.journey.agent_name].route == best.journey.route


# --------------------------------------------------------------------------------------
# MILPs: append-only insertions
# --------------------------------------------------------------------------------------


def test_milp_insertions_append_at_route_end_without_touching_the_source() -> None:
    ctx = make_ctx(generated(1, nr_agents=2, nr_requests=4), "milp3")
    solution, (held,) = partially_filled(ctx, nr_held=1)
    before = {
        k: (list(j.route), dict(j.request_pickup)) for k, j in solution.journeys.items()
    }

    candidates = milp_feasible_insertions(solution, held)

    assert {c.journey.agent_name for c in candidates} == set(ctx.agent_names)
    for c in candidates:
        route = c.journey.route
        assert route[-3:-1] == [ctx.pickup_node[held], ctx.delivery_node[held]]
        assert route[:-3] == solution.journeys[c.journey.agent_name].route[:-1]
        assert c.journey._is_feasible()
    after = {
        k: (list(j.route), dict(j.request_pickup)) for k, j in solution.journeys.items()
    }
    assert after == before


@pytest.mark.parametrize(
    "enumerate_", [milp_feasible_insertions, alns_feasible_insertions]
)
def test_insertions_respect_capacity_and_agent_filter(enumerate_) -> None:
    agents = [ModemsAgent("h", 1, load_max=2), ModemsAgent("h", 1, load_max=6)]
    requests = [ModemsRequest(1, 2, load=4, request_id="a")]
    strategy = "alns" if enumerate_ is alns_feasible_insertions else "milp3"
    base, _ = preprocess(make_ctx(line_scenario(requests, agents=agents), strategy))
    assert {c.journey.agent_name for c in enumerate_(base, "request_1")} == {"agent_2"}
    assert enumerate_(base, "request_1", agents=["agent_1"]) == []
    assert enumerate_(base, "request_1", agents=[]) == []


# --------------------------------------------------------------------------------------
# ALNS: all feasible insertions
# --------------------------------------------------------------------------------------


def assert_matches_oracle(solution: ModemsSolution, request_name: str) -> int:
    """Compare Algorithm 3 against the V0 oracle, return the number of candidates"""
    fast = {
        (c.journey.agent_name, tuple(c.journey.route)): c
        for c in alns_feasible_insertions(solution, request_name)
    }
    slow = {
        (c.journey.agent_name, tuple(c.journey.route)): c
        for c in variant_v0_naive(solution, request_name)
    }
    assert fast.keys() == slow.keys()
    base_obj = solution.objective()
    for key, candidate in fast.items():
        reference = slow[key]
        assert math.isclose(
            candidate.delta_obj, reference.delta_obj, rel_tol=1e-9, abs_tol=1e-7
        ), key
        for actual, expected in zip(
            candidate.journey.states, reference.journey.states, strict=True
        ):
            assert (actual.node, actual.z_arr, actual.z_dep) == (
                expected.node,
                expected.z_arr,
                expected.z_dep,
            )
            for field in STATE_FIELDS:
                assert math.isclose(
                    getattr(actual, field),
                    getattr(expected, field),
                    rel_tol=1e-9,
                    abs_tol=1e-7,
                ), (key, actual.node, field)
        assert candidate.journey.request_pickup == reference.journey.request_pickup
        assert candidate.journey.request_delivery == reference.journey.request_delivery
        assert candidate.journey._is_feasible()
        adopted = solution.copy()
        adopted.journeys[candidate.journey.agent_name] = candidate.journey
        adopted.accepted.add(request_name)
        assert math.isclose(
            adopted.objective() - base_obj,
            candidate.delta_obj,
            rel_tol=1e-9,
            abs_tol=1e-7,
        )
    return len(fast)


@pytest.mark.parametrize("objective", ["closed", "open"])
@pytest.mark.parametrize("seed", range(1, 11))
def test_algorithm3_matches_the_oracle_on_filled_routes(
    seed: int, objective: str
) -> None:
    scenario = generated(seed, nr_agents=1 + seed % 3, nr_requests=6 + seed % 8)
    solution, held = partially_filled(
        make_ctx(scenario, objective=objective), nr_held=3
    )
    for request_name in held:
        assert_matches_oracle(solution, request_name)


@pytest.mark.parametrize("scenario_type", ["clustered", "mixed"])
@pytest.mark.parametrize("timing", ["uniform", "peaks"])
def test_algorithm3_matches_the_oracle_across_spatial_and_temporal_shapes(
    scenario_type: str, timing: str
) -> None:
    scenario = generated(
        5,
        nr_agents=2,
        nr_requests=12,
        scenario_type=scenario_type,
        scenario_timing=timing,
    )
    solution, held = partially_filled(make_ctx(scenario), nr_held=3)
    for request_name in held:
        assert_matches_oracle(solution, request_name)


def test_algorithm3_matches_the_oracle_when_soc_is_scarce() -> None:
    """Low starting SoC makes the energy-based pruning bounds decide the outcome"""
    counts = []
    for seed in range(1, 6):
        scenario = generated(seed, nr_agents=1, nr_requests=10, soc_lb=0.4, soc_ub=0.5)
        solution, held = partially_filled(make_ctx(scenario), nr_held=4)
        counts += [assert_matches_oracle(solution, r) for r in held]
    assert 0 in counts and sum(counts) > 0  # both pruned and surviving probes occur


def test_algorithm3_matches_the_oracle_with_tight_ride_times() -> None:
    """rho close to 1 makes the ride-time early exit decide the outcome"""
    solution, held = partially_filled(
        make_ctx(generated(4, nr_agents=1, nr_requests=9), rho=1.2), nr_held=4
    )
    assert sum(assert_matches_oracle(solution, r) for r in held) > 0


def test_algorithm3_matches_the_oracle_for_non_selective_problems() -> None:
    ctx = ProblemContext(
        generated(3, nr_agents=2, nr_requests=8), "closed_non_selective", "alns"
    )
    solution, held = partially_filled(ctx, nr_held=3)
    for request_name in held:
        assert_matches_oracle(solution, request_name)


def test_algorithm3_exercises_mid_route_deliveries() -> None:
    solution, held = partially_filled(
        make_ctx(generated(1, nr_agents=1, nr_requests=10)), nr_held=3
    )
    mid_route = [
        c
        for r in held
        for c in alns_feasible_insertions(solution, r)
        if c.journey.request_delivery[r] < len(c.journey.route) - 2
    ]
    assert mid_route


@pytest.mark.parametrize("strategy", ["milp1", "milp2", "milp3"])
def test_algorithm3_rejects_non_alns_strategies(strategy: str) -> None:
    base, _ = preprocess(make_ctx(line_scenario([NEW]), strategy))
    with pytest.raises(ValueError, match="requires SolverStrategy.alns"):
        alns_feasible_insertions(base, "request_1")


def test_algorithm3_candidates_are_independent_of_each_other_and_the_source() -> None:
    solution, (held, *_) = partially_filled(
        make_ctx(generated(5, nr_agents=1, nr_requests=8)), nr_held=3
    )
    source = solution.journeys["agent_1"]
    source_snapshot = (list(source.route), [s.to_dict() for s in source.states])

    candidates = alns_feasible_insertions(solution, held)
    assert len(candidates) > 1
    assert len({id(c.journey.states) for c in candidates}) == len(candidates)
    assert len({id(c.journey.request_pickup) for c in candidates}) == len(candidates)
    for candidate in candidates:
        candidate.journey.remove_request(held)
    assert (list(source.route), [s.to_dict() for s in source.states]) == source_snapshot


def test_algorithm3_is_deterministic() -> None:
    solution, held = partially_filled(
        make_ctx(generated(42, nr_agents=2, nr_requests=8)), nr_held=3
    )
    for request_name in held:
        runs = [
            [
                (c.journey.agent_name, c.journey.route, c.delta_obj)
                for c in alns_feasible_insertions(solution, request_name)
            ]
            for _ in range(2)
        ]
        assert runs[0] == runs[1]
