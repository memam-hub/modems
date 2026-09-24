from __future__ import annotations

import math

import pytest

from modems import ModemsScenarioGenerator
from modems.algorithms import (
    InsertionCandidate,
    _sort_key,
    alns_feasible_insertions,
    greedy_complete,
    milp_feasible_insertions,
    preprocess,
)
from modems.core import ModemsScenario, ProblemContext, ProblemType, SolverStrategy
from modems.insertion_ablation import variant_v0_naive
from modems.solution import ModemsJourney, ModemsSolution

PARAMS: dict = {"eps": 0.01, "zeta": 1.0, "eta": 100.0, "rho": 2.0, "omega": 5.0}


def _preprocess(
    scenario: ModemsScenario,
    problem_type: ProblemType,
    strategy: SolverStrategy,
    model_params: dict = PARAMS,
) -> tuple[ModemsSolution | None, set[str]]:
    """Build a ProblemContext from raw arguments, then run Algorithm 1 on it"""
    return preprocess(ProblemContext(scenario, problem_type, strategy, model_params))


def _candidate_map(
    candidates: list[InsertionCandidate],
) -> dict[tuple[str, tuple[str, ...]], float]:
    """Map candidates by (agent, route) -> delta_obj for order-independent comparison"""
    return {
        (c.journey.agent_name, tuple(c.journey.route)): c.delta_obj for c in candidates
    }


def _candidates_by_route(
    candidates: list[InsertionCandidate],
) -> dict[tuple[str, tuple[str, ...]], InsertionCandidate]:
    """Index candidates by (agent, route) -> the candidate itself"""
    return {(c.journey.agent_name, tuple(c.journey.route)): c for c in candidates}


def _ordered(ctx: ProblemContext, requests: set[str]) -> list[str]:
    """Sort requests by greedy_complete()'s own key for deterministic test ordering"""
    return sorted(requests, key=lambda r: _sort_key(ctx, r))


def _assert_matches_reference(solution: ModemsSolution, request_name: str) -> None:
    """Assert alns_feasible_insertions() and the brute-force reference match exactly"""
    new = _candidates_by_route(alns_feasible_insertions(solution, request_name))
    ref = _candidates_by_route(variant_v0_naive(solution, request_name))
    assert set(new.keys()) == set(ref.keys())
    for key in new:
        candidate = new[key]
        reference = ref[key]
        assert math.isclose(
            candidate.delta_obj,
            reference.delta_obj,
            rel_tol=1e-9,
            abs_tol=1e-7,
        ), f"{key}: new={candidate.delta_obj!r} ref={reference.delta_obj!r}"
        assert candidate.journey.request_pickup == reference.journey.request_pickup
        assert candidate.journey.request_delivery == reference.journey.request_delivery
        assert len(candidate.journey.states) == len(reference.journey.states)
        for actual_state, reference_state in zip(
            candidate.journey.states, reference.journey.states
        ):
            assert actual_state.node == reference_state.node
            assert actual_state.z_arr == reference_state.z_arr
            assert actual_state.z_dep == reference_state.z_dep
            for attribute in (
                "t_arr",
                "t_wait",
                "t_start",
                "t_dep",
                "tau",
                "phi_arr",
                "phi_dep",
            ):
                assert math.isclose(
                    getattr(actual_state, attribute),
                    getattr(reference_state, attribute),
                    rel_tol=1e-9,
                    abs_tol=1e-7,
                ), f"{key} state {actual_state.node} attribute {attribute}"

        adopted = solution.copy()
        adopted.journeys[candidate.journey.agent_name] = candidate.journey
        adopted.accepted.add(request_name)
        adopted.rejected.discard(request_name)
        assert math.isclose(
            adopted.objective() - solution.objective(),
            candidate.delta_obj,
            rel_tol=1e-9,
            abs_tol=1e-7,
        )


# --------------------------------------------------------------------------------------
# Algorithm 1 -- preprocess()
# --------------------------------------------------------------------------------------


def test_preprocess_succeeds_with_all_new_requests(
    small_scenario: ModemsScenario,
) -> None:
    """With no scheduled requests, preprocess() always returns a feasible base plan"""
    ctx = ProblemContext(
        small_scenario,
        ProblemType.closed_selective,
        SolverStrategy.alns,
        PARAMS,
    )
    sol, unassigned = preprocess(ctx)
    assert sol is not None
    assert unassigned == set(sol.ctx.request_names)


# --------------------------------------------------------------------------------------
# Algorithm 2 -- greedy_complete()
# --------------------------------------------------------------------------------------


def test_greedy_complete_accepts_feasible_requests(
    small_scenario: ModemsScenario,
) -> None:
    """greedy_complete() buckets every request into accepted or rejected"""
    ctx = ProblemContext(
        small_scenario,
        ProblemType.closed_selective,
        SolverStrategy.alns,
        PARAMS,
    )
    base_plan, r_unassigned = preprocess(ctx)
    assert base_plan is not None
    sol, _, ok = greedy_complete(base_plan, r_unassigned)
    assert ok
    assert sol.accepted | sol.rejected == set(sol.ctx.request_names)


def test_greedy_complete_rejected_set_is_not_stale() -> None:
    """
    Regression test for a real bug: greedy_complete()'s local rejection tracking must
    appear in both its own return value and solution.rejected
    """
    scenario = ModemsScenarioGenerator(seed=1).generate_random_scenario(
        nr_agents=1, nr_requests=10
    )
    scenario.agents[0].load_max = 2  # force some impossible requests
    ctx = ProblemContext(
        scenario, ProblemType.closed_selective, SolverStrategy.alns, PARAMS
    )
    base_plan, r_unassigned = preprocess(ctx)
    assert base_plan is not None
    sol, r_rejected, ok = greedy_complete(base_plan, r_unassigned)
    assert ok
    assert len(sol.accepted) < len(ctx.request_names)  # fixture must force a rejection
    assert r_rejected == sol.pending()
    assert sol.rejected == sol.pending()


def test_greedy_complete_milp1_fails_if_any_request_unservable() -> None:
    """A non-selective strategy fails outright if one request can't be served"""
    # duration_max just barely covers the idle hop to the nearest final hub
    # (so preprocessing/base-plan feasibility still succeeds) but leaves no
    # budget at all for actually serving any request
    generator = ModemsScenarioGenerator(seed=1)
    scenario = generator.generate_random_scenario(nr_agents=1, nr_requests=3)
    ctx0 = ProblemContext(
        scenario,
        ProblemType.closed_non_selective,
        SolverStrategy.milp1,
        PARAMS,
    )
    agent_name = ctx0.agent_names[0]
    idle_hop = ModemsJourney(ctx0, agent_name).total_travel_time()
    scenario.agents[0].duration_max = idle_hop

    ctx = ProblemContext(
        scenario,
        ProblemType.closed_non_selective,
        SolverStrategy.milp1,
        PARAMS,
    )
    base_plan, r_unassigned = preprocess(ctx)
    assert base_plan is not None
    _, _, ok = greedy_complete(base_plan, r_unassigned)
    assert not ok


def test_greedy_completion_adopts_candidate_without_insert_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    greedy_complete() (ALNS) adopts a ready InsertionCandidate directly,
    no re-insertion
    """
    scenario = ModemsScenarioGenerator(seed=8).generate_random_scenario(
        nr_agents=1,
        nr_requests=3,
    )
    base_plan, r_unassigned = _preprocess(
        scenario,
        ProblemType.closed_selective,
        SolverStrategy.alns,
        PARAMS,
    )
    assert base_plan is not None

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("the ready candidate must be adopted directly")

    monkeypatch.setattr(ModemsJourney, "insert_at", fail_if_called)
    sol, _, ok = greedy_complete(base_plan, r_unassigned)
    assert ok
    assert sol.accepted


# --------------------------------------------------------------------------------------
# Algorithm 3 -- alns_feasible_insertions(): direct unit tests
# --------------------------------------------------------------------------------------


def test_feasible_insertions_returns_candidates_for_first_request(
    small_scenario: ModemsScenario,
) -> None:
    """Every returned candidate is a feasible InsertionCandidate for this context"""
    ctx = ProblemContext(
        small_scenario,
        ProblemType.closed_selective,
        SolverStrategy.alns,
        PARAMS,
    )
    base_plan, r_unassigned = preprocess(ctx)
    assert base_plan is not None
    ctx = base_plan.ctx
    r = next(iter(r_unassigned))
    candidates = alns_feasible_insertions(base_plan, r)
    assert len(candidates) > 0
    assert all(isinstance(c, InsertionCandidate) for c in candidates)
    assert all(c.journey.agent_name in ctx.agent_names for c in candidates)
    assert all(c.journey._is_feasible() for c in candidates)


def test_feasible_insertions_respects_capacity() -> None:
    """No candidates are returned when the request load exceeds every agent capacity"""
    generator = ModemsScenarioGenerator(seed=1)
    scenario = generator.generate_random_scenario(nr_agents=1, nr_requests=1)
    scenario.agents[0].load_max = 1
    scenario.requests[0].load = 5  # exceeds capacity
    ctx = ProblemContext(
        scenario,
        ProblemType.closed_selective,
        SolverStrategy.alns,
        PARAMS,
    )
    base_plan, r_unassigned = preprocess(ctx)
    assert base_plan is not None
    r = next(iter(r_unassigned))
    candidates = alns_feasible_insertions(base_plan, r)
    assert candidates == []


def test_feasible_insertions_respects_empty_agent_filter() -> None:
    """Passing agents=[] returns no candidates regardless of feasibility"""
    scenario = ModemsScenarioGenerator(seed=3).generate_random_scenario(
        nr_agents=1,
        nr_requests=1,
    )
    ctx = ProblemContext(
        scenario,
        ProblemType.closed_selective,
        SolverStrategy.alns,
        PARAMS,
    )
    solution = ModemsSolution(ctx)
    assert alns_feasible_insertions(solution, ctx.request_names[0], agents=[]) == []


def test_milp_feasible_insertions_does_not_corrupt_source_journey() -> None:
    """
    Regression test for a real bug: ModemsJourney.copy() aliases request_pickup/
    request_delivery to the same dict objects as the source journey (copy-on-write)
    until something reassigns them. milp_feasible_insertions() used to silently mutate
    the original, still-live journey dict too; it must rebuild fresh dicts instead
    """
    scenario = ModemsScenarioGenerator(seed=1).generate_random_scenario(
        nr_agents=1, nr_requests=4
    )
    ctx = ProblemContext(
        scenario, ProblemType.closed_non_selective, SolverStrategy.milp1, PARAMS
    )
    base_plan, r_unassigned = preprocess(ctx)
    assert base_plan is not None
    agent_name = ctx.agent_names[0]
    source_journey = base_plan.journeys[agent_name]
    pickup_before = dict(source_journey.request_pickup)
    delivery_before = dict(source_journey.request_delivery)

    r = next(iter(r_unassigned))
    candidates = milp_feasible_insertions(base_plan, r)

    assert source_journey.request_pickup == pickup_before
    assert source_journey.request_delivery == delivery_before
    assert r not in source_journey.request_pickup
    if candidates:
        assert r in candidates[0].journey.request_pickup


# --------------------------------------------------------------------------------------
# Broad cross-validation against the reference (pre-rewrite) implementation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(1, 15))
def test_matches_reference_across_seeds_alns(seed: int) -> None:
    """alns_feasible_insertions() matches the brute-force reference across many seeds"""
    generator = ModemsScenarioGenerator(seed=seed)
    n_agents = 1 + seed % 3
    n_requests = 6 + seed % 8
    scenario = generator.generate_random_scenario(
        nr_agents=n_agents, nr_requests=n_requests
    )
    base_plan, r_unassigned = _preprocess(
        scenario, ProblemType.closed_selective, SolverStrategy.alns, PARAMS
    )
    if base_plan is None:
        pytest.skip("infeasible base plan for this seed")
    ctx = base_plan.ctx
    nr_seed = max(1, n_requests - 3)
    ordered = _ordered(ctx, r_unassigned)
    sol, _, _ = greedy_complete(base_plan, set(ordered[:nr_seed]))
    for r in ordered[nr_seed:]:
        _assert_matches_reference(sol, r)


@pytest.mark.parametrize("ptype", list(ProblemType))
def test_matches_reference_across_alns_problem_types(ptype: ProblemType) -> None:
    """alns_feasible_insertions() matches the reference under every ProblemType"""
    generator = ModemsScenarioGenerator(seed=3)
    scenario = generator.generate_random_scenario(nr_agents=2, nr_requests=8)
    ctx = ProblemContext(scenario, ptype, SolverStrategy.alns, PARAMS)
    sol = ModemsSolution(ctx)
    r_assigned = []
    for r in ctx.request_names[:5]:
        for k in ctx.agent_names:
            candidate = sol.journeys[k]._append_request_direct(r)
            if candidate is not None:
                sol.journeys[k] = candidate
                sol.accepted.add(r)
                r_assigned.append(r)
                break
    for r in [r for r in ctx.request_names if r not in r_assigned]:
        _assert_matches_reference(sol, r)


# --------------------------------------------------------------------------------------
# Targeted regressions for bugs found while building the current
# alns_feasible_insertions() implementation
# --------------------------------------------------------------------------------------


def test_modified_segment_intermediate_node_shift_affects_objective() -> None:
    """
    Regression for a real bug: inserting p^u before an intermediate node n_b (a < b)
    shifts n_b's own service-start time, and if n_b is itself a request node, that
    shift changes its own g_i(t) contribution to the objective, which is distinct from
    (and in addition to) the separately-handled suffix shift past b. The first version
    of this rewrite silently dropped this term entirely
    """
    generator = ModemsScenarioGenerator(seed=1)
    scenario = generator.generate_random_scenario(nr_agents=1, nr_requests=10)
    base_plan, r_unassigned = _preprocess(
        scenario, ProblemType.closed_selective, SolverStrategy.alns, PARAMS
    )
    assert base_plan is not None
    ctx = base_plan.ctx
    r_ordered = _ordered(ctx, r_unassigned)
    sol, _, _ = greedy_complete(base_plan, set(r_ordered[:7]))
    r_remaining = r_ordered[7:]

    found_b_gt_a = False
    for r in r_remaining:
        new = alns_feasible_insertions(sol, r)
        for c in new:
            pickup_index = c.journey.request_pickup[r]
            delivery_index = c.journey.request_delivery[r]
            if delivery_index > pickup_index + 1:
                found_b_gt_a = True
        _assert_matches_reference(sol, r)
    assert (
        found_b_gt_a
    ), "test scenario never exercised a b>a candidate; strengthen the fixture"


def test_non_selective_problem_has_no_rejection_penalty_in_delta() -> None:
    """
    Regression for a real bug: a non-selective problem has no rejection penalty in
    the objective at all, so accepting a request must not subtract eta from delta_obj.
    The first version of this rewrite unconditionally subtracted eta regardless of
    the problem type
    """
    generator = ModemsScenarioGenerator(seed=1)
    scenario = generator.generate_random_scenario(nr_agents=2, nr_requests=6)
    ctx = ProblemContext(
        scenario,
        ProblemType.closed_non_selective,
        SolverStrategy.alns,
        PARAMS,
    )
    sol = ModemsSolution(ctx)
    r0 = ctx.request_names[0]
    for k in ctx.agent_names:
        candidate = sol.journeys[k]._append_request_direct(r0)
        if candidate is not None:
            sol.journeys[k] = candidate
            sol.accepted.add(r0)
            break

    r_remaining = ctx.request_names[1]
    candidates = alns_feasible_insertions(sol, r_remaining)
    assert candidates, "expected at least one feasible candidate"
    for c in candidates:
        # a bare insertion delta should be a modest, request-scale adjustment
        # (mission time + eps terms), never offset by -100 (eta)
        assert c.delta_obj > -50.0, f"delta_obj={c.delta_obj} looks eta-contaminated"
    _assert_matches_reference(sol, r_remaining)


def test_candidate_journey_shares_state_until_committed_mutation() -> None:
    """A candidate journey aliases the source untouched states (copy-on-write)"""
    scenario = ModemsScenarioGenerator(seed=8).generate_random_scenario(
        nr_agents=1,
        nr_requests=3,
    )
    base_plan, r_unassigned = _preprocess(
        scenario,
        ProblemType.closed_selective,
        SolverStrategy.alns,
        PARAMS,
    )
    assert base_plan is not None
    request_name = _ordered(base_plan.ctx, r_unassigned)[0]
    source_journey = base_plan.journeys[base_plan.ctx.agent_names[0]]
    candidate = alns_feasible_insertions(base_plan, request_name)[0]

    assert candidate.journey.states[0] is source_journey.states[0]
    source_route = list(source_journey.route)
    source_states = [state.to_dict() for state in source_journey.states]

    candidate.journey.remove_request(request_name)

    assert list(source_journey.route) == source_route
    assert [state.to_dict() for state in source_journey.states] == source_states


def test_candidates_use_independent_builtin_containers() -> None:
    """Each candidate owns its own route/states/index-map containers (no aliasing)"""
    scenario = ModemsScenarioGenerator(seed=5).generate_random_scenario(
        nr_agents=1,
        nr_requests=8,
    )
    base_plan, r_unassigned = _preprocess(
        scenario, ProblemType.closed_selective, SolverStrategy.alns, PARAMS
    )
    assert base_plan is not None
    r_ordered = _ordered(base_plan.ctx, r_unassigned)
    sol, _, ok = greedy_complete(base_plan, set(r_ordered[:4]))
    assert ok
    candidates = alns_feasible_insertions(sol, r_ordered[4])

    assert len(candidates) > 1
    assert all(type(candidate.journey.route) is list for candidate in candidates)
    assert all(type(candidate.journey.states) is list for candidate in candidates)
    assert all(
        type(candidate.journey.request_pickup) is dict for candidate in candidates
    )
    assert all(
        type(candidate.journey.request_delivery) is dict for candidate in candidates
    )
    assert len({id(candidate.journey.route) for candidate in candidates}) == len(
        candidates
    )
    assert len({id(candidate.journey.states) for candidate in candidates}) == len(
        candidates
    )
    assert all(
        candidate.journey.route == [state.node for state in candidate.journey.states]
        for candidate in candidates
    )


# --------------------------------------------------------------------------------------
# New pruning optimizations
# --------------------------------------------------------------------------------------


def test_energy_lower_bound_prunes_when_terminal_slack_is_tight() -> None:
    """
    The energy lower bound (monotone on-board distance while carrying u) should
    terminate the delivery scan early once even the minimum extra cost would exceed
    the terminal SoC slack: verified indirectly by checking the candidate set still
    matches the reference exactly even when xi^k is deliberately made very tight
    """
    generator = ModemsScenarioGenerator(seed=2)
    scenario = generator.generate_random_scenario(nr_agents=1, nr_requests=6)
    base_plan, r_unassigned = _preprocess(
        scenario, ProblemType.closed_selective, SolverStrategy.alns, PARAMS
    )
    assert base_plan is not None
    ctx = base_plan.ctx
    r_ordered = _ordered(ctx, r_unassigned)
    sol, _, _ = greedy_complete(base_plan, set(r_ordered[:3]))
    # tighten the agent's operational SoC floor so slack is scarce
    for j in sol.journeys.values():
        j.agent.soc_min_operational = min(0.9, j.states[-1].phi_arr - 1e-3)
    for r in r_ordered[3:]:
        _assert_matches_reference(sol, r)


def test_ride_time_violation_for_u_terminates_scan_not_just_candidate() -> None:
    """
    u's own ride time is monotone non-decreasing as b increases (more detour before
    its own delivery), so once it exceeds the max-ride-time bound, every larger b is
    equally hopeless
    """
    generator = ModemsScenarioGenerator(seed=4)
    scenario = generator.generate_random_scenario(nr_agents=1, nr_requests=8)
    base_plan, r_unassigned = _preprocess(
        scenario, ProblemType.closed_selective, SolverStrategy.alns, PARAMS
    )
    assert base_plan is not None
    ctx = base_plan.ctx
    r_ordered = _ordered(ctx, r_unassigned)
    sol, _, _ = greedy_complete(base_plan, r_ordered[:5])
    for r in r_ordered[5:]:
        _assert_matches_reference(sol, r)


# --------------------------------------------------------------------------------------
# Determinism note
# --------------------------------------------------------------------------------------


def test_feasible_insertions_is_deterministic_given_sorted_input() -> None:
    """
    alns_feasible_insertions() itself must be a pure function of its inputs, verified
    by calling it twice on the same (ctx, solution, request) and checking bit-identical
    output. Full-process ALNS determinism is covered separately in test_alns.py
    """
    generator = ModemsScenarioGenerator(seed=42)
    scenario = generator.generate_random_scenario(nr_agents=2, nr_requests=8)
    base_plan, r_unassigned = _preprocess(
        scenario, ProblemType.closed_selective, SolverStrategy.alns, PARAMS
    )
    assert base_plan is not None
    ctx = base_plan.ctx
    r_ordered = _ordered(ctx, r_unassigned)
    sol, _, _ = greedy_complete(base_plan, set(r_ordered[:5]))
    for r in r_ordered[5:]:
        first = _candidate_map(alns_feasible_insertions(sol, r))
        second = _candidate_map(alns_feasible_insertions(sol, r))
        assert first.keys() == second.keys()
        for key in first:
            assert first[key] == second[key]
