"""
Tests for modems.alns. Operator tests call each destroy/repair operator directly on
a greedy solution and check what every ALNS state must keep: feasible journeys, with 
routed requests equal to the accepted set, and accepted/rejected request partitioning. 
Integration tests run the full search through the alns package
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from modems.alns import ModemsAlns
from modems.core import ModemsAgent, ModemsRequest, ModemsScenario
from modems.solution import (
    ModemsInstance,
    ModemsJourney,
    ModemsSolution,
    SolutionStatus,
)

from .builders import generated, line_network, line_scenario, make_ctx

DESTROY_OPS = [
    "destroy_random",
    "destroy_random_zone",
    "destroy_lowest_demand",
    "destroy_worst_cost",
    "destroy_worst_waiting",
    "destroy_worst_energy",
]
REPAIR_OPS = [
    "repair_random",
    "repair_greedy",
    "repair_regret_2",
    "repair_longest_trip",
    "repair_most_constrained",
]


def assert_consistent(state: ModemsSolution) -> None:
    ctx = state.ctx
    assert all(journey._is_feasible() for journey in state.journeys.values())
    routed = [r for journey in state.journeys.values() for r in journey.request_pickup]
    assert len(routed) == len(set(routed))
    assert set(routed) == state.accepted
    assert state.accepted.isdisjoint(state.rejected)
    assert state.accepted | state.rejected == set(ctx.request_names)


@pytest.fixture
def alns_and_state() -> tuple[ModemsAlns, ModemsSolution]:
    """ALNS wrapper and its greedy initial state on a 2-agent, 10-request scenario"""
    alns = ModemsAlns(generated(5, nr_agents=2, nr_requests=10))
    state = alns._create_initial_solution()
    assert len(state.accepted) >= 5
    return alns, state


# --------------------------------------------------------------------------------------
# Construction and argument validation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("ptype", ["closed_non_selective", "open_non_selective"])
def test_alns_rejects_non_selective_problems(ptype: str) -> None:
    with pytest.raises(ValueError, match="selective"):
        ModemsAlns(line_scenario([ModemsRequest(1, 2)]), ptype)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"params_rd": {"min_pct": 0.5, "max_pct": 0.4}}, "destroy percentages"),
        ({"params_rd": {"min_pct": 0.0}}, "destroy percentages"),
        ({"params_rd": {"max_pct": 1.0}}, "destroy percentages"),
        ({"max_iter": 0}, "iterations"),
    ],
)
def test_solve_rejects_invalid_search_parameters(kwargs: dict, match: str) -> None:
    alns = ModemsAlns(line_scenario([ModemsRequest(1, 2)]))
    with pytest.raises(ValueError, match=match):
        alns.solve(seed=1, **kwargs)


def test_scheduled_requests_require_a_partial_plan() -> None:
    scenario = line_scenario([ModemsRequest(1, 2, status="scheduled")])
    with pytest.raises(RuntimeError, match="Infeasible base plan"):
        ModemsAlns(scenario).solve(seed=1)


def test_solve_before_plotting_raises() -> None:
    alns = ModemsAlns(line_scenario([ModemsRequest(1, 2)]))
    with pytest.raises(NameError):
        alns.plot_routes(show=False)
    with pytest.raises(NameError):
        alns.plot_metrics("unused", "unused")


# --------------------------------------------------------------------------------------
# Trivial instances (the search is never run)
# --------------------------------------------------------------------------------------


def test_no_requests_returns_the_idle_fleet_without_searching() -> None:
    alns = ModemsAlns(line_scenario([]))
    assert alns.solve(seed=1) is None
    info = alns.instance.solution_info
    assert (info.status, info.objective) == (SolutionStatus.optimal, 0.0)
    assert info.solver_diagnostics == {"iterations": 0}
    assert alns.result is None


def test_no_agents_rejects_every_request_without_searching() -> None:
    requests = [
        ModemsRequest(1, 2, request_id="a"),
        ModemsRequest(2, 3, request_id="b"),
    ]
    alns = ModemsAlns(ModemsScenario([], requests, line_network([0.0, 1.0, 2.0, 3.0])))
    assert alns.solve(seed=1) is None
    assert alns.instance.solution.rejected == {"request_1", "request_2"}
    assert alns.instance.solution_info.objective == 2 * alns.ctx.eta


# --------------------------------------------------------------------------------------
# Destroy operators
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pct", [(0.1, 0.2), (0.2, 0.4), (0.5, 0.9)])
def test_destroy_size_is_a_fraction_of_accepted_at_least_one(pct: tuple) -> None:
    alns = ModemsAlns(line_scenario([ModemsRequest(1, 2)]))
    alns._destroy_min_pct, alns._destroy_max_pct = pct
    rng = np.random.default_rng(0)
    state = ModemsSolution(alns.ctx)
    assert alns._destroy_size(state, rng) == 0
    state.accepted = {f"request_{i}" for i in range(10)}
    sizes = {alns._destroy_size(state, rng) for _ in range(200)}
    assert min(sizes) >= max(1, int(pct[0] * 10)) and max(sizes) <= int(pct[1] * 10)


@pytest.mark.parametrize("operator", DESTROY_OPS)
def test_destroy_removes_requests_and_keeps_the_state_consistent(
    alns_and_state: tuple[ModemsAlns, ModemsSolution], operator: str
) -> None:
    alns, state = alns_and_state
    snapshot = state.to_dict()
    for seed in range(5):
        rng = np.random.default_rng(seed)
        expected = alns._destroy_size(state, np.random.default_rng(seed))
        destroyed = getattr(alns, operator)(state, rng)
        removed = state.accepted - destroyed.accepted
        assert destroyed.accepted <= state.accepted
        assert len(removed) >= expected  # random-zone can remove a whole station
        assert_consistent(destroyed)
    assert state.to_dict() == snapshot  # operators never mutate their input


@pytest.mark.parametrize("operator", DESTROY_OPS)
def test_destroy_on_an_empty_solution_is_a_copy(operator: str) -> None:
    alns = ModemsAlns(line_scenario([ModemsRequest(1, 2)]))
    state = ModemsSolution(alns.ctx)
    state.recompute_rejected()
    destroyed = getattr(alns, operator)(state, np.random.default_rng(0))
    assert destroyed is not state and destroyed.to_dict() == state.to_dict()


def test_lowest_demand_removes_the_smallest_loads_first() -> None:
    requests = [
        ModemsRequest(i, i + 1, load=load, earliest_pickup=10.0 * i, request_id=str(i))
        for i, load in zip(range(1, 5), [3, 1, 2, 1])
    ]
    alns = ModemsAlns(line_scenario(requests, positions=[float(i) for i in range(7)]))
    state = alns._create_initial_solution()
    assert state.accepted == set(alns.ctx.request_names)
    alns._destroy_min_pct, alns._destroy_max_pct = 0.5, 0.6  # removes exactly 2
    destroyed = alns.destroy_lowest_demand(state, np.random.default_rng(0))
    assert state.accepted - destroyed.accepted == {"request_2", "request_4"}


# --------------------------------------------------------------------------------------
# Repair operators
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("repair", REPAIR_OPS)
@pytest.mark.parametrize("destroy", ["destroy_random", "destroy_worst_cost"])
def test_repair_restores_a_consistent_state_without_loosing_insertable_requests(
    alns_and_state: tuple[ModemsAlns, ModemsSolution], destroy: str, repair: str
) -> None:
    alns, state = alns_and_state
    for seed in range(3):
        destroyed = getattr(alns, destroy)(state, np.random.default_rng(seed))
        repaired = getattr(alns, repair)(destroyed, np.random.default_rng(seed))
        assert_consistent(repaired)
        assert destroyed.accepted <= repaired.accepted
        # repair stops only when no pending request has a feasible insertion (left)
        for r in repaired.rejected:
            assert not alns_feasible(repaired, r)


def alns_feasible(state: ModemsSolution, request_name: str) -> bool:
    from modems.algorithms import alns_feasible_insertions

    return bool(alns_feasible_insertions(state, request_name))


@pytest.mark.parametrize("repair", REPAIR_OPS)
def test_repair_restores_scheduled_requests_or_marks_state_infeasible(
    repair: str,
) -> None:
    """Scheduled requests must be re-inserted; otherwise, the state is discarded"""
    requests = [
        ModemsRequest(
            1, 3, load=2, earliest_pickup=5.0, status="scheduled", request_id="s"
        ),
        ModemsRequest(2, 4, earliest_pickup=6.0, request_id="n"),
    ]
    alns = ModemsAlns(line_scenario(requests))
    route = ["a_1_h_1", "r_1_p_1", "r_1_d_3", "h_1"]
    plan = ModemsSolution(alns.ctx)
    plan.journeys["agent_1"] = ModemsJourney.from_route(alns.ctx, "agent_1", route)
    plan.accepted = {"request_1"}
    state = alns._create_initial_solution(partial_plan=plan)

    destroyed = alns._remove_requests(state, ["request_1"])
    repaired = getattr(alns, repair)(destroyed, np.random.default_rng(0))
    assert "request_1" in repaired.accepted and not repaired._infeasible

    blocked = destroyed.copy()  # the agent can no longer seat the scheduled rider
    blocked.journeys["agent_1"].agent = ModemsAgent("h", 1, load_max=1)
    repaired = getattr(alns, repair)(blocked, np.random.default_rng(0))
    assert repaired._infeasible and repaired.objective() >= 1e37


# --------------------------------------------------------------------------------------
# Integration: full search
# --------------------------------------------------------------------------------------


@pytest.mark.integration
def test_search_finds_the_single_request_optimum() -> None:
    """Same hand-built instance as the MILP optimum test: 20.24 (closed)"""
    alns = ModemsAlns(
        line_scenario([ModemsRequest(2, 5, load=2, earliest_pickup=10.0)])
    )
    alns.solve(seed=1, max_iter=20)
    assert alns.instance.solution_info.objective == pytest.approx(20.24)


@pytest.mark.integration
@pytest.mark.parametrize("objective", ["closed", "open"])
def test_search_improves_on_construction_and_stays_consistent(objective: str) -> None:
    scenario = generated(5, nr_agents=2, nr_requests=10)
    alns = ModemsAlns(scenario, make_ctx(scenario, objective=objective).problem_type)
    initial = alns._create_initial_solution().objective()
    alns.solve(seed=3, max_iter=100)
    info = alns.instance.solution_info
    assert_consistent(alns.best_solution)
    assert info.status is SolutionStatus.feasible
    assert info.objective == pytest.approx(alns.best_solution.objective())
    assert info.objective <= initial + 1e-9
    assert info.lower_bound is info.upper_bound is None
    assert info.solver_diagnostics["iterations"] >= 100
    assert info.solver_options["seed"] == 3 and info.solver_options["max_iter"] == 100


@pytest.mark.integration
def test_search_keeps_scheduled_requests_from_the_partial_plan() -> None:
    scenario = generated(5, nr_agents=2, nr_requests=6, nr_scheduled=2)
    ctx = make_ctx(scenario)
    plan = ModemsSolution(ctx)
    for r in ("request_1", "request_2"):
        for k in ctx.agent_names:
            candidate = plan.journeys[k]._append_request_direct(r)
            if candidate is not None:
                plan.journeys[k] = candidate
                plan.accepted.add(r)
                break
    assert plan.accepted == {"request_1", "request_2"}
    alns = ModemsAlns(scenario)
    alns.solve(seed=1, max_iter=50, partial_plan=plan)
    assert {"request_1", "request_2"} <= alns.best_solution.accepted
    assert_consistent(alns.best_solution)


@pytest.mark.integration
def test_search_is_seed_reproducible_and_round_trips(tmp_path) -> None:
    scenario = generated(42, nr_agents=2, nr_requests=8)
    runs = []
    for _ in range(2):
        alns = ModemsAlns(scenario)
        alns.solve(seed=7, max_iter=60)
        runs.append(alns)
    assert runs[0].best_solution.to_dict() == runs[1].best_solution.to_dict()
    instance = runs[0].instance
    restored = ModemsInstance.from_json(instance.to_json(str(tmp_path / "alns.json")))
    assert restored.to_dict() == instance.to_dict()
    runs[0].plot_metrics(str(tmp_path), "run")
    assert (tmp_path / "run_objectives.png").exists()


@pytest.mark.integration
def test_search_is_independent_of_the_python_hash_seed() -> None:
    """Set iteration order varies with PYTHONHASHSEED, only across processes"""
    script = (
        "from modems import ModemsAlns, ModemsScenarioGenerator\n"
        "s = ModemsScenarioGenerator(seed=42).generate_random_scenario(2, 8)\n"
        "a = ModemsAlns(s)\n"
        "a.solve(seed=42, max_iter=60)\n"
        "print(sorted(a.best_solution.get_agent_routes().items()))\n"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            check=True,
        ).stdout
        for hash_seed in ("0", "1", "2")
    }
    assert len(outputs) == 1
