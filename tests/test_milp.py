"""
Tests for modems.milp. Model-level tests: checking the formulation vs. solution model,
a greedy plan loaded as a warm start must satisfy every MILP constraint and reproduce
the same objective. Integration tests: solve tiny instances with CBC (and HiGHS when
installed) and check exact, hand-derived optima
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from typing import Any

import pyomo.environ as pyo
import pytest

import modems.milp as milp_module
from modems.algorithms import (
    _sort_key,
    greedy_complete,
    milp_feasible_insertions,
    preprocess,
)
from modems.core import ModemsAgent, ModemsRequest, ModemsScenario, SolverStrategy
from modems.milp import MilpType, ModemsMilp
from modems.solution import FLOAT_INF, ModemsInstance, SolutionStatus

from .builders import generated, greedy_solution, line_network, line_scenario, make_ctx

TIMELIMIT = 30.0
requires_highs = pytest.mark.skipif(
    importlib.util.find_spec("highspy") is None, reason="highspy not installed"
)

# one request s2 -> s5 (pickup window [10, 15]) on the line network: serving at
# t=10 is optimal for every variant, T = 20 (closed) or 15 (open), eps * (10 + 14)
ONE_REQUEST = [ModemsRequest(2, 5, load=2, service_time=1.0, earliest_pickup=10.0)]


def solve(
    scenario: ModemsScenario,
    milp_type: str,
    objective: str = "closed",
    selective: bool | None = None,
    warm_start: bool = True,
    solver: tuple[str, str] = ("cbc", "cbc"),
    **model_params: Any,
) -> ModemsMilp:
    ctx = make_ctx(scenario, milp_type, objective, selective, **model_params)
    model = ModemsMilp(scenario, milp_type, ctx.problem_type, model_params or None)
    model.solve(
        *solver,
        solver_options={"timelimit": TIMELIMIT},
        warm_start_solution=greedy_solution(ctx) if warm_start else None,
    )
    return model


# --------------------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("milp_type", list(MilpType))
def test_milp_type_maps_to_its_strategy(milp_type: MilpType) -> None:
    assert milp_type.kappa is SolverStrategy(milp_type.value)


@pytest.mark.parametrize(
    "milp_type, problem_type, match",
    [
        ("milp1", "closed_selective", "MILP1"),
        ("milp1", "open_selective", "MILP1"),
        ("milp2", "closed_non_selective", "MILP2"),
        ("milp2", "open_non_selective", "MILP2"),
    ],
)
def test_constructor_rejects_unsupported_selectivity(
    milp_type: str, problem_type: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        ModemsMilp(line_scenario(ONE_REQUEST), milp_type, problem_type)


def test_constructor_rejects_non_positive_big_m() -> None:
    with pytest.raises(ValueError, match="big_m"):
        ModemsMilp(
            line_scenario(ONE_REQUEST), "milp3", "closed_selective", {"big_m": 0.0}
        )


def test_constructor_works_on_a_private_copy_of_the_scenario() -> None:
    scenario = line_scenario(ONE_REQUEST)
    model = ModemsMilp(scenario, "milp3", "closed_selective")
    assert model.ctx.scenario is not scenario
    scenario.requests[0].earliest_pickup = 40.0
    assert model.ctx.requests["request_1"].earliest_pickup == 10.0


# --------------------------------------------------------------------------------------
# Solver option mapping (fake solver, no optimization)
# --------------------------------------------------------------------------------------


class _Stop(Exception):
    pass


@pytest.mark.parametrize(
    "config, expected",
    [
        ("cbc", {"seconds": 12.0, "mipgap": 0.1}),
        ("gurobi", {"timelimit": 12.0, "threads": 2, "mipgap": 0.1}),
        ("highs", {"time_limit": 12.0, "threads": 2, "mipgap": 0.1}),
    ],
)
def test_generic_solver_options_are_mapped_per_config(
    monkeypatch: pytest.MonkeyPatch, config: str, expected: dict
) -> None:
    class FakeSolver:
        def __init__(self) -> None:
            self.options: dict[str, Any] = {}

        def available(self, exception_flag: bool = False) -> bool:
            return True

        def solve(self, *args: Any, **kwargs: Any) -> None:
            raise _Stop

    fake = FakeSolver()
    monkeypatch.setattr(milp_module, "SolverFactory", lambda *a, **k: fake)
    model = ModemsMilp(line_scenario(ONE_REQUEST), "milp3", "closed_selective")
    with pytest.raises(_Stop):
        model.solve("any", config, {"timelimit": 12.0, "threads": 2, "mipgap": 0.1})
    assert fake.options == expected


# --------------------------------------------------------------------------------------
# Formulation vs. solution model (no solver call)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("objective", ["closed", "open"])
@pytest.mark.parametrize(
    "milp_type, selective",
    [("milp1", False), ("milp2", True), ("milp3", True), ("milp3", False)],
)
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_greedy_plan_is_a_valid_milp_assignment_with_equal_objective(
    seed: int, milp_type: str, selective: bool, objective: str
) -> None:
    scenario = generated(seed, nr_agents=2, nr_requests=5, nr_scheduled=0)
    ctx = make_ctx(scenario, milp_type, objective, selective)
    plan = greedy_solution(ctx)
    model = ModemsMilp(scenario, milp_type, ctx.problem_type)
    model._warm_start(plan)
    assert model._loaded_solution_is_valid()
    assert pyo.value(model.model.objective) == pytest.approx(plan.objective())


def test_warm_start_rejected_requests_are_deselected() -> None:
    requests = [
        ModemsRequest(1, 2, load=5, request_id="a"),
        # arrives at s6 at t=6, window [0, 1]: rejected under hard time windows
        ModemsRequest(6, 1, load=3, earliest_pickup=0.0, tw_length=1.0, request_id="b"),
    ]
    ctx = make_ctx(line_scenario(requests), "milp2")
    plan = greedy_solution(ctx)
    assert plan.rejected == {"request_2"}
    model = ModemsMilp(ctx.scenario, "milp2", ctx.problem_type)
    model._warm_start(plan)
    assert model.model.var_y["request_1"].value == 1
    assert model.model.var_y["request_2"].value == 0
    assert model._loaded_solution_is_valid()


def _warm_started(milp_type: str = "milp3") -> ModemsMilp:
    scenario = generated(1, nr_agents=2, nr_requests=5)
    ctx = make_ctx(scenario, milp_type)
    model = ModemsMilp(scenario, milp_type, ctx.problem_type)
    model._warm_start(greedy_solution(ctx))
    assert model._loaded_solution_is_valid()
    return model


def test_a_fractional_point_is_not_a_valid_incumbent(monkeypatch) -> None:
    """
    The midpoint of two valid plans satisfies every (linear) constraint, exactly like
    an LP relaxation point loaded after a time limit, but it is not integral -> invalid
    """
    scenario = generated(1, nr_agents=2, nr_requests=5)
    ctx = make_ctx(scenario, "milp3")
    base, unassigned = preprocess(ctx)
    ordered = sorted(unassigned, key=lambda r: _sort_key(ctx, r))
    partial, _, _ = greedy_complete(base, set(ordered[:-1]))
    points = []
    for candidate in milp_feasible_insertions(partial, ordered[-1])[:2]:
        plan = partial.copy()
        plan.journeys[candidate.journey.agent_name] = candidate.journey
        plan.accepted.add(ordered[-1])
        model = ModemsMilp(scenario, "milp3", ctx.problem_type)
        model._warm_start(plan)
        assert model._loaded_solution_is_valid()
        points.append(
            {v.name: v.value for v in model.model.component_data_objects(pyo.Var)}
        )
    assert len(points) == 2 and points[0] != points[1]

    for var in model.model.component_data_objects(pyo.Var):
        a, b = points[0][var.name], points[1][var.name]
        var.value = None if a is None or b is None else (a + b) / 2
    monkeypatch.setattr(milp_module, "INTEGRALITY_TOL", 1.0)  # constraints only
    assert model._loaded_solution_is_valid()
    monkeypatch.undo()
    assert not model._loaded_solution_is_valid()


def test_invalid_incumbent_is_reported_as_unknown_without_an_objective() -> None:
    """For example, a time limit reached before any integer solution was found"""
    model = _warm_started()
    arc = next(a for a in model.model.set_arcs if model.model.var_x[a].value == 1)
    model.model.var_x[arc].value = 0.6
    model.results = SimpleNamespace(
        problem=SimpleNamespace(lower_bound=50.0, upper_bound=56.4),
        solver=SimpleNamespace(
            status=pyo.SolverStatus.aborted,
            termination_condition=pyo.TerminationCondition.maxTimeLimit,
        ),
    )
    model.problem_done, model.solver_type, model.solver_options = True, "cbc", {}
    model._build_instance(solution_time=3.0)
    info = model.instance.solution_info
    assert info.status is SolutionStatus.unknown
    assert info.objective is None and info.upper_bound is None
    assert info.lower_bound == 50.0  # the solver bound remains valid
    assert model.instance.solution.accepted == set()


# --------------------------------------------------------------------------------------
# Trivial instances (the optimizer is never invoked)
# --------------------------------------------------------------------------------------


def _no_solver(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("the optimizer must not be invoked")


def test_no_requests_is_solved_exactly_without_the_optimizer(monkeypatch) -> None:
    monkeypatch.setattr(milp_module, "SolverFactory", _no_solver)
    model = ModemsMilp(line_scenario([]), "milp3", "closed_selective")
    assert model.is_trivial
    assert model.solve("cbc", "cbc") is None
    info = model.instance.solution_info
    assert (info.status, info.objective) == (SolutionStatus.optimal, 0.0)
    assert info.lower_bound is info.upper_bound is None
    assert info.solver_diagnostics == {"trivial_solution": 1}


@pytest.mark.parametrize(
    "milp_type, problem_type, status, objective, rejected",
    [
        ("milp3", "closed_selective", SolutionStatus.optimal, 100.0, {"request_1"}),
        ("milp1", "closed_non_selective", SolutionStatus.infeasible, FLOAT_INF, set()),
    ],
)
def test_no_agents_rejects_all_or_is_infeasible(
    monkeypatch, milp_type, problem_type, status, objective, rejected
) -> None:
    monkeypatch.setattr(milp_module, "SolverFactory", _no_solver)
    scenario = ModemsScenario(
        [], ONE_REQUEST, line_network([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    )
    model = ModemsMilp(scenario, milp_type, problem_type)
    model.solve("cbc", "cbc")
    info = model.instance.solution_info
    assert (info.status, info.objective) == (status, objective)
    assert model.instance.solution.rejected == rejected
    assert model.instance.solution.accepted == set()


# --------------------------------------------------------------------------------------
# Integration: exact optima on hand-built instances
# --------------------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("objective, expected", [("closed", 20.24), ("open", 15.24)])
@pytest.mark.parametrize(
    "milp_type, selective",
    [("milp1", False), ("milp2", True), ("milp3", True), ("milp3", False)],
)
def test_single_request_optimum(
    milp_type: str, selective: bool, objective: str, expected: float
) -> None:
    model = solve(line_scenario(ONE_REQUEST), milp_type, objective, selective)
    info, solution = model.instance.solution_info, model.instance.solution
    assert info.status is SolutionStatus.optimal
    assert info.objective == pytest.approx(expected)
    assert solution.objective() == pytest.approx(expected)
    assert solution.journeys["agent_1"].route == [
        "a_1_h_1",
        "r_1_p_2",
        "r_1_d_5",
        "h_1",
    ]
    assert solution.journeys["agent_1"].states[1].t_start == pytest.approx(10.0)
    assert info.lower_bound == pytest.approx(expected)


@pytest.mark.integration
@pytest.mark.parametrize(
    "milp_type, selective, accepted, expected",
    [
        ("milp1", False, {"request_1"}, 20.24),
        ("milp3", False, {"request_1"}, 20.24),
        ("milp2", True, set(), 5.0),
        ("milp3", True, set(), 5.0),
    ],
)
def test_cheap_rejection_is_taken_only_by_selective_variants(
    milp_type: str, selective: bool, accepted: set, expected: float
) -> None:
    """With eta = 5 < 20.24, rejecting the only request is optimal when allowed"""
    model = solve(line_scenario(ONE_REQUEST), milp_type, selective=selective, eta=5.0)
    assert model.instance.solution_info.status is SolutionStatus.optimal
    assert model.instance.solution.accepted == accepted
    assert model.instance.solution_info.objective == pytest.approx(expected)


@pytest.mark.integration
def test_optimum_trades_capacity_across_two_agents() -> None:
    """Two simultaneous 4-passenger requests cannot share one 6-seat agent"""
    requests = [
        ModemsRequest(1, 3, load=4, earliest_pickup=5.0, request_id="a"),
        ModemsRequest(2, 4, load=4, earliest_pickup=5.0, request_id="b"),
    ]
    agents = [ModemsAgent("h", 1, agent_id="k1"), ModemsAgent("h", 1, agent_id="k2")]
    model = solve(line_scenario(requests, agents=agents), "milp3")
    solution = model.instance.solution
    assert model.instance.solution_info.status is SolutionStatus.optimal
    assert solution.accepted == {"request_1", "request_2"}
    assert {solution.agent_of("request_1"), solution.agent_of("request_2")} == {
        "agent_1",
        "agent_2",
    }


# --------------------------------------------------------------------------------------
# Integration: generated instances
# --------------------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize(
    "milp_type, selective",
    [("milp1", False), ("milp2", True), ("milp3", True), ("milp3", False)],
)
def test_generated_instance_solves_to_a_consistent_optimum(
    tiny_scenario: ModemsScenario, milp_type: str, selective: bool
) -> None:
    ctx = make_ctx(tiny_scenario, milp_type, selective=selective)
    baseline = greedy_solution(ctx).objective()
    model = solve(tiny_scenario, milp_type, selective=selective)
    info, solution = model.instance.solution_info, model.instance.solution
    assert info.status is SolutionStatus.optimal
    assert info.objective == pytest.approx(solution.objective())
    assert info.objective <= baseline + 1e-6
    assert all(journey._is_feasible() for journey in solution.journeys.values())
    if not selective:
        assert solution.accepted == set(ctx.request_names)
    assert info.solver_diagnostics["nr_variables"] is not None


@pytest.mark.integration
@pytest.mark.parametrize("milp_type, selective", [("milp1", False), ("milp3", True)])
def test_open_optimum_never_exceeds_closed(
    small_scenario: ModemsScenario, milp_type: str, selective: bool
) -> None:
    closed = solve(small_scenario, milp_type, "closed", selective).instance
    opened = solve(small_scenario, milp_type, "open", selective).instance
    assert (
        opened.solution_info.status
        is closed.solution_info.status
        is SolutionStatus.optimal
    )
    assert opened.solution_info.objective <= closed.solution_info.objective + 1e-6


@pytest.mark.integration
def test_solved_instance_json_round_trip(
    tmp_path, tiny_scenario: ModemsScenario
) -> None:
    instance = solve(tiny_scenario, "milp3").instance
    restored = ModemsInstance.from_json(instance.to_json(str(tmp_path / "milp.json")))
    assert restored.to_dict() == instance.to_dict()


@requires_highs
@pytest.mark.integration
def test_cbc_and_highs_reach_the_same_optimum(small_scenario: ModemsScenario) -> None:
    cbc = solve(small_scenario, "milp3").instance.solution_info
    highs = solve(
        small_scenario, "milp3", solver=("appsi_highs", "highs")
    ).instance.solution_info
    assert cbc.status is highs.status is SolutionStatus.optimal
    assert cbc.objective == pytest.approx(highs.objective, abs=1e-4)
