from __future__ import annotations

import pytest
from pyomo.opt import SolverFactory

from modems import ModemsInstance, ModemsScenario
from modems.algorithms import greedy_complete, preprocess
from modems.core import ProblemContext, ProblemType
from modems.milp import DEFAULT_MILP_SOLVER_DATA, MilpType, ModemsMilp, SolverConfigType
from modems.solution import DEFAULT_PARAMS_MILP, ModemsSolution, SolutionStatus

PARAMS: dict = DEFAULT_PARAMS_MILP
TIMELIMIT = 15.0

# HiGHS (via pyomo's appsi_highs interface) is optional, skip when it is not installed
_HIGHS_AVAILABLE = SolverFactory("appsi_highs").available(exception_flag=False)
requires_highs = pytest.mark.skipif(
    not _HIGHS_AVAILABLE, reason="HiGHS (appsi_highs) is not installed/available"
)


def test_milp_enums_are_string_enums() -> None:
    """SolverConfigType/MilpType members equal and coerce from their literal values"""
    assert SolverConfigType.cbc == "cbc"
    assert MilpType.milp1 == "milp1"
    assert MilpType("milp3") is MilpType.milp3


def _build_and_solve(
    scenario: ModemsScenario,
    milp_type: MilpType,
    problem_type: ProblemType,
    solver_name: str = DEFAULT_MILP_SOLVER_DATA[0],
    solver_config_type: SolverConfigType = DEFAULT_MILP_SOLVER_DATA[1],
    warm_start: ModemsSolution | None = None,
) -> ModemsMilp:
    """Build, warm-start (if given), and solve a MILP model end to end"""
    m = ModemsMilp(
        scenario=scenario,
        milp_type=milp_type,
        problem_type=problem_type,
        milp_params=PARAMS,
    )
    m.solve(
        solver_name=solver_name,
        solver_config_type=solver_config_type,
        solver_options={"timelimit": TIMELIMIT},
        warm_start_solution=warm_start,
    )
    return m


def _baseline(
    scenario: ModemsScenario, milp_type: MilpType, problem_type: ProblemType
) -> ModemsSolution | None:
    """Compute the strategy-matched constructive baseline used to warm-start"""
    ctx = ProblemContext(scenario, problem_type, milp_type.kappa, PARAMS)
    base_plan, r_unassigned = preprocess(ctx)
    if base_plan is None:
        return None
    sol, _, ok = greedy_complete(base_plan, r_unassigned)
    return sol if ok else None


# --------------------------------------------------------------------------------------
# Build + solve, per variant (CBC -- always exercised)
# --------------------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize(
    "milp_type,problem_type",
    [
        (MilpType.milp1, ProblemType.closed_non_selective),
        (MilpType.milp2, ProblemType.closed_selective),
        (MilpType.milp3, ProblemType.closed_selective),
    ],
)
def test_milp_solves_tiny_scenario(
    tiny_scenario: ModemsScenario, milp_type: MilpType, problem_type: ProblemType
) -> None:
    """Every MILP variant reaches Optimal or Feasible on a tiny scenario with CBC"""
    warm_start = _baseline(tiny_scenario, milp_type, problem_type)
    m = _build_and_solve(tiny_scenario, milp_type, problem_type, warm_start=warm_start)
    info = m.instance.solution_info
    assert info.status in (SolutionStatus.optimal, SolutionStatus.feasible)


@pytest.mark.integration
@pytest.mark.parametrize(
    "milp_type,problem_type",
    [
        (MilpType.milp1, ProblemType.closed_non_selective),
        (MilpType.milp2, ProblemType.closed_selective),
        (MilpType.milp3, ProblemType.closed_selective),
    ],
)
def test_milp_reported_objective_matches_propagated_solution(
    tiny_scenario: ModemsScenario, milp_type: MilpType, problem_type: ProblemType
) -> None:
    """The solver-reported objective matches ModemsSolution.objective() exactly"""
    warm_start = _baseline(tiny_scenario, milp_type, problem_type)
    m = _build_and_solve(tiny_scenario, milp_type, problem_type, warm_start=warm_start)
    info = m.instance.solution_info
    if info.status not in (SolutionStatus.optimal, SolutionStatus.feasible):
        pytest.skip("no incumbent found within the time limit")
    assert info.objective == pytest.approx(m.instance.solution.objective(), abs=1e-6)


@pytest.mark.integration
def test_milp1_is_non_selective(tiny_scenario: ModemsScenario) -> None:
    """MILP1 accepts every request whenever a proven-optimal solution exists"""
    warm_start = _baseline(
        tiny_scenario, MilpType.milp1, ProblemType.closed_non_selective
    )
    m = _build_and_solve(
        tiny_scenario,
        MilpType.milp1,
        ProblemType.closed_non_selective,
        warm_start=warm_start,
    )
    if m.instance.solution_info.status == SolutionStatus.optimal:
        # non-selective: every request must be accepted when a solution exists
        assert len(m.instance.solution.accepted) == len(tiny_scenario.requests)


@pytest.mark.integration
def test_milp3_honors_non_selective_problem_type(tiny_scenario: ModemsScenario) -> None:
    """MILP3 also accepts every request when built with a non-selective ProblemType"""
    problem_type = ProblemType.closed_non_selective
    warm_start = _baseline(tiny_scenario, MilpType.milp3, problem_type)
    m = _build_and_solve(
        tiny_scenario,
        MilpType.milp3,
        problem_type,
        warm_start=warm_start,
    )
    if m.instance.solution_info.status in (
        SolutionStatus.optimal,
        SolutionStatus.feasible,
    ):
        assert len(m.instance.solution.accepted) == len(tiny_scenario.requests)


# --------------------------------------------------------------------------------------
# Build + solve via HiGHS (optional -- skipped automatically if unavailable)
# --------------------------------------------------------------------------------------


@requires_highs
@pytest.mark.integration
def test_milp3_solves_tiny_scenario_with_highs(tiny_scenario: ModemsScenario) -> None:
    """MILP3 reaches Optimal or Feasible on the real HiGHS backend (appsi_highs)"""
    problem_type = ProblemType.closed_selective
    warm_start = _baseline(tiny_scenario, MilpType.milp3, problem_type)
    m = _build_and_solve(
        tiny_scenario,
        MilpType.milp3,
        problem_type,
        solver_name="appsi_highs",
        solver_config_type=SolverConfigType.highs,
        warm_start=warm_start,
    )
    info = m.instance.solution_info
    assert info.status in (SolutionStatus.optimal, SolutionStatus.feasible)
    assert info.objective == pytest.approx(m.instance.solution.objective(), abs=1e-6)


@requires_highs
@pytest.mark.integration
def test_cbc_and_highs_agree_on_optimal_objective(
    tiny_scenario: ModemsScenario,
) -> None:
    """CBC and HiGHS reach the same objective when both prove optimality"""
    problem_type = ProblemType.closed_selective
    warm_start = _baseline(tiny_scenario, MilpType.milp3, problem_type)
    cbc_model = _build_and_solve(
        tiny_scenario, MilpType.milp3, problem_type, warm_start=warm_start
    )
    highs_model = _build_and_solve(
        tiny_scenario,
        MilpType.milp3,
        problem_type,
        solver_name="appsi_highs",
        solver_config_type=SolverConfigType.highs,
        warm_start=warm_start,
    )
    cbc_info, highs_info = (
        cbc_model.instance.solution_info,
        highs_model.instance.solution_info,
    )
    if (
        cbc_info.status != SolutionStatus.optimal
        or highs_info.status != SolutionStatus.optimal
    ):
        pytest.skip(
            "at least one backend did not prove optimality within the time limit"
        )
    assert cbc_info.objective == pytest.approx(highs_info.objective, abs=1e-4)


# --------------------------------------------------------------------------------------
# Empty agents/requests -- optimizer must not be invoked (fast, no solver call)
# --------------------------------------------------------------------------------------


def test_empty_requests_never_invokes_solver(tiny_scenario: ModemsScenario) -> None:
    """A scenario with zero requests short-circuits to _solve_trivial(); no CBC call"""
    empty = ModemsScenario(
        agents=tiny_scenario.agents, requests=[], network=tiny_scenario.network
    )
    m = ModemsMilp(
        scenario=empty,
        milp_type=MilpType.milp3,
        problem_type=ProblemType.closed_selective,
        milp_params=PARAMS,
    )
    result = m.solve(
        solver_name=DEFAULT_MILP_SOLVER_DATA[0],
        solver_config_type=DEFAULT_MILP_SOLVER_DATA[1],
        solver_options={"timelimit": TIMELIMIT},
    )
    assert result is None  # _solve_trivial path returns None, never calls the solver
    assert m.instance.solution_info.status == SolutionStatus.optimal
    assert m.instance.solution_info.objective == 0.0
    # trivial solves are exact by construction, but per convention we still
    # don't report a bound: only a real optimizer call populates these.
    assert m.instance.solution_info.lower_bound is None
    assert m.instance.solution_info.upper_bound is None


@pytest.mark.integration
def test_real_milp_solve_populates_numeric_bounds(
    tiny_scenario: ModemsScenario,
) -> None:
    """Unlike the trivial/baseline paths, a real (non-trivial) MILP solve reports
    actual numeric lower_bound/upper_bound from the solver, not None"""
    warm_start = _baseline(tiny_scenario, MilpType.milp3, ProblemType.closed_selective)
    m = _build_and_solve(
        tiny_scenario,
        MilpType.milp3,
        ProblemType.closed_selective,
        warm_start=warm_start,
    )
    info = m.instance.solution_info
    if info.status not in (SolutionStatus.optimal, SolutionStatus.feasible):
        pytest.skip("no incumbent found within the time limit")
    assert info.lower_bound is not None
    assert info.upper_bound is not None


@pytest.mark.integration
def test_empty_agents_selective_rejects_everyone(tiny_scenario: ModemsScenario) -> None:
    """A scenario with zero agents rejects every request under a selective type"""
    empty = ModemsScenario(
        agents=[], requests=tiny_scenario.requests, network=tiny_scenario.network
    )
    m = ModemsMilp(
        scenario=empty,
        milp_type=MilpType.milp3,
        problem_type=ProblemType.closed_selective,
        milp_params=PARAMS,
    )
    m.solve(
        solver_name=DEFAULT_MILP_SOLVER_DATA[0],
        solver_config_type=DEFAULT_MILP_SOLVER_DATA[1],
        solver_options={"timelimit": TIMELIMIT},
    )
    assert m.instance.solution_info.status == SolutionStatus.optimal
    assert m.instance.solution_info.objective == PARAMS["eta"] * len(
        tiny_scenario.requests
    )


@pytest.mark.integration
def test_empty_agents_non_selective_is_infeasible(
    tiny_scenario: ModemsScenario,
) -> None:
    """A scenario with zero agents is proven infeasible under a non-selective type"""
    empty = ModemsScenario(
        agents=[], requests=tiny_scenario.requests, network=tiny_scenario.network
    )
    m = ModemsMilp(
        scenario=empty,
        milp_type=MilpType.milp1,
        problem_type=ProblemType.closed_non_selective,
        milp_params=PARAMS,
    )
    m.solve(
        solver_name="cbc",
        solver_config_type=SolverConfigType.cbc,
        solver_options={"timelimit": TIMELIMIT},
    )
    assert m.instance.solution_info.status == SolutionStatus.infeasible


# --------------------------------------------------------------------------------------
# JSON round-trip
# --------------------------------------------------------------------------------------


@pytest.mark.integration
def test_milp_instance_round_trips_exactly(tiny_scenario: ModemsScenario) -> None:
    """to_dict()/from_dict() on a solved instance preserve the objective exactly"""
    warm_start = _baseline(tiny_scenario, MilpType.milp3, ProblemType.closed_selective)
    m = _build_and_solve(
        tiny_scenario,
        MilpType.milp3,
        ProblemType.closed_selective,
        warm_start=warm_start,
    )
    d = m.instance.to_dict()
    inst2 = ModemsInstance.from_dict(d)
    assert inst2.solution.objective() == pytest.approx(
        m.instance.solution.objective(), abs=1e-6
    )
