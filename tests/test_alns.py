from __future__ import annotations

import subprocess
import sys

import pytest

from modems import ModemsAlns, ModemsInstance, ModemsScenario, ModemsScenarioGenerator
from modems.algorithms import greedy_complete, preprocess
from modems.core import ProblemContext, ProblemType, SolverStrategy
from modems.solution import ModemsSolution, SolutionStatus

PARAMS_OBJ: dict = {"eps": 0.01, "zeta": 1.0, "eta": 100.0, "rho": 2.5}
PARAMS_RW: dict = {"scores": [5, 2, 1, 0.5], "decay": 0.8}
PARAMS_SA: dict = {"start_temp": 1000, "end_temp": 0.1, "cooling_rate": 0.995}


def _solve(
    scenario: ModemsScenario,
    seed: int = 1,
    max_iter: int = 300,
    partial_plan: ModemsSolution | None = None,
    problem_type: ProblemType = ProblemType.closed_selective,
) -> ModemsAlns:
    """Build and run a ModemsAlns search end to end via the real alns package."""
    model_alns = ModemsAlns(
        scenario, problem_type=problem_type, model_params=PARAMS_OBJ
    )
    model_alns.solve(
        params_rw=PARAMS_RW,
        params_sa=PARAMS_SA,
        seed=seed,
        max_iter=max_iter,
        partial_plan=partial_plan,
    )
    return model_alns


def _scheduled_partial_plan(scenario: ModemsScenario) -> ModemsSolution:
    """Place every scheduled request via direct insertion to build a partial plan."""
    ctx = ProblemContext(
        scenario,
        ProblemType.closed_selective,
        SolverStrategy.alns,
        PARAMS_OBJ,
    )
    base_sol = ModemsSolution(ctx)
    for r_name in ctx.request_names:
        if not ctx.requests[r_name].is_scheduled():
            continue
        for agent_name in ctx.agent_names:
            candidate = base_sol.journeys[agent_name]._append_request_direct(r_name)
            if candidate is not None:
                base_sol.journeys[agent_name] = candidate
                base_sol.accepted.add(r_name)
                break
        else:
            raise AssertionError("test fixture could not place scheduled request")
    return base_sol


@pytest.fixture
def small_scenario() -> ModemsScenario:
    """Override conftest's small_scenario: ALNS needs more requests to have
    real destroy/repair room to work with (2 agents, 5 requests, seed=1)."""
    generator = ModemsScenarioGenerator(seed=1)
    return generator.generate_random_scenario(nr_agents=2, nr_requests=5)


# --------------------------------------------------------------------------------------
# Basic solve correctness
# --------------------------------------------------------------------------------------


@pytest.mark.integration
def test_alns_solves_and_produces_feasible_solution(
    small_scenario: ModemsScenario,
) -> None:
    """A full ALNS search on the real alns package yields a feasible solution."""
    model_alns = _solve(small_scenario)
    best_sol = model_alns.best_solution
    assert best_sol.accepted | best_sol.rejected == set(best_sol.ctx.request_names)
    for j in best_sol.journeys.values():
        assert j._is_feasible()


@pytest.mark.integration
def test_alns_reports_no_bounds() -> None:
    """ALNS proves no lower/upper bound, so both stay None (never 0.0/objective --
    that would misreport a specific, meaningless duality gap for every result)."""
    scenario = ModemsScenarioGenerator(seed=1).generate_random_scenario(
        nr_agents=1, nr_requests=2
    )
    model_alns = _solve(scenario, max_iter=20)
    info = model_alns.instance.solution_info
    assert info.lower_bound is None
    assert info.upper_bound is None


@pytest.mark.integration
def test_alns_never_regresses_below_constructive(
    small_scenario: ModemsScenario,
) -> None:
    """The best ALNS objective never exceeds the constructive objective."""
    model_alns = _solve(small_scenario, max_iter=300)
    ctx = ProblemContext(
        small_scenario,
        ProblemType.closed_selective,
        SolverStrategy.alns,
        PARAMS_OBJ,
    )
    base_plan, r_unassigned = preprocess(ctx)
    assert base_plan is not None
    sol, _, _ = greedy_complete(base_plan, r_unassigned)
    sol_obj = sol.objective()
    assert model_alns.best_solution.objective() <= sol_obj + 1e-9


# --------------------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------------------


@pytest.mark.integration
def test_alns_is_deterministic_given_fixed_seed(
    small_scenario: ModemsScenario,
) -> None:
    """Two runs with the same seed produce the same objective and routes."""
    model_alns_1 = _solve(small_scenario, seed=7, max_iter=300)
    model_alns_2 = _solve(small_scenario, seed=7, max_iter=300)
    assert (
        model_alns_1.best_solution.objective() == model_alns_2.best_solution.objective()
    )
    assert (
        model_alns_1.best_solution.get_agent_routes()
        == model_alns_2.best_solution.get_agent_routes()
    )


# --------------------------------------------------------------------------------------
# Empty agents/requests -- optimizer must not be invoked
# --------------------------------------------------------------------------------------


def test_alns_empty_requests_never_runs_search(small_scenario: ModemsScenario) -> None:
    """A scenario with zero requests short-circuits without running alns.iterate()."""
    empty_scenario = ModemsScenario(
        agents=small_scenario.agents, requests=[], network=small_scenario.network
    )
    model_alns = _solve(empty_scenario)
    assert model_alns.instance.solution_info.status == SolutionStatus.optimal
    assert model_alns.instance.solution_info.objective == 0.0


def test_alns_empty_agents_rejects_everyone(small_scenario: ModemsScenario) -> None:
    """A scenario with zero agents rejects every request without running the search."""
    empty_scenario = ModemsScenario(
        agents=[], requests=small_scenario.requests, network=small_scenario.network
    )
    model_alns = _solve(empty_scenario)
    assert model_alns.instance.solution_info.status == SolutionStatus.optimal
    assert model_alns.instance.solution_info.objective == PARAMS_OBJ["eta"] * len(
        small_scenario.requests
    )


# --------------------------------------------------------------------------------------
# timed partial-plan excerpt
# --------------------------------------------------------------------------------------


@pytest.mark.integration
def test_alns_solves_with_scheduled_requests_given_partial_plan() -> None:
    """A scheduled request carried via partial_plan remains accepted after solving."""
    generator = ModemsScenarioGenerator(seed=5)
    scenario = generator.generate_random_scenario(
        nr_agents=2, nr_requests=3, nr_scheduled=1
    )
    a = _solve(scenario, partial_plan=_scheduled_partial_plan(scenario))
    assert "request_1" in a.best_solution.accepted


def test_alns_fails_without_partial_plan_when_scheduled_present() -> None:
    """solve() raises RuntimeError if scheduled requests exist with no partial_plan."""
    generator = ModemsScenarioGenerator(seed=5)
    scenario = generator.generate_random_scenario(
        nr_agents=2, nr_requests=3, nr_scheduled=1
    )
    with pytest.raises(RuntimeError):
        _solve(scenario, partial_plan=None)


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_alns_instance_round_trips_exactly(small_scenario: ModemsScenario) -> None:
    """to_dict()/from_dict() on a solved instance preserve objective and routes."""
    a = _solve(small_scenario)
    d = a.instance.to_dict()
    inst2 = ModemsInstance.from_dict(d)
    assert inst2.solution.objective() == pytest.approx(
        a.best_solution.objective(), abs=1e-9
    )
    assert inst2.solution.get_agent_routes() == a.best_solution.get_agent_routes()


# --------------------------------------------------------------------------------------
# Determinism regression: destroy operators sorting by a tie-prone key directly on the
# the raw (hash-ordered) accepted set, without a total-order tie-breaker, silently
# reintroduced hash-randomization-dependent behavior despite using sorted()
# --------------------------------------------------------------------------------------


@pytest.mark.integration
def test_alns_full_solve_is_deterministic_across_process_launches() -> None:
    """
    Runs the exact same ModemsAlns.solve() in fresh subprocesses (not just
    within one Python process) and asserts identical routes every time --
    this is the only way to actually catch PYTHONHASHSEED-dependent bugs,
    since hash randomization only varies BETWEEN process launches.
    """
    script = (
        "from modems import ModemsScenarioGenerator, ModemsAlns\n"
        "generator = ModemsScenarioGenerator(seed=42)\n"
        "scenario = generator.generate_random_scenario(nr_agents=2, nr_requests=8)\n"
        "a = ModemsAlns(scenario, model_params="
        "{'eps':0.01,'zeta':1.0,'eta':100.0,'rho':2.0})\n"
        "a.solve("
        " params_rw={'scores':[5,2,1,0.5],'decay':0.8},"
        " params_sa={'start_temp':1000,'end_temp':0.1,'cooling_rate':0.995},"
        " seed=42, max_iter=400)\n"
        "for k in sorted(a.best_solution.journeys.keys()):\n"
        "    print(k, a.best_solution.journeys[k].route)\n"
    )
    outputs = set()
    for _ in range(4):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        outputs.add(result.stdout)
    assert len(outputs) == 1, f"non-deterministic across process launches: {outputs}"


def test_destroy_sort_keys_are_total_orders_even_with_ties() -> None:
    """
    destroy_lowest_demand/destroy_worst_waiting sort state.accepted (a set) directly
    by a key that plausibly ties (small integer load range; wait is often exactly 0.0).
    Abare sorted(some_set, key=...) without a tie-breaker is only deterministic if the
    key never ties, since sorted()'s stability otherwise falls back to the set
    hash-randomized iteration order
    """
    generator = ModemsScenarioGenerator(seed=5)
    scenario = generator.generate_random_scenario(nr_agents=2, nr_requests=10)
    params = {"eps": 0.01, "zeta": 1.0, "eta": 100.0, "rho": 2.0}
    ctx = ProblemContext(
        scenario, ProblemType.closed_selective, SolverStrategy.alns, params
    )
    base_plan, r_unassigned = preprocess(ctx)
    assert base_plan is not None
    ctx = base_plan.ctx
    sol, _, _ = greedy_complete(base_plan, r_unassigned)

    # deliberately force real ties: two requests with identical load
    loads = [ctx.requests[r].load for r in sol.accepted]
    assert len(set(loads)) < len(
        loads
    ), "fixture needs at least one load tie to be meaningful"

    load_keys = [(ctx.requests[r].load, r) for r in sol.accepted]
    assert len(set(load_keys)) == len(
        load_keys
    ), "load+name key must be a total order (no duplicates)"

    def wait_of(r: str) -> float:
        """Return this request's pickup waiting time in the current solution."""
        k = sol.agent_of(r)
        j = sol.journeys[k]
        p_idx = j.request_pickup[r]
        return j.states[p_idx].t_wait

    wait_keys = [(wait_of(r), r) for r in sol.accepted]
    assert len(set(wait_keys)) == len(
        wait_keys
    ), "wait+name key must be a total order (no duplicates)"
