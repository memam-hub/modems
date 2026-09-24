from __future__ import annotations

import math

import numpy as np
import pytest

from modems.algorithms import greedy_complete, preprocess
from modems.core import (
    DEFAULT_PARAMS_OBJ,
    ModemsAgent,
    ModemsRequest,
    ModemsScenario,
    ProblemContext,
    ProblemType,
    SolverStrategy,
)
from modems.generator import ModemsScenarioGenerator
from modems.milp import DEFAULT_MILP_SOLVER_DATA
from modems.network import NetworkNodeType, RoadNetwork
from modems.solution import (
    ModemsInstance,
    ModemsJourney,
    ModemsSolution,
    ModemsSolutionInfo,
    SolutionStatus,
)

# --------------------------------------------------------------------------------------
# Journey propagation
# --------------------------------------------------------------------------------------


def test_journey_starts_idle_at_agent_position(
    small_scenario: ModemsScenario,
) -> None:
    """A journey built with no states is a dummy idle route to the nearest hub"""
    ctx = ProblemContext(
        small_scenario,
        ProblemType.closed_selective,
        SolverStrategy.alns,
        DEFAULT_PARAMS_OBJ,
    )
    agent_name = ctx.agent_names[0]
    j = ModemsJourney(ctx, agent_name)
    assert not j.is_active
    assert len(j.route) == 2
    assert j.states[0].t_start == ctx.agents[agent_name].time_initial


def test_journey_append_request_direct_produces_valid_pair(
    small_scenario: ModemsScenario,
) -> None:
    """_append_request_direct() inserts (pickup, delivery) in visit order"""
    ctx = ProblemContext(
        small_scenario,
        ProblemType.closed_selective,
        SolverStrategy.alns,
        DEFAULT_PARAMS_OBJ,
    )
    agent_name = ctx.agent_names[0]
    r = ctx.request_names[0]
    j = ModemsJourney(ctx, agent_name)._append_request_direct(r)
    assert j is not None
    assert j.is_active
    assert r in j.request_pickup
    p_idx = j.request_pickup[r]
    d_idx = j.request_delivery[r]
    assert p_idx < d_idx
    # delivery must not occur before pickup
    assert j.states[d_idx].t_start >= j.states[p_idx].t_start


def test_journey_waits_until_earliest_pickup_when_arriving_early(
    small_scenario: ModemsScenario,
) -> None:
    """Arriving before earliest_pickup produces waiting up to that floor"""
    ctx = ProblemContext(
        small_scenario,
        ProblemType.closed_selective,
        SolverStrategy.alns,
        DEFAULT_PARAMS_OBJ,
    )
    agent_name = ctx.agent_names[0]
    r = ctx.request_names[0]
    j = ModemsJourney(ctx, agent_name)._append_request_direct(r)
    assert j is not None
    p_idx = j.request_pickup[r]
    st = j.states[p_idx]
    e = ctx.node_earliest_p[st.node]
    if st.t_arr < e:
        assert math.isclose(st.t_start, e, rel_tol=1e-6) or st.t_wait > 0
    assert st.t_arr + st.t_wait == pytest.approx(st.t_start)


def test_journey_remove_request_clears_positions(
    small_scenario: ModemsScenario,
) -> None:
    """remove_request() drops the request from both position-index maps"""
    ctx = ProblemContext(
        small_scenario,
        ProblemType.closed_selective,
        SolverStrategy.alns,
        DEFAULT_PARAMS_OBJ,
    )
    agent_name = ctx.agent_names[0]
    r = ctx.request_names[0]
    j = ModemsJourney(ctx, agent_name)._append_request_direct(r)
    assert j is not None
    j.remove_request(r)
    assert not j.is_active
    assert r not in j.request_pickup
    assert r not in j.request_delivery


@pytest.mark.parametrize(
    ("rho", "expected_pickup_start", "expected_tau"),
    [(4.0, 3.0, 2.0), (5.0, 2.0, 1.0)],
)
def test_remove_request_uses_ride_slack_to_preserve_schedule(
    rho: float,
    expected_pickup_start: float,
    expected_tau: float,
) -> None:
    """A surviving pickup advances by at most its own current ride-time slack"""
    travel_times = np.ones((7, 7), dtype=float)
    np.fill_diagonal(travel_times, 0.0)
    network = RoadNetwork(
        nr_hubs=1,
        nr_stations=6,
        locations=np.zeros((7, 2)),
        travel_times=travel_times,
    )
    scenario = ModemsScenario(
        agents=[ModemsAgent(NetworkNodeType.hub, 1)],
        requests=[
            ModemsRequest(1, 2, service_time=0.0, earliest_pickup=0.0),
            ModemsRequest(
                3,
                4,
                service_time=0.0,
                earliest_pickup=0.0,
                tw_length=1.0,
            ),
            ModemsRequest(5, 6, service_time=0.0, earliest_pickup=5.0),
        ],
        network=network,
    )
    ctx = ProblemContext(
        scenario,
        ProblemType.closed_selective,
        SolverStrategy.alns,
        {**DEFAULT_PARAMS_OBJ, "rho": rho},
    )
    removed, survivor, waiting_request = ctx.request_names
    route = [
        ctx.agent_initial_node["agent_1"],
        ctx.pickup_node[removed],
        ctx.delivery_node[removed],
        ctx.pickup_node[survivor],
        ctx.pickup_node[waiting_request],
        ctx.delivery_node[waiting_request],
        ctx.delivery_node[survivor],
        ctx.final_depot_names[0],
    ]
    journey = ModemsJourney.from_route(ctx, "agent_1", route)
    pickup_before = journey.states[journey.request_pickup[survivor]]
    delivery_before = journey.states[journey.request_delivery[survivor]]
    ride_before = delivery_before.t_start - pickup_before.t_start
    ride_slack_before = (
        rho * ctx.t_travel(pickup_before.node, delivery_before.node) - ride_before
    )

    journey.remove_request(removed)

    pickup_after = journey.states[journey.request_pickup[survivor]]
    delivery_after = journey.states[journey.request_delivery[survivor]]
    assert pickup_before.t_start == pytest.approx(3.0)
    assert ride_slack_before == pytest.approx(rho - 4.0)
    assert pickup_after.t_start == pytest.approx(expected_pickup_start)
    assert pickup_after.tau == pytest.approx(expected_tau)
    assert delivery_after.t_start == pytest.approx(7.0)
    assert journey._is_feasible()


def test_journey_constructor_validates_supplied_states(
    small_scenario: ModemsScenario,
) -> None:
    """Passing explicit states validates feasibility and rejects inconsistency"""
    ctx = ProblemContext(
        small_scenario,
        ProblemType.closed_selective,
        SolverStrategy.alns,
        DEFAULT_PARAMS_OBJ,
    )
    agent_name = ctx.agent_names[0]
    request_name = ctx.request_names[0]
    journey = ModemsJourney(ctx, agent_name)._append_request_direct(request_name)
    assert journey is not None

    rebuilt = ModemsJourney(ctx, agent_name, journey.states)
    assert rebuilt.route == journey.route
    assert rebuilt.request_pickup == journey.request_pickup
    assert rebuilt.states is not journey.states

    invalid_states = [state.copy() for state in journey.states]
    invalid_states[1].t_arr += 1.0
    with pytest.raises(ValueError, match="infeasible or inconsistent states"):
        ModemsJourney(ctx, agent_name, invalid_states)


def test_journey_from_route_matches_manual_construction(
    small_scenario: ModemsScenario,
) -> None:
    """from_route() re-propagates a route to the same timing/load as the original"""
    ctx = ProblemContext(
        small_scenario,
        ProblemType.closed_selective,
        SolverStrategy.alns,
        DEFAULT_PARAMS_OBJ,
    )
    agent_name = ctx.agent_names[0]
    r = ctx.request_names[0]
    j = ModemsJourney(ctx, agent_name)._append_request_direct(r)
    assert j is not None
    j2 = ModemsJourney.from_route(ctx, agent_name, j.route)
    assert j2.route == j.route
    for s1, s2 in zip(j.states, j2.states):
        assert s1.t_start == pytest.approx(s2.t_start)
        assert s1.z_dep == s2.z_dep


def test_journey_from_route_with_tau_reduces_wait(
    small_scenario: ModemsScenario,
) -> None:
    """Supplying tau_of consumes wait first, reducing t_wait by the given slack"""
    ctx = ProblemContext(
        small_scenario,
        ProblemType.closed_selective,
        SolverStrategy.milp3,
        DEFAULT_PARAMS_OBJ,
    )
    agent_name = ctx.agent_names[0]
    r = ctx.request_names[0]
    j = ModemsJourney(ctx, agent_name)._append_request_direct(r)
    assert j is not None
    p_idx = j.request_pickup[r]
    p_node = j.route[p_idx]
    no_tau_wait = j.states[p_idx].t_wait
    if no_tau_wait > 1.0:
        j_tau = ModemsJourney.from_route(ctx, agent_name, j.route, tau_of={p_node: 1.0})
        assert j_tau.states[p_idx].t_wait == pytest.approx(no_tau_wait - 1.0)
        assert j_tau.states[p_idx].tau == 1.0


def test_alns_tau_never_permits_early_service(
    small_scenario: ModemsScenario,
) -> None:
    """Under ALNS, pickup service never starts before earliest_pickup"""
    ctx = ProblemContext(
        small_scenario,
        ProblemType.closed_selective,
        SolverStrategy.alns,
        DEFAULT_PARAMS_OBJ,
    )
    agent_name = ctx.agent_names[0]
    request_name = ctx.request_names[0]
    journey = ModemsJourney(ctx, agent_name)._append_request_direct(request_name)
    assert journey is not None
    pickup_index = journey.request_pickup[request_name]
    pickup_node = journey.route[pickup_index]

    rebuilt = ModemsJourney.from_route(
        ctx,
        agent_name,
        journey.route,
        tau_of={pickup_node: 10.0},
    )

    assert rebuilt.states[pickup_index].t_start >= ctx.node_earliest_p[pickup_node]
    assert rebuilt.states[pickup_index].tau == pytest.approx(
        max(0.0, rebuilt.states[pickup_index].t_start - ctx.node_latest_p[pickup_node])
    )


def test_infeasible_insert_at_is_atomic() -> None:
    """A failed insert_at() leaves the journey byte-for-byte unchanged"""
    scenario = ModemsScenarioGenerator(seed=1).generate_random_scenario(
        nr_agents=1,
        nr_requests=1,
    )
    scenario.agents[0].load_max = 1
    scenario.requests[0].load = 2
    ctx = ProblemContext(
        scenario,
        ProblemType.closed_selective,
        SolverStrategy.alns,
        DEFAULT_PARAMS_OBJ,
    )
    journey = ModemsJourney(ctx, ctx.agent_names[0])
    route_before = list(journey.route)
    states_before = [state.to_dict() for state in journey.states]

    with pytest.raises(ValueError, match="selected insertion"):
        journey.insert_at(ctx.request_names[0], 0, 0)

    assert journey.route == route_before
    assert [state.to_dict() for state in journey.states] == states_before
    assert journey.request_pickup == {}
    assert journey.request_delivery == {}


# --------------------------------------------------------------------------------------
# Solution
# --------------------------------------------------------------------------------------


def test_solution_objective_zero_when_all_idle_and_non_selective(
    small_scenario: ModemsScenario,
) -> None:
    """An idle, non-selective (milp1) fleet has objective 0 (T=0, no penalties)"""
    ctx = ProblemContext(
        small_scenario,
        ProblemType.closed_non_selective,
        SolverStrategy.milp1,
        DEFAULT_PARAMS_OBJ,
    )
    sol = ModemsSolution(ctx)
    # milp1 has no selectivity/rejection penalty at all, so pending requests
    # contribute nothing and an idle fleet has t_mission=0
    assert sol.objective() == 0.0


def test_solution_objective_counts_pending_as_rejected_under_selective_kappa(
    small_scenario: ModemsScenario,
) -> None:
    """Every pending request contributes eta when the problem is selective"""
    ctx = ProblemContext(
        small_scenario,
        ProblemType.closed_selective,
        SolverStrategy.alns,
        DEFAULT_PARAMS_OBJ,
    )
    sol = ModemsSolution(ctx)
    # nothing accepted yet -- every request is implicitly "would be rejected"
    assert sol.objective() == ctx.eta * len(ctx.request_names)


def test_solution_objective_matches_manual_calc_for_one_request(
    small_scenario: ModemsScenario,
) -> None:
    """objective() equals mission_time + eps/zeta terms + eta * rejected count"""
    ctx = ProblemContext(
        small_scenario,
        ProblemType.closed_selective,
        SolverStrategy.alns,
        DEFAULT_PARAMS_OBJ,
    )
    agent_name = ctx.agent_names[0]
    r = ctx.request_names[0]
    sol = ModemsSolution(ctx)
    candidate = sol.journeys[agent_name]._append_request_direct(r)
    assert candidate is not None
    sol.journeys[agent_name] = candidate
    sol.accepted.add(r)
    sol.recompute_rejected()  # the other requests are implicitly rejected

    j = sol.journeys[agent_name]
    p_idx = j.request_pickup[r]
    d_idx = j.request_delivery[r]
    t_p, t_d = j.states[p_idx].t_start, j.states[d_idx].t_start
    tau_p = max(
        0.0,
        ctx.node_earliest_p[j.states[p_idx].node] - t_p,
        t_p - ctx.node_latest_p[j.states[p_idx].node],
    )
    T = j.states[-1].t_arr
    n_rejected = len(ctx.request_names) - 1
    expected = T + ctx.eps * (t_p + t_d) + ctx.zeta * tau_p + ctx.eta * n_rejected
    assert sol.objective() == pytest.approx(expected)


def test_solution_pending_and_recompute_rejected(
    small_scenario: ModemsScenario,
) -> None:
    """pending() is all-minus-accepted; recompute_rejected() sets rejected to it"""
    ctx = ProblemContext(
        small_scenario,
        ProblemType.closed_selective,
        SolverStrategy.alns,
        DEFAULT_PARAMS_OBJ,
    )
    sol = ModemsSolution(ctx)
    assert sol.pending() == set(ctx.request_names)
    r = ctx.request_names[0]
    sol.accepted.add(r)
    assert r not in sol.pending()
    sol.recompute_rejected()
    assert sol.rejected == sol.pending()


def test_solution_to_dict_from_dict_round_trip(
    small_scenario: ModemsScenario,
) -> None:
    """to_dict()/from_dict() JSON round-trips persist"""
    ctx = ProblemContext(
        small_scenario,
        ProblemType.closed_selective,
        SolverStrategy.alns,
        DEFAULT_PARAMS_OBJ,
    )
    agent_name = ctx.agent_names[0]
    r = ctx.request_names[0]
    sol = ModemsSolution(ctx)
    candidate = sol.journeys[agent_name]._append_request_direct(r)
    assert candidate is not None
    sol.journeys[agent_name] = candidate
    sol.accepted.add(r)
    sol.recompute_rejected()

    d = sol.to_dict()
    sol2 = ModemsSolution.from_dict(ctx, d)
    assert sol2.accepted == sol.accepted
    assert sol2.rejected == sol.rejected
    assert sol2.objective() == pytest.approx(sol.objective())


def test_solution_from_dict_rejects_requests_routed_on_two_agents(
    small_scenario: ModemsScenario,
) -> None:
    """from_dict() raises when the same request appears in two journeys"""
    ctx = ProblemContext(
        small_scenario,
        ProblemType.closed_selective,
        SolverStrategy.alns,
        DEFAULT_PARAMS_OBJ,
    )
    agent_1, agent_2 = ctx.agent_names[0], ctx.agent_names[1]
    r = ctx.request_names[0]
    sol = ModemsSolution(ctx)
    candidate_1 = sol.journeys[agent_1]._append_request_direct(r)
    candidate_2 = sol.journeys[agent_2]._append_request_direct(r)
    assert candidate_1 is not None and candidate_2 is not None
    sol.journeys[agent_1] = candidate_1
    sol.journeys[agent_2] = candidate_2  # same request served twice, on two agents
    sol.accepted.add(r)
    with pytest.raises(ValueError, match="multiple journeys"):
        ModemsSolution.from_dict(ctx, sol.to_dict())


# --------------------------------------------------------------------------------------
# ModemsSolutionInfo / ModemsInstance
# --------------------------------------------------------------------------------------


def test_solution_info_to_dict_from_dict_round_trip() -> None:
    """to_dict()/from_dict() preserve every ModemsSolutionInfo field"""
    info = ModemsSolutionInfo(
        status=SolutionStatus.optimal,
        solution_time=1.5,
        objective=42.0,
        lower_bound=40.0,
        upper_bound=42.0,
        solver_name=DEFAULT_MILP_SOLVER_DATA[0],
        solver_options={"timelimit": 60},
        solver_diagnostics={"nr_variables": 123},
    )
    d = info.to_dict()
    assert SolutionStatus.optimal == "optimal"
    assert str(SolutionStatus.optimal) == "optimal"
    assert d["status"] == "optimal"
    info2 = ModemsSolutionInfo.from_dict(d)
    assert info2.to_dict() == d


def test_solution_info_bounds_default_to_none() -> None:
    """lower_bound/upper_bound default to None; meaningful only for MILPs"""
    info = ModemsSolutionInfo(status=SolutionStatus.optimal, objective=10.0)
    assert info.lower_bound is None
    assert info.upper_bound is None
    d = info.to_dict()
    assert d["lower_bound"] is None
    assert d["upper_bound"] is None
    info2 = ModemsSolutionInfo.from_dict(d)
    assert info2.lower_bound is None
    assert info2.upper_bound is None


def test_modems_instance_requires_matching_context(
    small_scenario: ModemsScenario,
) -> None:
    """__init__ raises ValueError if solution.ctx is not the same object as ctx"""
    ctx1 = ProblemContext(
        small_scenario,
        ProblemType.closed_selective,
        SolverStrategy.alns,
        DEFAULT_PARAMS_OBJ,
    )
    ctx2 = ProblemContext(
        small_scenario,
        ProblemType.closed_selective,
        SolverStrategy.alns,
        DEFAULT_PARAMS_OBJ,
    )
    sol = ModemsSolution(ctx1)
    info = ModemsSolutionInfo(status=SolutionStatus.optimal, objective=0.0)
    with pytest.raises(ValueError, match="same ProblemContext"):
        ModemsInstance(ctx2, sol, info)


def test_modems_instance_to_dict_from_dict_round_trip(
    small_scenario: ModemsScenario,
) -> None:
    """to_dict() top-level shape matches spec and from_dict() round-trips"""
    ctx = ProblemContext(
        small_scenario,
        ProblemType.closed_selective,
        SolverStrategy.alns,
        DEFAULT_PARAMS_OBJ,
    )
    base, r_unassigned = preprocess(ctx)
    assert base is not None
    ctx = base.ctx
    sol, _, _ = greedy_complete(base, r_unassigned)
    info = ModemsSolutionInfo(
        status=SolutionStatus.feasible,
        solution_time=0.1,
        objective=sol.objective(),
        solver_diagnostics={"iterations": 25},
    )
    inst = ModemsInstance(
        ctx,
        sol,
        info,
    )

    d = inst.to_dict()
    assert set(d) == {
        "problem_context",
        "solution",
        "solution_info",
        "diagnostics",
    }
    assert set(d["problem_context"]) == {
        "scenario",
        "problem_type",
        "strategy",
        "model_params",
    }
    assert set(d["diagnostics"]) == {
        "duality_gap",
        "solver",
        "agents",
        "requests",
    }
    assert d["diagnostics"]["duality_gap"] is None
    assert d["diagnostics"]["solver"] == {"iterations": 25}
    inst2 = ModemsInstance.from_dict(d)
    assert inst2.solution.objective() == pytest.approx(inst.solution.objective())
    assert inst2.solution_info.objective == info.objective
