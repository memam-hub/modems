from __future__ import annotations

import math
from typing import Any

import pytest

from modems import (
    ModemsAgent,
    ModemsScenarioGenerator,
    ProblemType,
    RollingHorizonSimulator,
    WorkdaySimulation,
    compare_solvers_one_workday,
    compare_solvers_over_workdays,
)
from modems.algorithms import greedy_complete, preprocess
from modems.core import ModemsRequest, ProblemContext, RequestStatus, SolverStrategy
from modems.network import NetworkNodeType
from modems.rolling_horizon import (
    PLANNING_HORIZON_P,
    REPLANNING_THRESHOLD_W,
    SOC_CHARGE_TARGET,
    AgentAdvanceResult,
    AgentOperationalState,
    EpochLog,
    _collect_request_metrics,
    advance_agent_state,
    charging_duration_min,
    summarize_workday,
)
from modems.solution import (
    DEFAULT_PARAMS_MILP,
    ModemsInstance,
    ModemsJourney,
    ModemsSolution,
    ModemsSolutionInfo,
    NodeState,
    SolutionStatus,
)

# --------------------------------------------------------------------------------------
# charging_duration_min
# --------------------------------------------------------------------------------------


def test_agent_operational_state_is_string_enum() -> None:
    """AgentOperationalState members equal and stringify to literal values"""
    assert AgentOperationalState.available == "available"
    assert str(AgentOperationalState.charging) == "charging"


def test_charging_full_cycle_is_five_hours() -> None:
    """A full 0%->100% charge takes exactly the calibrated 5 hours"""
    assert math.isclose(charging_duration_min(0.0, 1.0), 300.0, abs_tol=1e-6)


def test_charging_cc_segment() -> None:
    """The constant-current segment (below 80%) uses the higher charge rate"""
    assert math.isclose(charging_duration_min(0.4, 0.8), 100.0, abs_tol=1e-6)


def test_charging_cv_segment() -> None:
    """The constant-voltage segment (above 80%) uses the lower charge rate"""
    assert math.isclose(charging_duration_min(0.8, 1.0), 100.0, abs_tol=1e-6)


def test_charging_mixed_segment() -> None:
    """A charge spanning both segments sums both charge rate durations"""
    assert math.isclose(charging_duration_min(0.0, 0.8), 200.0, abs_tol=1e-6)


def test_charging_no_op_when_already_there() -> None:
    """charging_duration_min() is 0 whenever soc_to is at or below soc_from"""
    assert charging_duration_min(0.8, 0.8) == 0.0
    assert charging_duration_min(0.9, 0.5) == 0.0


# --------------------------------------------------------------------------------------
# Test fixture: a small real scenario/context, with hand-built Journey states
# so each branch of the state machine can be fully checked
# --------------------------------------------------------------------------------------


@pytest.fixture
def ctx_and_agent() -> ProblemContext:
    """Build a real ProblemContext (1 agent, 2 requests) for hand-built journeys"""
    generator = ModemsScenarioGenerator(seed=1)
    scenario = generator.generate_random_scenario(nr_agents=1, nr_requests=2)
    ctx = ProblemContext(
        scenario,
        ProblemType.closed_selective,
        SolverStrategy.alns,
        DEFAULT_PARAMS_MILP,
    )
    return ctx


def _mk_state(
    node: str,
    t_arr: float = 0.0,
    wait: float = 0.0,
    t: float = 0.0,
    t_dep: float = 0.0,
    z_arr: int = 0,
    z_dep: int = 0,
    phi_arr: float = 1.0,
    phi_dep: float = 1.0,
) -> NodeState:
    """Build a NodeState with short, test-friendly keyword names"""
    return NodeState(
        node=node,
        t_arr=0.0 if t_arr is None else t_arr,
        t_wait=wait,
        t_start=t,
        t_dep=t_dep,
        z_arr=z_arr,
        z_dep=z_dep,
        phi_arr=phi_arr,
        phi_dep=phi_dep,
    )


def _mk_journey(
    ctx: ProblemContext, agent_name: str, route: list[str], states: list[NodeState]
) -> ModemsJourney:
    """Build a ModemsJourney directly from a hand-crafted route/states pair"""
    j = ModemsJourney.__new__(ModemsJourney)
    j.ctx = ctx
    j.agent_name = agent_name
    j.agent = ctx.agents[agent_name]
    j.route = route
    j.states = states
    j._rebuild_request_indices()
    return j


# --------------------------------------------------------------------------------------
# Branch: idle agent (never left home)
# --------------------------------------------------------------------------------------


def test_idle_agent_high_soc_is_available(ctx_and_agent: ProblemContext) -> None:
    """An idle agent above the robustness threshold reports available, delta=0"""
    ctx = ctx_and_agent
    agent_name = ctx.agent_names[0]
    v_k = ctx.agent_initial_node[agent_name]
    hf = ctx.nearest_final_depot(v_k)
    route = [v_k, hf]
    states = [_mk_state(v_k, t_arr=0.0, t=0.0, t_dep=0.0, phi_dep=0.9), _mk_state(hf)]
    journey = _mk_journey(ctx, agent_name, route, states)

    result = advance_agent_state(ctx, agent_name, journey, t_elapsed=0.0)
    assert result.operation_state == AgentOperationalState.available
    assert result.included is True
    assert result.absorbed == set()
    assert result.replan == set()


def test_soc_exactly_at_robustness_threshold_is_available(
    ctx_and_agent: ProblemContext,
) -> None:
    """SoC exactly at soc_min + margin is still classified as available (>=, not >)"""
    ctx = ctx_and_agent
    agent_name = ctx.agent_names[0]
    v_k = ctx.agent_initial_node[agent_name]
    hf = ctx.nearest_final_depot(v_k)
    threshold = ctx.agents[agent_name].soc_min_operational + 0.20
    journey = _mk_journey(
        ctx,
        agent_name,
        [v_k, hf],
        [_mk_state(v_k, phi_dep=threshold), _mk_state(hf)],
    )

    result = advance_agent_state(ctx, agent_name, journey, t_elapsed=0.0)

    assert result.operation_state == AgentOperationalState.available


def test_idle_agent_low_soc_charges(ctx_and_agent: ProblemContext) -> None:
    """An idle agent below the robustness threshold travels to its hub and charges"""
    ctx = ctx_and_agent
    agent_name = ctx.agent_names[0]
    v_k = ctx.agent_initial_node[agent_name]
    hf = ctx.nearest_final_depot(v_k)
    travel_time = ctx.t_travel(v_k, hf)
    agent = ctx.agents[agent_name]
    sigma_at_hub = 0.3 - travel_time * agent.soc_alpha
    route = [v_k, hf]
    states = [
        _mk_state(v_k, t_arr=0.0, t=0.0, t_dep=0.0, phi_dep=0.3),
        _mk_state(hf, t_arr=travel_time, phi_arr=sigma_at_hub),
    ]
    journey = _mk_journey(ctx, agent_name, route, states)

    result = advance_agent_state(ctx, agent_name, journey, t_elapsed=0.0)
    assert result.operation_state == AgentOperationalState.charging
    assert result.sigma == SOC_CHARGE_TARGET
    assert result.delta == pytest.approx(
        travel_time + charging_duration_min(sigma_at_hub, 0.8)
    )
    assert result.node == hf
    assert result.travel_time == pytest.approx(travel_time)


# --------------------------------------------------------------------------------------
# Branch: en route to final hub (route complete)
# --------------------------------------------------------------------------------------


def test_en_route_to_hub_high_soc_available(ctx_and_agent: ProblemContext) -> None:
    """Reaching the final hub with high SoC reports available and absorbs the trip"""
    ctx = ctx_and_agent
    agent_name = ctx.agent_names[0]
    r1 = ctx.request_names[0]
    p1, d1 = ctx.pickup_node[r1], ctx.delivery_node[r1]
    v_k = ctx.agent_initial_node[agent_name]
    hf = ctx.nearest_final_depot(v_k)
    route = [v_k, p1, d1, hf]
    states = [
        _mk_state(v_k, t_arr=0.0, t=0.0, t_dep=0.0),
        _mk_state(p1, t_arr=5.0, t=5.0, t_dep=6.0, z_dep=2),
        _mk_state(d1, t_arr=10.0, t=10.0, t_dep=11.0, z_dep=0),
        _mk_state(hf, t_arr=15.0, phi_arr=0.9),
    ]
    journey = _mk_journey(ctx, agent_name, route, states)

    result = advance_agent_state(ctx, agent_name, journey, t_elapsed=12.0)
    assert result.operation_state == AgentOperationalState.available
    assert result.delta == pytest.approx(3.0)  # 15 - 12
    assert result.absorbed == {r1}
    assert result.replan == set()


def test_en_route_to_hub_low_soc_charges_and_excluded_if_too_long(
    ctx_and_agent: ProblemContext,
) -> None:
    """Reaching the hub with very low SoC charges long enough to be excluded"""
    ctx = ctx_and_agent
    agent_name = ctx.agent_names[0]
    r1 = ctx.request_names[0]
    p1, d1 = ctx.pickup_node[r1], ctx.delivery_node[r1]
    v_k = ctx.agent_initial_node[agent_name]
    hf = ctx.nearest_final_depot(v_k)
    # route length > 2 (a completed request) so this exercises _resolve's hub
    # branch rather than the "never left home" idle shortcut
    route = [v_k, p1, d1, hf]
    states = [
        _mk_state(v_k, t_arr=0.0, t=0.0, t_dep=0.0),
        _mk_state(p1, t_arr=2.0, t=2.0, t_dep=3.0, z_dep=1),
        _mk_state(d1, t_arr=8.0, t=8.0, t_dep=9.0, z_dep=0),
        _mk_state(hf, t_arr=15.0, phi_arr=0.15),  # well below threshold
    ]
    journey = _mk_journey(ctx, agent_name, route, states)

    result = advance_agent_state(ctx, agent_name, journey, t_elapsed=10.0)
    assert result.operation_state == AgentOperationalState.charging
    # 0.15 -> 0.8 needs a lot of charge time, plus 5 min travel remaining
    assert result.delta > PLANNING_HORIZON_P
    assert result.included is False
    assert result.absorbed == {r1}


# --------------------------------------------------------------------------------------
# Branch: en route with passengers onboard (z_hat > 0) -- recursion
# --------------------------------------------------------------------------------------


def test_en_route_with_passengers_absorbs_through_first_empty_delivery(
    ctx_and_agent: ProblemContext,
) -> None:
    """With no further pickup before the hub, recursion absorbs through to the hub"""
    ctx = ctx_and_agent
    agent_name = ctx.agent_names[0]
    r1, r2 = ctx.request_names[0], ctx.request_names[1]
    p1, d1 = ctx.pickup_node[r1], ctx.delivery_node[r1]
    p2, d2 = ctx.pickup_node[r2], ctx.delivery_node[r2]
    v_k = ctx.agent_initial_node[agent_name]
    hf = ctx.nearest_final_depot(v_k)
    # already picked up r1; next planned node (idx) is p2, with z=1 onboard already
    route = [v_k, p1, p2, d1, d2, hf]
    states = [
        _mk_state(v_k, t_arr=0.0, t=0.0, t_dep=0.0),
        _mk_state(
            p1, t_arr=2.0, t=2.0, t_dep=3.0, z_dep=1
        ),  # already done (t_elapsed will be after this)
        _mk_state(p2, t_arr=6.0, t=6.0, t_dep=7.0, z_dep=2, wait=0.0),
        _mk_state(d1, t_arr=10.0, t=10.0, t_dep=11.0, z_dep=1),
        _mk_state(d2, t_arr=14.0, t=14.0, t_dep=15.0, z_dep=0),
        _mk_state(hf, t_arr=20.0, phi_arr=0.9),
    ]
    journey = _mk_journey(ctx, agent_name, route, states)

    # t_elapsed=4: we've departed p1 (t_dep=3) but not yet reached p2 (t_dep=7)
    # -> _find_next_index lands on p2 (idx=2); z_hat = states[1].z_dep = 1 > 0
    # Nothing (no further pickup) follows d2 except the final hub, so the
    # recursion correctly continues through to resolving the hub outcome too
    # (matching the flowchart's loop-back into the n_{i+1} in H_f check)
    result = advance_agent_state(ctx, agent_name, journey, t_elapsed=4.0)
    assert result.absorbed == {r1, r2}
    assert result.node == hf
    assert result.operation_state == AgentOperationalState.available
    assert result.delta == pytest.approx(20.0 - 4.0)


def test_en_route_with_passengers_stops_at_next_pickup_not_hub(
    ctx_and_agent: ProblemContext,
) -> None:
    """A new pickup before the hub stops the recursion there, not at the hub"""
    ctx = ctx_and_agent
    agent_name = ctx.agent_names[0]
    r1, r2 = ctx.request_names[0], ctx.request_names[1]
    p1, d1 = ctx.pickup_node[r1], ctx.delivery_node[r1]
    p2, d2 = ctx.pickup_node[r2], ctx.delivery_node[r2]
    v_k = ctx.agent_initial_node[agent_name]
    hf = ctx.nearest_final_depot(v_k)
    # onboard r1 already, heading to d1 (zero-occupancy point), then a NEW
    # pickup p2 follows before the hub; recursion should stop at p2, not hf.
    route = [v_k, p1, d1, p2, d2, hf]
    states = [
        _mk_state(v_k, t_arr=0.0, t=0.0, t_dep=0.0),
        _mk_state(p1, t_arr=2.0, t=2.0, t_dep=3.0, z_dep=1),
        _mk_state(d1, t_arr=6.0, t=6.0, t_dep=7.0, z_dep=0),
        _mk_state(
            p2,
            t_arr=12.0,
            t=12.0,
            t_dep=13.0,
            z_dep=2,
            wait=REPLANNING_THRESHOLD_W + 5.0,
        ),
        _mk_state(d2, t_arr=18.0, t=18.0, t_dep=19.0, z_dep=0),
        _mk_state(hf, t_arr=24.0, phi_arr=0.9),
    ]
    journey = _mk_journey(ctx, agent_name, route, states)

    # t_elapsed=4: departed p1 (t_dep=3), not yet at d1 (t_dep=7) -> idx=2 (d1);
    # z_hat = states[1].z_dep = 1 > 0 -> absorb through d1, recurse at p2 (idx=3)
    result = advance_agent_state(ctx, agent_name, journey, t_elapsed=4.0)
    assert r1 in result.absorbed
    assert result.node == p2
    assert result.replan == {r2}


# --------------------------------------------------------------------------------------
# Branch: en route empty to a pickup -- release vs retain
# --------------------------------------------------------------------------------------


def test_en_route_empty_release_when_slack_above_threshold(
    ctx_and_agent: ProblemContext,
) -> None:
    """Waiting slack above the threshold releases the pickup for replanning"""
    ctx = ctx_and_agent
    agent_name = ctx.agent_names[0]
    r1 = ctx.request_names[0]
    p1, d1 = ctx.pickup_node[r1], ctx.delivery_node[r1]
    v_k = ctx.agent_initial_node[agent_name]
    hf = ctx.nearest_final_depot(v_k)
    route = [v_k, p1, d1, hf]
    states = [
        _mk_state(v_k, t_arr=0.0, t=0.0, t_dep=0.0),
        _mk_state(
            p1, t_arr=5.0, t=5.0, t_dep=6.0, z_dep=2, wait=REPLANNING_THRESHOLD_W + 5.0
        ),
        _mk_state(d1, t_arr=10.0, t=10.0, t_dep=11.0, z_dep=0),
        _mk_state(hf, t_arr=15.0),
    ]
    journey = _mk_journey(ctx, agent_name, route, states)

    result = advance_agent_state(ctx, agent_name, journey, t_elapsed=0.0)
    assert result.node == p1
    assert result.absorbed == set()
    assert result.replan == {r1}


def test_en_route_empty_retain_when_slack_at_or_below_threshold(
    ctx_and_agent: ProblemContext,
) -> None:
    """Waiting slack at or below the threshold keeps resolving past the pickup"""
    ctx = ctx_and_agent
    agent_name = ctx.agent_names[0]
    r1 = ctx.request_names[0]
    p1, d1 = ctx.pickup_node[r1], ctx.delivery_node[r1]
    v_k = ctx.agent_initial_node[agent_name]
    hf = ctx.nearest_final_depot(v_k)
    route = [v_k, p1, d1, hf]
    states = [
        _mk_state(v_k, t_arr=0.0, t=0.0, t_dep=0.0),
        _mk_state(
            p1, t_arr=5.0, t=5.0, t_dep=6.0, z_dep=2, wait=REPLANNING_THRESHOLD_W - 5.0
        ),
        _mk_state(d1, t_arr=10.0, t=10.0, t_dep=11.0, z_dep=0),
        _mk_state(hf, t_arr=15.0, phi_arr=0.9),
    ]
    journey = _mk_journey(ctx, agent_name, route, states)

    result = advance_agent_state(ctx, agent_name, journey, t_elapsed=0.0)
    # retained through p1, then recurse: z becomes 1 after p1 (not in states here,
    # but the recursive call re-checks states[idx-1].z_dep at the NEXT index, i.e.
    # states[1].z_dep=2 when resolving idx=2 (d1)) -> treated as onboard, absorbs
    # through d1 (first zero-occupancy delivery), then resolves the final hub
    assert r1 in result.absorbed
    assert result.replan == set()


def test_waiting_decision_uses_remaining_not_original_wait(
    ctx_and_agent: ProblemContext,
) -> None:
    """The threshold check uses the wait remaining after t_elapsed, not the original"""
    ctx = ctx_and_agent
    agent_name = ctx.agent_names[0]
    r1 = ctx.request_names[0]
    p1, d1 = ctx.pickup_node[r1], ctx.delivery_node[r1]
    v_k = ctx.agent_initial_node[agent_name]
    hf = ctx.nearest_final_depot(v_k)
    route = [v_k, p1, d1, hf]
    states = [
        _mk_state(v_k, t=0.0, t_dep=0.0),
        _mk_state(p1, t_arr=0.0, wait=20.0, t=20.0, t_dep=21.0, z_dep=1),
        _mk_state(d1, t_arr=25.0, t=25.0, t_dep=26.0, z_dep=0),
        _mk_state(hf, t_arr=30.0, phi_arr=0.9),
    ]
    journey = _mk_journey(ctx, agent_name, route, states)

    result = advance_agent_state(ctx, agent_name, journey, t_elapsed=15.0)

    assert r1 in result.absorbed
    assert result.replan == set()


# --------------------------------------------------------------------------------------
# preprocess() with a timed partial-plan excerpt
# --------------------------------------------------------------------------------------


def _scheduled_partial_plan(
    scenario: Any, strategy: SolverStrategy = SolverStrategy.alns
) -> ModemsSolution:
    """Place every scheduled request via direct insertion to build a partial plan"""
    ctx = ProblemContext(
        scenario,
        ProblemType.closed_selective,
        strategy,
        DEFAULT_PARAMS_MILP,
    )
    plan = ModemsSolution(ctx)
    for request_name in ctx.request_names:
        if not ctx.requests[request_name].is_scheduled():
            continue
        for agent_name in ctx.agent_names:
            candidate = plan.journeys[agent_name]._append_request_direct(request_name)
            if candidate is not None:
                plan.journeys[agent_name] = candidate
                plan.accepted.add(request_name)
                break
        else:
            raise AssertionError("test fixture could not place scheduled request")
    return plan


def test_preprocess_requires_partial_plan_when_scheduled_present() -> None:
    """preprocess() returns solution=None if a scheduled request has no partial_plan"""
    generator = ModemsScenarioGenerator(seed=5)
    scenario = generator.generate_random_scenario(
        nr_agents=2, nr_requests=3, nr_scheduled=1
    )

    ctx = ProblemContext(
        scenario, ProblemType.open_selective, SolverStrategy.alns, DEFAULT_PARAMS_MILP
    )
    solution, unassigned = preprocess(ctx)
    assert solution is None
    assert unassigned == {"request_2", "request_3"}


def test_preprocess_inserts_scheduled_request_via_partial_plan() -> None:
    """A scheduled request carried via partial_plan is accepted by preprocess()"""
    generator = ModemsScenarioGenerator(seed=5)
    scenario = generator.generate_random_scenario(
        nr_agents=2, nr_requests=3, nr_scheduled=1
    )

    partial_plan = _scheduled_partial_plan(scenario)
    ctx = ProblemContext(
        scenario,
        ProblemType.closed_selective,
        SolverStrategy.alns,
        DEFAULT_PARAMS_MILP,
    )
    sol, unassigned = preprocess(ctx, partial_plan=partial_plan)
    assert sol is not None
    assert "request_1" in sol.accepted
    assert unassigned == {"request_2", "request_3"}

    sol2, rejected, ok2 = greedy_complete(sol, list(unassigned))
    assert ok2 is True
    assert "request_1" in sol2.accepted  # still accepted after Algorithm 2 runs


def test_preprocess_preserves_partial_route_and_milp3_timing() -> None:
    """preprocess() preserves exact service-start times/slack from an inherited route"""
    generator = ModemsScenarioGenerator(seed=5)
    scenario = generator.generate_random_scenario(
        nr_agents=1,
        nr_requests=2,
        nr_scheduled=1,
    )
    partial_plan = _scheduled_partial_plan(scenario, SolverStrategy.milp3)
    inherited = partial_plan.journeys["agent_1"]
    request_name = "request_1"
    pickup_index = inherited.request_pickup[request_name]
    delivery_index = inherited.request_delivery[request_name]
    pickup = inherited.route[pickup_index]
    delivery = inherited.route[delivery_index]
    pickup_start = partial_plan.ctx.node_latest_p[pickup] + 2.0
    delivery_start = (
        pickup_start
        + partial_plan.ctx.node_t_service[pickup]
        + partial_plan.ctx.t_travel(pickup, delivery)
    )
    delayed_starts = {pickup: pickup_start, delivery: delivery_start}
    partial_plan.journeys["agent_1"] = ModemsJourney.from_route(
        partial_plan.ctx,
        "agent_1",
        inherited.route,
        tau_of={pickup: 2.0},
        t_start_of=delayed_starts,
    )

    ctx = ProblemContext(
        scenario,
        ProblemType.closed_selective,
        SolverStrategy.milp3,
        DEFAULT_PARAMS_MILP,
    )
    rebuilt, unassigned = preprocess(ctx, partial_plan=partial_plan)

    assert rebuilt is not None
    rebuilt_journey = rebuilt.journeys["agent_1"]
    assert rebuilt_journey.route == partial_plan.journeys["agent_1"].route
    assert rebuilt_journey.states[pickup_index].t_start == pytest.approx(
        delayed_starts[pickup]
    )
    assert rebuilt_journey.states[delivery_index].t_start == pytest.approx(
        delayed_starts[delivery]
    )
    assert rebuilt_journey.states[pickup_index].tau == pytest.approx(2.0)
    assert unassigned == {"request_2"}


def test_preprocess_fails_on_incompatible_partial_plan_agents() -> None:
    """preprocess() returns solution=None if partial_plan is missing an agent"""
    generator = ModemsScenarioGenerator(seed=5)
    scenario = generator.generate_random_scenario(
        nr_agents=2, nr_requests=3, nr_scheduled=1
    )
    params = {"eps": 0.01, "zeta": 1.0, "eta": 100.0, "rho": 2.0}

    partial_plan = _scheduled_partial_plan(scenario)
    partial_plan.journeys.pop("agent_2")
    ctx = ProblemContext(
        scenario,
        ProblemType.closed_selective,
        SolverStrategy.alns,
        params,
    )
    solution, _ = preprocess(ctx, partial_plan=partial_plan)
    assert solution is None


def test_rolling_excerpt_shifts_and_preserves_milp3_service_times() -> None:
    """_build_next_scenario() shifts time by t_elapsed and preprocess() accepts it"""
    params = {"eps": 0.01, "zeta": 1.0, "eta": 100.0, "rho": 2.0}
    scenario = ModemsScenarioGenerator(seed=5).generate_random_scenario(
        nr_agents=1,
        nr_requests=1,
    )
    ctx = ProblemContext(
        scenario,
        ProblemType.closed_selective,
        SolverStrategy.milp3,
        params,
    )
    request_name = "request_1"
    agent_name = "agent_1"
    journey = ModemsJourney(ctx, agent_name)._append_request_direct(request_name)
    assert journey is not None
    pickup_index = journey.request_pickup[request_name]
    delivery_index = journey.request_delivery[request_name]
    pickup = journey.route[pickup_index]
    delivery = journey.route[delivery_index]
    pickup_start = ctx.node_earliest_p[pickup] + 2.0
    delivery_start = (
        pickup_start + ctx.node_t_service[pickup] + ctx.t_travel(pickup, delivery)
    )
    journey = ModemsJourney.from_route(
        ctx,
        agent_name,
        journey.route,
        t_start_of={pickup: pickup_start, delivery: delivery_start},
    )
    t_elapsed = 5.0
    advance = AgentAdvanceResult(
        node=pickup,
        delta=max(0.0, journey.states[pickup_index].t_arr - t_elapsed),
        sigma=journey.states[pickup_index].phi_arr,
        operation_state=AgentOperationalState.en_route,
        included=True,
        absorbed=set(),
        replan={request_name},
    )
    simulator = RollingHorizonSimulator.__new__(RollingHorizonSimulator)
    simulator.previous_ctx = ctx
    simulator.journeys = {agent_name: journey}
    simulator.time_since_last_adoption = t_elapsed
    simulator.network = scenario.network
    simulator.model_params = params

    updated_scenario, partial_plan = simulator._build_next_scenario(
        {agent_name: (advance, ctx.agents[agent_name])},
        [],
    )
    next_ctx = ProblemContext(
        updated_scenario,
        ProblemType.closed_selective,
        SolverStrategy.milp3,
        params,
    )
    rebuilt, unassigned = preprocess(next_ctx, partial_plan=partial_plan)

    assert rebuilt is not None
    assert unassigned == set()
    inherited = rebuilt.journeys[agent_name]
    assert inherited.states[pickup_index].t_start == pytest.approx(
        pickup_start - t_elapsed
    )
    assert inherited.states[delivery_index].t_start == pytest.approx(
        delivery_start - t_elapsed
    )


# --------------------------------------------------------------------------------------
# RollingHorizonSimulator (integration-level, state machine is covered above)
# --------------------------------------------------------------------------------------


def _tiny_simulator(
    solver_mode: str = "operational",
) -> tuple[RollingHorizonSimulator, ModemsScenarioGenerator]:
    """Build a minimal (1 hub-based agent) RollingHorizonSimulator for tests"""
    generator = ModemsScenarioGenerator(seed=1)
    agents = [
        ModemsAgent(
            node_type=NetworkNodeType.hub,
            node_index=1,
            load_max=6,
            time_initial=0.0,
            soc_initial=0.9,
        )
    ]
    rh_simulator = RollingHorizonSimulator(
        generator.network,
        agents,
        solver_mode=solver_mode,
        milp_timelimit=20.0,
        alns_max_iter=200,
    )
    return rh_simulator, generator


def test_simulator_initializes_idle_and_feasible() -> None:
    """A fresh simulator starts with one idle agent and zero t_elapsed time"""
    rh_simulator, _ = _tiny_simulator()
    assert list(rh_simulator.agents.keys()) == ["agent_1"]
    assert rh_simulator.time_since_last_adoption == 0.0


@pytest.mark.integration
def test_simulator_advances_one_epoch_successfully() -> None:
    """advance_epoch() with 'operational' mode (real MILP3+ALNS) adopts a plan"""
    rh_simulator, generator = _tiny_simulator()
    new_reqs = [
        generator.generate_random_request(
            load_lb=1,
            load_ub=3,
            time_lb=15.0,
            time_ub=55.0,
            tw_length=5.0,
            status=RequestStatus.new,
        )
        for _ in range(2)
    ]
    record = rh_simulator.advance_epoch(new_reqs)
    assert record.adopted_strategy is not None
    assert rh_simulator.time_since_last_adoption == 0.0


def test_simulator_fallback_keeps_old_plan_and_rejects_new(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If every solver fails, the old plan is kept and new requests are rejected"""
    rh_simulator, generator = _tiny_simulator()
    old_journeys = dict(rh_simulator.journeys)

    def _fail_all(scenario: Any, partial_plan: Any) -> dict:
        rh_simulator.last_solver_errors = {}
        return {}

    monkeypatch.setattr(rh_simulator, "_solve_all", _fail_all)

    new_reqs = [
        generator.generate_random_request(
            load_lb=1,
            load_ub=3,
            time_lb=15.0,
            time_ub=55.0,
            tw_length=5.0,
            status=RequestStatus.new,
        )
        for _ in range(2)
    ]
    record = rh_simulator.advance_epoch(new_reqs)
    assert record.adopted_strategy is None
    assert len(rh_simulator.rejected_log) == 2
    assert all(
        rh_simulator.journeys[k].route == old_journeys[k].route for k in old_journeys
    )
    assert rh_simulator.time_since_last_adoption == 10.0


def test_workday_urgent_arrival_is_not_hidden_by_earlier_nonurgent_arrival() -> None:
    """A later but urgent arrival still triggers before a later periodic epoch"""

    class RecordingSimulator:
        """Fake simulator that just records each advance_epoch() arguments"""

        def __init__(self) -> None:
            self.epoch_log: list[EpochLog] = []
            self.calls: list[tuple[float, list[ModemsRequest]]] = []

        def advance_epoch(
            self, requests: list[ModemsRequest], t_elapsed: float
        ) -> EpochLog:
            """Record the call and append a minimal (unadopted) epoch record"""
            self.calls.append((t_elapsed, requests))
            record = EpochLog(adopted_strategy=None)
            self.epoch_log.append(record)
            return record

    simulator = RecordingSimulator()
    nonurgent = ModemsRequest(1, 2, earliest_pickup=100.0, request_id="later")
    urgent = ModemsRequest(2, 3, earliest_pickup=20.0, request_id="urgent")
    workday = WorkdaySimulation(
        simulator,
        [(1.0, nonurgent), (2.0, urgent)],
        workday_length=3.0,
    )

    workday.run()

    assert simulator.calls[0][0] == pytest.approx(2.0)
    assert [request.request_id for request in simulator.calls[0][1]] == ["urgent"]


# --------------------------------------------------------------------------------------
# _collect_request_metrics / summarize_workday
# --------------------------------------------------------------------------------------


def test_collect_request_metrics_matches_manual_calc() -> None:
    """_collect_request_metrics() matches a hand-computed wait/ride/tardiness calc"""
    generator = ModemsScenarioGenerator(seed=1)
    scenario = generator.generate_random_scenario(nr_agents=1, nr_requests=1)
    ctx = ProblemContext(
        scenario,
        ProblemType.closed_selective,
        SolverStrategy.alns,
        {"eps": 0.01, "zeta": 1.0, "eta": 100.0, "rho": 2.0},
    )
    agent_name = ctx.agent_names[0]
    r = ctx.request_names[0]
    journey = ModemsJourney(ctx, agent_name)._append_request_direct(r)
    assert journey is not None

    metrics = _collect_request_metrics(ctx, journey, r)
    p_idx = journey.request_pickup[r]
    d_idx = journey.request_delivery[r]
    p_state, d_state = journey.states[p_idx], journey.states[d_idx]
    assert metrics["pickup_time"] == p_state.t_start
    assert metrics["delivery_time"] == d_state.t_start
    assert metrics["waiting_time"] == p_state.t_wait
    expected_ride = d_state.t_start - p_state.t_start - ctx.node_t_service[p_state.node]
    assert metrics["ride_time"] == pytest.approx(expected_ride)
    assert metrics["excess_ride_time"] == pytest.approx(
        expected_ride - metrics["direct_time"]
    )


def test_collect_request_metrics_none_for_unrelated_request() -> None:
    """_collect_request_metrics() returns None for a request not in this journey"""
    generator = ModemsScenarioGenerator(seed=1)
    scenario = generator.generate_random_scenario(nr_agents=1, nr_requests=2)
    ctx = ProblemContext(
        scenario,
        ProblemType.closed_selective,
        SolverStrategy.alns,
        {"eps": 0.01, "zeta": 1.0, "eta": 100.0, "rho": 2.0},
    )
    agent_name = ctx.agent_names[0]
    journey = ModemsJourney(ctx, agent_name)._append_request_direct(
        ctx.request_names[0]
    )
    assert journey is not None
    # request_2 was never inserted into this journey
    assert _collect_request_metrics(ctx, journey, ctx.request_names[1]) is None


def test_summarize_workday_aggregates_across_epochs() -> None:
    """summarize_workday() correctly totals/means/maxes a hand-built epoch_log"""
    from types import SimpleNamespace

    def fake_instance(solve_time: float) -> SimpleNamespace:
        return SimpleNamespace(solution_info=SimpleNamespace(solution_time=solve_time))

    epoch_log = [
        EpochLog(
            adopted_strategy="milp3",
            results={"milp3": fake_instance(5.0)},
            rejected_new={"request_2"},
            completed_requests={
                "request_1": {
                    "waiting_time": 2.0,
                    "excess_ride_time": 1.0,
                    "delay_time": 0.5,
                }
            },
            energy_consumed={"agent_1": 0.01},
            charging_events={},
            agent_travel_time={"agent_1": 4.0},
        ),
        EpochLog(
            adopted_strategy="alns",
            results={"alns": fake_instance(0.2)},
            rejected_new=set(),
            completed_requests={
                "request_1": {
                    "waiting_time": 3.0,
                    "excess_ride_time": 0.0,
                    "delay_time": 0.0,
                }
            },
            energy_consumed={"agent_1": 0.02},
            charging_events={"agent_1": {"soc_before": 0.3, "soc_after": 0.8}},
            agent_travel_time={"agent_1": 6.0},
        ),
        EpochLog(
            adopted_strategy=None,
            rejected_new={"request_5"},
            completed_requests={},
            energy_consumed={},
            charging_events={},
            agent_travel_time={},
        ),
    ]
    summary = summarize_workday(epoch_log)
    assert summary["nr_epochs"] == 3
    assert summary["nr_epochs_no_adoption"] == 1
    assert summary["nr_adopted"] == {"milp3": 1, "alns": 1}
    assert summary["nr_requests_completed"] == 2
    assert summary["nr_requests_rejected"] == 2
    assert summary["acceptance_rate"] == pytest.approx(0.5)
    assert summary["total_waiting_time"] == pytest.approx(5.0)
    assert summary["mean_waiting_time"] == pytest.approx(2.5)
    assert summary["max_waiting_time"] == pytest.approx(3.0)
    assert summary["total_excess_ride_time"] == pytest.approx(1.0)
    assert summary["max_excess_ride_time"] == pytest.approx(1.0)
    assert summary["total_delay_time"] == pytest.approx(0.5)
    assert summary["max_delay_time"] == pytest.approx(0.5)
    assert summary["total_agent_travel_time"] == pytest.approx(10.0)
    assert summary["total_energy_consumed"] == pytest.approx(0.03)
    assert summary["nr_charging_events"] == 1
    assert summary["total_soc_gained_from_charging"] == pytest.approx(0.5)
    assert summary["total_solve_time"] == pytest.approx(5.2)
    assert summary["mean_solve_time_per_epoch"] == pytest.approx(2.6)


def test_summarize_workday_handles_empty_log() -> None:
    """summarize_workday([]) returns zero counts and None for undefined rates/means"""
    summary = summarize_workday([])
    assert summary["nr_epochs"] == 0
    assert summary["acceptance_rate"] is None
    assert summary["mean_waiting_time"] is None


@pytest.mark.integration
def test_completed_and_rejected_requests_keyed_by_persistent_request_id() -> None:
    """Every id seen across a real workday's epoch_log traces to a generated request"""
    generator = ModemsScenarioGenerator(seed=3)
    agents = [
        ModemsAgent(
            node_type=NetworkNodeType.hub,
            node_index=1,
            load_max=6,
            time_initial=0.0,
            soc_initial=0.9,
        )
    ]
    rh_simulator = RollingHorizonSimulator(
        generator.network,
        agents,
        solver_mode="operational",
        milp_timelimit=10.0,
        alns_max_iter=100,
    )
    arrivals = generator.generate_workday_requests(
        workday_length=40.0, base_rate_per_hour=6.0, nr_surges=0
    )
    generated_ids = {r.request_id for _, r in arrivals}

    workday = WorkdaySimulation(rh_simulator, arrivals, workday_length=40.0)
    log = workday.run()

    seen_ids: set[str] = set()
    for rec in log:
        seen_ids |= set(rec.completed_requests.keys())
        seen_ids |= rec.rejected_new
    # every seen id maps to a real, originally-generated request
    assert seen_ids <= generated_ids


# --------------------------------------------------------------------------------------
# Bug fixes found while adding travel-time tracking: (i) retroactive absorption of
# already-completed pairs preceding the search's starting index, (ii) correct
# travel_time when t_elapsed exceeds the entire journey
# --------------------------------------------------------------------------------------


def test_retroactive_absorption_of_earlier_completed_pair_mid_ride(
    ctx_and_agent: ProblemContext,
) -> None:
    """An earlier-delivered request is absorbed alongside the current one"""
    ctx = ctx_and_agent
    agent_name = ctx.agent_names[0]
    r1, r2 = ctx.request_names[0], ctx.request_names[1]
    p1, d1 = ctx.pickup_node[r1], ctx.delivery_node[r1]
    p2, d2 = ctx.pickup_node[r2], ctx.delivery_node[r2]
    v_k = ctx.agent_initial_node[agent_name]
    hf = ctx.nearest_final_depot(v_k)
    route = [v_k, p1, d1, p2, d2, hf]
    states = [
        _mk_state(v_k, t_arr=0.0, t=0.0, t_dep=0.0),
        _mk_state(p1, t_arr=2.0, t=2.0, t_dep=3.0, z_dep=1),
        _mk_state(d1, t_arr=6.0, t=6.0, t_dep=7.0, z_dep=0),
        _mk_state(p2, t_arr=12.0, t=12.0, t_dep=13.0, z_dep=2),
        _mk_state(d2, t_arr=18.0, t=18.0, t_dep=19.0, z_dep=0),
        _mk_state(hf, t_arr=24.0, phi_arr=0.9),
    ]
    journey = _mk_journey(ctx, agent_name, route, states)

    # t_elapsed=15: departed p2 (t_dep=13) but not yet at d2 (t_dep=19) -> idx=4
    # (d2), z_hat=states[3].z_dep=2>0 -- r1 was already fully delivered
    # earlier and must also be absorbed
    result = advance_agent_state(ctx, agent_name, journey, t_elapsed=15.0)
    assert r1 in result.absorbed
    assert r2 in result.absorbed


def test_travel_time_when_elapsed_exceeds_entire_journey(
    ctx_and_agent: ProblemContext,
) -> None:
    """When t_elapsed exceeds the whole journey, travel_time sums the full route"""
    ctx = ctx_and_agent
    agent_name = ctx.agent_names[0]
    r1 = ctx.request_names[0]
    p1, d1 = ctx.pickup_node[r1], ctx.delivery_node[r1]
    v_k = ctx.agent_initial_node[agent_name]
    hf = ctx.nearest_final_depot(v_k)
    route = [v_k, p1, d1, hf]
    states = [
        _mk_state(v_k, t_arr=0.0, t=0.0, t_dep=0.0),
        _mk_state(p1, t_arr=2.0, t=2.0, t_dep=3.0, z_dep=1),
        _mk_state(d1, t_arr=6.0, t=6.0, t_dep=7.0, z_dep=0),
        _mk_state(hf, t_arr=10.0, phi_arr=0.9),
    ]
    journey = _mk_journey(ctx, agent_name, route, states)

    full_route_time = sum(
        ctx.t_travel(route[i], route[i + 1]) for i in range(len(route) - 1)
    )
    result = advance_agent_state(ctx, agent_name, journey, t_elapsed=1000.0)
    assert result.node == hf
    assert result.travel_time == pytest.approx(full_route_time)


def test_travel_time_matches_single_arc_when_stopping_early(
    ctx_and_agent: ProblemContext,
) -> None:
    """When resolve stops at the first pickup, travel_time is only that one arc"""
    ctx = ctx_and_agent
    agent_name = ctx.agent_names[0]
    r1 = ctx.request_names[0]
    p1, d1 = ctx.pickup_node[r1], ctx.delivery_node[r1]
    v_k = ctx.agent_initial_node[agent_name]
    hf = ctx.nearest_final_depot(v_k)
    route = [v_k, p1, d1, hf]
    states = [
        _mk_state(v_k, t_arr=0.0, t=0.0, t_dep=0.0),
        _mk_state(
            p1, t_arr=2.0, t=2.0, t_dep=3.0, z_dep=1, wait=REPLANNING_THRESHOLD_W + 5.0
        ),
        _mk_state(d1, t_arr=6.0, t=6.0, t_dep=7.0, z_dep=0),
        _mk_state(hf, t_arr=10.0),
    ]
    journey = _mk_journey(ctx, agent_name, route, states)

    result = advance_agent_state(ctx, agent_name, journey, t_elapsed=0.0)
    assert result.node == p1  # released for replanning, stops at the pickup
    assert result.travel_time == pytest.approx(ctx.t_travel(v_k, p1))


# --------------------------------------------------------------------------------------
# solver_mode="single" and the same-arrivals MILP3-vs-ALNS comparison driver
# --------------------------------------------------------------------------------------


def test_single_solver_mode_requires_valid_single_solver() -> None:
    """raises ValueError for solver_mode='single' with no/invalid single_solver"""
    generator = ModemsScenarioGenerator(seed=1)
    agents = [
        ModemsAgent(
            node_type=NetworkNodeType.hub,
            node_index=1,
            load_max=6,
            time_initial=0.0,
            soc_initial=0.9,
        )
    ]
    with pytest.raises(ValueError):
        RollingHorizonSimulator(
            generator.network, agents, solver_mode="single", single_solver=None
        )
    with pytest.raises(ValueError):
        RollingHorizonSimulator(
            generator.network, agents, solver_mode="single", single_solver="milp1"
        )


@pytest.mark.integration
def test_single_solver_mode_only_calls_the_chosen_solver() -> None:
    """solver_mode='single' with single_solver='alns' only ever invokes real alns"""
    generator = ModemsScenarioGenerator(seed=1)
    agents = [
        ModemsAgent(
            node_type=NetworkNodeType.hub,
            node_index=1,
            load_max=6,
            time_initial=0.0,
            soc_initial=0.9,
        )
    ]
    rh_simulator = RollingHorizonSimulator(
        generator.network,
        agents,
        solver_mode="single",
        single_solver="alns",
        milp_timelimit=10.0,
        alns_max_iter=100,
    )
    new_reqs = [
        generator.generate_random_request(
            load_lb=1,
            load_ub=3,
            time_lb=15.0,
            time_ub=55.0,
            tw_length=5.0,
            status=RequestStatus.new,
        )
        for _ in range(2)
    ]
    record = rh_simulator.advance_epoch(new_reqs)
    assert set(record.results.keys()) <= {"alns"}
    assert record.adopted_strategy in (None, "alns")


@pytest.mark.integration
def test_compare_solvers_one_workday_uses_identical_arrivals() -> None:
    """compare_solvers_one_workday() runs the same arrivals through both solvers"""
    generator = ModemsScenarioGenerator(seed=2)
    agents = [
        ModemsAgent(
            node_type=NetworkNodeType.hub,
            node_index=1,
            load_max=6,
            time_initial=0.0,
            soc_initial=0.9,
        )
    ]
    # generous duration relative to the 15-45min lead so arrivals have time to resolve
    request_submissions = generator.generate_workday_requests(
        workday_length=20.0, base_rate_per_hour=6.0, nr_surges=0
    )

    summaries = compare_solvers_one_workday(
        generator.network,
        agents,
        request_submissions,
        workday_length=90.0,
        milp_timelimit=10.0,
        alns_max_iter=100,
    )
    assert set(summaries.keys()) == {"milp3", "alns"}
    for summary in summaries.values():
        assert summary["nr_requests_completed"] + summary[
            "nr_requests_rejected"
        ] == len(request_submissions)


@pytest.mark.integration
def test_compare_solvers_over_workdays_is_seeded_reproducible() -> None:
    """compare_solvers_over_workdays() with the same base_seed reproduces exactly"""
    generator = ModemsScenarioGenerator(seed=1)
    agents = [
        ModemsAgent(
            node_type=NetworkNodeType.hub,
            node_index=1,
            load_max=6,
            time_initial=0.0,
            soc_initial=0.9,
        )
    ]

    kwargs = dict(
        network=generator.network,
        agents=agents,
        nr_workdays=2,
        workday_length=20.0,
        base_seed=7,
        generator_kwargs={"base_rate_per_hour": 6.0, "nr_surges": 0},
        milp_timelimit=10.0,
        alns_max_iter=100,
    )
    results1 = compare_solvers_over_workdays(**kwargs)
    results2 = compare_solvers_over_workdays(**kwargs)
    assert len(results1) == 2
    for r1, r2 in zip(results1, results2):
        assert r1["seed"] == r2["seed"]
        assert (
            r1["milp3"]["nr_requests_completed"] == r2["milp3"]["nr_requests_completed"]
        )
        assert (
            r1["alns"]["nr_requests_completed"] == r2["alns"]["nr_requests_completed"]
        )


@pytest.mark.integration
def test_compare_solvers_over_workdays_checkpoint_callback() -> None:
    """on_workday_done() is called once per completed workday, in order"""
    generator = ModemsScenarioGenerator(seed=1)
    agents = [
        ModemsAgent(
            node_type=NetworkNodeType.hub,
            node_index=1,
            load_max=6,
            time_initial=0.0,
            soc_initial=0.9,
        )
    ]
    seen: list[int] = []
    compare_solvers_over_workdays(
        generator.network,
        agents,
        nr_workdays=2,
        workday_length=20.0,
        base_seed=3,
        generator_kwargs={"base_rate_per_hour": 6.0, "nr_surges": 0},
        milp_timelimit=8.0,
        alns_max_iter=80,
        on_workday_done=lambda i, record: seen.append(i),
    )
    assert seen == [0, 1]


# --------------------------------------------------------------------------------------
# agent_id: persistent cross-epoch identity
# --------------------------------------------------------------------------------------


def test_to_agent_propagates_template_agent_id() -> None:
    """
    AgentAdvanceResult.to_agent() carries the template's agent_id forward, so the
    agent built in the next epoch's scenario is recognized/mapped correctly
    """
    template = ModemsAgent(
        node_type=NetworkNodeType.hub, node_index=1, agent_id="vehicle-7"
    )
    result = AgentAdvanceResult(
        node="h_1",
        delta=5.0,
        sigma=0.7,
        operation_state=AgentOperationalState.available,
        included=True,
        absorbed=set(),
        replan=set(),
    )
    new_agent = result.to_agent(template)
    assert new_agent.agent_id == "vehicle-7"


def test_simulator_rejects_duplicate_agent_ids() -> None:
    """
    Two fleet agents sharing an agent_id would silently corrupt the cross-epoch
    tracking; reject the fleet upfront
    """
    generator = ModemsScenarioGenerator(seed=1)
    agents = [
        ModemsAgent(node_type=NetworkNodeType.hub, node_index=1, agent_id="dup"),
        ModemsAgent(node_type=NetworkNodeType.hub, node_index=2, agent_id="dup"),
    ]
    with pytest.raises(ValueError, match="agent_id"):
        RollingHorizonSimulator(generator.network, agents)


def test_advance_parked_agents_stays_parked_until_within_horizon() -> None:
    """
    A parked agent ages by t_elapsed each call and only rejoins once its
    remaining charge-then-wait time is under PLANNING_HORIZON_P
    """

    simulator = RollingHorizonSimulator.__new__(RollingHorizonSimulator)
    simulator.parked_agents = {
        "vehicle-1": {"node": "h_1", "remaining_delta": 90.0, "sigma": 0.8}
    }

    rejoined = simulator._advance_parked_agents(t_elapsed=10.0)
    assert rejoined == {}
    assert simulator.parked_agents["vehicle-1"]["remaining_delta"] == pytest.approx(
        80.0
    )

    rejoined = simulator._advance_parked_agents(t_elapsed=25.0)
    assert "vehicle-1" not in simulator.parked_agents
    assert set(rejoined) == {"vehicle-1"}
    result = rejoined["vehicle-1"]
    assert result.node == "h_1"
    assert result.sigma == 0.8
    assert result.delta == pytest.approx(55.0)
    assert result.included is True
    assert result.operation_state == AgentOperationalState.available


def test_excluded_agent_is_parked_then_rejoins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Regression test: an agent needing more than PLANNING_HORIZON_P of charging used
    to be dropped from new_agents and never introduced again. Instead, it must be
    parked and, once enough real time elapses for its SoC to fit within the planning
    horizon, rejoin the active fleet as the same physical agent (same agent_id)
    """
    generator = ModemsScenarioGenerator(seed=1)
    scenario = generator.generate_random_scenario(nr_agents=2, nr_requests=1)
    ctx = ProblemContext(
        scenario, ProblemType.closed_selective, SolverStrategy.alns, DEFAULT_PARAMS_MILP
    )
    agent_1_name, agent_2_name = ctx.agent_names[0], ctx.agent_names[1]
    agent_1, agent_2 = ctx.agents[agent_1_name], ctx.agents[agent_2_name]
    r1 = ctx.request_names[0]
    p1, d1 = ctx.pickup_node[r1], ctx.delivery_node[r1]
    v_k = ctx.agent_initial_node[agent_1_name]
    hf = ctx.nearest_final_depot(v_k)
    # agent_1 finishes its route at the final hub with very low SoC
    route_1 = [v_k, p1, d1, hf]
    states_1 = [
        _mk_state(v_k, t_arr=0.0, t=0.0, t_dep=0.0),
        _mk_state(p1, t_arr=2.0, t=2.0, t_dep=3.0, z_dep=1),
        _mk_state(d1, t_arr=8.0, t=8.0, t_dep=9.0, z_dep=0),
        _mk_state(hf, t_arr=15.0, phi_arr=0.15),  # well below threshold
    ]
    journey_1 = _mk_journey(ctx, agent_1_name, route_1, states_1)
    # agent_2 stays idle the whole time, healthy SoC; keeps every epoch sceanrio
    # non-empty to disentangle this test from zero-agent trivial-scenario short-circuit
    journey_2 = ModemsJourney(ctx, agent_2_name)

    simulator = RollingHorizonSimulator.__new__(RollingHorizonSimulator)
    simulator.network = scenario.network
    simulator.model_params = DEFAULT_PARAMS_MILP
    simulator.agents = {agent_1_name: agent_1, agent_2_name: agent_2}
    simulator.agent_templates = {
        agent_1.agent_id: agent_1.copy(),
        agent_2.agent_id: agent_2.copy(),
    }
    simulator.parked_agents = {}
    simulator.previous_ctx = ctx
    simulator.journeys = {agent_1_name: journey_1, agent_2_name: journey_2}
    simulator.time_since_last_adoption = 0.0
    simulator.rejected_log = []
    simulator.epoch_log = []
    simulator._last_confirmed_node = {}
    simulator._energy_checkpoint = {
        agent_1.agent_id: agent_1.soc_initial,
        agent_2.agent_id: agent_2.soc_initial,
    }

    def _fake_solve_all(next_scenario: Any, partial_plan: Any) -> dict:
        next_ctx = ProblemContext(
            next_scenario,
            ProblemType.closed_selective,
            SolverStrategy.alns,
            DEFAULT_PARAMS_MILP,
        )
        base_plan, r_unassigned = preprocess(next_ctx, partial_plan=partial_plan)
        assert base_plan is not None
        solution, _, ok = greedy_complete(base_plan, r_unassigned)
        assert ok
        info = ModemsSolutionInfo(
            status=SolutionStatus.optimal, objective=solution.objective()
        )
        simulator.last_solver_errors = {}
        return {SolverStrategy.alns: ModemsInstance(solution.ctx, solution, info)}

    monkeypatch.setattr(simulator, "_solve_all", _fake_solve_all)

    # epoch 1: cumulative t_elapsed=20 reaches the hub (t_arr=15) with very low SoC,
    # 0.15 -> 0.8 needs t_charge well over PLANNING_HORIZON_P
    record1 = simulator.advance_epoch([], t_elapsed=20.0)
    assert record1.adopted_strategy is not None
    assert agent_1.agent_id in simulator.parked_agents
    remaining_after_epoch1 = simulator.parked_agents[agent_1.agent_id][
        "remaining_delta"
    ]
    assert remaining_after_epoch1 > PLANNING_HORIZON_P
    # agent_1 does not appear in the active fleet, only agent_2 remains
    active_ids = {a.agent_id for a in simulator.previous_ctx.agents.values()}
    assert agent_1.agent_id not in active_ids
    assert agent_2.agent_id in active_ids

    # epoch 2: enough real time has elpased for the remaining charge-then-wait tim
    # to be within PLANNING_HORIZON_P; agent_1 should rejoin
    t2 = remaining_after_epoch1 - (PLANNING_HORIZON_P - 10.0)
    record2 = simulator.advance_epoch([], t_elapsed=t2)
    assert record2.adopted_strategy is not None
    assert agent_1.agent_id not in simulator.parked_agents
    active_ids_2 = {a.agent_id for a in simulator.previous_ctx.agents.values()}
    assert agent_1.agent_id in active_ids_2


def test_energy_consumed_is_incremental_not_cumulative_across_no_adoption_epochs(
    ctx_and_agent: ProblemContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Regression test: energy_consumed must report the delta since the last epoch with
    the agent itself, independent of when the active plan was last adopted. A plan
    can persist across several no-adoption epochs (every solver failing, or nothing
    usable); reporting the since-plan-adoption total re-counts the same distance
    """
    ctx = ctx_and_agent
    agent_name = ctx.agent_names[0]
    agent = ctx.agents[agent_name]
    r1 = ctx.request_names[0]
    journey = ModemsJourney(ctx, agent_name)._append_request_direct(r1)
    assert journey is not None
    pickup_index = journey.request_pickup[r1]
    delivery_index = journey.request_delivery[r1]
    pickup = journey.route[pickup_index]
    delivery = journey.route[delivery_index]
    pickup_start = ctx.node_earliest_p[pickup] + 2.0
    delivery_start = (
        pickup_start + ctx.node_t_service[pickup] + ctx.t_travel(pickup, delivery)
    )
    journey = ModemsJourney.from_route(
        ctx,
        agent_name,
        journey.route,
        t_start_of={pickup: pickup_start, delivery: delivery_start},
    )

    simulator = RollingHorizonSimulator.__new__(RollingHorizonSimulator)
    simulator.network = ctx.scenario.network
    simulator.model_params = DEFAULT_PARAMS_MILP
    simulator.agents = {agent_name: agent}
    simulator.agent_templates = {agent.agent_id: agent.copy()}
    simulator.parked_agents = {}
    simulator.previous_ctx = ctx
    simulator.journeys = {agent_name: journey}
    simulator.time_since_last_adoption = 0.0
    simulator.rejected_log = []
    simulator.epoch_log = []
    simulator._last_confirmed_node = {}
    simulator._energy_checkpoint = {agent.agent_id: agent.soc_initial}

    def _fail_all(next_scenario: Any, partial_plan: Any) -> dict:
        simulator.last_solver_errors = {}
        return {}

    monkeypatch.setattr(simulator, "_solve_all", _fail_all)

    t1 = pickup_start + 3.0
    t2 = 50.0
    record1 = simulator.advance_epoch([], t_elapsed=t1)
    record2 = simulator.advance_epoch([], t_elapsed=t2)
    # no adoption happened at either epochs, the same plan persisted throughout
    assert record1.adopted_strategy is None
    assert record2.adopted_strategy is None

    total_reported = record1.energy_consumed.get(
        agent.agent_id, 0.0
    ) + record2.energy_consumed.get(agent.agent_id, 0.0)

    # ground truth: a single direct call at the combined elapsed time gives exactly
    # the physical energy used along the (unchanged) route
    direct = advance_agent_state(ctx, agent_name, journey, t_elapsed=t1 + t2)
    result_idx = journey.route.index(direct.node)
    expected_total = agent.soc_initial - journey.states[result_idx].phi_arr

    assert record1.energy_consumed.get(agent.agent_id, 0.0) > 1e-6
    assert total_reported == pytest.approx(expected_total, abs=1e-9)
