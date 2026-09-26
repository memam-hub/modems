"""
Unit tests for modems.solution: state propagation (ModemsJourney), route mutations,
the objective (ModemsSolution), and persistence/diagnostics (ModemsInstance). Most
tests use the hand-built line network to simplify propagated value evaluation
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from modems.core import ModemsAgent, ModemsRequest, ModemsScenario, ProblemContext
from modems.network import RoadNetwork
from modems.solution import (
    FLOAT_INF,
    ModemsInstance,
    ModemsJourney,
    ModemsSolution,
    ModemsSolutionInfo,
    NodeState,
    SolutionStatus,
)

from .builders import greedy_solution, line_scenario, make_ctx

ALPHA, BETA = 6.65e-3, 0.15e-3  # ModemsAgent default discharge rates

# request_1: s2 -> s5, 2 passengers, 1 min service, pickup window [10, 15]
REQUEST_1 = ModemsRequest(2, 5, load=2, service_time=1.0, earliest_pickup=10.0)
ROUTE_1 = ["a_1_h_1", "r_1_p_2", "r_1_d_5", "h_1"]

# request_1 (s1 -> s4) and request_2 (s2 -> s3), which can nest inside request_1
TWO_REQUESTS = [
    ModemsRequest(1, 4, earliest_pickup=1.0, service_time=0.0, request_id="a"),
    ModemsRequest(2, 3, earliest_pickup=2.0, service_time=0.0, request_id="b"),
]


def one_request_ctx(strategy: str = "alns", **kwargs) -> ProblemContext:
    return make_ctx(line_scenario([REQUEST_1]), strategy, **kwargs)


def served(ctx: ProblemContext, *routes: list[str]) -> ModemsSolution:
    """A solution with agents following given routes, all routed requests accepted"""
    solution = ModemsSolution(ctx)
    for agent_name, route in zip(ctx.agent_names, routes):
        journey = ModemsJourney.from_route(ctx, agent_name, route)
        solution.journeys[agent_name] = journey
        solution.accepted |= set(journey.request_pickup)
    solution.recompute_rejected()
    return solution


def times(journey: ModemsJourney) -> list[tuple[float, float, float, float]]:
    return [(s.t_arr, s.t_wait, s.t_start, s.t_dep) for s in journey.states]


# --------------------------------------------------------------------------------------
# NodeState
# --------------------------------------------------------------------------------------


def test_node_state_copy_and_dict_round_trip() -> None:
    state = NodeState("r_1_p_2", 1.0, 2.0, 3.0, 4.0, 0.5, 1, 3, 0.9, 0.9)
    clone = state.copy()
    clone.t_arr = 99.0
    assert state.t_arr == 1.0
    assert NodeState.from_dict(state.to_dict()) == state


# --------------------------------------------------------------------------------------
# ModemsJourney: propagation
# --------------------------------------------------------------------------------------


def test_idle_journey_goes_straight_to_the_nearest_hub() -> None:
    scenario = line_scenario(
        [REQUEST_1], agents=[ModemsAgent("s", 4, time_initial=3.0, soc_initial=0.9)]
    )
    journey = ModemsJourney(make_ctx(scenario), "agent_1")
    assert journey.route == ["a_1_s_4", "h_1"]
    assert not journey.is_active
    assert times(journey) == [(3.0, 0.0, 3.0, 3.0), (7.0, 0.0, 7.0, 7.0)]
    assert journey.states[-1].phi_arr == pytest.approx(0.9 - 4 * ALPHA)
    assert journey.request_pickup == journey.request_delivery == {}


@pytest.mark.parametrize(
    "strategy, pickup, delivery, hub, tau",
    [
        # ALNS/MILP2 wait until e^r=10;
        # MILP1/MILP3 serve a new request early, but no earlier than e^r - omega = 5
        ("alns", (2.0, 8.0, 10.0, 11.0), (14.0, 0.0, 14.0, 15.0), 20.0, 0.0),
        ("milp2", (2.0, 8.0, 10.0, 11.0), (14.0, 0.0, 14.0, 15.0), 20.0, 0.0),
        ("milp1", (2.0, 3.0, 5.0, 6.0), (9.0, 0.0, 9.0, 10.0), 15.0, 5.0),
        ("milp3", (2.0, 3.0, 5.0, 6.0), (9.0, 0.0, 9.0, 10.0), 15.0, 5.0),
    ],
)
def test_from_route_propagates_time_by_strategy_scheduling_rule(
    strategy: str, pickup: tuple, delivery: tuple, hub: float, tau: float
) -> None:
    journey = ModemsJourney.from_route(one_request_ctx(strategy), "agent_1", ROUTE_1)
    assert times(journey) == [
        (0.0, 0.0, 0.0, 0.0),
        pickup,
        delivery,
        (hub, 0.0, hub, hub),
    ]
    assert journey.states[1].tau == tau
    assert journey._is_feasible()


def test_from_route_propagates_load_and_load_dependent_energy() -> None:
    journey = ModemsJourney.from_route(one_request_ctx(), "agent_1", ROUTE_1)
    assert [(s.z_arr, s.z_dep) for s in journey.states] == [
        (0, 0),
        (0, 2),
        (2, 0),
        (0, 0),
    ]
    soc = np.cumsum([1.0, -2 * ALPHA, -3 * (ALPHA + 2 * BETA), -5 * ALPHA])
    assert [s.phi_arr for s in journey.states] == pytest.approx(list(soc))
    assert all(s.phi_dep == s.phi_arr for s in journey.states)
    assert journey.total_travel_time() == 10.0
    assert journey.total_energy() == pytest.approx(1.0 - soc[-1])
    assert journey.terminal_soc_slack() == pytest.approx(soc[-1] - 0.25)


def test_scheduled_request_is_never_served_early_even_under_milp3() -> None:
    request = ModemsRequest(2, 5, load=2, earliest_pickup=10.0, status="scheduled")
    ctx = make_ctx(line_scenario([request]), "milp3")
    journey = ModemsJourney.from_route(ctx, "agent_1", ROUTE_1)
    assert journey.states[1].t_start == 10.0 and journey.states[1].tau == 0.0


def test_from_route_honors_given_service_starts_and_slack() -> None:
    ctx = one_request_ctx("milp3")
    journey = ModemsJourney.from_route(
        ctx,
        "agent_1",
        ROUTE_1,
        tau_of={"r_1_p_2": 1.0},
        t_start_of={"r_1_p_2": 9.0, "r_1_d_5": 13.0},
    )
    assert times(journey)[1:3] == [(2.0, 7.0, 9.0, 10.0), (13.0, 0.0, 13.0, 14.0)]
    assert journey.states[1].tau == 1.0
    assert journey._is_feasible()
    with pytest.raises(ValueError, match="precedes its propagated arrival"):
        ModemsJourney.from_route(ctx, "agent_1", ROUTE_1, t_start_of={"r_1_p_2": 1.0})


def test_early_service_beyond_omega_is_infeasible() -> None:
    """Serving a new MILP3 pickup more than omega before e^r breaks the TW"""
    ctx = one_request_ctx("milp3")
    ok = {"r_1_p_2": 5.0, "r_1_d_5": 9.0}
    too_early = {"r_1_p_2": 4.0, "r_1_d_5": 8.0}
    tau = {"r_1_p_2": 6.0}
    assert ModemsJourney.from_route(
        ctx, "agent_1", ROUTE_1, tau_of=tau, t_start_of=ok
    )._is_feasible()
    journey = ModemsJourney.from_route(
        ctx, "agent_1", ROUTE_1, tau_of=tau, t_start_of=too_early
    )
    assert not journey._is_alns_feasible()


def test_alns_tau_is_tardiness_only_and_never_permits_early_service() -> None:
    journey = ModemsJourney.from_route(
        one_request_ctx(), "agent_1", ROUTE_1, tau_of={"r_1_p_2": 5.0}
    )
    assert journey.states[1].t_start == 10.0
    assert journey.states[1].tau == 0.0


def test_late_arrival_accumulates_tardiness() -> None:
    late = ModemsRequest(6, 1, earliest_pickup=0.0, tw_length=2.0)  # s6 at t=6 > l=2
    ctx = make_ctx(line_scenario([late]))
    journey = ModemsJourney.from_route(
        ctx, "agent_1", ["a_1_h_1", "r_1_p_6", "r_1_d_1", "h_1"]
    )
    assert journey.states[1].t_start == 6.0 and journey.states[1].tau == 4.0


@pytest.mark.parametrize(
    "route, match",
    [
        (["a_1_h_1", "r_1_d_5", "r_1_p_2", "h_1"], "delivery precedes pickup"),
        (["a_1_h_1", "r_1_p_2", "h_1"], "incomplete"),
        (["a_1_h_1", "r_1_p_2", "r_1_p_2", "r_1_d_5", "h_1"], "duplicate"),
        (["a_1_h_1", "r_1_p_3", "r_1_d_5", "h_1"], "unknown service node"),
        (["a_1_h_1", "h_1", "h_1"], "unknown service node"),
        (["a_1_s_1", "h_1"], "inconsistent starting nodes"),
    ],
)
def test_from_route_rejects_malformed_routes(route: list[str], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ModemsJourney.from_route(one_request_ctx(), "agent_1", route)


def test_constructor_with_states_validates_them() -> None:
    ctx = one_request_ctx()
    reference = ModemsJourney.from_route(ctx, "agent_1", ROUTE_1)
    rebuilt = ModemsJourney(ctx, "agent_1", reference.states)
    assert [s.to_dict() for s in rebuilt.states] == [
        s.to_dict() for s in reference.states
    ]
    assert rebuilt.states is not reference.states
    for field, delta in [
        ("t_arr", 1.0),
        ("z_dep", 1),
        ("phi_arr", -0.1),
        ("tau", -1.0),
    ]:
        tampered = [s.copy() for s in reference.states]
        setattr(tampered[2], field, getattr(tampered[2], field) + delta)
        with pytest.raises(ValueError, match="infeasible or inconsistent"):
            ModemsJourney(ctx, "agent_1", tampered)


# --------------------------------------------------------------------------------------
# ModemsJourney: feasibility constraints
# --------------------------------------------------------------------------------------


def test_capacity_is_enforced() -> None:
    """Two single riders on board at once exceed a one-seat agent"""
    scenario = line_scenario(TWO_REQUESTS, agents=[ModemsAgent("h", 1, load_max=1)])
    ctx = make_ctx(scenario)
    one_by_one = ["a_1_h_1", "r_1_p_1", "r_1_d_4", "r_2_p_2", "r_2_d_3", "h_1"]
    nested = ["a_1_h_1", "r_1_p_1", "r_2_p_2", "r_2_d_3", "r_1_d_4", "h_1"]
    assert ModemsJourney.from_route(ctx, "agent_1", one_by_one)._is_feasible()
    assert not ModemsJourney.from_route(ctx, "agent_1", nested)._is_alns_feasible()


def test_max_ride_time_is_enforced() -> None:
    """request_2 rides s1 -> s3 (direct 2) via s5 and s6: ride = 9 > rho * 2 = 5"""
    requests = [
        ModemsRequest(1, 3, earliest_pickup=0.0, service_time=0.0, request_id="a"),
        ModemsRequest(5, 6, earliest_pickup=0.0, service_time=0.0, request_id="b"),
    ]
    ctx = make_ctx(line_scenario(requests))
    detour = ["a_1_h_1", "r_1_p_1", "r_2_p_5", "r_2_d_6", "r_1_d_3", "h_1"]
    assert not ModemsJourney.from_route(ctx, "agent_1", detour)._is_alns_feasible()
    ctx_lenient = make_ctx(line_scenario(requests), rho=5.0)
    assert ModemsJourney.from_route(ctx_lenient, "agent_1", detour)._is_feasible()


def test_extended_soc_floor_is_enforced_at_every_node() -> None:
    agent = ModemsAgent("h", 1, soc_initial=0.26, soc_min_operational=0.25)
    ctx = make_ctx(line_scenario([REQUEST_1], agents=[agent]))
    assert not ModemsJourney.from_route(ctx, "agent_1", ROUTE_1)._is_alns_feasible()


def test_milp1_uses_the_driving_duration_budget_instead_of_soc() -> None:
    """duration_max = 0.0275 / 0.0075 ~ 3.67 min < 10 min of driving"""
    agent = ModemsAgent(
        "h",
        1,
        soc_initial=0.2775,
        soc_min_operational=0.25,
        soc_alpha=0.0066,
        soc_beta=0.00015,
    )
    ctx = make_ctx(line_scenario([REQUEST_1], agents=[agent]), "milp1")
    assert not ModemsJourney.from_route(ctx, "agent_1", ROUTE_1)._is_alns_feasible()


def test_milp2_hard_time_window_rejects_tardiness() -> None:
    late = ModemsRequest(6, 1, earliest_pickup=0.0, tw_length=2.0)
    route = ["a_1_h_1", "r_1_p_6", "r_1_d_1", "h_1"]
    for strategy, feasible in [("milp2", False), ("milp3", True), ("alns", True)]:
        ctx = make_ctx(line_scenario([late]), strategy)
        assert (
            ModemsJourney.from_route(ctx, "agent_1", route)._is_alns_feasible()
            == feasible
        )


# --------------------------------------------------------------------------------------
# ModemsJourney: mutations
# --------------------------------------------------------------------------------------


def test_insert_at_places_pickup_and_delivery_after_the_given_positions() -> None:
    ctx = make_ctx(line_scenario(TWO_REQUESTS))
    journey = ModemsJourney.from_route(
        ctx, "agent_1", ["a_1_h_1", "r_1_p_1", "r_1_d_4", "h_1"]
    )
    journey.insert_at("request_2", 1, 1)  # insert in-between ride of request_1
    assert journey.route == [
        "a_1_h_1",
        "r_1_p_1",
        "r_2_p_2",
        "r_2_d_3",
        "r_1_d_4",
        "h_1",
    ]
    assert journey.request_pickup == {"request_1": 1, "request_2": 2}
    assert journey.request_delivery == {"request_2": 3, "request_1": 4}
    assert journey.onboard_at(2) == {"request_1", "request_2"}
    assert journey.onboard_at(3) == {"request_1"}
    assert journey._is_feasible()


def test_insert_at_rejects_bad_input_and_is_atomic_on_infeasibility() -> None:
    scenario = line_scenario(TWO_REQUESTS, agents=[ModemsAgent("h", 1, load_max=1)])
    route = ["a_1_h_1", "r_1_p_1", "r_1_d_4", "h_1"]
    journey = ModemsJourney.from_route(make_ctx(scenario), "agent_1", route)
    before = [s.to_dict() for s in journey.states]
    with pytest.raises(ValueError, match="invalid insertion positions"):
        journey.insert_at("request_2", 2, 1)
    with pytest.raises(ValueError, match="selected insertion"):
        journey.insert_at("request_2", 1, 1)  # in-between: 2 riders > 1 seat
    assert [s.to_dict() for s in journey.states] == before
    assert journey.route == route and set(journey.request_pickup) == {"request_1"}


def test_insert_at_rejects_a_request_already_in_the_journey() -> None:
    journey = ModemsJourney.from_route(one_request_ctx(), "agent_1", ROUTE_1)
    with pytest.raises(ValueError, match="already in this journey"):
        journey.insert_at("request_1", 0, 0)


def test_append_request_direct_reoptimizes_the_final_hub() -> None:
    scenario = ModemsScenario(
        [ModemsAgent("h", 1)],
        [ModemsRequest(1, 2, earliest_pickup=0.0)],
        line_network_two_hubs(),
    )
    journey = ModemsJourney(make_ctx(scenario), "agent_1")
    assert journey.route == ["a_1_h_1", "h_1"]
    appended = journey._append_request_direct("request_1")
    assert appended is not None
    assert appended.route == ["a_1_h_1", "r_1_p_1", "r_1_d_2", "h_2"]
    assert journey.route == ["a_1_h_1", "h_1"]  # stays the same


def line_network_two_hubs() -> RoadNetwork:
    from .builders import line_network

    return line_network([0.0, 10.0, 8.0, 9.0], nr_hubs=2)  # h_1=0, h_2=10


def test_remove_request_restores_idle_journey_and_ignores_foreign_requests() -> None:
    journey = ModemsJourney.from_route(one_request_ctx(), "agent_1", ROUTE_1)
    journey.remove_request("request_99")
    assert journey.route == ROUTE_1
    journey.remove_request("request_1")
    assert journey.route == ["a_1_h_1", "h_1"] and not journey.is_active
    assert journey._is_feasible()


@pytest.mark.parametrize(
    "rho, expected_pickup_start, expected_tau",
    [(4.0, 3.0, 2.0), (5.0, 2.0, 1.0)],
)
def test_remove_request_advances_survivors_by_at_most_their_ride_slack(
    rho: float, expected_pickup_start: float, expected_tau: float
) -> None:
    """
    Unit travel times everywhere. Before removal, surviving request is picked up at t=3
    and delivered at t=7 (ride 4). Removing the first request lets its pickup move
    up to t=1, but only rho-4 of ride slack is available, the rest is waiting
    """
    travel_times = np.ones((7, 7)) - np.eye(7)
    network = RoadNetwork(1, 6, locations=np.zeros((7, 2)), travel_times=travel_times)
    scenario = ModemsScenario(
        [ModemsAgent("h", 1)],
        [
            ModemsRequest(1, 2, service_time=0.0, earliest_pickup=0.0),
            ModemsRequest(3, 4, service_time=0.0, earliest_pickup=0.0, tw_length=1.0),
            ModemsRequest(5, 6, service_time=0.0, earliest_pickup=5.0),
        ],
        network,
    )
    ctx = make_ctx(scenario, rho=rho)
    route = [
        "a_1_h_1",
        "r_1_p_1",
        "r_1_d_2",
        "r_2_p_3",
        "r_3_p_5",
        "r_3_d_6",
        "r_2_d_4",
        "h_1",
    ]
    journey = ModemsJourney.from_route(ctx, "agent_1", route)
    assert journey.states[3].t_start == 3.0 and journey.states[6].t_start == 7.0

    journey.remove_request("request_1")

    pickup = journey.states[journey.request_pickup["request_2"]]
    delivery = journey.states[journey.request_delivery["request_2"]]
    assert pickup.t_start == pytest.approx(expected_pickup_start)
    assert pickup.tau == pytest.approx(expected_tau)
    assert delivery.t_start == pytest.approx(7.0)
    assert journey._is_feasible()


# --------------------------------------------------------------------------------------
# ModemsSolution
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy, objective, selective, expected",
    [
        # closed: T = hub arrival; open: T = last delivery departure
        ("alns", "closed", True, 20.0 + 0.01 * (10 + 14)),
        ("alns", "open", True, 15.0 + 0.01 * (10 + 14)),
        ("milp2", "closed", True, 20.0 + 0.01 * (10 + 14)),
        # early service at t=5: tau = 5, zeta = 2
        ("milp1", "closed", False, 15.0 + 0.01 * (5 + 9) + 2.0 * 5.0),
        ("milp3", "open", False, 10.0 + 0.01 * (5 + 9) + 2.0 * 5.0),
    ],
)
def test_objective_on_a_single_served_request(
    strategy: str, objective: str, selective: bool, expected: float
) -> None:
    ctx = one_request_ctx(strategy, objective=objective, selective=selective)
    solution = served(ctx, ROUTE_1)
    assert solution.objective() == pytest.approx(expected)
    assert solution.objective(include_rejection=False) == pytest.approx(expected)


def test_objective_charges_eta_per_unserved_request_only_when_selective() -> None:
    scenario = line_scenario(TWO_REQUESTS)
    selective = ModemsSolution(make_ctx(scenario, "milp3"))
    non_selective = ModemsSolution(make_ctx(scenario, "milp3", selective=False))
    assert selective.objective() == 2 * selective.ctx.eta
    assert selective.objective(include_rejection=False) == 0.0
    assert non_selective.objective() == 0.0


def test_mission_time_is_the_fleet_maximum_over_active_agents() -> None:
    scenario = line_scenario(
        TWO_REQUESTS,
        agents=[ModemsAgent("h", 1), ModemsAgent("h", 1, time_initial=50.0)],
    )
    ctx = make_ctx(scenario)
    solution = served(ctx, ["a_1_h_1", "r_1_p_1", "r_1_d_4", "h_1"], ["a_2_h_1", "h_1"])
    assert solution.mission_time() == 8.0  # idle agent_2 (t=50) is ignored
    solution.journeys["agent_2"] = ModemsJourney.from_route(
        ctx, "agent_2", ["a_2_h_1", "r_2_p_2", "r_2_d_3", "h_1"]
    )
    solution.accepted.add("request_2")
    assert solution.mission_time() == 56.0


def test_request_bookkeeping_and_copy_independence() -> None:
    ctx = make_ctx(line_scenario(TWO_REQUESTS))
    solution = served(ctx, ["a_1_h_1", "r_1_p_1", "r_1_d_4", "h_1"])
    assert solution.agent_of("request_1") == "agent_1"
    assert solution.agent_of("request_2") is None
    assert solution.pending() == solution.rejected == {"request_2"}

    clone = solution.copy()
    clone.journeys["agent_1"].insert_at("request_2", 1, 1)
    clone.accepted.add("request_2")
    assert solution.journeys["agent_1"].route == [
        "a_1_h_1",
        "r_1_p_1",
        "r_1_d_4",
        "h_1",
    ]
    assert solution.accepted == {"request_1"}
    assert solution.total_energy() < clone.total_energy()


# --------------------------------------------------------------------------------------
# ModemsSolution persistence
# --------------------------------------------------------------------------------------


def test_solution_dict_round_trip(small_scenario: ModemsScenario) -> None:
    solution = greedy_solution(make_ctx(small_scenario))
    restored = ModemsSolution.from_dict(solution.ctx, solution.to_dict())
    assert restored.to_dict() == solution.to_dict()
    assert restored.objective() == solution.objective()


def _corrupt(data: dict, mutation: str) -> dict:
    if mutation == "unknown_agent":
        data["journeys"]["agent_9"] = data["journeys"].pop("agent_2")
    elif mutation == "twice_routed":
        data["journeys"]["agent_2"] = data["journeys"]["agent_1"] | {}
        data["journeys"]["agent_2"]["states"] = [
            {**s, "node": s["node"].replace("a_1_", "a_2_")}
            for s in data["journeys"]["agent_1"]["states"]
        ]
    elif mutation == "accepted_not_routed":
        data["accepted"].append("request_2")
        data["rejected"].remove("request_2")
    elif mutation == "overlap":
        data["rejected"].append("request_1")
    elif mutation == "unknown_request":
        data["rejected"].append("request_9")
    return data


@pytest.mark.parametrize(
    "mutation, match",
    [
        ("unknown_agent", "do not match the ProblemContext"),
        ("twice_routed", "multiple journeys"),
        ("accepted_not_routed", "do not match the loaded journeys"),
        ("overlap", "overlap"),
        ("unknown_request", "unknown requests"),
    ],
)
def test_solution_from_dict_rejects_inconsistent_data(
    mutation: str, match: str
) -> None:
    requests = [
        ModemsRequest(1, 2, earliest_pickup=0.0, request_id="a"),
        ModemsRequest(3, 4, request_id="b"),
    ]
    scenario = line_scenario(
        requests, agents=[ModemsAgent("h", 1), ModemsAgent("h", 1)]
    )
    ctx = make_ctx(scenario)
    solution = served(ctx, ["a_1_h_1", "r_1_p_1", "r_1_d_2", "h_1"], ["a_2_h_1", "h_1"])
    data = _corrupt(solution.to_dict(), mutation)
    with pytest.raises(ValueError, match=match):
        ModemsSolution.from_dict(ctx, data)


# --------------------------------------------------------------------------------------
# ModemsSolutionInfo / ModemsInstance
# --------------------------------------------------------------------------------------


def test_solution_info_parses_status_and_round_trips() -> None:
    info = ModemsSolutionInfo(
        status="optimal", objective=4.0, lower_bound=3.0, upper_bound=4.0
    )
    assert info.status is SolutionStatus.optimal
    assert info.solver_options == {} and info.solver_diagnostics == {}
    assert ModemsSolutionInfo.from_dict(info.to_dict()) == info
    with pytest.raises(ValueError):
        ModemsSolutionInfo(status="solved")


def test_instance_requires_the_solution_context() -> None:
    ctx, other = one_request_ctx(), one_request_ctx()
    with pytest.raises(ValueError, match="same ProblemContext"):
        ModemsInstance(other, ModemsSolution(ctx), ModemsSolutionInfo())


@pytest.mark.parametrize(
    "lower, upper, gap",
    [(15.0, 20.0, 0.25), (None, 20.0, None), (15.0, FLOAT_INF, None), (0.0, 0.0, None)],
)
def test_instance_diagnostics(
    lower: float | None, upper: float | None, gap: float | None
) -> None:
    ctx = make_ctx(line_scenario(TWO_REQUESTS))
    solution = served(ctx, ["a_1_h_1", "r_1_p_1", "r_1_d_4", "h_1"])
    info = ModemsSolutionInfo(
        lower_bound=lower, upper_bound=upper, solver_diagnostics={"n": 1}
    )
    diagnostics = ModemsInstance(ctx, solution, info).diagnostics()
    assert diagnostics["duality_gap"] == gap
    assert diagnostics["solver"] == {"n": 1}
    agent = diagnostics["agents"]["agent_1"]
    assert agent["requests"] == ["request_1"] and not agent["is_idle"]
    assert agent["travel_times"] == [1.0, 3.0, 4.0]
    served_request = diagnostics["requests"]["request_1"]
    assert served_request["assigned_agent"] == "agent_1"
    assert (served_request["pickup_time"], served_request["delivery_time"]) == (
        1.0,
        4.0,
    )
    assert served_request["ride_time"] == 3.0 and served_request["path_time"] == 3.0
    rejected = diagnostics["requests"]["request_2"]
    assert not rejected["is_accepted"] and rejected["pickup_time"] is None


def test_instance_json_round_trip(tmp_path, small_scenario: ModemsScenario) -> None:
    solution = greedy_solution(make_ctx(small_scenario))
    info = ModemsSolutionInfo(
        status="feasible", objective=solution.objective(), solver_name="x"
    )
    instance = ModemsInstance(solution.ctx, solution, info)
    restored = ModemsInstance.from_json(
        instance.to_json(str(tmp_path / "instance.json"))
    )
    assert restored.to_dict() == instance.to_dict()


@pytest.mark.parametrize("strategy, has_soc_plot", [("alns", True), ("milp1", False)])
def test_instance_plot_writes_every_diagnostic_figure(
    tmp_path, strategy: str, has_soc_plot: bool
) -> None:
    ctx = make_ctx(line_scenario(TWO_REQUESTS), strategy)
    instance = ModemsInstance(ctx, greedy_solution(ctx), ModemsSolutionInfo())
    paths = instance.plot(str(tmp_path / "plots"), name="case")
    expected = {"network_plot", "routes_plot", "timing_plot", "load_plot"}
    assert set(paths) == expected | ({"soc_plot"} if has_soc_plot else set())
    assert sorted(os.listdir(tmp_path / "plots")) == sorted(
        os.path.basename(p) for p in paths.values()
    )
