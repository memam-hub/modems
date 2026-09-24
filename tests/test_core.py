from __future__ import annotations

import math

import pytest

from modems.core import (
    ModemsAgent,
    ModemsRequest,
    ModemsScenario,
    ProblemContext,
    ProblemType,
    RequestStatus,
    ScenarioSize,
    ScenarioTiming,
    ScenarioType,
    SolverStrategy,
)
from modems.generator import ModemsScenarioGenerator
from modems.network import NetworkNodeType

# --------------------------------------------------------------------------------------
# RequestStatus
# --------------------------------------------------------------------------------------


def test_request_status_classification() -> None:
    """RequestStatus members equal and stringify to their literal values"""
    assert RequestStatus.new == "new"
    assert RequestStatus.scheduled == "scheduled"
    assert str(RequestStatus.new) == "new"
    assert str(RequestStatus.scheduled) == "scheduled"


# --------------------------------------------------------------------------------------
# ModemsRequest
# --------------------------------------------------------------------------------------


def test_request_latest_pickup_is_earliest_plus_tw_length() -> None:
    """latest_pickup is inferred as earliest_pickup + tw_length, not stored"""
    r = ModemsRequest(
        node_pickup_index=1, node_delivery_index=2, earliest_pickup=10.0, tw_length=5.0
    )
    assert r.latest_pickup == 15.0


def test_request_status_methods() -> None:
    """is_new()/is_scheduled() are mutually exclusive per RequestStatus"""
    r_new = ModemsRequest(
        node_pickup_index=1, node_delivery_index=2, status=RequestStatus.new
    )
    r_sched = ModemsRequest(
        node_pickup_index=1, node_delivery_index=2, status=RequestStatus.scheduled
    )
    assert r_new.is_new() and not r_new.is_scheduled()
    assert r_sched.is_scheduled() and not r_sched.is_new()


def test_request_to_dict_from_dict_round_trip() -> None:
    """to_dict()/from_dict() preserve every field losslessly"""
    r = ModemsRequest(
        node_pickup_index=3,
        node_delivery_index=7,
        load=4,
        service_time=2.0,
        earliest_pickup=20.0,
        tw_length=5.0,
        status=RequestStatus.scheduled,
    )
    d = r.to_dict()
    r2 = ModemsRequest.from_dict(d)
    assert r2.to_dict() == d
    assert r2.node_pickup_index == 3
    assert r2.node_delivery_index == 7
    assert r2.load == 4
    assert r2.earliest_pickup == 20.0
    assert r2.tw_length == 5.0
    assert r2.status == RequestStatus.scheduled


def test_request_node_names_are_1_based_and_use_own_indices() -> None:
    """Pickup/delivery node names embed the request index and station index"""
    r = ModemsRequest(node_pickup_index=5, node_delivery_index=9)
    assert r.make_pickup_node_name(2) == "r_2_p_5"
    assert r.make_delivery_node_name(2) == "r_2_d_9"


def test_request_copy_is_independent() -> None:
    """copy() returns a deep copy; mutating it does not impact the original"""
    r = ModemsRequest(node_pickup_index=1, node_delivery_index=2, load=3)
    r2 = r.copy()
    r2.load = 99
    assert r.load == 3


# --------------------------------------------------------------------------------------
# ModemsAgent
# --------------------------------------------------------------------------------------


def test_agent_duration_max_formula() -> None:
    """duration_max = (soc_initial - soc_min) / (alpha + beta * load_max)"""
    agent = ModemsAgent(
        node_type=NetworkNodeType.hub,
        node_index=1,
        load_max=6,
        soc_min_operational=0.2,
        soc_initial=0.8,
        soc_alpha=0.01,
        soc_beta=0.001,
    )
    expected = (0.8 - 0.2) / (0.01 + 0.001 * 6)
    assert math.isclose(agent.duration_max, expected)


def test_agent_to_dict_from_dict_round_trip() -> None:
    """to_dict()/from_dict() preserve every field losslessly"""
    agent = ModemsAgent(
        node_type=NetworkNodeType.station, node_index=4, load_max=8, soc_initial=0.9
    )
    d = agent.to_dict()
    agent2 = ModemsAgent.from_dict(d)
    assert agent2.to_dict() == d
    assert agent2.node_index == 4
    assert agent2.load_max == 8


def test_agent_node_name_uses_own_type_and_index() -> None:
    """The agent's start-node name embeds its own node type and index"""
    agent = ModemsAgent(node_type=NetworkNodeType.hub, node_index=2, load_max=6)
    assert agent.make_node_name(1) == "a_1_h_2"


# --------------------------------------------------------------------------------------
# agent_id (persistent, deterministic, unique identifier -- mirrors request_id)
# --------------------------------------------------------------------------------------


def test_agent_id_fallback_is_deterministic_given_same_fields() -> None:
    """Two agents with identical fields and no explicit id collide (by design)"""
    a_args = dict(
        node_type=NetworkNodeType.hub,
        node_index=1,
        time_initial=0.0,
        load_max=6,
        soc_initial=0.9,
    )
    a1 = ModemsAgent(**a_args)
    a2 = ModemsAgent(**a_args)
    assert a1.agent_id == a2.agent_id  # deterministic fallback (same fields -> hash)


def test_agent_id_explicit_is_preserved() -> None:
    """An explicitly supplied agent_id is kept verbatim"""
    a = ModemsAgent(
        node_type=NetworkNodeType.hub, node_index=1, agent_id="my-custom-agent"
    )
    assert a.agent_id == "my-custom-agent"


def test_agent_id_survives_copy_and_mutation() -> None:
    """agent_id stays stable across copy() and field mutation"""
    a = ModemsAgent(node_type=NetworkNodeType.hub, node_index=1, agent_id="stable-id")
    a2 = a.copy()
    a2.time_initial = 123.0  # simulate the epoch-advance mutation pattern
    a2.soc_initial = 0.5
    assert a2.agent_id == "stable-id"


def test_agent_id_included_in_to_dict_round_trip() -> None:
    """agent_id is serialized and restored by to_dict()/from_dict()"""
    a = ModemsAgent(node_type=NetworkNodeType.hub, node_index=1, agent_id="abc123")
    d = a.to_dict()
    assert d["agent_id"] == "abc123"
    a2 = ModemsAgent.from_dict(d)
    assert a2.agent_id == "abc123"


# --------------------------------------------------------------------------------------
# request_id (persistent, deterministic, unique identifier)
# --------------------------------------------------------------------------------------


def test_request_id_fallback_is_deterministic_given_same_fields() -> None:
    """Two requests with identical fields and no explicit id collide (by design)"""
    r_args = dict(
        node_pickup_index=int(1),
        node_delivery_index=int(2),
        load=int(3),
        service_time=float(1.0),
        earliest_pickup=float(10.0),
        tw_length=float(5.0),
    )
    r1 = ModemsRequest(**r_args)
    r2 = ModemsRequest(**r_args)
    assert (
        r1.request_id == r2.request_id
    )  # deterministic fallback (same fields -> same hash)


def test_request_id_explicit_is_preserved() -> None:
    """An explicitly supplied request_id is kept verbatim"""
    r = ModemsRequest(
        node_pickup_index=1, node_delivery_index=2, request_id="my-custom-id"
    )
    assert r.request_id == "my-custom-id"


def test_request_id_survives_copy_and_mutation() -> None:
    """request_id stays stable across copy() and field mutation"""
    r = ModemsRequest(
        node_pickup_index=1, node_delivery_index=2, request_id="stable-id"
    )
    r2 = r.copy()
    r2.earliest_pickup = (
        999.0  # simulate the epoch-shift mutation done in rolling horizon
    )
    r2.status = RequestStatus("scheduled")
    assert r2.request_id == "stable-id"


def test_request_id_included_in_to_dict_round_trip() -> None:
    """request_id is serialized and restored by to_dict()/from_dict()"""
    r = ModemsRequest(node_pickup_index=1, node_delivery_index=2, request_id="abc123")
    d = r.to_dict()
    assert d["request_id"] == "abc123"
    r2 = ModemsRequest.from_dict(d)
    assert r2.request_id == "abc123"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"node_pickup_index": 0, "node_delivery_index": 1},
        {"node_pickup_index": 1, "node_delivery_index": 1, "load": 0},
        {"node_pickup_index": 1, "node_delivery_index": 1, "service_time": -1},
        {"node_pickup_index": 1, "node_delivery_index": 1, "tw_length": 0},
    ],
)
def test_request_rejects_values_outside_theory_domain(kwargs: dict) -> None:
    """__init__ raises ValueError for any field outside its valid domain"""
    with pytest.raises(ValueError):
        ModemsRequest(**kwargs)


def test_agent_rejects_invalid_soc_interval() -> None:
    """__init__ raises ValueError when soc_initial < soc_min_operational"""
    with pytest.raises(ValueError, match="soc_initial"):
        ModemsAgent(
            node_type=NetworkNodeType.hub,
            node_index=1,
            soc_min_operational=0.5,
            soc_initial=0.4,
        )


# --------------------------------------------------------------------------------------
# ProblemType / SolverStrategy
# --------------------------------------------------------------------------------------


def test_problem_type_open_closed() -> None:
    """is_open()/is_closed() classify each of the four ProblemType values"""
    assert ProblemType.is_open(ProblemType.open_selective)
    assert ProblemType.is_open(ProblemType.open_non_selective)
    assert not ProblemType.is_open(ProblemType.closed_selective)
    assert ProblemType.is_closed(ProblemType.closed_non_selective)


def test_problem_type_selective_non_selective() -> None:
    """is_selective()/is_non_selective() classify each ProblemType value"""
    assert ProblemType.is_selective(ProblemType.closed_selective)
    assert not ProblemType.is_selective(ProblemType.closed_non_selective)
    assert ProblemType.is_non_selective(ProblemType.open_non_selective)


def test_kappa_feasibility_flags() -> None:
    """Each SolverStrategy reports the correct soft-TW/selectivity/SoC flags"""
    assert SolverStrategy.milp1.has_soft_tw()
    assert not SolverStrategy.milp1.has_selectivity()
    assert not SolverStrategy.milp1.has_extended_soc()

    assert not SolverStrategy.milp2.has_soft_tw()
    assert SolverStrategy.milp2.has_selectivity()
    assert SolverStrategy.milp2.has_extended_soc()

    assert SolverStrategy.milp3.has_soft_tw()
    assert SolverStrategy.milp3.has_selectivity()
    assert SolverStrategy.milp3.has_extended_soc()

    assert SolverStrategy.alns.has_soft_tw()
    assert SolverStrategy.alns.has_selectivity()
    assert SolverStrategy.alns.has_extended_soc()


# --------------------------------------------------------------------------------------
# ScenarioType / ScenarioTiming / ScenarioSize
# --------------------------------------------------------------------------------------


def test_scenario_enums_are_string_enums_with_letter_values() -> None:
    """
    ScenarioType/ScenarioTiming/ScenarioSize values are the indicative capital
    letters used in scenario naming; .name is the full descriptive word
    """
    assert ScenarioType.random == "R"
    assert ScenarioType.random.name == "random"
    assert ScenarioTiming.tight == "T"
    assert ScenarioTiming.tight.name == "tight"
    assert ScenarioSize.small == "S"
    assert ScenarioSize.small.name == "small"
    assert str(ScenarioType.clustered) == "C"
    assert str(ScenarioTiming.loose) == "L"


# --------------------------------------------------------------------------------------
# ModemsScenario
# --------------------------------------------------------------------------------------


def test_scenario_to_dict_from_dict_round_trip(small_scenario: ModemsScenario) -> None:
    """to_dict()/from_dict() reconstruct an identical scenario"""
    d = small_scenario.to_dict()
    scenario2 = ModemsScenario.from_dict(d)
    assert scenario2.to_dict() == d


def test_scenario_to_dict_properties_use_full_names(
    small_scenario: ModemsScenario,
) -> None:
    """
    to_dict()'s properties use the full enum member name (e.g., 'random'), not
    the single-letter value ('R') used in make_scenario_name(); from_dict() must
    parse them back the same way (by name, not by value)
    """
    d = small_scenario.to_dict()
    properties = d["properties"]
    assert properties["type"] == small_scenario.type.name
    assert properties["timing"] == small_scenario.timing.name
    assert properties["size"] == small_scenario.size.name
    assert len(properties["type"]) > 1  # a full word, not the single-letter value
    scenario2 = ModemsScenario.from_dict(d)
    assert scenario2.type == small_scenario.type
    assert scenario2.timing == small_scenario.timing
    assert scenario2.size == small_scenario.size


def test_scenario_new_and_scheduled_request_properties() -> None:
    """new_requests/scheduled_requests correctly filter by RequestStatus"""
    gen = ModemsScenarioGenerator(seed=1)
    scenario = gen.generate_random_scenario(nr_agents=1, nr_requests=5, nr_scheduled=2)
    assert len(scenario.new_requests) == 3
    assert len(scenario.scheduled_requests) == 2
    assert all(r.is_new() for r in scenario.new_requests)
    assert all(r.is_scheduled() for r in scenario.scheduled_requests)


def test_scenario_name_reflects_size() -> None:
    """make_scenario_name() embeds the agent and request counts"""
    gen = ModemsScenarioGenerator(seed=1)
    scenario = gen.generate_random_scenario(nr_agents=2, nr_requests=4)
    name = scenario.make_scenario_name(i_rep=0)
    assert "a2" in name and "r4" in name


def test_scenario_rejects_duplicate_request_ids() -> None:
    """__init__ raises ValueError when two requests share a request_id"""
    from modems.network import RoadNetwork

    network = RoadNetwork(nr_hubs=1, nr_stations=4)
    duplicate = ModemsRequest(1, 2, request_id="same-id")
    with pytest.raises(ValueError, match="request_id"):
        ModemsScenario(
            agents=[ModemsAgent(NetworkNodeType.hub, 1)],
            requests=[duplicate, duplicate.copy()],
            network=network,
        )


# --------------------------------------------------------------------------------------
# ProblemContext
# --------------------------------------------------------------------------------------


def test_problem_context_builds_consistent_node_maps(
    small_scenario: ModemsScenario, default_model_params: dict
) -> None:
    """Pickup/delivery node maps agree with is_pickup/is_delivery/request_of"""
    ctx = ProblemContext(
        small_scenario,
        ProblemType.closed_selective,
        SolverStrategy.alns,
        default_model_params,
    )
    assert len(ctx.agent_names) == 2
    assert len(ctx.request_names) == 3
    for r in ctx.request_names:
        p, d = ctx.pickup_node[r], ctx.delivery_node[r]
        assert ctx.is_pickup(p)
        assert ctx.is_delivery(d)
        assert ctx.request_of(p) == r
        assert ctx.request_of(d) == r
    for k in ctx.agent_names:
        start = ctx.agent_initial_node[k]
        assert any(i == start or j == start for i, j in ctx.travel_times)


def test_problem_context_travel_time_symmetric(
    small_scenario: ModemsScenario, default_model_params: dict
) -> None:
    """t_travel is symmetric on this network and zero for a node to itself"""
    ctx = ProblemContext(
        small_scenario,
        ProblemType.closed_selective,
        SolverStrategy.alns,
        default_model_params,
    )
    r = ctx.request_names[0]
    p, d = ctx.pickup_node[r], ctx.delivery_node[r]
    assert math.isclose(ctx.t_travel(p, d), ctx.t_travel(d, p))
    assert ctx.t_travel(p, p) == 0.0


def test_problem_context_to_dict_from_dict_round_trip(
    small_scenario: ModemsScenario, default_model_params: dict
) -> None:
    """to_dict()/from_dict() rebuild every derived lookup table identically"""
    ctx = ProblemContext(
        small_scenario,
        ProblemType.closed_selective,
        SolverStrategy.alns,
        {**default_model_params, "extra_option": True},
    )
    rebuilt = ProblemContext.from_dict(ctx.to_dict())
    assert rebuilt.to_dict() == ctx.to_dict()


def test_problem_context_rejects_invalid_penalty_ordering(
    small_scenario: ModemsScenario,
) -> None:
    """__init__ raises ValueError unless 0 < eps < zeta < eta"""
    with pytest.raises(ValueError, match="eps"):
        ProblemContext(
            small_scenario,
            ProblemType.closed_selective,
            SolverStrategy.alns,
            {"eps": 1.0, "zeta": 1.0, "eta": 100.0, "rho": 2.0},
        )


def test_problem_context_rejects_rho_below_one(
    small_scenario: ModemsScenario,
) -> None:
    """__init__ raises ValueError when rho < 1 (max-ride-time multiplier)"""
    with pytest.raises(ValueError, match="rho"):
        ProblemContext(
            small_scenario,
            ProblemType.closed_selective,
            SolverStrategy.alns,
            {"eps": 0.01, "zeta": 1.0, "eta": 100.0, "rho": 0.5},
        )
