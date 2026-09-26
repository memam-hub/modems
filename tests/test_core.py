"""Unit tests for modems.core: enums, requests, agents, scenarios, problem context"""

from __future__ import annotations

import math

import pytest

import modems  # noqa: F401  (registers every SmartStrEnum subclass)
from modems.core import (
    DEFAULT_PARAMS_OBJ,
    ModemsAgent,
    ModemsRequest,
    ModemsScenario,
    ObjectiveType,
    ProblemContext,
    ProblemType,
    RequestStatus,
    ScenarioSize,
    ScenarioType,
    SmartStrEnum,
    SolverStrategy,
    resolve_problem_type,
    scenario_bucket,
    scenario_size_of,
)
from modems.network import NetworkNodeType

from .builders import line_network, line_scenario, make_ctx

# every SmartStrEnum in the package (importing modems registers all subclasses)
SMART_ENUMS = sorted(SmartStrEnum.__subclasses__(), key=lambda enum: enum.__name__)

# --------------------------------------------------------------------------------------
# SmartStrEnum: one contract, checked for every subclass and member
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "member", [m for enum in SMART_ENUMS for m in enum], ids=lambda m: repr(m)
)
def test_smart_enum_parses_value_name_and_letter_case_insensitively(
    member: SmartStrEnum,
) -> None:
    enum = type(member)
    assert member.letter == member.value[0].upper()
    for alias in (member.value, member.name, member.letter, member.value.upper()):
        assert enum(alias) is member


@pytest.mark.parametrize("enum", SMART_ENUMS)
def test_smart_enum_letters_are_unique(enum: type[SmartStrEnum]) -> None:
    """Letters double as parse aliases and scenario-name codes, so must not collide"""
    letters = [m.letter for m in enum]
    assert len(letters) == len(set(letters))


@pytest.mark.parametrize("value", ["unknown", "", 1, None])
def test_smart_enum_rejects_unknown_values(value: object) -> None:
    with pytest.raises(ValueError):
        ScenarioType(value)


# --------------------------------------------------------------------------------------
# ModemsRequest
# --------------------------------------------------------------------------------------


def test_request_latest_pickup_and_status() -> None:
    r = ModemsRequest(1, 2, earliest_pickup=12.0, tw_length=3.0, status="scheduled")
    assert r.latest_pickup == 15.0
    assert r.status is RequestStatus.scheduled
    assert r.is_scheduled() and not r.is_new()


def test_request_node_names_embed_request_and_station_indices() -> None:
    r = ModemsRequest(node_pickup_index=5, node_delivery_index=9)
    assert r.make_request_name(2) == "request_2"
    assert r.make_pickup_node_name(2) == "r_2_p_5"
    assert r.make_delivery_node_name(2) == "r_2_d_9"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"node_pickup_index": 0},
        {"node_delivery_index": 0},
        {"node_delivery_index": 1},
        {"node_pickup_index": 1.0},
        {"load": 0},
        {"load": True},
        {"service_time": -0.1},
        {"earliest_pickup": -1.0},
        {"tw_length": 0.0},
        {"status": "unknown"},
    ],
)
def test_request_rejects_invalid_fields(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ModemsRequest(**{"node_pickup_index": 1, "node_delivery_index": 2, **kwargs})


def test_request_fallback_id_is_a_deterministic_function_of_its_fields() -> None:
    fields = dict(node_pickup_index=1, node_delivery_index=2, earliest_pickup=10.0)
    assert ModemsRequest(**fields).request_id == ModemsRequest(**fields).request_id
    shifted = ModemsRequest(**{**fields, "earliest_pickup": 11.0})
    assert shifted.request_id != ModemsRequest(**fields).request_id
    assert ModemsRequest(**fields, request_id="rid").request_id == "rid"


@pytest.mark.parametrize("request_id", ["rid", None])
def test_request_dict_round_trip(request_id: str | None) -> None:
    r = ModemsRequest(3, 7, 4, 2.0, 20.0, 6.0, "scheduled", request_id=request_id)
    restored = ModemsRequest.from_dict(r.to_dict())
    assert restored.to_dict() == r.to_dict()
    assert restored.request_id == r.request_id


def test_request_copy_is_independent_and_keeps_id() -> None:
    r = ModemsRequest(1, 2, request_id="rid")
    clone = r.copy()
    clone.earliest_pickup = 99.0
    assert r.earliest_pickup == 10.0
    assert clone.request_id == "rid"


# --------------------------------------------------------------------------------------
# ModemsAgent
# --------------------------------------------------------------------------------------


def test_agent_duration_max_is_worst_case_driving_time() -> None:
    """D^k = (sigma - sigma_min) / (alpha + beta * Q)"""
    agent = ModemsAgent(
        "h", 1, load_max=6, soc_min_operational=0.2, soc_initial=0.8,
        soc_alpha=0.01, soc_beta=0.001,
    )  # fmt: skip
    assert math.isclose(agent.duration_max, 0.6 / 0.016)


@pytest.mark.parametrize(
    "node_type, stored",
    [
        ("h", NetworkNodeType.hub),
        ("s", NetworkNodeType.station),
        ("p", NetworkNodeType.station),
        ("d", NetworkNodeType.station),
    ],
)
def test_agent_start_node_type_is_normalized_to_hub_or_station(
    node_type: str, stored: NetworkNodeType
) -> None:
    agent = ModemsAgent(node_type, 2)
    assert agent.node_type is stored
    assert agent.make_node_name(1) == f"a_1_{stored}_2"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"node_type": "a"},
        {"node_type": "r"},
        {"node_index": 0},
        {"time_initial": -1.0},
        {"load_max": 0},
        {"load_max": True},
        {"soc_min_operational": 0.0},
        {"soc_min_operational": 0.5, "soc_initial": 0.4},
        {"soc_initial": 1.1},
        {"soc_alpha": 0.0},
        {"soc_beta": -1e-3},
    ],
)
def test_agent_rejects_invalid_fields(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ModemsAgent(**{"node_type": "h", "node_index": 1, **kwargs})


def test_agent_fallback_id_is_a_deterministic_function_of_its_fields() -> None:
    assert ModemsAgent("h", 1).agent_id == ModemsAgent("h", 1).agent_id
    assert ModemsAgent("h", 1).agent_id != ModemsAgent("h", 2).agent_id
    assert ModemsAgent("h", 1, agent_id="k").agent_id == "k"


@pytest.mark.parametrize("agent_id", ["k", None])
def test_agent_dict_round_trip(agent_id: str | None) -> None:
    agent = ModemsAgent("s", 4, 3.0, 8, 0.3, 0.9, 0.005, 0.0002, agent_id=agent_id)
    restored = ModemsAgent.from_dict(agent.to_dict())
    assert restored.to_dict() == agent.to_dict()


# --------------------------------------------------------------------------------------
# Problem taxonomy: ProblemType, SolverStrategy, resolve_problem_type
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ptype, objective, selective",
    [
        (ProblemType.open_non_selective, ObjectiveType.open, False),
        (ProblemType.open_selective, ObjectiveType.open, True),
        (ProblemType.closed_non_selective, ObjectiveType.closed, False),
        (ProblemType.closed_selective, ObjectiveType.closed, True),
    ],
)
def test_problem_type_decomposes_and_recomposes(
    ptype: ProblemType, objective: ObjectiveType, selective: bool
) -> None:
    assert ProblemType.objective_of(ptype) is objective
    assert ProblemType.is_open(ptype) == (objective is ObjectiveType.open)
    assert ProblemType.is_closed(ptype) == (objective is ObjectiveType.closed)
    assert ProblemType.is_selective(ptype) == selective
    assert ProblemType.is_non_selective(ptype) == (not selective)
    assert ProblemType.from_parts(objective.letter, selective) is ptype


@pytest.mark.parametrize(
    "strategy, soft_tw, extended_soc, supported",
    [
        (SolverStrategy.milp1, True, False, (False,)),
        (SolverStrategy.milp2, False, True, (True,)),
        (SolverStrategy.milp3, True, True, (True, False)),
        (SolverStrategy.alns, True, True, (True,)),
    ],
)
def test_solver_strategy_capabilities(
    strategy: SolverStrategy,
    soft_tw: bool,
    extended_soc: bool,
    supported: tuple[bool, ...],
) -> None:
    assert strategy.has_soft_tw() == soft_tw
    assert strategy.has_extended_soc() == extended_soc
    assert strategy.has_selectivity() == (True in supported)
    assert strategy.supported_selectivity() == supported


@pytest.mark.parametrize("objective", list(ObjectiveType))
@pytest.mark.parametrize("strategy", list(SolverStrategy))
def test_resolve_problem_type_default_and_explicit_selectivity(
    strategy: SolverStrategy, objective: ObjectiveType
) -> None:
    supported = strategy.supported_selectivity()
    default = resolve_problem_type(strategy, objective)
    assert default is ProblemType.from_parts(objective, supported[0])
    for selective in (True, False):
        if selective in supported:
            ptype = resolve_problem_type(strategy.value, objective.value, selective)
            assert ptype is ProblemType.from_parts(objective, selective)
        else:
            with pytest.raises(ValueError, match="does not support"):
                resolve_problem_type(strategy, objective, selective)


# --------------------------------------------------------------------------------------
# Scenario size buckets
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "nr_agents, nr_requests, size",
    [
        (1, 6, ScenarioSize.small),
        (1, 7, ScenarioSize.medium),
        (2, 10, ScenarioSize.medium),
        (2, 11, ScenarioSize.large),
        (3, 1, ScenarioSize.large),
        (9, 99, ScenarioSize.large),
    ],
)
def test_scenario_size_of_uses_bucket_upper_bounds(
    nr_agents: int, nr_requests: int, size: ScenarioSize
) -> None:
    assert scenario_size_of(nr_agents, nr_requests) is size


def test_scenario_buckets_are_consistent_with_size_of() -> None:
    for size in ScenarioSize:
        nr_agents, (lo, hi) = scenario_bucket(size.letter)
        assert lo <= hi
        assert scenario_size_of(nr_agents, hi) is size


# --------------------------------------------------------------------------------------
# ModemsScenario
# --------------------------------------------------------------------------------------


def test_scenario_rejects_out_of_network_indices_and_duplicate_ids() -> None:
    network = line_network([0.0, 1.0, 2.0])  # 1 hub, 2 stations
    hub_agent = ModemsAgent("h", 1)
    with pytest.raises(ValueError, match="exceeds"):
        ModemsScenario([ModemsAgent("h", 2)], [], network)
    with pytest.raises(ValueError, match="exceeds"):
        ModemsScenario([ModemsAgent("s", 3)], [], network)
    with pytest.raises(ValueError, match="pickup"):
        ModemsScenario([hub_agent], [ModemsRequest(3, 1)], network)
    with pytest.raises(ValueError, match="delivery"):
        ModemsScenario([hub_agent], [ModemsRequest(1, 3)], network)
    twin = ModemsRequest(1, 2, request_id="same")
    with pytest.raises(ValueError, match="request_id"):
        ModemsScenario([hub_agent], [twin, twin.copy()], network)
    agents = [ModemsAgent("h", 1, load_max=5), ModemsAgent("h", 1, load_max=8)]
    ModemsScenario(agents, [ModemsRequest(1, 2, load=8)], network)
    with pytest.raises(ValueError, match="largest agent capacity"):
        ModemsScenario(agents, [ModemsRequest(1, 2, load=9)], network)
    # without agents nothing can be served, and every request is simply rejected
    ModemsScenario([], [ModemsRequest(1, 2, load=99)], network)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda s: setattr(s.requests[0], "load", 7), "largest agent capacity"),
        (lambda s: setattr(s.requests[0], "node_delivery_index", 9), "delivery index"),
        (lambda s: setattr(s.agents[0], "node_index", 2), "agent node index"),
        (lambda s: s.requests.append(s.requests[0].copy()), "request_id"),
    ],
)
def test_context_revalidates_the_scenario_on_creation(mutate, match: str) -> None:
    scenario = line_scenario([ModemsRequest(1, 2)])
    make_ctx(scenario)
    mutate(scenario)
    with pytest.raises(ValueError, match=match):
        make_ctx(scenario)


def test_scenario_owns_deep_copies_of_its_inputs() -> None:
    agent, request = ModemsAgent("h", 1), ModemsRequest(1, 2)
    scenario = line_scenario([request], [agent])
    agent.time_initial, request.load = 50.0, 3
    assert scenario.agents[0].time_initial == 0.0
    assert scenario.requests[0].load == 1


def test_scenario_partitions_requests_by_status() -> None:
    requests = [
        ModemsRequest(1, 2, status="scheduled", request_id="s1"),
        ModemsRequest(2, 3, request_id="n1"),
        ModemsRequest(3, 4, request_id="n2"),
    ]
    scenario = line_scenario(requests)
    assert [r.request_id for r in scenario.new_requests] == ["n1", "n2"]
    assert [r.request_id for r in scenario.scheduled_requests] == ["s1"]


def test_scenario_name_encodes_classification_and_counts() -> None:
    scenario = ModemsScenario(
        [ModemsAgent("h", 1)],
        [ModemsRequest(1, 2)],
        line_network([0.0, 1.0, 2.0, 3.0], nr_hubs=2),
        type="clustered",
        timing="p",
    )
    assert scenario.make_scenario_name(i_rep=4) == "S_SCP4_a1_r1_s2_h2"


def test_scenario_json_round_trip(tmp_path, small_scenario: ModemsScenario) -> None:
    path = str(tmp_path / "scenario.json")
    small_scenario.to_json(path)
    restored = ModemsScenario.from_json(path)
    assert restored.to_dict() == small_scenario.to_dict()
    assert (restored.size, restored.type, restored.timing) == (
        small_scenario.size,
        small_scenario.type,
        small_scenario.timing,
    )


# --------------------------------------------------------------------------------------
# ProblemContext
# --------------------------------------------------------------------------------------


def test_context_lookup_tables_on_a_hand_built_scenario() -> None:
    scenario = line_scenario(
        [
            ModemsRequest(2, 5, load=3, service_time=0.5, earliest_pickup=8.0),
            ModemsRequest(4, 1, load=1, service_time=1.0, earliest_pickup=20.0),
        ],
        agents=[ModemsAgent("s", 3, time_initial=2.0)],
    )
    ctx = make_ctx(scenario)
    assert ctx.agent_names == ["agent_1"]
    assert ctx.agent_initial_node == {"agent_1": "a_1_s_3"}
    assert ctx.request_names == ["request_1", "request_2"]
    assert ctx.pickup_node == {"request_1": "r_1_p_2", "request_2": "r_2_p_4"}
    assert ctx.delivery_node == {"request_1": "r_1_d_5", "request_2": "r_2_d_1"}
    assert ctx.node_load == {"r_1_p_2": 3, "r_1_d_5": -3, "r_2_p_4": 1, "r_2_d_1": -1}
    assert ctx.node_earliest_p == {"r_1_p_2": 8.0, "r_2_p_4": 20.0}
    assert ctx.node_latest_p == {"r_1_p_2": 13.0, "r_2_p_4": 25.0}
    assert ctx.node_t_service["r_1_d_5"] == 0.5
    assert ctx.final_depot_names == ["h_1"]
    # line positions: hub at 0, station i at i
    assert ctx.t_travel("a_1_s_3", "r_1_d_5") == 2.0
    assert ctx.t_travel("r_2_d_1", "h_1") == 1.0
    assert ctx.t_travel("r_1_p_2", "r_1_p_2") == 0.0
    assert ctx.request_of("r_2_p_4") == "request_2"
    assert ctx.request_of("h_1") is None
    assert ctx.is_pickup("r_1_p_2") and not ctx.is_pickup("r_1_d_5")
    assert ctx.is_delivery("r_1_d_5") and ctx.is_station("r_1_d_5")
    assert not ctx.is_station("h_1")


def test_context_nearest_final_depot() -> None:
    scenario = ModemsScenario(
        [ModemsAgent("h", 1)],
        [ModemsRequest(1, 3)],
        line_network([0.0, 10.0, 1.0, 5.0, 9.0], nr_hubs=2),  # hubs at 0 and 10
    )
    ctx = make_ctx(scenario)
    assert ctx.nearest_final_depot("r_1_p_1") == "h_1"
    assert ctx.nearest_final_depot("r_1_d_3") == "h_2"


def test_context_merges_model_params_over_defaults() -> None:
    ctx = make_ctx(line_scenario([ModemsRequest(1, 2)]), rho=3.0)
    assert ctx.model_params == {**DEFAULT_PARAMS_OBJ, "rho": 3.0}
    assert ctx.rho == 3.0 and ctx.eta == DEFAULT_PARAMS_OBJ["eta"]


@pytest.mark.parametrize(
    "params, match",
    [
        ({"eps": 0.0}, "eps"),
        ({"eps": 2.0, "zeta": 2.0}, "eps"),
        ({"zeta": 200.0}, "eps"),
        ({"rho": 0.99}, "rho"),
        ({"omega": 0.5}, "omega"),
    ],
)
def test_context_rejects_invalid_model_params(params: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        make_ctx(line_scenario([ModemsRequest(1, 2)]), **params)


@pytest.mark.parametrize("ptype", list(ProblemType))
def test_context_flags_and_dict_round_trip(
    ptype: ProblemType, small_scenario: ModemsScenario
) -> None:
    ctx = ProblemContext(small_scenario, ptype, "milp3", {"extra_option": True})
    assert ctx.is_open() == ProblemType.is_open(ptype)
    assert ctx.is_selective() == ProblemType.is_selective(ptype)
    rebuilt = ProblemContext.from_dict(ctx.to_dict())
    assert rebuilt.to_dict() == ctx.to_dict()
    assert rebuilt.travel_times == ctx.travel_times
