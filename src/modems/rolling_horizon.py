from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .algorithms import greedy_complete, preprocess
from .alns import ModemsAlns
from .benchmark import stable_seed
from .core import (
    DEFAULT_BASE_SEED,
    ModemsAgent,
    ModemsRequest,
    ModemsScenario,
    ProblemContext,
    ProblemType,
    RequestStatus,
    SolverStrategy,
)
from .generator import ModemsScenarioGenerator
from .milp import DEFAULT_MILP_SOLVER_DATA, MilpType, ModemsMilp, SolverConfigType
from .network import NetworkNodeName, NetworkNodeType, RoadNetwork
from .solution import (
    DEFAULT_MILP_TIMELIMIT,
    DEFAULT_PARAMS_ALNS,
    DEFAULT_PARAMS_MILP,
    FLOAT_TOL,
    ModemsInstance,
    ModemsJourney,
    ModemsSolution,
    NodeState,
    SolutionStatus,
)

# Charging model: two-segment piecewise linear splitting at 80% ([0.0-0.8], [0.8-1.0])
# calibrated to a measured 5h 0->100% AC charge with an assumed 2:1 rate
CHARGE_RATE_HOUR = (0.24, 0.12)  # SoC/hour, ([0.0-0.8], [0.8-1.0])
SOC_CHARGE_TARGET = 0.80

# robustness margin (epsilon_sigma) to avoid re-routing agents with relatively low SoC
SOC_ROBUSTNESS_MARGIN = 0.10

# periodic replanning interval (minutes) for invoking planning epochs
DEFAULT_REPLANNING_INTERVAL = 10.0

# W: waiting-time threshold (minutes) for retaining/releasing requests
REPLANNING_THRESHOLD_W = 10.0

# P: full planning horizon (minutes), include submitted requests within this timeframe
PLANNING_HORIZON_P = 60.0


def charging_duration_min(soc_from: float, soc_to: float) -> float:
    """Get required charging duration (min) to reach soc_to starting at soc_from"""
    soc_from = max(0.0, min(1.0, soc_from))
    soc_to = max(0.0, min(1.0, soc_to))
    if soc_to <= soc_from:
        return 0.0
    seg_1 = max(0.0, min(soc_to, 0.8) - min(soc_from, 0.8)) / CHARGE_RATE_HOUR[0]
    seg_2 = max(0.0, max(soc_to, 0.8) - max(soc_from, 0.8)) / CHARGE_RATE_HOUR[1]
    return 60.0 * (seg_1 + seg_2)


class AgentOperationalState(StrEnum):
    """Agent operational state, as specified in the operational workflow (flowchart)"""

    out_of_service = "out_of_service"
    available = "available"
    charging = "charging"
    en_route = "en_route"
    waiting = "waiting"


class SolverMode(StrEnum):
    """
    The current solver mode, benchmarking for the 4-solver battle royal, single for
    one solver only (useful for workday) and operational for MILP3-vs-ALNS
    """

    benchmarking = "benchmarking"
    single = "single"
    operational = "operational"


class AgentAdvanceResult:
    """
    The result of advancing an agent journey using the operational workflow (flowchart)
    to match the current problem context

    Attributes:
        node: The starting node v^k
        delta: The starting temporal delay delta^k
        sigma: The starting SoC sigma^k
        operation_state: The agent operational state, check AgentOperationalState
        included: Whether the agent is included in the current
        absorbed: The set of requests absorbed by the agent previous/continued journey
        replan: The set of requests released/freed for scheduling re-optimization
        travel_time: Total agent travel time across the entire workday
    """

    __slots__ = (
        "absorbed",
        "delta",
        "included",
        "node",
        "operation_state",
        "replan",
        "sigma",
        "travel_time",
    )
    node: str
    delta: float
    sigma: float
    operation_state: AgentOperationalState
    included: bool
    absorbed: set[str]
    replan: set[str]
    travel_time: float

    def __init__(
        self,
        node: str,
        delta: float,
        sigma: float,
        operation_state: AgentOperationalState | str,
        included: bool,
        absorbed: set[str],
        replan: set[str],
        travel_time: float = 0.0,
    ) -> None:
        self.node = node
        self.delta = delta
        self.sigma = sigma
        self.operation_state = AgentOperationalState(operation_state)
        self.included = included
        self.absorbed = absorbed
        self.replan = replan
        self.travel_time = travel_time

    def to_agent(self, template_agent: ModemsAgent) -> ModemsAgent:
        """Migrate agent to the new context"""
        node_type = (
            NetworkNodeType.hub
            if NetworkNodeType.is_hub(NetworkNodeName.get_node_type(self.node))
            else NetworkNodeType.station
        )
        node_index = NetworkNodeName.get_node_index(self.node)

        return ModemsAgent(
            node_type=node_type,
            node_index=node_index,
            time_initial=self.delta,
            load_max=template_agent.load_max,
            soc_min_operational=template_agent.soc_min_operational,
            soc_initial=self.sigma,
            soc_alpha=template_agent.soc_alpha,
            soc_beta=template_agent.soc_beta,
            agent_id=template_agent.agent_id,
        )


def _route_requests(
    ctx: ProblemContext,
    route: list[str],
    i_start: int = 0,
    i_end: int | None = None,
) -> set[str]:
    """Get the visited request names in the given route[i_start:i_end]"""
    i_end = len(route) if i_end is None else i_end
    r_names = set()
    for node in route[i_start:i_end]:
        r_name = ctx.request_of(node)
        if r_name is not None:
            r_names.add(r_name)
    return r_names


def _find_next_index(states: list[NodeState], t_elapsed: float) -> int | None:
    """
    Loop on states[1:] to find the next (n+1) node index, which is visited after
    t_elapsed (i.e., t_dep > t_elapsed); None if journey finishes before t_elapsed
    """
    for i in range(1, len(states)):
        st = states[i]
        if st.t_dep > t_elapsed:
            return i
    return None  # the entire journey is completed before t_elapsed


def _resolve_at_depot(
    ctx: ProblemContext,
    agent: ModemsAgent,
    route: list[str],
    state: NodeState,
    idx: int,
    t_elapsed: float,
) -> AgentAdvanceResult:
    """
    Resolve an agent status at a start depot (re-planning all route requests) or at a
    a final depot (absorbing all route requests), then set the availability based on
    the SoC left, robustness margin, and possible charging duration
    """
    node = route[idx]
    sigma = state.phi_arr
    delta = max(0.0, state.t_arr - t_elapsed)
    # either absorb or replan all route requests
    if idx == 0:
        r_absorbed = set()
        r_replan = _route_requests(ctx, route)
    else:
        r_absorbed = _route_requests(ctx, route)
        r_replan = set()
    if sigma >= agent.soc_min_operational + SOC_ROBUSTNESS_MARGIN + FLOAT_TOL:
        return AgentAdvanceResult(
            node,
            delta,
            sigma,
            AgentOperationalState.available,
            True,
            r_absorbed,
            r_replan,
        )
    total_delta = delta + charging_duration_min(sigma, SOC_CHARGE_TARGET)
    included = total_delta < PLANNING_HORIZON_P + FLOAT_TOL
    return AgentAdvanceResult(
        node,
        total_delta,
        SOC_CHARGE_TARGET,
        AgentOperationalState.charging,
        included,
        r_absorbed,
        r_replan,
    )


def _resolve(
    ctx: ProblemContext,
    agent: ModemsAgent,
    route: list[str],
    states: list[NodeState],
    idx: int,
    t_elapsed: float,
) -> AgentAdvanceResult:
    """
    Implementation of the flowchart (workflow) to determine agent initial status and
    availability, extending it for dynamic operation using the actual elapsed time.
    Start at n_{i+1} is route[idx], states[idx]
    """
    node = route[idx]
    state = states[idx]

    if not ctx.is_station(node) and idx > 0:
        # final hub: route complete, everything on it is absorbed
        return _resolve_at_depot(ctx, agent, route, state, idx, t_elapsed)

    # if agent has on-board passengers, loop until the earliest node where all disembark
    z_hat = states[idx - 1].z_dep if idx > 0 else 0
    if z_hat > 0:
        j = idx
        while j < len(states) - 1 and states[j].z_dep != 0:
            j += 1
        r_absorbed = _route_requests(ctx, route, idx, j + 1)
        agent_adv_res = _resolve(ctx, agent, route, states, j + 1, t_elapsed)  # recurse
        agent_adv_res.absorbed |= r_absorbed
        return agent_adv_res

    # agent has no on-board requests, and is either en-route or waiting somewhere.
    # Compile and evaluate (remaining) waiting-time
    remain_t_wait = state.t_wait + state.t_arr - t_elapsed
    if idx == 0:
        if len(route) <= 2:
            remain_t_wait += REPLANNING_THRESHOLD_W  # trivial case, resolve at depot
        else:
            # include travel to the next service node and waiting time there
            remain_t_wait += ctx.t_travel(route[0], route[1]) + states[1].t_wait
    # If there is sufficient waiting time, release all next requests
    if remain_t_wait > REPLANNING_THRESHOLD_W + FLOAT_TOL:
        if not ctx.is_station(node):
            return _resolve_at_depot(ctx, agent, route, state, idx, t_elapsed)
        # resolve at service node, previous requests already absorbed, replan the rest
        delta = max(0.0, state.t_arr - t_elapsed)
        sigma = state.phi_arr
        r_replan = _route_requests(ctx, route, idx, len(route) - 1)
        return AgentAdvanceResult(
            route[idx],
            delta,
            sigma,
            AgentOperationalState.available,
            True,
            set(),
            r_replan,
        )
    # insufficient waiting, retain request and resolve at its successor
    r_absorbed = _route_requests(ctx, route, idx, idx + 1)
    agent_adv_res = _resolve(ctx, agent, route, states, idx + 1, t_elapsed)  # recurse
    agent_adv_res.absorbed |= r_absorbed
    return agent_adv_res


def advance_agent_state(
    ctx: ProblemContext, agent_name: str, journey: ModemsJourney, t_elapsed: float
) -> AgentAdvanceResult:
    """
    Advance one agent state across an elapsed real-time gap, given its previously
    cached ModemsJourney. Returns an AgentAdvanceResult with the new (v^k, delta^k,
    sigma^k), whether the agent is included in K this planning epoch, and the
    absorbed/excluded vs released for scheduling re-opt (R_s) request sets
    """
    agent = ctx.agents[agent_name]
    idx = _find_next_index(journey.states, t_elapsed)
    journey_complete = idx is None
    if journey_complete:
        idx = len(journey.states) - 1  # resolve at the final depot
    agent_adv_res = _resolve(ctx, agent, journey.route, journey.states, idx, t_elapsed)

    # Any request with t_delivery before the search even started is already absorbed
    agent_adv_res.absorbed |= {
        r_name
        for r_name, delivery_index in journey.request_delivery.items()
        if delivery_index < idx
    }

    # travel time actually incurred this planning cycle
    idx_end = journey.route.index(agent_adv_res.node)
    idx_start = 1 if journey_complete else idx
    agent_adv_res.travel_time = sum(
        ctx.t_travel(journey.route[i], journey.route[i + 1])
        for i in range(idx_start - 1, idx_end)
    )
    return agent_adv_res


# --------------------------------------------------------------------------------------
# RollingHorizonSimulator
# --------------------------------------------------------------------------------------


class RequestOutcome(StrEnum):
    """The realized outcome of a submitted request by the end of the workday"""

    accepted = "accepted"
    rejected = "rejected"
    unresolved = "unresolved"  # still pending when the workday ended


@dataclass(slots=True)
class RequestRecord:
    """
    A full-day record of a submitted request: the request itself, when it was submitted
    (offset from workday clock-start), its realized service outcome, and, if accepted,
    the timing metrics: pickup_time/delivery_time/wait/delay/excess_ride_time; they are
    set to None otherwise
    """

    request: ModemsRequest
    submission_time: float
    outcome: RequestOutcome
    pickup_time: float | None = None
    delivery_time: float | None = None
    wait: float | None = None
    delay: float | None = None
    excess_ride_time: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "submission_time": self.submission_time,
            "outcome": self.outcome,
            "pickup_time": self.pickup_time,
            "delivery_time": self.delivery_time,
            "wait": self.wait,
            "delay": self.delay,
            "excess_ride_time": self.excess_ride_time,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RequestRecord:
        return cls(
            request=ModemsRequest.from_dict(data["request"]),
            submission_time=data["submission_time"],
            outcome=RequestOutcome(data["outcome"]),
            pickup_time=data.get("pickup_time"),
            delivery_time=data.get("delivery_time"),
            wait=data.get("wait"),
            delay=data.get("delay"),
            excess_ride_time=data.get("excess_ride_time"),
        )


@dataclass(slots=True)
class AgentNodeVisit:
    """
    One agent node visit with arrival/departure time and SoC. Extract the data from
    NodeState (t_arr/t_start/phi_arr/phi_dep) at the moment when the node is absorbed
    by advance_epoch() using epoch-local time, later shifted to absolute time by
    build_workday_log() using offset from workday clock-start, similar to RequestRecord
    """

    node: str
    arrival_time: float
    departure_time: float
    soc_arrival: float
    soc_departure: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "arrival_time": self.arrival_time,
            "departure_time": self.departure_time,
            "soc_arrival": self.soc_arrival,
            "soc_departure": self.soc_departure,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentNodeVisit:
        return cls(**data)


@dataclass(slots=True)
class EpochLog:
    """
    The outcome of one planning epoch. clock_start/clock_end are absolute offset from
    workday clock-start times specified by WorkdaySimulation. Currently active plan:
    completed_requests and agent_node_visits have (plan_reference_clock)-local time,
    and are later shifted during build_workday_log() to absolute times for readability.
    results is a per-strategy dict of the ModemsInstance data solved this epoch; it is
    opt-in for serialization with (include_results=False default) to have the data live
    while keeping the full workday exports (hundreds of epochs) contained
    """

    clock_start: float | None = None
    clock_end: float | None = None
    # absolute clock time of the currently active plan, set by WorkdaySimulation.run()
    # a plan can be adopted in a previous epoch and still be active/running, so using
    # the epoch clock_start is not necessarily sufficient/correct
    plan_reference_clock: float | None = None
    adopted_strategy: SolverStrategy | None = None
    rejected_new: set[str] = field(default_factory=set)
    completed_requests: dict[str, dict[str, float]] = field(default_factory=dict)
    agent_node_visits: dict[str, list[AgentNodeVisit]] = field(default_factory=dict)
    energy_consumed: dict[str, float] = field(default_factory=dict)
    charging_events: dict[str, dict[str, float]] = field(default_factory=dict)
    agent_travel_time: dict[str, float] = field(default_factory=dict)
    solver_errors: dict[str, str] = field(default_factory=dict)
    results: dict[SolverStrategy, ModemsInstance] | None = None

    def to_dict(self, include_results: bool = False) -> dict[str, Any]:
        data = {
            "clock_start": self.clock_start,
            "clock_end": self.clock_end,
            "plan_reference_clock": self.plan_reference_clock,
            "adopted_strategy": self.adopted_strategy,
            "rejected_new": sorted(self.rejected_new),
            "completed_requests": self.completed_requests,
            "agent_node_visits": {
                agent_id: [v.to_dict() for v in node_visits]
                for agent_id, node_visits in self.agent_node_visits.items()
            },
            "energy_consumed": self.energy_consumed,
            "charging_events": self.charging_events,
            "agent_travel_time": self.agent_travel_time,
            "solver_errors": self.solver_errors,
        }
        if include_results and self.results is not None:
            data["results"] = {str(k): v.to_dict() for k, v in self.results.items()}
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpochLog:
        results = None
        if data.get("results") is not None:
            results = {
                SolverStrategy(k): ModemsInstance.from_dict(v)
                for k, v in data["results"].items()
            }
        return cls(
            clock_start=data.get("clock_start"),
            clock_end=data.get("clock_end"),
            plan_reference_clock=data.get("plan_reference_clock"),
            adopted_strategy=(
                SolverStrategy(data["adopted_strategy"])
                if data.get("adopted_strategy")
                else None
            ),
            rejected_new=set(data.get("rejected_new", [])),
            completed_requests=data.get("completed_requests", {}),
            agent_node_visits={
                agent_id: [AgentNodeVisit.from_dict(v) for v in node_visits]
                for agent_id, node_visits in data.get("agent_node_visits", {}).items()
            },
            energy_consumed=data.get("energy_consumed", {}),
            charging_events=data.get("charging_events", {}),
            agent_travel_time=data.get("agent_travel_time", {}),
            solver_errors=data.get("solver_errors", {}),
            results=results,
        )


def _collect_request_metrics(
    ctx: ProblemContext, journey: ModemsJourney, request_name: str
) -> dict[str, float] | None:
    """
    Realized per-request metrics (waiting, ride time vs. direct time, tardiness)
    from the journey it was actually served on, at the moment it is fully absorbed
    """
    if request_name not in journey.request_pickup:
        return None
    p_idx = journey.request_pickup[request_name]
    d_idx = journey.request_delivery[request_name]
    p_state, d_state = journey.states[p_idx], journey.states[d_idx]
    direct_time = ctx.t_travel(p_state.node, d_state.node)
    ride_time = d_state.t_start - p_state.t_start - ctx.node_t_service[p_state.node]
    return {
        "pickup_time": p_state.t_start,
        "delivery_time": d_state.t_start,
        "waiting_time": p_state.t_wait,
        "delay_time": max(0.0, p_state.t_start - ctx.node_latest_p[p_state.node]),
        "ride_time": ride_time,
        "direct_time": direct_time,
        "excess_ride_time": ride_time - direct_time,
        "excess_ride_pct": (ride_time - direct_time) / direct_time,
    }


class RollingHorizonSimulator:
    """
    Epoch-by-epoch rolling-horizon driver. Request submissions are stubbed and
    advance_epoch() takes an explicit list of new ModemsRequest objects each call.
    Workday request generation and auotare handled separately)
    """

    network: RoadNetwork
    model_params: dict[str, Any]
    solver_mode: SolverMode
    single_solver: SolverStrategy | None
    milp_timelimit: float
    alns_max_iter: int
    solver_name: str
    solver_config_type: SolverConfigType
    agents: dict[str, ModemsAgent]
    agent_templates: dict[str, ModemsAgent]
    parked_agents: dict[str, dict[str, Any]]
    previous_ctx: ProblemContext
    journeys: dict[str, ModemsJourney]
    time_since_last_adoption: float
    rejected_log: list[ModemsRequest]
    epoch_log: list[EpochLog]
    _last_confirmed_node: dict[str, str]
    _energy_checkpoint: dict[str, float]

    def __init__(
        self,
        network: RoadNetwork,
        agents: list[ModemsAgent],
        model_params: dict[str, Any] | None = None,
        solver_mode: SolverMode | str = SolverMode.benchmarking,
        single_solver: SolverStrategy | str | None = None,
        milp_timelimit: float = DEFAULT_MILP_TIMELIMIT,
        alns_max_iter: int = DEFAULT_PARAMS_ALNS["max_iter"],
        solver_name: str = DEFAULT_MILP_SOLVER_DATA[0],
        solver_config_type: SolverConfigType = DEFAULT_MILP_SOLVER_DATA[1],
    ) -> None:
        """
        solver_mode: "benchmarking" (all 4, recorded, best adopted), "operational"
            (MILP3+ALNS race, best adopted), or "single" (exactly one solver decides
            every epoch, used for MILP3-vs-ALNS comparison: run the identical request
            stream on two separate simulators, and compare via summarize_workday()
        single_solver: Optional, required if solver_mode="single". must be
            SolverStrategy.milp3 or SolverStrategy.alns (or their string values)
        """
        single_solver = (
            SolverStrategy(single_solver) if single_solver is not None else None
        )
        if solver_mode not in [n for n in SolverMode]:
            raise ValueError(f"solver_mode must be {[n for n in SolverMode]}'")
        solver_mode = SolverMode(solver_mode)
        if solver_mode == SolverMode.single and single_solver not in (
            SolverStrategy.milp3,
            SolverStrategy.alns,
        ):
            raise ValueError(
                f"single_solver must be {SolverStrategy.milp3} or {SolverStrategy.alns}"
                f"when solver_mode={SolverMode.single}"
            )
        self.network = network
        self.model_params = {**DEFAULT_PARAMS_MILP, **(model_params or {})}
        self.solver_mode = solver_mode
        self.single_solver = single_solver
        self.milp_timelimit = milp_timelimit
        self.alns_max_iter = alns_max_iter
        self.solver_name = solver_name
        self.solver_config_type = solver_config_type

        agent_ids = [a.agent_id for a in agents]
        if len(agent_ids) != len(set(agent_ids)):
            raise ValueError(
                "agents passed to RollingHorizonSimulator must have unique agent_id "
                "values. Provide an explicit agent_id to every ModemsAgent in the "
                "fleet (ModemsScenarioGenerator does this automatically)"
            )

        self.agents = {a.make_agent_name(i): a for i, a in enumerate(agents, start=1)}
        self.agent_templates = {a.agent_id: a.copy() for a in agents}
        empty_scenario = ModemsScenario(
            agents=list(self.agents.values()), requests=[], network=network
        )
        initial_ctx = ProblemContext(
            empty_scenario,
            ProblemType.closed_selective,
            SolverStrategy.alns,
            self.model_params,
        )
        solution, _ = preprocess(initial_ctx)
        if solution is None:
            raise ValueError("initial inactive fleet must be feasible")
        self.previous_ctx = solution.ctx
        self.journeys = dict(solution.journeys)
        self.time_since_last_adoption = 0.0
        self.rejected_log = []
        self.epoch_log = []
        # parked_agents are excluded from the active scenario since their remaining
        # charging time exceeds PLANNING_HORIZON_P, keyed by agent_id. They may appear
        # after a few epochs when the remaining charging duration to SOC_CHARGE_TARGET
        # lies within the PLANNING_HORIZON_P
        self.parked_agents = {}
        # confirmed last visited (arrived + departed) node name per agent_id
        self._last_confirmed_node = {}
        # SoC checkpoint per agent_id, used to measure consumed energy across epochs
        self._energy_checkpoint = {a.agent_id: a.soc_initial for a in agents}

    def _advance_active_agents(self) -> dict[str, AgentAdvanceResult]:
        """Advance all active agents (states)"""
        agent_adv_results: dict[str, AgentAdvanceResult] = {}
        for a_name, journey in self.journeys.items():
            agent_adv_results[a_name] = advance_agent_state(
                self.previous_ctx, a_name, journey, self.time_since_last_adoption
            )
        return agent_adv_results

    def _advance_parked_agents(self, t_elapsed: float) -> dict[str, AgentAdvanceResult]:
        """
        Advance all parked (currently charging, excluded from the active set) agents
        by t_elapsed real minutes, returning an AgentAdvanceResult (keyed by agent_id)
        of agents that can rejoin the active set when their remaining charging duration
        is below the PLANNING_HORIZON_P
        """
        agent_adv_results: dict[str, AgentAdvanceResult] = {}
        for agent_id, state in list(self.parked_agents.items()):
            remaining_t_charge = max(0.0, state["remaining_delta"] - t_elapsed)
            if remaining_t_charge < PLANNING_HORIZON_P + FLOAT_TOL:
                agent_adv_results[agent_id] = AgentAdvanceResult(
                    state["node"],
                    remaining_t_charge,
                    state["sigma"],
                    AgentOperationalState.available,
                    True,
                    set(),
                    set(),
                )
                del self.parked_agents[agent_id]
            else:
                state["remaining_delta"] = remaining_t_charge
        return agent_adv_results

    def _build_next_scenario(
        self,
        mapped_agent_data: dict[str, tuple[AgentAdvanceResult, ModemsAgent]],
        new_requests: list[ModemsRequest],
    ) -> tuple[ModemsScenario, ModemsSolution]:
        """
        Build a complete scenario from the updated (advanced) agent states and a list
        of new (R_n) requests. Rejoining agents are delayed+available agents without
        previous route/requests to inherit
        """
        updated_agents, agent_name_map = [], {}
        for agent_id, (agent_adv_res, a_template) in mapped_agent_data.items():
            if not agent_adv_res.included:
                continue
            updated_agents.append(agent_adv_res.to_agent(a_template))
            agent_name_map[agent_id] = len(updated_agents)  # 1-based

        # Preserve request order and timing decisions for each retained route
        rs_entries: list[tuple[ModemsRequest, str, str]] = []  # update/rename R_s
        for agent_id, (agent_adv_res, _) in mapped_agent_data.items():
            old_journey = self.journeys.get(agent_id)
            if (
                not agent_adv_res.included
                or agent_id not in agent_name_map
                or old_journey is None
            ):
                continue
            ordered_replan: list[str] = []
            seen_requests: set[str] = set()
            # map requests across contexts
            for node in old_journey.route:
                r_name = self.previous_ctx.request_of(node)
                if r_name in agent_adv_res.replan and r_name not in seen_requests:
                    ordered_replan.append(r_name)
                    seen_requests.add(r_name)
            for r_name in ordered_replan:
                old_request = self.previous_ctx.requests[r_name]
                updated_request = old_request.copy()
                updated_request.status = RequestStatus.scheduled
                updated_request.earliest_pickup = max(
                    0.0, old_request.earliest_pickup - self.time_since_last_adoption
                )
                rs_entries.append((updated_request, agent_id, r_name))

        requests = [request for request, _, _ in rs_entries] + list(new_requests)
        scenario = ModemsScenario(
            agents=updated_agents, requests=requests, network=self.network
        )

        new_request_index = {
            r_old_name: i for i, (_, _, r_old_name) in enumerate(rs_entries, start=1)
        }
        partial_ctx = ProblemContext(
            scenario,
            self.previous_ctx.problem_type,
            self.previous_ctx.strategy,
            self.model_params,
        )
        partial_plan = ModemsSolution(partial_ctx)
        for agent_id, (agent_adv_res, _) in mapped_agent_data.items():
            old_journey = self.journeys.get(agent_id)
            if (
                not agent_adv_res.included
                or agent_id not in agent_name_map
                or old_journey is None
            ):
                continue
            service_nodes: list[str] = []
            t_start_of: dict[str, float] = {}
            tau_of: dict[str, float] = {}
            # map states of service nodes across contexts
            for old_state in old_journey.states[1:-1]:
                old_node = old_state.node
                old_request_name = self.previous_ctx.request_of(old_node)
                if old_request_name not in agent_adv_res.replan:
                    continue
                new_idx = new_request_index[old_request_name]
                updated_request = rs_entries[new_idx - 1][0]
                if self.previous_ctx.is_pickup(old_node):
                    new_node = updated_request.make_pickup_node_name(new_idx)
                    tau_of[new_node] = old_state.tau
                else:
                    new_node = updated_request.make_delivery_node_name(new_idx)
                service_nodes.append(new_node)
                t_start_of[new_node] = old_state.t_start - self.time_since_last_adoption
            if service_nodes:
                new_agent_name = f"agent_{agent_name_map[agent_id]}"
                route = [
                    partial_ctx.agent_initial_node[new_agent_name],
                    *service_nodes,
                    old_journey.route[-1],
                ]
                inherited = ModemsJourney.from_route(
                    partial_ctx,
                    new_agent_name,
                    route,
                    tau_of=tau_of,
                    t_start_of=t_start_of,
                )
                partial_plan.journeys[new_agent_name] = inherited
                partial_plan.accepted.update(inherited.request_pickup)
        return scenario, partial_plan

    def _solve_all(
        self, scenario: ModemsScenario, partial_plan: ModemsSolution
    ) -> dict[SolverStrategy, ModemsInstance]:
        """Solve a scenario+partial plan with all (benchmarking) or a single strategy"""
        results = {}
        self.last_solver_errors: dict[str, str] = {}
        solvers: list[SolverStrategy] = []
        if self.solver_mode == SolverMode.benchmarking:
            solvers = [
                SolverStrategy.milp1,
                SolverStrategy.milp2,
                SolverStrategy.milp3,
                SolverStrategy.alns,
            ]
        elif self.solver_mode == SolverMode.single:
            solvers = [SolverStrategy(self.single_solver)]
        else:  # "operational", MILP3 + ALNS race, adopt whichever is better
            solvers = [SolverStrategy.milp3, SolverStrategy.alns]
        for key in solvers:
            try:
                if key == SolverStrategy.alns:
                    model = ModemsAlns(
                        scenario,
                        problem_type=ProblemType.closed_selective,
                        model_params=self.model_params,
                    )
                    model.solve(
                        seed=DEFAULT_BASE_SEED,
                        max_iter=self.alns_max_iter,
                        partial_plan=partial_plan,
                    )
                    results[key] = model.instance
                else:
                    milp_type = {
                        SolverStrategy.milp1: MilpType.milp1,
                        SolverStrategy.milp2: MilpType.milp2,
                        SolverStrategy.milp3: MilpType.milp3,
                    }[SolverStrategy(key)]
                    problem_type = (
                        ProblemType.closed_non_selective
                        if milp_type == MilpType.milp1
                        else ProblemType.closed_selective
                    )
                    ctx = ProblemContext(
                        scenario,
                        problem_type,
                        milp_type.kappa,
                        self.model_params,
                    )
                    base_plan, r_unassigned = preprocess(ctx, partial_plan=partial_plan)
                    warm_start = None
                    if base_plan is not None:
                        baseline, _, ok = greedy_complete(base_plan, r_unassigned)
                        warm_start = baseline if ok else None
                    model = ModemsMilp(
                        scenario=scenario,
                        milp_type=milp_type,
                        problem_type=problem_type,
                        milp_params=self.model_params,
                    )
                    model.solve(
                        solver_name=self.solver_name,
                        solver_config_type=self.solver_config_type,
                        solver_options={"timelimit": self.milp_timelimit},
                        warm_start_solution=warm_start,
                    )
                    results[key] = model.instance
            except Exception as excp:
                self.last_solver_errors[key] = f"{type(excp).__name__}: {excp}"
                continue
        return results

    def advance_epoch(
        self,
        new_requests: list[ModemsRequest],
        t_elapsed: float = DEFAULT_REPLANNING_INTERVAL,
    ) -> EpochLog:
        """Advance instance to the current planning epoch"""
        self.time_since_last_adoption += t_elapsed
        active_agent_adv_results = self._advance_active_agents()
        rejoining_agent_adv_results = self._advance_parked_agents(t_elapsed)

        completed_requests: dict[str, dict[str, float]] = {}
        agent_node_visits: dict[str, list[AgentNodeVisit]] = {}
        energy_consumed: dict[str, float] = {}
        charging_events: dict[str, dict[str, float]] = {}
        agent_travel_time: dict[str, float] = {}
        for agent_name, agent_adv_res in active_agent_adv_results.items():
            agent_id = self.previous_ctx.agents[agent_name].agent_id
            old_journey = self.journeys[agent_name]
            old_sigma = self._energy_checkpoint.get(
                agent_id, old_journey.states[0].phi_dep
            )
            for r_name in agent_adv_res.absorbed:
                metrics = _collect_request_metrics(
                    self.previous_ctx, old_journey, r_name
                )
                if metrics is not None:
                    request_id = self.previous_ctx.requests[r_name].request_id
                    completed_requests[request_id] = metrics
            adv_res_idx = old_journey.route.index(agent_adv_res.node)
            # get newly visited nodes in this epoch, skipped for the very first epoch
            visit_start_idx = (
                1
                if old_journey.states[0].node == self._last_confirmed_node.get(agent_id)
                else 0
            )
            # get the last confirmed agent visited (arrived + departed) node
            visit_end_idx = (
                adv_res_idx
                if self.time_since_last_adoption < old_journey.states[adv_res_idx].t_arr
                else adv_res_idx + 1
            )
            # compile node visits if agent actually moved across epochs
            if visit_end_idx > visit_start_idx:
                self._last_confirmed_node[agent_id] = old_journey.states[
                    visit_end_idx - 1
                ].node
                agent_node_visits[agent_id] = [
                    AgentNodeVisit(
                        node=state.node,
                        arrival_time=state.t_arr,
                        departure_time=state.t_dep,
                        soc_arrival=state.phi_arr,
                        soc_departure=state.phi_dep,
                    )
                    for state in old_journey.states[visit_start_idx:visit_end_idx]
                ]
            # energy consumed since the last-recorded checkpoint for this agent_id
            sigma_at_result_node = old_journey.states[adv_res_idx].phi_arr
            energy_consumed[agent_id] = max(0.0, old_sigma - sigma_at_result_node)
            # compile charging events
            if agent_adv_res.operation_state == AgentOperationalState.charging:
                charging_events[agent_id] = {
                    "soc_before": sigma_at_result_node,
                    "soc_after": agent_adv_res.sigma,
                }
            if not agent_adv_res.included:
                # parked agent, excluded for this epoch
                self.parked_agents[agent_id] = {
                    "node": agent_adv_res.node,
                    "remaining_delta": agent_adv_res.delta,
                    "sigma": agent_adv_res.sigma,
                }
            # compile agent travel time
            agent_travel_time[agent_id] = agent_adv_res.travel_time
            # checkpoint SoC for the next epoch baseline at result.sigma, either the
            # actual SoC if agent is available, or the charge target if charging
            self._energy_checkpoint[agent_id] = agent_adv_res.sigma

        for agent_id, agent_adv_res in rejoining_agent_adv_results.items():
            # nothing physically happens while parked; only the checkpoint is
            # carried forward so a later epoch's energy delta stays correct
            self._energy_checkpoint[agent_id] = agent_adv_res.sigma

        # combine currently active and rejoining agents
        mapped_agent_data: dict[str, tuple[AgentAdvanceResult, ModemsAgent]] = {
            agent_id: (agent_adv_res, self.previous_ctx.agents[agent_id])
            for agent_id, agent_adv_res in active_agent_adv_results.items()
        }
        for agent_id, agent_adv_res in rejoining_agent_adv_results.items():
            mapped_agent_data[agent_id] = (
                agent_adv_res,
                self.agent_templates[agent_id],
            )
        # build scenario and solve with all (desired) strategies
        scenario, partial_plan = self._build_next_scenario(
            mapped_agent_data, new_requests
        )
        results = self._solve_all(scenario, partial_plan)
        usable_results = {
            solver_strategy: instance
            for solver_strategy, instance in results.items()
            if instance.solution_info.status
            in (SolutionStatus.optimal, SolutionStatus.feasible)
        }
        # advance simulation using adopted strategy result
        adopted_strategy = (
            min(usable_results, key=lambda k: usable_results[k].solution_info.objective)
            if usable_results
            else None
        )

        # completed_requests/agent_node_visits (below) are keyed by PERSISTENT
        # request_id values, stable across epochs to enable workday tracking, unlike
        # the volatile, snapshot-local/per-epoch, positional names "request_<idx>"
        epoch_log = EpochLog(
            results=results,
            adopted_strategy=adopted_strategy,
            rejected_new={r.request_id for r in new_requests},
            completed_requests=completed_requests,
            agent_node_visits=agent_node_visits,
            energy_consumed=energy_consumed,
            charging_events=charging_events,
            agent_travel_time=agent_travel_time,
            solver_errors=dict(self.last_solver_errors),
        )

        if adopted_strategy is None:
            self.rejected_log.extend(new_requests)
            # any agent that would have rejoined this epoch never actually did/entered
            # a solved plan, retain it as parked
            for agent_id, agent_adv_res in rejoining_agent_adv_results.items():
                self.parked_agents[agent_id] = {
                    "node": agent_adv_res.node,
                    "remaining_delta": agent_adv_res.delta,
                    "sigma": agent_adv_res.sigma,
                }
            self.epoch_log.append(epoch_log)
            return epoch_log

        adopted_instance = results[adopted_strategy]
        self.previous_ctx = adopted_instance.solution.ctx
        self.journeys = dict(adopted_instance.solution.journeys)
        self.time_since_last_adoption = 0.0
        rejected_ids = {
            adopted_instance.solution.ctx.requests[r_name].request_id
            for r_name in adopted_instance.solution.rejected
        }
        new_by_id = {request.request_id: request for request in new_requests}
        epoch_log.rejected_new = rejected_ids & set(new_by_id)
        self.rejected_log.extend(
            new_by_id[request_id] for request_id in epoch_log.rejected_new
        )
        self.epoch_log.append(epoch_log)
        return epoch_log


# --------------------------------------------------------------------------------------
# WorkdaySimulation: walk RollingHorizonSimulator through a deterministic, pre-defined
# request stream from ModemsScenarioGenerator.generate_workday_requests, triggering
# epochs periodically or early when a request submission mandates re-planning, i.e.,
# its earliest_pickup lies within the planning horizon P
# --------------------------------------------------------------------------------------


class WorkdaySimulation:
    simulator: RollingHorizonSimulator
    request_submissions: list[tuple[float, ModemsRequest]]
    workday_length: float
    clock_start: float
    _next_idx: int
    _pending: list[ModemsRequest]

    def __init__(
        self,
        simulator: RollingHorizonSimulator,
        request_submissions: list[tuple[float, ModemsRequest]],
        workday_length: float,
    ) -> None:
        self.simulator = simulator
        # sort requests by their submission/arrival time
        self.request_submissions = sorted(request_submissions, key=lambda x: x[0])
        self.workday_length = workday_length
        self.clock_start = 0.0
        self._next_idx = 0
        # unresolved requests at the end of the workday
        self._pending = []
        # absolute clock time for when the currently-active plan was last adopte, i.e.,
        # the reference point for epoch-local completed_requests/agent_node_visits
        # compiled by advance_epoch(). Stays 0.0 until the first successful adoption
        self._last_adoption_clock = 0.0

    def _next_trigger_time(self) -> float:
        periodic = self.clock_start + DEFAULT_REPLANNING_INTERVAL
        # A newly received request triggers an immediate epoch when its pickup time at
        # submission lies within P. A non-urgent request must not hide another (LATER)
        # urgent request before the next periodic epoch
        for t_submit, request in self.request_submissions[self._next_idx :]:
            if t_submit > periodic:
                break  # remaining requests are outside planning horizon
            if (request.earliest_pickup - t_submit) < PLANNING_HORIZON_P + FLOAT_TOL:
                return max(t_submit, self.clock_start)  # urgent request, return (now)
        return periodic

    def run(self, max_epochs: int = 10_000) -> list[EpochLog]:
        """Simulate the workday for max_epochs, return list of epoch logs"""
        for _ in range(max_epochs):
            if self.clock_start >= self.workday_length:
                break
            t_trigger = min(self._next_trigger_time(), self.workday_length)
            t_elapsed = max(0.0, t_trigger - self.clock_start)

            # collect relevant requests
            while (
                self._next_idx < len(self.request_submissions)
                and self.request_submissions[self._next_idx][0] <= t_trigger
            ):
                _, request = self.request_submissions[self._next_idx]
                self._pending.append(request.copy())
                self._next_idx += 1

            r_ready: list[ModemsRequest] = []
            r_pending: list[ModemsRequest] = []
            for request in self._pending:
                if request.earliest_pickup - t_trigger < PLANNING_HORIZON_P + FLOAT_TOL:
                    r_copy = request.copy()
                    r_copy.earliest_pickup = max(
                        0.0, request.earliest_pickup - t_trigger
                    )
                    r_ready.append(r_copy)
                else:
                    r_pending.append(request)
            self._pending = r_pending

            record = self.simulator.advance_epoch(r_ready, t_elapsed=t_elapsed)
            record.clock_start = self.clock_start
            record.clock_end = t_trigger
            # the plan is actually determined at the _last_adoption_clock time
            record.plan_reference_clock = self._last_adoption_clock
            if record.adopted_strategy is not None:
                self._last_adoption_clock = t_trigger
            self.clock_start = t_trigger
            if t_trigger >= self.workday_length and self._next_idx >= len(
                self.request_submissions
            ):
                break
        return self.simulator.epoch_log


# --------------------------------------------------------------------------------------
# summarize_workday: aggregate metrics across a full epoch_log, either from
# RollingHorizonSimulator, driven directly, or via WorkdaySimulation
# --------------------------------------------------------------------------------------


def summarize_workday(epoch_log: list[EpochLog]) -> dict[str, Any]:
    """
    Extract workday-level metrics from an epoch_log: (waiting time, excess ride time,
    tardiness) per completed request, agent travel time, instance solve time, energy
    consumed and charging events (per agent-epoch), request acceptance rate, and which
    solver was adopted/best how often. Returns a plain dict for serialization

    Intended for a same-stream, single-solver comparison; run the identical request
    submission stream with two RollingHorizonSimulator instances, one per
    solver_mode="single" choice, and compare their summaries
    """
    nr_completed = 0
    total_wait = total_excess_ride = total_delay_time = 0.0
    max_wait = max_excess_ride = max_delay_time = 0.0
    total_energy = 0.0
    nr_charging_events = 0
    total_soc_gained = 0.0
    nr_rejected = 0
    nr_epochs = len(epoch_log)
    nr_epochs_no_adoption = 0
    adopted_counts = {}
    total_travel_time = 0.0
    total_solve_time = 0.0
    nr_solved_epochs = 0

    for record in epoch_log:
        adopted_key = record.adopted_strategy
        if adopted_key is None:
            nr_epochs_no_adoption += 1
        else:
            adopted_counts[adopted_key] = adopted_counts.get(adopted_key, 0) + 1
            adopted_info = (record.results or {}).get(adopted_key)
            solve_time = getattr(
                getattr(adopted_info, "solution_info", None), "solution_time", None
            )
            if solve_time is not None:
                total_solve_time += solve_time
                nr_solved_epochs += 1

        nr_rejected += len(record.rejected_new)

        for metrics in record.completed_requests.values():
            nr_completed += 1
            total_wait += metrics["waiting_time"]
            total_excess_ride += metrics["excess_ride_time"]
            total_delay_time += metrics["delay_time"]
            max_wait = max(max_wait, metrics["waiting_time"])
            max_excess_ride = max(max_excess_ride, metrics["excess_ride_time"])
            max_delay_time = max(max_delay_time, metrics["delay_time"])

        for energy in record.energy_consumed.values():
            total_energy += energy

        for event in record.charging_events.values():
            nr_charging_events += 1
            total_soc_gained += max(0.0, event["soc_after"] - event["soc_before"])

        for travel_time in record.agent_travel_time.values():
            total_travel_time += travel_time

    nr_seen = nr_completed + nr_rejected
    return {
        "nr_epochs": nr_epochs,
        "nr_epochs_no_adoption": nr_epochs_no_adoption,
        "nr_adopted": adopted_counts,
        "nr_requests_completed": nr_completed,
        "nr_requests_rejected": nr_rejected,
        "acceptance_rate": (nr_completed / nr_seen) if nr_seen else None,
        "total_waiting_time": total_wait,
        "mean_waiting_time": (total_wait / nr_completed) if nr_completed else None,
        "max_waiting_time": max_wait if nr_completed else None,
        "total_excess_ride_time": total_excess_ride,
        "mean_excess_ride_time": (
            (total_excess_ride / nr_completed) if nr_completed else None
        ),
        "max_excess_ride_time": max_excess_ride if nr_completed else None,
        "total_delay_time": total_delay_time,
        "mean_delay_time": (total_delay_time / nr_completed) if nr_completed else None,
        "max_delay_time": max_delay_time if nr_completed else None,
        "total_agent_travel_time": total_travel_time,
        "total_energy_consumed": total_energy,
        "nr_charging_events": nr_charging_events,
        "total_soc_gained_from_charging": total_soc_gained,
        "total_solve_time": total_solve_time,
        "mean_solve_time_per_epoch": (
            (total_solve_time / nr_solved_epochs) if nr_solved_epochs else None
        ),
    }


# --------------------------------------------------------------------------------------
# build_workday_log: full-day request-list + agent-walk aggregation
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class WorkdayLog:
    """
    Full-day aggregation of a RollingHorizonSimulator.epoch_log, built by
    build_workday_log(): every submitted request across the whole day, its final
    outcome (accepted/rejected/unresolved) with absolute time (offset from workday
    clock-start), as well as the full node-by-node walk of every agent in absolute time
    """

    clock_start: float
    clock_end: float
    requests: list[RequestRecord] = field(default_factory=list)
    agent_node_visits: dict[str, list[AgentNodeVisit]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "clock_start": self.clock_start,
            "clock_end": self.clock_end,
            "requests": [r.to_dict() for r in self.requests],
            "agent_node_visits": {
                agent_name: [v.to_dict() for v in visits]
                for agent_name, visits in self.agent_node_visits.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkdayLog:
        return cls(
            clock_start=data["clock_start"],
            clock_end=data["clock_end"],
            requests=[RequestRecord.from_dict(r) for r in data["requests"]],
            agent_node_visits={
                agent_name: [AgentNodeVisit.from_dict(v) for v in visits]
                for agent_name, visits in data["agent_node_visits"].items()
            },
        )


def build_workday_log(
    epoch_log: list[EpochLog],
    request_submissions: list[tuple[float, ModemsRequest]],
) -> WorkdayLog:
    """
    Aggregate a full epoch_log plus request_submissions from WorkdaySimulation into one
    WorkdayLog: outcome of every submitted request, and the full-day node-by-node walk
    of every agent, both in absolute time (offset from workday clock-start)
    """
    clock_start = 0.0
    clock_end = max(
        (e.clock_end for e in epoch_log if e.clock_end is not None), default=0.0
    )

    completed_by_id: dict[str, dict[str, float]] = {}
    rejected_ids: set[str] = set()
    agent_node_visits: dict[str, list[AgentNodeVisit]] = {}

    for epoch in epoch_log:
        t_shift = epoch.plan_reference_clock or 0.0
        for request_id, metrics in epoch.completed_requests.items():
            completed_by_id[request_id] = {
                **metrics,
                "pickup_time": metrics["pickup_time"] + t_shift,
                "delivery_time": metrics["delivery_time"] + t_shift,
            }
        rejected_ids |= epoch.rejected_new
        for a_name, node_visits in epoch.agent_node_visits.items():
            shifted = [
                AgentNodeVisit(
                    node=node_visit.node,
                    arrival_time=node_visit.arrival_time + t_shift,
                    departure_time=node_visit.departure_time + t_shift,
                    soc_arrival=node_visit.soc_arrival,
                    soc_departure=node_visit.soc_departure,
                )
                for node_visit in node_visits
            ]
            agent_node_visits.setdefault(a_name, []).extend(shifted)

    r_records: list[RequestRecord] = []
    for t_submit, request in request_submissions:
        request_id = request.request_id
        if request_id in completed_by_id:
            metrics = completed_by_id[request_id]
            r_records.append(
                RequestRecord(
                    request=request,
                    submission_time=t_submit,
                    outcome=RequestOutcome.accepted,
                    pickup_time=metrics["pickup_time"],
                    delivery_time=metrics["delivery_time"],
                    wait=metrics["waiting_time"],
                    delay=metrics["delay_time"],
                    excess_ride_time=metrics["excess_ride_time"],
                )
            )
        elif request_id in rejected_ids:
            r_records.append(
                RequestRecord(
                    request=request,
                    submission_time=t_submit,
                    outcome=RequestOutcome.rejected,
                )
            )
        else:
            r_records.append(
                RequestRecord(
                    request=request,
                    submission_time=t_submit,
                    outcome=RequestOutcome.unresolved,
                )
            )

    return WorkdayLog(
        clock_start=clock_start,
        clock_end=clock_end,
        requests=r_records,
        agent_node_visits=agent_node_visits,
    )


# --------------------------------------------------------------------------------------
# Same-arrivals MILP3-vs-ALNS comparison: two independent single-solver simulations
# against the identical request demand
# --------------------------------------------------------------------------------------


def compare_solvers_one_workday(
    network: RoadNetwork,
    agents: list[ModemsAgent],
    request_submissions: list[tuple[float, ModemsRequest]],
    workday_length: float,
    model_params: dict[str, Any] | None = None,
    milp_timelimit: float = DEFAULT_MILP_TIMELIMIT,
    alns_max_iter: int = DEFAULT_PARAMS_ALNS["max_iter"],
    solver_name: str = DEFAULT_MILP_SOLVER_DATA[0],
    solver_config_type: SolverConfigType = DEFAULT_MILP_SOLVER_DATA[1],
    workday_logs_out: dict[SolverStrategy, WorkdayLog] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Run the identical submitted requests stream through two independent, single-solver
    RollingHorizonSimulator instances (one MILP3, one ALNS) for a fair comparison.
    Return {"milp3": summary_dict, "alns": summary_dict}, each from summarize_workday()

    If given, workday_logs_out is populated with {solver_strategy: WorkdayLog} in place
    for the full per-request/per-agent day log of a detailed export. None by default
    """
    summaries = {}
    for solver_strategy in (SolverStrategy.milp3, SolverStrategy.alns):
        agents_copy = [a.copy() for a in agents]
        rh_simulator = RollingHorizonSimulator(
            network,
            agents_copy,
            model_params=model_params,
            solver_mode=SolverMode.single,
            single_solver=solver_strategy,
            milp_timelimit=milp_timelimit,
            alns_max_iter=alns_max_iter,
            solver_name=solver_name,
            solver_config_type=solver_config_type,
        )
        workday = WorkdaySimulation(rh_simulator, request_submissions, workday_length)
        log = workday.run()
        summaries[solver_strategy] = summarize_workday(log)
        if workday_logs_out is not None:
            workday_logs_out[solver_strategy] = build_workday_log(
                log, request_submissions
            )
    return summaries


def compare_solvers_over_workdays(
    network: RoadNetwork,
    agents: list[ModemsAgent],
    nr_workdays: int,
    workday_length: float,
    base_seed: int = DEFAULT_BASE_SEED,
    generator_kwargs: dict[str, Any] | None = None,
    model_params: dict[str, Any] | None = None,
    milp_timelimit: float = DEFAULT_MILP_TIMELIMIT,
    alns_max_iter: int = DEFAULT_PARAMS_ALNS["max_iter"],
    solver_name: str = DEFAULT_MILP_SOLVER_DATA[0],
    solver_config_type: SolverConfigType = DEFAULT_MILP_SOLVER_DATA[1],
    on_workday_done: Callable[[int, dict[str, Any]], Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Generate nr_workdays independent submission streams (one ModemsScenarioGenerator
    per workday, seeded deterministically with base_seed) and run each through
    compare_solvers_one_workday(). Return a list of per-workday dicts: {"workday": idx,
    "milp3": summary, "alns": summary}. If given, on_workday_done(workday_index, result)
    is called after each workday completes to checkpoint progress to disk, since a full
    run can take a long time with CBC (faster with highs or gurobi)
    """
    results = []
    generator_kwargs = generator_kwargs or {}
    for idx in range(nr_workdays):
        seed = stable_seed(base_seed, "workday", idx)
        generator = ModemsScenarioGenerator(seed=seed, network=network)
        request_submissions = generator.generate_workday_requests(
            workday_length=workday_length, **generator_kwargs
        )
        summaries = compare_solvers_one_workday(
            network,
            agents,
            request_submissions,
            workday_length,
            model_params=model_params,
            milp_timelimit=milp_timelimit,
            alns_max_iter=alns_max_iter,
            solver_name=solver_name,
            solver_config_type=solver_config_type,
        )
        record = {"workday": idx, "seed": seed, **summaries}
        results.append(record)
        if on_workday_done is not None:
            on_workday_done(idx, record)
    return results
