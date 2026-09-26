from __future__ import annotations

import copy
import hashlib
import json
from enum import StrEnum
from typing import Any

import numpy as np

from .network import NetworkNodeName, NetworkNodeType, RoadNetwork

# default (fixed) time window length
DEFAULT_TIME_WINDOW = 5.0

# default base seed for all stochastic elements/processes
DEFAULT_BASE_SEED = 42

# default parameters for objective caculation, time windows, and maximum ride-time
DEFAULT_PARAMS_OBJ = {
    "eps": 0.01,
    "zeta": 2.0,
    "eta": 100.0,
    "rho": 2.5,
    "omega": DEFAULT_TIME_WINDOW,
}


class SmartStrEnum(StrEnum):
    """An aliasable StrEnum for parsing/reporting purposes"""

    @classmethod
    def _missing_(cls, value: Any) -> None | SmartStrEnum:
        if not isinstance(value, str):
            return None
        normalized = value.casefold()
        for member in cls:
            aliases = {
                member.name.casefold(),
                str(member.value).casefold(),
                member.name[0].casefold(),
            }
            if normalized in aliases:
                return member
        return None

    @property
    def letter(self) -> str:
        return self.value[0].upper()


class RequestStatus(StrEnum):
    """Request status enumeration class"""

    # requests in R_n, which the routing strategy may accept or reject
    new = "new"
    # requests in R_s, previously accepted and must be serviced (enforce y_r = 1)
    scheduled = "scheduled"


class ModemsRequest:
    """
    MODEMS Request class, encapsulates the tuple (p^r, d^r, q^r, s^r, [e^r, l^r]).
    Earliest pickup time is specified, latest pickup time = e^r + omega, where omega
    is the fixed time-window length (shared by the whole model)
    """

    node_pickup_index: int
    node_delivery_index: int
    load: int
    service_time: float
    earliest_pickup: float
    tw_length: float
    status: RequestStatus
    request_id: str

    def __init__(
        self,
        node_pickup_index: int,
        node_delivery_index: int,
        load: int = 1,
        service_time: float = 1.0,
        earliest_pickup: float = 10.0,
        tw_length: float = DEFAULT_TIME_WINDOW,
        status: RequestStatus | str = RequestStatus.new,
        request_id: str | None = None,
    ) -> None:
        """
        Initialize the MODEMS Request.

        Args:
            node_pickup_index: Index of the pickup node (p^r)
            node_delivery_index: Index of the delivery node (d^r)
            load: Number of passengers (q^r)
            service_time: Required service time at pickup and delivery (s^r)
            earliest_pickup: Desired earliest start of service at pickup (e^r)
            tw_length: Fixed pickup TW length (omega), l^r = e^r + omega
            status: "new" (R_n) or "scheduled" (R_s)
            request_id: Optional, persistent, unique identifier for this request,
                stable across cloning/renaming and multiple epochs. If empty, fallback
                to a hash of this request's own field values: deterministic, but only
                unique up to those values (i.e., two requests with identical fields
                would collide). Must be provided for workdays/multi-epoch runs
        """
        if not isinstance(node_pickup_index, int) or node_pickup_index < 1:
            raise ValueError("node_pickup_index must be a positive 1-based index")
        if not isinstance(node_delivery_index, int) or node_delivery_index < 1:
            raise ValueError("node_delivery_index must be a positive 1-based index")
        if not isinstance(load, int) or isinstance(load, bool) or load <= 0:
            raise ValueError("load must be a positive integer")
        if service_time < 0:
            raise ValueError("service_time must be non-negative")
        if earliest_pickup < 0:
            raise ValueError("earliest_pickup must be non-negative")
        if tw_length <= 0:
            raise ValueError("tw_length must be positive")

        self.node_pickup_index = node_pickup_index
        self.node_delivery_index = node_delivery_index
        self.load = load
        self.service_time = service_time
        self.earliest_pickup = earliest_pickup
        self.tw_length = tw_length
        self.status = RequestStatus(status)
        self.request_id = request_id or self._fallback_request_id()

    def _fallback_request_id(self) -> str:
        """Fallback request ID, based on a hash of the request's own field values"""
        key = (
            f"{self.node_pickup_index}:"
            f"{self.node_delivery_index}:"
            f"{self.load}:"
            f"{self.service_time}:"
            f"{self.earliest_pickup}:"
            f"{self.tw_length}"
        )
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    @property
    def latest_pickup(self) -> float:
        """Latest pickup service start time, l^r = e^r + omega"""
        return self.earliest_pickup + self.tw_length

    def is_new(self) -> bool:
        """Check if request is new (in R_n)."""
        return self.status == RequestStatus.new

    def is_scheduled(self) -> bool:
        """Check if request is scheduled (in R_s)"""
        return self.status == RequestStatus.scheduled

    def make_request_name(self, idx: int) -> str:
        """Return the request name from its index"""
        return f"request_{idx}"

    def make_pickup_node_name(self, idx: int) -> str:
        """Return the pickup node name from its index"""
        return NetworkNodeName.make_request_pickup_node_name(
            idx, self.node_pickup_index
        )

    def make_delivery_node_name(self, idx: int) -> str:
        """Return the delivery node name from its index"""
        return NetworkNodeName.make_request_delivery_node_name(
            idx, self.node_delivery_index
        )

    def copy(self) -> ModemsRequest:
        """Return a deepcopy of the Request"""
        return copy.deepcopy(self)

    def __str__(self) -> str:
        """Return a string representation of the Request"""
        return (
            "ModemsRequest:\n"
            f"  request_id: {self.request_id}\n"
            f"  node pickup index: {self.node_pickup_index}\n"
            f"  node delivery index: {self.node_delivery_index}\n"
            f"  load: {self.load}\n"
            f"  service time: {self.service_time:.2f} minutes\n"
            f"  status: {self.status}\n"
            f"  earliest pickup (e^r): {self.earliest_pickup:.2f} minutes\n"
            f"  latest pickup (l^r): {self.latest_pickup:.2f} minutes\n"
        )

    def to_dict(self) -> dict:
        """Return a dictionary representation of the Request"""
        return {
            "node_pickup_index": int(self.node_pickup_index),
            "node_delivery_index": int(self.node_delivery_index),
            "load": int(self.load),
            "service_time": float(self.service_time),
            "earliest_pickup": float(self.earliest_pickup),
            "tw_length": float(self.tw_length),
            "status": self.status,
            "request_id": self.request_id,
        }

    @classmethod
    def from_dict(cls, request_dict: dict) -> ModemsRequest:
        """Create a Request from the given dictionary"""
        return cls(
            node_pickup_index=request_dict["node_pickup_index"],
            node_delivery_index=request_dict["node_delivery_index"],
            load=request_dict["load"],
            service_time=request_dict["service_time"],
            earliest_pickup=request_dict["earliest_pickup"],
            tw_length=request_dict["tw_length"],
            status=request_dict.get("status", RequestStatus.new),
            request_id=request_dict.get("request_id"),
        )


class ModemsAgent:
    """
    MODEMS Agent class, encapsulates the extended tuple
    (v^k, delta^k, Q^k, sigma_min^k, sigma^k, alpha^k, beta^k)
    """

    node_type: NetworkNodeType
    node_index: int
    time_initial: float
    load_max: int
    soc_min_operational: float
    soc_initial: float
    soc_alpha: float
    soc_beta: float
    agent_id: str

    def __init__(
        self,
        node_type: NetworkNodeType | str,
        node_index: int,
        time_initial: float = 0.0,
        load_max: int = 6,
        soc_min_operational: float = 0.25,
        soc_initial: float = 1.0,
        soc_alpha: float = 6.65e-03,
        soc_beta: float = 0.15e-03,
        agent_id: str | None = None,
    ) -> None:
        """
        Initialize the MODEMS Agent

        Args:
            node_type: Type of initial node (hub/station)
            node_index: Index of initial node
            time_initial: Initial availability time (delta^k)
            load_max: Maximum passenger load (Q^k)
            soc_min_operational: Minimum operational SoC (sigma^k_min)
            soc_initial: Initial SoC (sigma^k)
            soc_alpha: default discharge rate, load-free agent (alpha^k)
            soc_beta: load-dependent (nr of passengers) discharge rate (beta^k)
            agent_id: Optional, persistent, unique identifier for this agent,
                stable across cloning/renaming and multiple epochs. If empty, fallback
                to a hash of this agent's own field values: deterministic, but only
                unique up to those values (i.e., two agents with identical fields
                would collide). Must be provided for workdays/multi-epoch runs
        """
        node_type = NetworkNodeType(node_type)
        if node_type == NetworkNodeType.hub:
            self.node_type = NetworkNodeType.hub
        elif NetworkNodeType.is_station(node_type):
            self.node_type = NetworkNodeType.station
        else:
            raise ValueError("Invalid node type")
        if not isinstance(node_index, int) or node_index < 1:
            raise ValueError("node_index must be a positive 1-based index")
        if time_initial < 0:
            raise ValueError("time_initial must be non-negative")
        if not isinstance(load_max, int) or isinstance(load_max, bool) or load_max <= 0:
            raise ValueError("load_max must be a positive integer")
        if not 0 < soc_min_operational <= 1:
            raise ValueError("soc_min_operational must lie in (0, 1]")
        if not soc_min_operational <= soc_initial <= 1:
            raise ValueError("soc_initial must lie in [soc_min_operational, 1]")
        if soc_alpha <= 0 or soc_beta <= 0:
            raise ValueError("soc_alpha and soc_beta must be positive")
        self.node_index = node_index
        self.time_initial = time_initial
        self.load_max = load_max
        self.soc_min_operational = soc_min_operational
        self.soc_initial = soc_initial
        self.soc_alpha = soc_alpha
        self.soc_beta = soc_beta
        self.agent_id = agent_id or self._fallback_agent_id()
        # remaining driving duration based on worst discharge
        # D^k = (sigma^k - sigma^k_min) / (alpha^k + beta^k * Q^k)
        self.duration_max = (soc_initial - soc_min_operational) / (
            soc_alpha + soc_beta * load_max
        )

    def _fallback_agent_id(self) -> str:
        """Fallback agent ID, based on a hash of the agent's own field values"""
        key = (
            f"{self.node_type}:"
            f"{self.node_index}:"
            f"{self.time_initial}:"
            f"{self.load_max}:"
            f"{self.soc_min_operational}:"
            f"{self.soc_initial}:"
            f"{self.soc_alpha}:"
            f"{self.soc_beta}"
        )
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    @staticmethod
    def make_agent_name(idx: int) -> str:
        """Return the agent name from its index"""
        return f"agent_{idx}"

    def make_node_name(self, idx: int) -> str:
        """Return the agent node name from its index"""
        return NetworkNodeName.make_agent_node_name(
            idx, self.node_type, self.node_index
        )

    def copy(self) -> ModemsAgent:
        """Return a deepcopy of the Agent"""
        return copy.deepcopy(self)

    def __str__(self) -> str:
        """Return a string representation of the Agent"""
        return (
            "ModemsAgent:\n"
            f"  agent_id: {self.agent_id}\n"
            f"  node type: {self.node_type}\n"
            f"  node index: {self.node_index}\n"
            f"  time initial: {self.time_initial:.2f} minutes\n"
            f"  load max: {self.load_max}\n"
            f"  SoC min operational: {self.soc_min_operational:.2f}\n"
            f"  SoC initial: {self.soc_initial:.2f}\n"
            f"  SoC default discharge rate (alpha): {self.soc_alpha:.4f}\n"
            f"  SoC load-dependent discharge rate (beta): {self.soc_beta:.4f}\n"
            f"  Duration max: {self.duration_max:.2f}\n"
        )

    def to_dict(self) -> dict:
        """Return a dictionary representation of the Agent"""
        return {
            "node_type": self.node_type,
            "node_index": int(self.node_index),
            "time_initial": float(self.time_initial),
            "load_max": int(self.load_max),
            "soc_min_operational": float(self.soc_min_operational),
            "soc_initial": float(self.soc_initial),
            "soc_alpha": float(self.soc_alpha),
            "soc_beta": float(self.soc_beta),
            "duration_max": float(self.duration_max),
            "agent_id": self.agent_id,
        }

    @classmethod
    def from_dict(cls, agent_dict: dict) -> ModemsAgent:
        """Create an Agent from the given dictionary"""
        return cls(
            node_type=agent_dict["node_type"],
            node_index=agent_dict["node_index"],
            time_initial=agent_dict["time_initial"],
            load_max=agent_dict["load_max"],
            soc_min_operational=agent_dict["soc_min_operational"],
            soc_initial=agent_dict["soc_initial"],
            soc_alpha=agent_dict["soc_alpha"],
            soc_beta=agent_dict["soc_beta"],
            agent_id=agent_dict.get("agent_id"),
        )


# --------------------------------------------------------------------------------------
# Problem taxonomy: routing-problem variant and active solving strategy
# --------------------------------------------------------------------------------------


class ObjectiveType(SmartStrEnum):
    """
    Routing objective: closed includes the final return-to-hub leg (for all agents)
    in the mission time, open ends the mission time at each agent last delivery. Both
    cases ensure that every journey ends at a reachable final hub (SoC feasibility)
    """

    closed = "closed"
    open = "open"


class ProblemType(StrEnum):
    """Routing problem type enumeration class"""

    open_non_selective = "open_non_selective"
    open_selective = "open_selective"
    closed_non_selective = "closed_non_selective"
    closed_selective = "closed_selective"

    @staticmethod
    def from_parts(objective: ObjectiveType | str, selective: bool) -> ProblemType:
        """Compose a problem type from its objective and selectivity"""
        suffix = "selective" if selective else "non_selective"
        return ProblemType(f"{ObjectiveType(objective)}_{suffix}")

    @staticmethod
    def objective_of(problem_type: str) -> ObjectiveType:
        """Get the objective (open/closed) of the given problem type"""
        return (
            ObjectiveType.open
            if ProblemType.is_open(problem_type)
            else ObjectiveType.closed
        )

    @staticmethod
    def is_open(problem_type: str) -> bool:
        """Check if problem variant is open"""
        return problem_type in (
            ProblemType.open_non_selective,
            ProblemType.open_selective,
        )

    @staticmethod
    def is_closed(problem_type: str) -> bool:
        """Check if problem variant is closed"""
        return not ProblemType.is_open(problem_type)

    @staticmethod
    def is_selective(problem_type: str) -> bool:
        """Check if problem variant is selective (requests may be rejected)"""
        return problem_type in (
            ProblemType.open_selective,
            ProblemType.closed_selective,
        )

    @staticmethod
    def is_non_selective(problem_type: str) -> bool:
        """Check if problem variant is non-selective (all requests must be accepted)"""
        return not ProblemType.is_selective(problem_type)


class SolverStrategy(StrEnum):
    """Active solution strategy, dictating the used feasibility conditions"""

    milp1 = "milp1"
    milp2 = "milp2"
    milp3 = "milp3"
    alns = "alns"

    def has_soft_tw(self) -> bool:
        """Strategy has soft time windows (TW violations permitted but penalized)"""
        return self in (SolverStrategy.milp1, SolverStrategy.milp3, SolverStrategy.alns)

    def has_selectivity(self) -> bool:
        """Requests may be rejected (selective routing problem)"""
        return self in (SolverStrategy.milp2, SolverStrategy.milp3, SolverStrategy.alns)

    def has_extended_soc(self) -> bool:
        """Strategy uses extended SoC model"""
        return self in (SolverStrategy.milp2, SolverStrategy.milp3, SolverStrategy.alns)

    def supported_selectivity(self) -> tuple[bool, ...]:
        """
        Selectivity values this strategy supports, the first being its default:
        milp1 is non-selective only, milp2 and alns are selective only, milp3 supports
        both (selective by default). Every strategy supports both objectives
        """
        if self == SolverStrategy.milp1:
            return (False,)
        if self == SolverStrategy.milp3:
            return (True, False)
        return (True,)


def resolve_problem_type(
    strategy: SolverStrategy | str,
    objective: ObjectiveType | str,
    selective: bool | None = None,
) -> ProblemType:
    """
    Resolve the problem type for the given strategy and objective. If selective is
    None, use the strategy default selectivity; otherwise validate that the strategy
    supports it and raise ValueError if not
    """
    strategy = SolverStrategy(strategy)
    supported = strategy.supported_selectivity()
    if selective is None:
        selective = supported[0]
    elif selective not in supported:
        kind = "selective" if selective else "non-selective"
        raise ValueError(f"{strategy} does not support {kind} routing problems")
    return ProblemType.from_parts(objective, selective)


# --------------------------------------------------------------------------------------
# Scenario container and precomputed problem context
# --------------------------------------------------------------------------------------


class ScenarioType(SmartStrEnum):
    """
    Spatial distribution of request pickup/delivery stations
      - random: pickup/delivery stations drawn uniformly at random.
      - clustered: drawn around a handful of sampled geographic centers --
        models "everyone leaving/entering the same building"
      - mixed: each endpoint independently clustered with probability
        mixed_probability, uniform otherwise
    """

    random = "random"
    clustered = "clustered"
    mixed = "mixed"


class ScenarioSize(SmartStrEnum):
    """Scenario size, check scenario_bucket() for corresponding nr_agents/nr_requests"""

    small = "small"
    medium = "medium"
    large = "large"


class ScenarioTiming(SmartStrEnum):
    """
    Temporal distribution (shape) of requests' earliest-pickup times over a horizon:
      - uniform: spread uniformly across the horizon.
      - peaks: half spread uniformly across the horizon, half within two plateaus
        of 1/6 of the horizon each, centered at 1/3 and 2/3 of it (rush hours).
    Workday surges (generate_workday_requests) come on top of either shape
    """

    uniform = "uniform"
    peaks = "peaks"


def scenario_bucket(size: ScenarioSize | str) -> tuple[int, tuple[int, int]]:
    """Return the (nr_agents, (min_requests, max_requests)) bucket data for the size"""
    size = ScenarioSize(size)
    if size == ScenarioSize.small:
        return (1, (4, 6))
    elif size == ScenarioSize.medium:
        return (2, (5, 10))
    else:
        return (3, (10, 15))


def scenario_size_of(nr_agents: int, nr_requests: int) -> ScenarioSize:
    """Return the scenario size for the given (nr_agents, nr_requests) data"""
    buckets = {s: scenario_bucket(s) for s in ScenarioSize}
    for s, (max_agents, (_, max_requests)) in buckets.items():
        if (nr_agents <= max_agents) and (nr_requests <= max_requests):
            return s
    return ScenarioSize.large  # default to large (max currently supported)


class ModemsScenario:
    """
    MODEMS test scenario. Contains agents, requests, and a road network (directed graph)
    """

    agents: list[ModemsAgent]
    requests: list[ModemsRequest]
    network: RoadNetwork
    size: ScenarioSize
    type: ScenarioType
    timing: ScenarioTiming

    def __init__(
        self,
        agents: list[ModemsAgent],
        requests: list[ModemsRequest],
        network: RoadNetwork,
        type: ScenarioType | str = ScenarioType.random,
        timing: ScenarioTiming | str = ScenarioTiming.uniform,
    ) -> None:
        """Initialize the scenario with agents, requests, and a road network"""
        for agent in agents:
            limit = (
                network.nr_hubs
                if agent.node_type == NetworkNodeType.hub
                else network.nr_stations
            )
            if agent.node_index > limit:
                raise ValueError(
                    f"agent node index {agent.node_index} exceeds the available "
                    f"{agent.node_type} nodes ({limit})"
                )
        for request in requests:
            if request.node_pickup_index > network.nr_stations:
                raise ValueError("request pickup index exceeds nr_stations")
            if request.node_delivery_index > network.nr_stations:
                raise ValueError("request delivery index exceeds nr_stations")
        request_ids = [request.request_id for request in requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError(
                "request_id values must be unique within a scenario; provide "
                "explicit IDs for otherwise identical requests"
            )
        self.agents = copy.deepcopy(agents)
        self.requests = copy.deepcopy(requests)
        self.network = copy.deepcopy(network)
        self.size = scenario_size_of(len(agents), len(requests))
        self.type = ScenarioType(type)
        self.timing = ScenarioTiming(timing)

    @property
    def new_requests(self) -> list[ModemsRequest]:
        """Requests in R_n (new, may be accepted or rejected)"""
        return [r for r in self.requests if r.is_new()]

    @property
    def scheduled_requests(self) -> list[ModemsRequest]:
        """Requests in R_s (scheduled, must remain accepted)"""
        return [r for r in self.requests if r.is_scheduled()]

    def make_scenario_name(self, i_rep: int = 0) -> str:
        """
        Make a scenario name from its size/type/timing letters, the given repetition
        index, and its agent/request/station/hub counts, e.g., S_SRL0_a1_r4_s15_h3
        """
        return (
            f"S_{self.size.letter}{self.type.letter}{self.timing.letter}{i_rep}_"
            f"a{len(self.agents)}_"
            f"r{len(self.requests)}_"
            f"s{self.network.nr_stations}_"
            f"h{self.network.nr_hubs}"
        )

    def to_dict(self) -> dict:
        """Return a dictionary representation of the Scenario"""
        return {
            "properties": {
                "size": self.size,
                "type": self.type,
                "timing": self.timing,
            },
            "agents": [
                {
                    "name": agent.make_agent_name(i),
                    **agent.to_dict(),
                }
                for i, agent in enumerate(self.agents, start=1)
            ],
            "requests": [
                {
                    "name": request.make_request_name(i),
                    **request.to_dict(),
                }
                for i, request in enumerate(self.requests, start=1)
            ],
            "network": {
                "nr_hubs": self.network.nr_hubs,
                "nr_stations": self.network.nr_stations,
                "locations": self.network.locations.tolist(),
                "travel_times": self.network.travel_times.tolist(),
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> ModemsScenario:
        """Read a Scenario from the given dictionary"""
        agents = [ModemsAgent.from_dict(a) for a in data["agents"]]
        requests = [ModemsRequest.from_dict(r) for r in data["requests"]]
        nr_hubs = data["network"]["nr_hubs"]
        nr_stations = data["network"]["nr_stations"]
        network = RoadNetwork(
            nr_hubs=nr_hubs,
            nr_stations=nr_stations,
            locations=np.array(data["network"]["locations"]),
            travel_times=np.array(data["network"]["travel_times"]),
        )
        type = ScenarioType[data["properties"]["type"]]
        timing = ScenarioTiming[data["properties"]["timing"]]
        return cls(agents, requests, network, type, timing)

    def to_json(self, outfile: str) -> None:
        """Export the Scenario to a JSON file"""
        with open(outfile, "w") as f:
            json.dump(self.to_dict(), f, indent=4, default=str)

    @classmethod
    def from_json(cls, outfile: str) -> ModemsScenario:
        """Import a test scenario from JSON file"""
        with open(outfile, "r") as file:
            data = json.load(file)
        return cls.from_dict(data)

    def copy(self) -> ModemsScenario:
        """Return a deepcopy of the Scenario"""
        return copy.deepcopy(self)


class ProblemContext:
    """
    Precomputed, read-only lookup tables for a scenario, shared by all solution
    operations. Build this once to avoid re-deriving node maps, travel times,
    and request/agent parameters during construction and/or search
    """

    scenario: ModemsScenario
    problem_type: ProblemType
    strategy: SolverStrategy
    model_params: dict[str, Any]
    eps: float
    zeta: float
    eta: float
    rho: float
    omega: float
    agent_names: list[str]
    request_names: list[str]
    agents: dict[str, ModemsAgent]
    requests: dict[str, ModemsRequest]
    agent_initial_node: dict[str, str]
    pickup_node: dict[str, str]
    delivery_node: dict[str, str]
    node_request: dict[str, str]
    node_load: dict[str, int]
    node_t_service: dict[str, float]
    node_earliest_p: dict[str, float]
    node_latest_p: dict[str, float]

    def __init__(
        self,
        scenario: ModemsScenario,
        problem_type: ProblemType | str,
        strategy: SolverStrategy | str,
        model_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize a problem context from the given scenario and parameters,
        validating the given model_params (or using defaults from DEFAULT_PARAMS_OBJ)
        """

        self.scenario = scenario
        self.problem_type = ProblemType(problem_type)
        self.strategy = SolverStrategy(strategy)
        model_params = {**DEFAULT_PARAMS_OBJ, **(model_params or {})}
        self.model_params = model_params
        self.eps = self.model_params["eps"]
        self.zeta = self.model_params["zeta"]
        self.eta = self.model_params["eta"]
        self.rho = self.model_params["rho"]
        self.omega = self.model_params["omega"]
        if not 0.0 < self.eps < self.zeta < self.eta:
            raise ValueError("model penalties must satisfy 0 < eps < zeta < eta")
        if self.rho < 1.0:
            raise ValueError("rho must be at least 1")
        if self.omega < 1.0:
            raise ValueError("omega must be at least 1")

        self.agent_names = [
            a.make_agent_name(i) for i, a in enumerate(scenario.agents, start=1)
        ]
        self.request_names = [
            r.make_request_name(i) for i, r in enumerate(scenario.requests, start=1)
        ]
        self.agents = dict(zip(self.agent_names, scenario.agents))
        self.requests = dict(zip(self.request_names, scenario.requests))

        self.agent_initial_node = {}
        self.pickup_node = {}
        self.delivery_node = {}
        self.node_request = {}
        self.node_load = {}
        self.node_t_service = {}
        self.node_earliest_p = {}
        self.node_latest_p = {}

        for i, a in enumerate(scenario.agents, start=1):
            k = self.agent_names[i - 1]
            self.agent_initial_node[k] = a.make_node_name(i)

        for i, r in enumerate(scenario.requests, start=1):
            r_name = self.request_names[i - 1]
            p = r.make_pickup_node_name(i)
            d = r.make_delivery_node_name(i)
            self.pickup_node[r_name] = p
            self.delivery_node[r_name] = d
            self.node_request[p] = r_name
            self.node_request[d] = r_name
            self.node_load[p] = r.load
            self.node_load[d] = -r.load
            self.node_t_service[p] = r.service_time
            self.node_t_service[d] = r.service_time
            self.node_earliest_p[p] = r.earliest_pickup
            self.node_latest_p[p] = r.latest_pickup

        self.final_depot_names = [
            NetworkNodeName.make_hub_name(i + 1)
            for i in range(scenario.network.nr_hubs)
        ]
        # make the complete graph
        self.travel_times = scenario.network.get_travel_times_dict(
            scenario.agents, scenario.requests
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the inputs needed to reconstruct this context"""
        return {
            "scenario": self.scenario.to_dict(),
            "problem_type": self.problem_type.value,
            "strategy": self.strategy.value,
            "model_params": self.model_params,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProblemContext:
        """Reconstruct a context and all of its derived lookup tables"""
        return cls(
            ModemsScenario.from_dict(data["scenario"]),
            ProblemType(data["problem_type"]),
            SolverStrategy(data["strategy"]),
            data["model_params"],
        )

    def t_travel(self, i: str, j: str) -> float:
        """Travel time c_ij between two virtual nodes"""
        if i == j:
            return 0.0
        return self.travel_times[(i, j)]

    def nearest_final_depot(self, node: str) -> str:
        """argmin_{h in H_f} t_travel(node, h)"""
        return min(self.final_depot_names, key=lambda h: self.t_travel(node, h))

    def is_pickup(self, node_name: str) -> bool:
        """Wrapper to check node type from its name"""
        return NetworkNodeType.is_pickup(NetworkNodeName.get_node_type(node_name))

    def is_delivery(self, node_name: str) -> bool:
        """Wrapper to check node type from its name"""
        return NetworkNodeType.is_delivery(NetworkNodeName.get_node_type(node_name))

    def is_station(self, node_name: str) -> bool:
        """Wrapper to check node type from its name"""
        return NetworkNodeType.is_station(NetworkNodeName.get_node_type(node_name))

    def request_of(self, node_name: str) -> str | None:
        """Get associated request from the given node name, if it exists"""
        return self.node_request.get(node_name)

    def is_open(self) -> bool:
        """Wrapper to check if problem objective is open (OVRP)"""
        return ProblemType.is_open(self.problem_type)

    def is_selective(self) -> bool:
        """Whether this routing-problem variant may reject new requests"""
        return ProblemType.is_selective(self.problem_type)
