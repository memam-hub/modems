from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from modems.network import NetworkNodeName, NetworkNodeType

from .core import DEFAULT_PARAMS_OBJ, ModemsAgent, ProblemContext, SolverStrategy

# float infinity, stored as a number to simplify JSON exports
FLOAT_INF: float = 1.0e38

# tolerance for comparing floating-point variables/parameters
FLOAT_TOL: float = 1.0e-8

# default MILP parameters including objective calculation and big_m. Deliberately
# excludes all solver_options since they have different mappings, check SolverConfigType
DEFAULT_PARAMS_MILP = {**DEFAULT_PARAMS_OBJ, "big_m": 150.0}

# default MILP timelimt
DEFAULT_MILP_TIMELIMIT = 60.0

# default roulette wheel parameters for ALNS
DEFAULT_PARAMS_RW = {"scores": [5, 2, 1, 0.5], "decay": 0.8}

# default Simulated Annealing parameters for ALNS
DEFAULT_PARAMS_SA = {"start_temp": 1000, "end_temp": 0.1, "cooling_rate": 0.995}

# default [min_pct, max_pct] for destroy operation in ALNS
DEFAULT_PARAMS_RD = {"min_pct": 0.2, "max_pct": 0.4}

# default MILP parameters including OBJ, RW, SA, request destroy, and max_iterations
DEFAULT_PARAMS_ALNS = {
    "max_iter": 1000,
    "obj": DEFAULT_PARAMS_OBJ,
    "params_rw": DEFAULT_PARAMS_RW,
    "params_sa": DEFAULT_PARAMS_SA,
    "params_rd": DEFAULT_PARAMS_RD,
}


class SolutionStatus(StrEnum):
    """Solution status, the solver's own termination outcome"""

    unknown = "unknown"
    infeasible = "infeasible"
    feasible = "feasible"
    optimal = "optimal"


@dataclass(slots=True)
class NodeState:
    """
    Arrival/departure state at one route node

    Attributes:
        node: node name
        t_arr: arrival time
        t_wait: waiting time
        t_start: start of service time t_start = t_arr + t_wait
        t_dep: departure time after service t_dep = t_start + t_service
        tau: slack for pickup TW violation (for MILP1/MILP3)
        z_arr: number of passengers on arrival at node
        z_dep: number of passengers on departure from node
        phi_arr: SoC on arrival at node
        phi_dep: SoC on departure from node

    For the agent starting node and for a final depot, just duplicate the departure
    and arrival state values, respectively, to avoid special handling/cases
    """

    node: str
    t_arr: float = 0.0
    t_wait: float = 0.0
    t_start: float = 0.0
    t_dep: float = 0.0
    tau: float = 0.0
    z_arr: int = 0
    z_dep: int = 0
    phi_arr: float = 1.0
    phi_dep: float = 1.0

    def copy(self) -> NodeState:
        """Return a shallow copy of the NodeState"""
        return copy.copy(self)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> NodeState:
        return cls(**data)


def energy_consmp(
    ctx: ProblemContext, agent: ModemsAgent, n_from: str, n_to: str, z_dep_from: int
) -> float:
    """arc-energy consumption w.r.t. load. E_ij^k(z) := c_ij (alpha^k + beta^k z)"""
    return ctx.t_travel(n_from, n_to) * (agent.soc_alpha + agent.soc_beta * z_dep_from)


def _node_objective(
    ctx: ProblemContext, node: str, t_start: float, soft_tw: bool
) -> float:
    """
    g_i(t): a single request node's own contribution to the objective, as a pure
    function of its service-start time. Matches ModemsSolution.objective()'s per-request
    term so that summing g_i deltas is equivalent to a full objective recompute
    """
    contrib = ctx.eps * t_start
    if soft_tw and ctx.is_pickup(node):
        violation = max(0.0, t_start - ctx.node_latest_p[node])
        request = ctx.requests[ctx.node_request[node]]
        if ctx.strategy != SolverStrategy.alns and request.is_new():
            violation = max(
                violation,
                ctx.node_earliest_p[node] - t_start,
            )
        contrib += ctx.zeta * violation
    return contrib


class ModemsJourney:
    """An agent route, its timing schedule, load profile, and SoC profile"""

    ctx: ProblemContext
    agent_name: str
    agent: ModemsAgent
    route: list[str]
    states: list[NodeState]
    request_pickup: dict[str, int]
    request_delivery: dict[str, int]

    def __init__(
        self,
        ctx: ProblemContext,
        agent_name: str,
        states: list[NodeState] | None = None,
    ) -> None:
        """
        Initialize the journey. If states is empty, create a dummy route instead from
        agent initial node to (propagated) nearest depot
        """
        self.ctx = ctx
        self.agent_name = agent_name
        self.agent = ctx.agents[agent_name]
        if states:
            self.route = [s.node for s in states]
            self.states = [s.copy() for s in states]
            self._rebuild_request_indices()
            if not self._is_feasible():
                raise ValueError(
                    f"infeasible or inconsistent states for agent {agent_name!r}"
                )
        else:
            v_k = ctx.agent_initial_node[agent_name]
            h_f = ctx.nearest_final_depot(v_k)
            self.route = [v_k, h_f]
            self.states = [
                NodeState(
                    node=v_k,
                    t_arr=self.agent.time_initial,
                    t_wait=0.0,
                    t_start=self.agent.time_initial,
                    t_dep=self.agent.time_initial,
                    tau=0.0,
                    z_arr=0,
                    z_dep=0,
                    phi_arr=self.agent.soc_initial,
                    phi_dep=self.agent.soc_initial,
                )
            ]
            self.states.append(self._make_update_node(self.states[0], h_f))
            self.request_pickup = {}
            self.request_delivery = {}

    def copy(self) -> ModemsJourney:
        """Return a trusted copy sharing immutable journey data until mutation."""
        journey = ModemsJourney.__new__(ModemsJourney)
        journey.ctx = self.ctx
        journey.agent_name = self.agent_name
        journey.agent = self.agent
        journey.route = self.route
        journey.states = self.states
        journey.request_pickup = self.request_pickup
        journey.request_delivery = self.request_delivery
        return journey

    @classmethod
    def _from_propagated(
        cls,
        ctx: ProblemContext,
        agent_name: str,
        route: list[str],
        states: list[NodeState],
    ) -> ModemsJourney:
        """Build an internally propagated journey without full validation"""
        journey = cls.__new__(cls)
        journey.ctx = ctx
        journey.agent_name = agent_name
        journey.agent = ctx.agents[agent_name]
        journey.route = route
        journey.states = states
        journey._rebuild_request_indices()
        return journey

    @classmethod
    def from_route(
        cls,
        ctx: ProblemContext,
        agent_name: str,
        route: list[str],
        tau_of: dict[str, float] | None = None,
        t_start_of: dict[str, float] | None = None,
    ) -> ModemsJourney:
        """
        Reconstruct a ModemsJourney by forward-propagating along an already-decided
        route (from a solved MILP/ALNS decision or a deserialized solution). When
        t_start_of is given, exact waiting is inferred as w_i = t_i - t_arr_i;
        otherwise, the active strategy's constructive scheduling rule is used.
        We preserve solved soft-TW slack using tau_of values (when applicable)
        """
        v_k = ctx.agent_initial_node[agent_name]
        if v_k != route[0]:
            raise ValueError(
                f"Inconsistent starting nodes agent {agent_name} should start at {v_k}"
                f"but the route starts at {route[0]}"
            )
        agent = ctx.agents[agent_name]
        first_state = NodeState(
            route[0],
            t_arr=agent.time_initial,
            t_start=agent.time_initial,
            t_dep=agent.time_initial,
            phi_arr=agent.soc_initial,
            phi_dep=agent.soc_initial,
        )
        journey = cls._from_propagated(ctx, agent_name, list(route), [first_state])
        for i in range(1, len(route)):
            journey.states.append(
                journey._make_update_node(
                    journey.states[i - 1],
                    route[i],
                    tau_of=tau_of,
                    t_start_of=t_start_of,
                )
            )
        journey._rebuild_request_indices()
        return journey

    @property
    def is_active(self) -> bool:
        """Return whether the agent is active or inactive (direct trip to hub)"""
        return len(self.route) > 2

    def _make_update_node(
        self,
        prev_state: NodeState,
        node: str,
        tau_of: dict[str, float] | None = None,
        t_start_of: dict[str, float] | None = None,
        pickup_t_start_floor_of: dict[str, float] | None = None,
    ) -> NodeState:
        """Following prev_state, create a new node and propagate/infer its states"""
        state = NodeState(node)
        self._propagate_node(
            prev_state,
            state,
            tau_of,
            t_start_of,
            pickup_t_start_floor_of,
        )
        return state

    def _propagate_node(
        self,
        prev_state: NodeState,
        state: NodeState,
        tau_of: dict[str, float] | None = None,
        t_start_of: dict[str, float] | None = None,
        pickup_t_start_floor_of: dict[str, float] | None = None,
    ) -> None:
        """ALNS Propagate arrival/departure state at node, following prev_state"""
        ctx = self.ctx
        agent = self.agent
        node = state.node
        state.t_arr = prev_state.t_dep + ctx.t_travel(prev_state.node, node)
        state.z_arr = prev_state.z_dep
        state.phi_arr = prev_state.phi_dep - energy_consmp(
            ctx, agent, prev_state.node, node, prev_state.z_dep
        )
        # get service start
        state.tau = 0.0
        state.t_wait = 0.0
        given_t_start = (t_start_of or {}).get(node)
        if given_t_start is not None and ctx.is_station(node):
            if given_t_start < state.t_arr - 1.0e-7:
                raise ValueError(
                    f"service start at {node} precedes its propagated arrival"
                )
            state.t_wait = max(0.0, given_t_start - state.t_arr)
            if ctx.is_pickup(node):
                state.tau = max(0.0, (tau_of or {}).get(node, 0.0))
        elif ctx.is_pickup(node):
            request = ctx.requests[ctx.node_request[node]]
            pickup_t_start_floor = (pickup_t_start_floor_of or {}).get(node)
            given_tau = (tau_of or {}).get(node)
            if pickup_t_start_floor is not None:
                state.t_wait = max(
                    0.0,
                    ctx.node_earliest_p[node] - state.t_arr,
                    pickup_t_start_floor - state.t_arr,
                )
            elif given_tau is not None:
                state.tau = max(0.0, given_tau)
                if (
                    request.is_scheduled()
                    or not ctx.strategy.has_soft_tw()
                    or ctx.strategy == SolverStrategy.alns
                ):
                    state.t_wait = max(0.0, ctx.node_earliest_p[node] - state.t_arr)
                else:
                    # early service is bounded: t_start >= e^r - omega
                    state.t_wait = max(
                        0.0,
                        ctx.node_earliest_p[node] - state.t_arr - state.tau,
                        ctx.node_earliest_p[node] - ctx.omega - state.t_arr,
                    )
            else:
                early_service_allowed = request.is_new() and ctx.strategy in (
                    SolverStrategy.milp1,
                    SolverStrategy.milp3,
                )
                # early service (MILP1/MILP3 new requests) starts at most omega
                # before e^r; everything else waits until e^r
                t_start_floor = ctx.node_earliest_p[node] - (
                    ctx.omega if early_service_allowed else 0.0
                )
                state.t_wait = max(0.0, t_start_floor - state.t_arr)
        # get departure state
        state.t_start = state.t_arr + state.t_wait
        if ctx.is_pickup(node) and ctx.strategy == SolverStrategy.alns:
            # ALNS never serves earlier than e^r; its only slack is tardiness
            state.tau = max(0.0, state.t_start - ctx.node_latest_p[node])
        elif (
            ctx.is_pickup(node)
            and tau_of is None
            and t_start_of is None
            and ctx.strategy.has_soft_tw()
        ):
            state.tau = max(
                0.0,
                ctx.node_earliest_p[node] - state.t_start,
                state.t_start - ctx.node_latest_p[node],
            )
        if ctx.is_station(node):
            state.t_dep = state.t_start + ctx.node_t_service[node]
            state.z_dep = state.z_arr + ctx.node_load[node]
        else:
            # final depot
            state.t_dep = state.t_arr
            state.z_dep = state.z_arr
        state.phi_dep = state.phi_arr

    def _make_update_from(
        self,
        i_start: int,
        tau_of: dict[str, float] | None = None,
        pickup_t_start_floor_of: dict[str, float] | None = None,
    ) -> None:
        """Make and propagate states for route[i_start:] trusting states[i_start-1]"""
        for i in range(i_start, len(self.route)):
            prev = self.states[i - 1]
            self.states.append(
                self._make_update_node(
                    prev,
                    self.route[i],
                    tau_of=tau_of,
                    pickup_t_start_floor_of=pickup_t_start_floor_of,
                )
            )

    def total_travel_time(self) -> float:
        """Sum of c_ij along the whole route, including the final-hub leg."""
        return sum(
            self.ctx.t_travel(self.route[i], self.route[i + 1])
            for i in range(len(self.route) - 1)
        )

    def total_energy(self) -> float:
        """Sum of E_ij^k(z_dep_i) along the whole route"""
        return sum(
            energy_consmp(
                self.ctx,
                self.agent,
                self.route[i],
                self.route[i + 1],
                self.states[i].z_dep,
            )
            for i in range(len(self.route) - 1)
        )

    def terminal_soc_slack(self) -> float:
        """xi^k := phi_arr_hf^k - sigma_min^k"""
        return self.states[-1].phi_arr - self.agent.soc_min_operational

    def _rebuild_request_indices(self) -> None:
        """Build and validate the cached pickup and delivery positions."""
        pickup: dict[str, int] = {}
        delivery: dict[str, int] = {}
        for index, node in enumerate(self.route[1:-1], start=1):
            request_name = self.ctx.request_of(node)
            if request_name is None:
                raise ValueError(
                    f"unknown service node {node!r} in route for {self.agent_name!r}"
                )
            target = pickup if self.ctx.is_pickup(node) else delivery
            expected = (
                self.ctx.pickup_node[request_name]
                if target is pickup
                else self.ctx.delivery_node[request_name]
            )
            if node != expected or request_name in target:
                raise ValueError(
                    f"inconsistent or duplicate node {node!r} for request "
                    f"{request_name!r}"
                )
            target[request_name] = index

        if pickup.keys() != delivery.keys():
            missing_pickups = sorted(delivery.keys() - pickup.keys())
            missing_deliveries = sorted(pickup.keys() - delivery.keys())
            raise ValueError(
                "route contains incomplete requests: "
                f"missing pickups={missing_pickups}, "
                f"missing deliveries={missing_deliveries}"
            )
        for request_name, pickup_index in pickup.items():
            if pickup_index >= delivery[request_name]:
                raise ValueError(
                    f"delivery precedes pickup for request {request_name!r}"
                )

        self.request_pickup = pickup
        self.request_delivery = delivery

    def onboard_at(self, idx: int) -> set[str]:
        """B_i^k: requests on-board when departing route position idx"""
        return {
            request_name
            for request_name, pickup_index in self.request_pickup.items()
            if pickup_index <= idx < self.request_delivery[request_name]
        }

    def _append_request_direct(self, request_name: str) -> ModemsJourney | None:
        """
        Naive approach: direct append-insertion candidate, extend journey by replacing
        the final depot h_f with (p^r, d^r, h_f^r). Works on a copy throughout,
        returning the updated candidate ModemsJourney if feasible; else None
        """
        ctx = self.ctx
        p = ctx.pickup_node[request_name]
        d = ctx.delivery_node[request_name]
        i_start = len(self.route) - 1

        # insert (p, d) before the final hub, then re-optimize the hub
        candidate = self.copy()
        h_f = self.ctx.nearest_final_depot(d)
        suffix = [p, d, h_f]
        candidate.route = candidate.route[:-1] + suffix
        candidate.states = candidate.states[:-1] + [NodeState(n) for n in suffix]
        for i in range(len(self.route) - 1, len(candidate.route)):
            candidate._propagate_node(candidate.states[i - 1], candidate.states[i])
        candidate._rebuild_request_indices()
        return candidate if candidate._is_alns_feasible(i_start) else None

    def remove_request(self, request_name: str) -> None:
        """
        ALNS remove a request while preserving a feasible schedule for survivors.
        A surviving pickup t_start may advance by at most its current max ride-time
        slack. Any remaining time becomes deliberate pickup waiting; deliveries and
        the final depot are propagated as early as possible
        """
        if request_name not in self.request_pickup:
            return  # was not removed from this journey
        ctx = self.ctx
        old_route = list(self.route)
        old_states = self.states
        removed_indices = {
            self.request_pickup[request_name],
            self.request_delivery[request_name],
        }
        t_start_old: dict[str, float] = {}
        tau_old: dict[str, float] = {}
        t_ride_slack: dict[str, float] = {}
        for r_survive, p_idx in self.request_pickup.items():  # cache r_survive t_data
            if r_survive == request_name:
                continue
            d_idx = self.request_delivery[r_survive]
            state_p_old = old_states[p_idx]
            state_d_old = old_states[d_idx]
            t_start_old[r_survive] = state_p_old.t_start
            tau_old[r_survive] = state_p_old.tau
            t_ride = (
                state_d_old.t_start
                - state_p_old.t_start
                - ctx.node_t_service[state_p_old.node]
            )
            max_t_ride = ctx.rho * ctx.t_travel(state_p_old.node, state_d_old.node)
            t_ride_slack[r_survive] = max(0.0, max_t_ride - t_ride)

        route_new = [n for idx, n in enumerate(old_route) if idx not in removed_indices]
        if route_new[-2] != old_route[-2]:  # re-opt final depot if needed
            route_new[-1] = ctx.nearest_final_depot(route_new[-2])

        state_0 = old_states[0].copy()
        candidate = self._from_propagated(ctx, self.agent_name, route_new, [state_0])
        for node in route_new[1:]:
            if ctx.is_pickup(node):
                r_survive = ctx.node_request[node]
                state_propg = candidate._make_update_node(candidate.states[-1], node)
                t_earliest = state_propg.t_start
                t_service = state_propg.t_dep - state_propg.t_start
                max_t_adv = max(0.0, t_start_old[r_survive] - t_earliest)
                t_advance = min(t_ride_slack[r_survive], max_t_adv)
                fix_t_start = t_start_old[r_survive] - t_advance
                state_propg.t_start = fix_t_start
                state_propg.t_wait = max(0.0, fix_t_start - state_propg.t_arr)
                state_propg.tau = max(0.0, tau_old[r_survive] - t_advance)
                state_propg.t_dep = state_propg.t_start + t_service
            else:
                state_propg = candidate._make_update_node(candidate.states[-1], node)
            candidate.states.append(state_propg)
        candidate._rebuild_request_indices()
        if not candidate._is_alns_feasible():
            raise ValueError(
                f"removing request {request_name!r} produced an infeasible "
                f"journey for agent {self.agent_name!r}"
            )

        self.route = candidate.route
        self.states = candidate.states
        self.request_pickup = candidate.request_pickup
        self.request_delivery = candidate.request_delivery

    def insert_at(self, request_name: str, a_idx: int, b_idx: int) -> None:
        """
        Naive approach: insert p^u after position a and d^u after shifted position b
        of the original route. Build and validate the selected mutation off-object
        so a failed insertion leaves this journey unchanged
        """
        if request_name in self.request_pickup:
            raise ValueError(f"request {request_name!r} is already in this journey")
        if not 0 <= a_idx <= b_idx < len(self.route) - 1:
            raise ValueError(
                f"invalid insertion positions a={a_idx}, b={b_idx} for route of "
                f"length {len(self.route)}"
            )
        ctx = self.ctx
        p_name = ctx.pickup_node[request_name]
        d_name = ctx.delivery_node[request_name]
        pickup_t_start_floor_of = {
            ctx.pickup_node[request_name]: self.states[pickup_index].t_start
            for request_name, pickup_index in self.request_pickup.items()
        }
        new_route = (
            self.route[: a_idx + 1]
            + [p_name]
            + self.route[a_idx + 1 : b_idx + 1]
            + [d_name]
            + self.route[b_idx + 1 :]
        )
        if new_route[-2] != self.route[-2]:
            new_route[-1] = ctx.nearest_final_depot(new_route[-2])

        candidate = self.copy()
        candidate.route = new_route
        candidate.states = candidate.states[: a_idx + 1]
        candidate._make_update_from(
            len(candidate.states), pickup_t_start_floor_of=pickup_t_start_floor_of
        )
        candidate._rebuild_request_indices()
        if not candidate._is_alns_feasible():
            raise ValueError(
                f"selected insertion of {request_name!r} is infeasible for "
                f"agent {self.agent_name!r}"
            )

        self.route = candidate.route
        self.states = candidate.states
        self.request_pickup = candidate.request_pickup
        self.request_delivery = candidate.request_delivery

    def _is_alns_feasible(self, i_start: int = 1, tol: float = FLOAT_TOL) -> bool:
        """Validate constraints that may change during ALNS propagation"""
        ctx = self.ctx
        agent = self.agent

        if not ctx.strategy.has_extended_soc():
            if self.total_travel_time() > agent.duration_max + tol:
                return False
        else:
            # extended SoC: every visited service/hub node keeps phi_arr >= sigma_min
            for s in self.states[i_start:]:
                if s.phi_arr < agent.soc_min_operational - tol:
                    return False

        for r_name, p_idx in self.request_pickup.items():
            d_idx = self.request_delivery[r_name]
            p_state, d_state = self.states[p_idx], self.states[d_idx]
            t_ride = (
                d_state.t_start - p_state.t_start - ctx.node_t_service[p_state.node]
            )
            if t_ride < ctx.t_travel(p_state.node, d_state.node) - tol:
                return False
            if t_ride > ctx.rho * ctx.t_travel(p_state.node, d_state.node) + tol:
                return False
        for s in self.states:
            if (s.z_dep > agent.load_max + tol) or (s.z_dep < -tol):
                return False

        for r_name, p_idx in self.request_pickup.items():
            p_state = self.states[p_idx]
            request = ctx.requests[r_name]
            if (
                request.is_scheduled() or not ctx.strategy.has_soft_tw()
            ) and p_state.t_start < ctx.node_earliest_p[p_state.node] - tol:
                return False
            # soft-TW early service is bounded: t_start >= e^r - omega
            if p_state.t_start < ctx.node_earliest_p[p_state.node] - ctx.omega - tol:
                return False
            if (
                not ctx.strategy.has_soft_tw()
                and p_state.t_start > ctx.node_latest_p[p_state.node] + tol
            ):
                return False
        return True

    def _is_feasible(self, tol: float = FLOAT_TOL) -> bool:
        """Fully validate route structure, propagated states, and constraints"""
        ctx = self.ctx
        agent = self.agent
        if len(self.route) != len(self.states) or len(self.route) < 2:
            return False
        if self.route[0] != ctx.agent_initial_node[self.agent_name]:
            return False
        if self.route[-1] not in ctx.final_depot_names:
            return False
        if len(set(self.route[1:-1])) != len(self.route[1:-1]):
            return False
        if any(state.node != node for state, node in zip(self.states, self.route)):
            return False

        expected_requests = set(self.request_pickup)
        if expected_requests != set(self.request_delivery):
            return False
        for r_name in expected_requests:
            pickup_index = self.request_pickup[r_name]
            delivery_index = self.request_delivery[r_name]
            if (
                pickup_index <= 0
                or delivery_index <= pickup_index
                or delivery_index >= len(self.route) - 1
                or self.route[pickup_index] != ctx.pickup_node[r_name]
                or self.route[delivery_index] != ctx.delivery_node[r_name]
            ):
                return False

        first_state = self.states[0]
        if (
            abs(first_state.t_arr - agent.time_initial) > tol
            or abs(first_state.t_start - agent.time_initial) > tol
            or abs(first_state.t_dep - agent.time_initial) > tol
            or first_state.z_arr != 0
            or first_state.z_dep != 0
            or abs(first_state.phi_arr - agent.soc_initial) > tol
            or abs(first_state.phi_dep - agent.soc_initial) > tol
        ):
            return False

        for index in range(1, len(self.states)):
            previous = self.states[index - 1]
            state = self.states[index]
            expected_arrival = previous.t_dep + ctx.t_travel(previous.node, state.node)
            expected_soc = previous.phi_dep - energy_consmp(
                ctx, agent, previous.node, state.node, previous.z_dep
            )
            if (
                abs(state.t_arr - expected_arrival) > tol
                or state.z_arr != previous.z_dep
                or abs(state.phi_arr - expected_soc) > tol
                or state.t_wait < -tol
                or abs(state.t_start - (state.t_arr + state.t_wait)) > tol
                or abs(state.phi_dep - state.phi_arr) > tol
                or state.tau < -tol
            ):
                return False
            if ctx.is_station(state.node):
                if (
                    abs(state.t_dep - (state.t_start + ctx.node_t_service[state.node]))
                    > tol
                    or state.z_dep != state.z_arr + ctx.node_load[state.node]
                ):
                    return False
            elif abs(state.t_dep - state.t_arr) > tol or state.z_dep != state.z_arr:
                return False

            if ctx.is_pickup(state.node):
                minimum_tau = max(
                    0.0,
                    state.t_start - ctx.node_latest_p[state.node],
                )
                request = ctx.requests[ctx.node_request[state.node]]
                if ctx.strategy != SolverStrategy.alns and request.is_new():
                    minimum_tau = max(
                        minimum_tau,
                        ctx.node_earliest_p[state.node] - state.t_start,
                    )
                if ctx.strategy.has_soft_tw() and state.tau < minimum_tau - tol:
                    return False

        return self._is_alns_feasible(tol=tol)


class ModemsSolution:
    """
    A complete (or partial) routing solution: one ModemsJourney per agent, in
    addition to the accepted/rejected request partition
    """

    ctx: ProblemContext
    journeys: dict[str, ModemsJourney]
    accepted: set[str]
    rejected: set[str]
    _infeasible: bool

    def __init__(self, ctx: ProblemContext) -> None:
        """
        Initialize a solution with dummy journeys for all context (scenario) agents,
        validating basic agent initial-to-hub feasibility
        """
        self.ctx = ctx
        self.journeys = {k: ModemsJourney(ctx, k) for k in ctx.agent_names}
        self.accepted = set()
        self.rejected = set()
        self._infeasible = False

    def copy(self) -> ModemsSolution:
        """Return a trusted hot-path copy without deepcopy"""
        sol = ModemsSolution.__new__(ModemsSolution)
        sol.ctx = self.ctx
        sol.journeys = {k: j.copy() for k, j in self.journeys.items()}
        sol.accepted = set(self.accepted)
        sol.rejected = set(self.rejected)
        sol._infeasible = self._infeasible
        return sol

    def agent_of(self, request_name: str) -> str | None:
        """return the agent name, where the request is currently assigned (if any)"""
        for k, j in self.journeys.items():
            if request_name in j.request_pickup:
                return k
        return None

    def pending(self) -> set[str]:
        """Requests neither accepted nor rejected (available for insertion, R_u)"""
        return set(self.ctx.request_names) - self.accepted

    def recompute_rejected(self) -> None:
        """Recompute R_r = R \\setminus R_a after a repair pass concludes"""
        self.rejected = self.pending()

    def mission_time(self) -> float:
        """T(Omega): the fleet-wide max completion time. For closed VRP, max over
        active agents journeys at fina-hub arrival times. For open, max over last
        request delivery departure time (after service). Extracted from objective()
        so callers wanting T_0 only can forego a full objective recompute
        """
        ctx = self.ctx
        is_open = ctx.is_open()
        mission_time = 0.0
        for j in self.journeys.values():
            if not j.is_active:
                continue
            if is_open:
                mission_time = max(mission_time, j.states[-2].t_dep)
            else:
                mission_time = max(mission_time, j.states[-1].t_arr)
        return mission_time

    def objective(self, include_rejection: bool = True) -> float:
        """
        F(Omega), the strategy-specific objective function, combining mission_time
        request arrival times, TW violations (for soft variants), rejected requests
        (for selective variants). Rejection weight (very large) can be excluded to
        avoid contaminating ALNS objectives during destroy/repair cycles
        """
        if getattr(self, "_infeasible", False):
            return FLOAT_INF
        ctx = self.ctx
        soft_tw = self.ctx.strategy.has_soft_tw()
        selective = self.ctx.is_selective()

        obj = self.mission_time()

        for r in ctx.request_names:
            if r not in self.accepted:
                if selective and include_rejection:
                    obj += ctx.eta
                continue
            k = self.agent_of(r)
            if k is None:
                continue
            journey = self.journeys[k]
            p_idx = journey.request_pickup[r]
            d_idx = journey.request_delivery[r]
            t_p, t_d = journey.states[p_idx].t_start, journey.states[d_idx].t_start
            obj += ctx.eps * (t_p + t_d)
            if soft_tw:
                p_node = journey.states[p_idx].node
                tau_p = max(
                    0.0,
                    ctx.node_earliest_p[p_node] - t_p,
                    t_p - ctx.node_latest_p[p_node],
                )
                obj += ctx.zeta * tau_p
        return obj

    def total_energy(self) -> float:
        """Total energy consumption of all active journeys"""
        return sum(j.total_energy() for j in self.journeys.values() if j.is_active)

    def get_agent_routes(self) -> dict[str, list[str]]:
        """Return the traversed agent routes, keyed by agent name"""
        return {k: list(j.route) for k, j in self.journeys.items()}

    # ----------------------------------------------------------------------------------
    # Plotting helpers: timing, SoC, and load over time
    # ----------------------------------------------------------------------------------

    def plot_timing(
        self,
        default_id: int = 0,
        legend: bool = True,
        show: bool = True,
        outfile: str = "",
    ) -> None:
        """
        Plot a timing diagram: each request pickup time window + inferred delivery time
        window from the direct trip time as horizontal bars, overlaid with each agent
        actual visit times as a connected line through the requests it visits
        """
        ctx = self.ctx
        network = ctx.scenario.network
        fig, ax = network._setup_fig_axis("Time (min)", "Request ID")

        # colors and markers
        p_color, p_marker = network._node_color(NetworkNodeType.pickup), "s"
        d_color, d_marker = network._node_color(NetworkNodeType.delivery), "s"
        for r_name, request in ctx.requests.items():
            r_idx = NetworkNodeName.get_request_index(r_name)
            p_name = ctx.pickup_node[r_name]
            d_name = ctx.delivery_node[r_name]
            t_direct_trip = ctx.t_travel(p_name, d_name)
            earliest_delivery = request.earliest_pickup + t_direct_trip
            latest_delivery = request.latest_pickup + t_direct_trip
            ax.plot(
                [request.earliest_pickup, request.latest_pickup],
                [r_idx, r_idx],
                color=p_color,
                marker=p_marker,
                markersize=14,
                linewidth=4,
            )
            ax.plot(
                [earliest_delivery, latest_delivery],
                [r_idx, r_idx],
                color=d_color,
                marker=d_marker,
                markersize=14,
                linewidth=4,
            )

        a_marker = "o"
        t_max = 0.0
        for a_idx, (a_name, a_sol) in enumerate(
            dict(sorted(self.journeys.items())).items()
        ):
            pts_t, pts_y = [], []
            for state in a_sol.states:
                r_name = ctx.request_of(state.node)
                pts_t.extend([state.t_arr, state.t_dep])
                if r_name is not None:
                    r_idx = NetworkNodeName.get_request_index(r_name)
                    pts_y.extend([r_idx, r_idx])
                else:
                    pts_y.extend([default_id, default_id])
                t_max = max(t_max, state.t_dep)
            if pts_t:
                ax.plot(
                    pts_t,
                    pts_y,
                    color=network._agent_color(a_idx),
                    marker=a_marker,
                    markersize=8,
                    linewidth=2,
                    label=a_name,
                )

        ax.set_ylim(-0.5, len(ctx.requests) + 0.5)
        ax.set_yticks(range(len(ctx.requests) + 1))
        t_max = 1.05 * t_max
        tick = max(10, int(round(t_max / 50) * 5))
        ax.set_xlim(0, t_max)
        ax.set_xticks(range(0, int(t_max), tick))

        if legend:
            handles = [
                Line2D(
                    [0],
                    [0],
                    color=p_color,
                    marker=p_marker,
                    linewidth=3,
                    label="Pickup service",
                ),
                Line2D(
                    [0],
                    [0],
                    color=d_color,
                    marker=d_marker,
                    linewidth=3,
                    label="Delivery service",
                ),
            ]
            labels = ["Pickup service", "Delivery service"]
            a_handles, a_labels = ax.get_legend_handles_labels()
            handles += a_handles
            labels += a_labels
            ax.legend(
                handles,
                labels,
                loc="best",
                framealpha=0.7,
                edgecolor="k",
                facecolor="w",
            )

        if outfile:
            plt.savefig(outfile, bbox_inches="tight", dpi=300)
        if show:
            plt.show()
        plt.close(fig)

    def _agent_soc_load_series(
        self,
    ) -> tuple[dict[str, dict[str, list[float]]], float, int]:
        """
        Build, per agent, the (time, SoC) and (time, load) step series shared by
        plot_soc and plot_load, so the node-by-node walk over each route is done once
        """
        series: dict[str, dict[str, list[float]]] = {}
        t_max: float = 0.0
        max_load: int = 1
        for a_name, a_journey in self.journeys.items():
            if not a_journey.states:
                continue
            t_soc: list[float] = []
            soc: list[float] = []
            t_load: list[float] = []
            load: list[float] = []
            for state in a_journey.states:
                t_soc.extend([state.t_arr, state.t_dep])
                soc.extend([state.phi_arr, state.phi_dep])
                node_type = NetworkNodeType(NetworkNodeName.get_node_type(state.node))
                if node_type == NetworkNodeType.pickup:
                    t_load.extend([state.t_arr, state.t_dep, state.t_dep])
                    load.extend([state.z_arr, state.z_arr, state.z_dep])
                elif node_type == NetworkNodeType.delivery:
                    t_load.extend([state.t_arr, state.t_arr, state.t_dep])
                    load.extend([state.z_arr, state.z_dep, state.z_dep])
                else:
                    t_load.extend([state.t_arr, state.t_dep])
                    load.extend([state.z_arr, state.z_dep])
                t_max = max(t_max, state.t_dep)
                max_load = max(max_load, state.z_arr, state.z_dep)
            series[a_name] = {
                "t_soc": t_soc,
                "soc": soc,
                "t_load": t_load,
                "load": load,
            }
        # small offset to improve visibility
        t_max *= 1.02
        return dict(sorted(series.items())), t_max, max_load

    def plot_soc(
        self, legend: bool = True, show: bool = True, outfile: str = ""
    ) -> None:
        """
        Plot SoC for all agents over time, along with the minimum allowed SoC

        Args:
            legend: Optional. Whether to create a legend for the plot. Default is True
            show: Optional. Whether to show the plot. Default is True
            outfile: File name to save plot figure. File is not created if empty
        """
        ctx = self.ctx
        network = ctx.scenario.network
        fig, ax = network._setup_fig_axis("Time (min)", "SoC", alpha=False)
        ax.grid(True, axis="x", alpha=0.5)

        series, t_max, _ = self._agent_soc_load_series()
        ax.set_xlim(0, max(1.0, t_max))
        ax.set_ylim(0.0, 1.01)
        soc_min_values = sorted({a.soc_min_operational for a in ctx.agents.values()})
        ax.set_yticks(sorted({0.0, *soc_min_values, 0.5, 0.75, 1.0}))

        handles, labels = [], []
        # soc data
        for a_idx, (a_name, a_series) in enumerate(series.items()):
            (h,) = ax.plot(
                a_series["t_soc"],
                a_series["soc"],
                color=network._agent_color(a_idx),
                marker="o",
                markersize=6,
                linewidth=2,
                label=f"{a_name}",
            )
            handles.append(h)
            labels.append(h.get_label())
        for idx, soc_min in enumerate(soc_min_values):
            min_line = ax.axhline(
                soc_min,
                c="k",
                alpha=0.7,
                ls="--",
                lw=3,
                label="min SoC" if idx == 0 else None,
            )
        handles.append(min_line)
        labels.append("min SoC" if len(soc_min_values) == 1 else "min SoC (per agent)")

        if legend:
            ax.legend(
                handles,
                labels,
                loc="best",
                framealpha=0.7,
                edgecolor="k",
                facecolor="w",
            )

        if outfile:
            plt.savefig(outfile, bbox_inches="tight", dpi=300)
        if show:
            plt.show()
        plt.close(fig)

    def plot_load(
        self, legend: bool = False, show: bool = True, outfile: str = ""
    ) -> None:
        """
        Plot passenger and equipment loads for all agents over time

        Args:
            legend: Optional. Whether to create a legend for the plot. Default is True
            show: Optional. Whether to show the plot. Default is True
            outfile: File name to save plot figure. File is not created if empty
        """
        ctx = self.ctx
        network = ctx.scenario.network
        fig, ax = network._setup_fig_axis("Time (min)", "Load")

        series, t_max, max_load = self._agent_soc_load_series()
        ax.set_xlim(0, max(1.0, t_max))
        ax.set_ylim(-0.1, max_load + 0.1)
        ax.set_yticks(range(0, max_load + 1))

        handles, labels = [], []
        # load data
        for a_idx, (a_name, a_series) in enumerate(series.items()):
            (h,) = ax.plot(
                a_series["t_load"],
                a_series["load"],
                color=network._agent_color(a_idx),
                marker="o",
                markersize=6,
                linewidth=2,
                label=f"{a_name}",
            )
            handles.append(h)
            labels.append(h.get_label())

        if legend:
            ax.legend(
                handles,
                labels,
                loc="best",
                framealpha=0.7,
                edgecolor="k",
                facecolor="w",
            )

        if outfile:
            plt.savefig(outfile, bbox_inches="tight", dpi=300)
        if show:
            plt.show()
        plt.close(fig)

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation"""
        journeys = {}
        for k, j in self.journeys.items():
            journeys[k] = {
                "route": list(j.route),
                "states": [s.to_dict() for s in j.states],
            }
        return {
            "journeys": journeys,
            "accepted": sorted(self.accepted),
            "rejected": sorted(self.rejected),
        }

    @classmethod
    def from_dict(cls, ctx: ProblemContext, data: dict) -> ModemsSolution:
        """
        Reconstruct a ModemsSolution against the given ProblemContext (which the
        caller must build from the same serialized ProblemContext -- see
        ModemsInstance.from_dict, which persists and restores it).
        Loads the stored route and states directly (no re-propagation --
        the file is the source of truth), then runs a feasibility check on
        every loaded journey and raises if it fails: a solution that claims
        to be valid without actually being one is worse than a loud failure,
        and the check is cheap relative to everything else in this path.
        """
        journey_names = set(data["journeys"])
        expected_agents = set(ctx.agent_names)
        if journey_names != expected_agents:
            raise ValueError(
                "loaded journey agents do not match the ProblemContext: "
                f"missing={sorted(expected_agents - journey_names)}, "
                f"unknown={sorted(journey_names - expected_agents)}"
            )
        solution = cls(ctx)
        routed: set[str] = set()
        for a_name, j_data in data["journeys"].items():
            states = [NodeState.from_dict(s) for s in j_data["states"]]
            journey = ModemsJourney(ctx, a_name, states)
            duplicates = routed.intersection(journey.request_pickup)
            if duplicates:
                raise ValueError(
                    f"requests assigned to multiple journeys: {sorted(duplicates)}"
                )
            routed.update(journey.request_pickup)
            solution.journeys[a_name] = journey
        solution.accepted = set(data["accepted"])
        solution.rejected = set(data["rejected"])
        if routed != solution.accepted:
            raise ValueError("accepted requests do not match the loaded journeys")
        if solution.accepted.intersection(solution.rejected):
            raise ValueError("accepted and rejected request sets overlap")
        unknown = (solution.accepted | solution.rejected) - set(ctx.request_names)
        if unknown:
            raise ValueError(f"solution contains unknown requests: {sorted(unknown)}")
        return solution


@dataclass(slots=True)
class ModemsSolutionInfo:
    """
    Solver-details overview of a ModemsSolution: original solver details, and solution
    quality and time. Always stored alongside the ModemsSolution and the ModemsScenario
    it was solved for (ModemsInstance), so a result is never separated from the context
    needed to interpret or reproduce it.
    """

    status: SolutionStatus | str = SolutionStatus.unknown
    solution_time: float = 0.0
    objective: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    solver_name: str = ""
    solver_options: dict[str, Any] | None = None
    solver_diagnostics: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.status = SolutionStatus(self.status)
        self.solver_options = dict(self.solver_options or {})
        self.solver_diagnostics = dict(self.solver_diagnostics or {})

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ModemsSolutionInfo:
        return cls(**data)


# --------------------------------------------------------------------------------------
# ModemsInstance: binds a ProblemContext, solution, and solution information
# --------------------------------------------------------------------------------------


class ModemsInstance:
    """
    Container binding a ProblemContext, the resulting ModemsSolution, and a
    ModemsSolutionInfo overview. Both ModemsMilp.solve() and ModemsAlns.solve()
    populate one of these
    """

    def __init__(
        self,
        ctx: ProblemContext,
        solution: ModemsSolution,
        solution_info: ModemsSolutionInfo,
    ) -> None:
        if solution.ctx is not ctx:
            raise ValueError("solution and instance must share the same ProblemContext")
        self.ctx = ctx
        self.solution = solution
        self.solution_info = solution_info

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable instance representation"""
        return {
            "problem_context": self.ctx.to_dict(),
            "solution": self.solution.to_dict(),
            "solution_info": self.solution_info.to_dict(),
            "diagnostics": self.diagnostics(),
        }

    def diagnostics(self) -> dict[str, Any]:
        """Return a solver-independent route and request diagnostic."""
        ctx = self.ctx
        solution = self.solution
        agents: dict[str, Any] = {}
        for agent_name, journey in solution.journeys.items():
            route = list(journey.route)
            r_names: list[str] = [
                ctx.request_of(node) for node in route if ctx.is_pickup(node)
            ]  # type: ignore : all requests exist, safe
            agents[agent_name] = {
                "name": agent_name,
                "is_idle": not journey.is_active,
                "requests": r_names,
                "route": route,
                "travel_times": [
                    ctx.t_travel(route[index], route[index + 1])
                    for index in range(len(route) - 1)
                ],
                "nodes": {
                    state.node: {
                        "arrival_time": state.t_arr,
                        "waiting_time": state.t_wait,
                        "service_start": state.t_start,
                        "departure_time": state.t_dep,
                        "time_window_violation": state.tau,
                        "arrival_load": state.z_arr,
                        "departure_load": state.z_dep,
                        "arrival_soc": state.phi_arr,
                        "departure_soc": state.phi_dep,
                    }
                    for state in journey.states
                },
            }

        requests: dict[str, Any] = {}
        for request_name in ctx.request_names:
            accepted = request_name in solution.accepted
            diagnostic: dict[str, Any] = {
                "name": request_name,
                "is_accepted": accepted,
                "assigned_agent": None,
                "path_time": ctx.t_travel(
                    ctx.pickup_node[request_name],
                    ctx.delivery_node[request_name],
                ),
                "pickup_time": None,
                "delivery_time": None,
                "ride_time": None,
                "waiting_time": None,
                "time_window_violation": None,
            }
            if accepted:
                agent_name = solution.agent_of(request_name)
                if agent_name is not None:
                    journey = solution.journeys[agent_name]
                    pickup = journey.states[journey.request_pickup[request_name]]
                    delivery = journey.states[journey.request_delivery[request_name]]
                    diagnostic.update(
                        {
                            "assigned_agent": agent_name,
                            "pickup_time": pickup.t_start,
                            "delivery_time": delivery.t_start,
                            "ride_time": (
                                delivery.t_start
                                - pickup.t_start
                                - ctx.node_t_service[pickup.node]
                            ),
                            "waiting_time": pickup.t_wait,
                            "time_window_violation": pickup.tau,
                        }
                    )
            requests[request_name] = diagnostic

        lower_bound = self.solution_info.lower_bound
        upper_bound = self.solution_info.upper_bound
        duality_gap = None
        if (
            lower_bound is not None
            and upper_bound is not None
            and lower_bound != FLOAT_INF
            and upper_bound != FLOAT_INF
            and math.isfinite(lower_bound)
            and math.isfinite(upper_bound)
            and upper_bound != 0
        ):
            duality_gap = (upper_bound - lower_bound) / upper_bound
        return {
            "duality_gap": duality_gap,
            "solver": dict(self.solution_info.solver_diagnostics or {}),
            "agents": agents,
            "requests": requests,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModemsInstance:
        """Reconstruct a full ModemsInstance from a dict"""
        ctx = ProblemContext.from_dict(data["problem_context"])
        solution = ModemsSolution.from_dict(ctx, data["solution"])
        solution_info = ModemsSolutionInfo.from_dict(data["solution_info"])
        return cls(ctx, solution, solution_info)

    def to_json(self, file_path: str) -> str:
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4, default=str)
        return str(file_path)

    @classmethod
    def from_json(cls, file_path: str) -> ModemsInstance:
        with open(file_path, "r") as f:
            return cls.from_dict(json.load(f))

    def plot(
        self,
        outdir: str,
        name: str | None = None,
        legend: bool = True,
        show: bool = False,
    ) -> dict[str, str]:
        """
        Plot the scenario network and every solution diagnostic (routes, timing, SoC,
        and load) in one call. Return a dict of the generated file paths; "soc_plot"
        is omitted for strategies without extended-SoC (MILP1)
        """
        import os

        os.makedirs(outdir, exist_ok=True)
        scenario = self.ctx.scenario
        base = os.path.join(outdir, name or scenario.make_scenario_name())

        paths = {}
        paths["network_plot"] = f"{base}_network.png"
        scenario.network.plot_network(
            legend=legend, show=show, outfile=paths["network_plot"]
        )

        is_open = self.ctx.is_open()
        routes = {
            k: (route[:-1] if is_open else route)
            for k, route in self.solution.get_agent_routes().items()
            if self.solution.journeys[k].is_active
        }
        paths["routes_plot"] = f"{base}_routes.png"
        scenario.network.plot_routes(
            routes, legend=legend, show=show, outfile=paths["routes_plot"]
        )

        paths["timing_plot"] = f"{base}_timing.png"
        self.solution.plot_timing(show=show, outfile=paths["timing_plot"])

        if self.solution.ctx.strategy.has_extended_soc():
            paths["soc_plot"] = f"{base}_soc.png"
            self.solution.plot_soc(legend=legend, show=show, outfile=paths["soc_plot"])

        paths["load_plot"] = f"{base}_loads.png"
        self.solution.plot_load(legend=legend, show=show, outfile=paths["load_plot"])

        return paths

    def print_summary(self) -> None:
        """Print a readable solution summary"""
        print(f"Objective Value: {self.solution_info.objective}")
        for a_name, journey in self.solution.journeys.items():
            print(f"Agent {a_name}:")
            if not journey.is_active:
                print("  idle")
                continue
            print("  route: " + " -> ".join(journey.route))
        for r_name in self.ctx.request_names:
            accepted = r_name in self.solution.accepted
            print(f"Request {r_name}: accepted={accepted}")
