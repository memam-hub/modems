from __future__ import annotations

from dataclasses import dataclass

from .core import ModemsRequest, ProblemContext, SolverStrategy
from .solution import (
    FLOAT_TOL,
    ModemsJourney,
    ModemsSolution,
    NodeState,
    _node_objective,
    energy_consmp,
)

# --------------------------------------------------------------------------------------
# Algorithm 1: Preprocessing and base plan generation
# --------------------------------------------------------------------------------------


def preprocess(
    ctx: ProblemContext,
    partial_plan: ModemsSolution | None = None,
) -> tuple[ModemsSolution | None, set[str]]:
    """
    Algorithm 1: Build the feasible base plan, return (solution, unassigned_requests).
    Failure flag is (solution is None). When scheduled requests exist, partial_plan
    must be an already-updated excerpt of the previous solution matching the current
    snapshot: its agent/request names, route order, service-start times, and soft-TW
    slack decisions are preserved exactly, whereas load, arrival, departure, and SoC
    states are safely re-propagated
    """
    r_scheduled = [r for r in ctx.request_names if ctx.requests[r].is_scheduled()]
    r_unassigned = {r for r in ctx.request_names if ctx.requests[r].is_new()}

    # check that a partial_plan exists if there are r_scheduled
    if r_scheduled and partial_plan is None:
        return None, r_unassigned

    # assert basic feasibility (all agents reache their nearest depot)
    solution = ModemsSolution(ctx)
    for journey in solution.journeys.values():
        if not journey._is_feasible():
            return None, r_unassigned

    # all new requests, return
    if partial_plan is None:
        return solution, r_unassigned

    # assert all agent journeys got updated in partial_plan
    if set(partial_plan.journeys) != set(ctx.agent_names):
        return None, r_unassigned

    # assert all inherited r_scheduled (not yet serviced) exist in current context
    for r_name in r_scheduled:
        request_inhrt = partial_plan.ctx.requests.get(r_name)
        if (
            request_inhrt is None
            or request_inhrt.request_id != ctx.requests[r_name].request_id
        ):
            return None, r_unassigned

    # parse all journeys in partial_plan, checking for inconsistencies
    seen_nodes: set[str] = set()
    for a_name in ctx.agent_names:
        journey_inhrt = partial_plan.journeys[a_name]
        if not journey_inhrt.is_active:
            continue
        service_nodes = journey_inhrt.route[1:-1]
        if any(node not in ctx.node_request for node in service_nodes):
            return None, r_unassigned
        if any(node in seen_nodes for node in service_nodes):
            return None, r_unassigned
        if any(
            not ctx.requests[ctx.node_request[node]].is_scheduled()
            for node in service_nodes
        ):
            return None, r_unassigned
        seen_nodes.update(service_nodes)

        # attempt to adopt the journey, propagating and validating feasibility
        route = [
            ctx.agent_initial_node[a_name],
            *service_nodes,
            journey_inhrt.route[-1],
        ]
        t_start_of = {state.node: state.t_start for state in journey_inhrt.states[1:-1]}
        tau_of = {
            state.node: state.tau
            for state in journey_inhrt.states[1:-1]
            if ctx.is_pickup(state.node)
        }
        try:
            journey = ModemsJourney.from_route(
                ctx,
                a_name,
                route,
                tau_of=tau_of,
                t_start_of=t_start_of,
            )
        except (KeyError, ValueError):
            return None, r_unassigned
        if not journey._is_feasible():
            return None, r_unassigned
        solution.journeys[a_name] = journey
        solution.accepted.update(journey.request_pickup)

    # assert all r_scheduled got accepted
    if not set(r_scheduled).issubset(solution.accepted):
        return None, r_unassigned

    return solution, r_unassigned


# --------------------------------------------------------------------------------------
# Algorithm 2: Greedy completion of a feasible base plan
# --------------------------------------------------------------------------------------


def _sort_key(ctx: ProblemContext, request_name: str) -> tuple[float, int, str]:
    """
    Return a guaranteed-unique (quasi-randomized) tie-breaker using r_name. Without
    this, two requests sharing the same (latest_pickup, load) would have to trust
    sorted() stability, which is not reliable
    """
    r = ctx.requests[request_name]
    return (r.latest_pickup, -r.load, request_name)


def greedy_complete(
    solution: ModemsSolution, r_unassigned: set[str]
) -> tuple[ModemsSolution, set[str], bool]:
    """
    Algorithm 2: complete the base plan, return (solution, r_rejected, success_flag).
    For MILPs, candidates are direct end-of-route insertions (one for each capacity-
    compatible agent): quick and cheap. For ALNS, the full feasible-insertion
    enumeration (Algorithm 3) is used to yield a good starting point for ALNS.
    """
    solution = solution.copy()
    r_rejected = set()
    # sort requests: latest pickup (earliest deadline) first, then highest demand
    r_ordered = sorted(r_unassigned, key=lambda r: _sort_key(solution.ctx, r))
    kappa = solution.ctx.strategy

    for r_name in r_ordered:
        selection_best: tuple[float, ModemsJourney] | None = None
        if kappa == SolverStrategy.alns:
            candidates = alns_feasible_insertions(solution, r_name)
        else:
            candidates = milp_feasible_insertions(solution, r_name)
        for candidate in candidates:
            if selection_best is None or candidate.delta_obj < selection_best[0]:
                selection_best = (candidate.delta_obj, candidate.journey)
        if selection_best is not None:
            solution.journeys[selection_best[1].agent_name] = selection_best[1]
            solution.accepted.add(r_name)
            solution.rejected.discard(r_name)
            continue

        # no feasible candidate found, break if strategy is not selective (no rejection)
        r_rejected.add(r_name)
        solution.rejected.add(r_name)
        if not solution.ctx.is_selective():
            return solution, r_rejected, False

    return solution, solution.rejected, True


# --------------------------------------------------------------------------------------
# Feasible request insertions and Algorithm 3: ALNS optimized enumeration
# --------------------------------------------------------------------------------------


def _collect_stats(stats: dict[str, int] | None, key: str, nr_key: int = 1) -> None:
    """
    Does nothing unless a stats dict was supplied; used to instrument Algorithm 3 for
    the insertion-ablation harness without additional productional cost
    """
    if stats is not None:
        stats[key] = stats.get(key, 0) + nr_key


@dataclass(slots=True)
class InsertionCandidate:
    """Wrapper for the insertion candidate: a journey with an objective difference"""

    journey: ModemsJourney
    delta_obj: float


def milp_feasible_insertions(
    solution: ModemsSolution,
    unassigned_name: str,
    agents: list[str] | None = None,
    tol: float = FLOAT_TOL,
) -> list[InsertionCandidate]:
    """Enumerate feasible MILP insertions (append only), use same signature as ALNS"""
    ctx = solution.ctx
    agents = ctx.agent_names if agents is None else agents
    compatible_journeys = [
        solution.journeys[k]
        for k in agents
        if (ctx.requests[unassigned_name].load <= solution.journeys[k].agent.load_max)
    ]
    if not compatible_journeys:
        return []

    candidates: list[InsertionCandidate] = []
    request = ctx.requests[unassigned_name]
    p_name = ctx.pickup_node[unassigned_name]
    d_name = ctx.delivery_node[unassigned_name]
    suffix = [p_name, d_name, ctx.nearest_final_depot(d_name)]
    soft_tw = ctx.strategy.has_soft_tw()
    extended_soc = ctx.strategy.has_extended_soc()
    for journey in compatible_journeys:
        if not soft_tw:
            # assert hard pickup deadline; arriving early is fine (the agent waits
            # until earliest_pickup during propagation)
            last_d = journey.states[-2]
            t_arr_p = last_d.t_dep + ctx.t_travel(last_d.node, p_name)
            if t_arr_p > request.latest_pickup:
                continue
        # copy journey, replace the final depot with (p, d, re-opt depot)
        candidate = journey.copy()
        p_idx = len(candidate.route) - 1
        candidate.route = candidate.route[:-1] + suffix
        candidate.states = candidate.states[:-1] + [NodeState(n) for n in suffix]
        # precedence, capacity, and max ride-time are guaranteed; propagate + check SoC
        delta_obj = 0
        soc_violated = False
        for i in range(p_idx, len(candidate.route)):
            state = candidate.states[i]
            candidate._propagate_node(candidate.states[i - 1], state)
            if extended_soc and state.phi_arr < journey.agent.soc_min_operational - tol:
                soc_violated = True
                break
            delta_obj += _node_objective(ctx, state.node, state.t_start, soft_tw)
        if soc_violated or (
            not extended_soc
            and candidate.total_travel_time() > journey.agent.duration_max + tol
        ):
            continue
        # update request nodes (indices)
        candidate._rebuild_request_indices()
        candidates.append(InsertionCandidate(candidate, delta_obj))

    return candidates


@dataclass(slots=True)
class _PropagatedSuffix:
    """Wrapper for the propagated suffix with an objective difference"""

    states: list[NodeState]
    delta_obj: float


@dataclass(slots=True)
class _PickupInsertionScan:
    """Mutable propagation state for one fixed pickup insertion position"""

    journey: ModemsJourney
    a_idx: int
    tentative_prefix: list[NodeState]
    current_state: NodeState
    pickup_start: float
    pickup_obj: float
    onboard_requests: set[str]
    pickup_t_start_of: dict[str, float]
    delta_energy: float
    delta_t_travel: float
    onboard_distance: float = 0.0
    segment_delta_obj: float = 0.0


class _InsertionEnumerator:
    """Enumerate Algorithm 3 candidates using one incremental prefix per pickup"""

    ctx: ProblemContext
    solution: ModemsSolution
    request_name: str
    request: ModemsRequest
    agent_names: list[str]
    tol: float
    pickup_node: str
    delivery_node: str
    soft_tw: bool
    extended_soc: bool
    rejection_penalty: float
    base_t_mission: float
    stats: dict[str, int] | None

    def __init__(
        self,
        solution: ModemsSolution,
        request_name: str,
        agents: list[str] | None,
        tol: float = FLOAT_TOL,
        stats: dict[str, int] | None = None,
    ) -> None:
        """Initialize internal variables, store the base (original) mission time"""
        self.ctx = solution.ctx
        self.solution = solution
        self.request_name = request_name
        self.request = self.ctx.requests[request_name]
        self.agent_names = self.ctx.agent_names if agents is None else agents
        self.tol = tol
        self.pickup_node = self.ctx.pickup_node[request_name]
        self.delivery_node = self.ctx.delivery_node[request_name]
        self.soft_tw = self.ctx.strategy.has_soft_tw()
        self.extended_soc = self.ctx.strategy.has_extended_soc()
        self.rejection_penalty = self.ctx.eta if self.ctx.is_selective() else 0.0
        self.base_t_mission = solution.mission_time()
        self.stats = stats

    def enumerate(self) -> list[InsertionCandidate]:
        """Wrapper for the agent-specific function to enumerate over all agents"""
        candidates: list[InsertionCandidate] = []
        for agent_name in self.agent_names:
            journey = self.solution.journeys[agent_name]
            if self.request.load > journey.agent.load_max:
                continue
            candidates.extend(self._enumerate_for_agent(journey))
        return candidates

    def _enumerate_for_agent(self, journey: ModemsJourney) -> list[InsertionCandidate]:
        """Agent-specific enumeration cycle, with caching, checking, and propagating"""
        candidates: list[InsertionCandidate] = []
        terminal_soc_slack = journey.terminal_soc_slack() if self.extended_soc else 0.0
        base_travel_time = journey.total_travel_time()

        for a_idx in range(len(journey.route) - 1):
            # find a feasible pickup and initialize its prefix when found
            scan = self._start_pickup_scan(journey, a_idx)
            if scan is None:
                _collect_stats(self.stats, "pickup_scans_rejected")
                continue
            _collect_stats(self.stats, "pickup_scans_started")

            # enumerate on possible delivery positions until an immediate failure occurs
            last_b_idx = a_idx - 1
            for b_idx in range(a_idx, len(journey.route) - 1):
                if b_idx > a_idx and not self._advance_prefix(scan, b_idx):
                    last_b_idx = b_idx
                    break
                candidate, stop_scan = self._try_delivery(
                    scan,
                    b_idx,
                    terminal_soc_slack,
                    base_travel_time,
                )
                _collect_stats(self.stats, "position_pairs_considered")
                if candidate is not None:
                    candidates.append(candidate)
                    _collect_stats(self.stats, "candidates_accepted")
                else:
                    _collect_stats(self.stats, "candidates_rejected")
                last_b_idx = b_idx
                if stop_scan:
                    break

            # every (a_idx, b_idx) pair strictly after last_b_idx was never attempted
            # -- either the loop reached the end of the route naturally (pruned=0),
            # or _advance_prefix/_try_delivery ended the scan early
            pruned = (len(journey.route) - 2) - last_b_idx
            if pruned > 0:
                _collect_stats(self.stats, "position_pairs_pruned", pruned)

        return candidates

    def _start_pickup_scan(
        self, journey: ModemsJourney, a_idx: int
    ) -> _PickupInsertionScan | None:
        """
        Find the first feasible pickup insertion position starting from a_idx.
        When found, insert the pickup after a and initialize the reusable prefix
        for subsequent delivery scans. If not found, return None
        """
        state_a = journey.states[a_idx]
        agent = journey.agent
        # check capacity limits
        if state_a.z_dep + self.request.load > agent.load_max:
            return None

        # ensure pickup is reachable (energy-wise)
        energy_to_pickup = energy_consmp(
            self.ctx,
            agent,
            state_a.node,
            self.pickup_node,
            state_a.z_dep,
        )
        if self.extended_soc and (
            state_a.phi_dep - energy_to_pickup < agent.soc_min_operational - self.tol
        ):
            return None

        # check hard TW (if we later decide to use this for other variants)
        pickup_state = journey._make_update_node(state_a, self.pickup_node)
        if not self.soft_tw and (
            pickup_state.t_start > self.ctx.node_latest_p[self.pickup_node] + self.tol
        ):
            return None

        # collect pickup times of on-board requests
        r_onboard = journey.onboard_at(a_idx)
        pickup_t_start_of = {}
        for r_name in r_onboard:
            pickup_idx = journey.request_pickup[r_name]
            pickup_t_start_of[r_name] = journey.states[pickup_idx].t_start
        r_onboard = r_onboard | {self.request_name}
        pickup_t_start_of[self.request_name] = pickup_state.t_start

        return _PickupInsertionScan(
            journey=journey,
            a_idx=a_idx,
            tentative_prefix=journey.states[: a_idx + 1] + [pickup_state],
            current_state=pickup_state,
            pickup_start=pickup_state.t_start,
            pickup_obj=_node_objective(
                self.ctx,
                self.pickup_node,
                pickup_state.t_start,
                self.soft_tw,
            ),
            onboard_requests=r_onboard,
            pickup_t_start_of=pickup_t_start_of,
            delta_energy=energy_to_pickup,
            delta_t_travel=self.ctx.t_travel(state_a.node, self.pickup_node),
        )

    def _advance_prefix(self, scan: _PickupInsertionScan, b_idx: int) -> bool:
        """
        Append original route[b] node to the tentative prefix and propagate it once
        to get the (new) states after pickup p^u insertion. The resulting state
        (updated route[b] in tentative prefix) is reused by every later delivery
        position for the same pickup position (in the same scan)
        """
        journey = scan.journey
        route = journey.route
        states = journey.states
        agent = journey.agent
        node_b = route[b_idx]
        state_prev = scan.current_state
        pickup_t_start_floor = (
            {node_b: states[b_idx].t_start} if self.ctx.is_pickup(node_b) else None
        )  # if route[b] is a pickup, use its original t_start in temporal propagation
        # insert pickup state, propagate it, and check immediate failures
        state_b = journey._make_update_node(
            state_prev,
            node_b,
            pickup_t_start_floor_of=pickup_t_start_floor,
        )
        _collect_stats(self.stats, "prefix_node_propagations")
        # check hard pickup time (if relevant), capacity, and SoC
        if (
            self.ctx.is_pickup(node_b)
            and not self.soft_tw
            and (state_b.t_start > self.ctx.node_latest_p[node_b] + self.tol)
        ):
            return False
        if state_b.z_dep > agent.load_max + self.tol:
            return False
        if self.extended_soc and state_b.phi_arr < agent.soc_min_operational - self.tol:
            return False

        # update internal variables
        scan.tentative_prefix.append(state_b)
        scan.current_state = state_b
        scan.delta_t_travel += self.ctx.t_travel(
            state_prev.node, node_b
        ) - self.ctx.t_travel(route[b_idx - 1], node_b)
        # consumption + on-board distance for the monotone energy lower bound
        scan.delta_energy += energy_consmp(
            self.ctx,
            agent,
            state_prev.node,
            node_b,
            state_prev.z_dep,
        ) - energy_consmp(
            self.ctx,
            agent,
            route[b_idx - 1],
            node_b,
            states[b_idx - 1].z_dep,
        )
        scan.onboard_distance += self.ctx.t_travel(state_prev.node, node_b)

        # check final depot reached
        request_name = self.ctx.request_of(node_b)
        if request_name is None:
            return True

        # get objective change (due to temporal differences between original/propagated)
        scan.segment_delta_obj += _node_objective(
            self.ctx,
            node_b,
            state_b.t_start,
            self.soft_tw,
        ) - _node_objective(
            self.ctx,
            node_b,
            states[b_idx].t_start,
            self.soft_tw,
        )

        # check max ride-time of on-board requests
        if self.ctx.is_pickup(node_b):
            # pickup reached: register (new) on-board request
            if request_name not in scan.onboard_requests:  # sanity check
                scan.onboard_requests.add(request_name)
                scan.pickup_t_start_of[request_name] = state_b.t_start
            return True
        # on-board request delivery reached: check max ride-time
        scan.onboard_requests.discard(request_name)
        pickup_t_start = scan.pickup_t_start_of[request_name]
        ride_time = (
            state_b.t_start
            - pickup_t_start
            - self.ctx.node_t_service[self.ctx.pickup_node[request_name]]
        )
        direct_time = self.ctx.t_travel(
            self.ctx.pickup_node[request_name],
            self.ctx.delivery_node[request_name],
        )
        return ride_time <= self.ctx.rho * direct_time + self.tol

    def _try_delivery(
        self,
        scan: _PickupInsertionScan,
        b_idx: int,
        terminal_soc_slack: float,
        base_travel_time: float,
    ) -> tuple[InsertionCandidate | None, bool]:
        """
        Insert the delivery node after the current tentative prefix, validating the
        insertion feasiblity. When feasible, return the insertion candidate and allow
        scan to continue. If infeasible, the boolean result indicates whether a later
        delivery insertion is plausible (based on SoC and timing variables), allowing
        us to stop the scan early
        """
        journey = scan.journey
        route = journey.route
        agent = journey.agent
        current = scan.current_state

        if self.extended_soc:
            # get carried_distance for a direct or indirect trip
            if b_idx == scan.a_idx:
                carried_distance = self.ctx.t_travel(
                    self.pickup_node, self.delivery_node
                )
            else:
                carried_distance = scan.onboard_distance + self.ctx.t_travel(
                    current.node, self.delivery_node
                )
            # verify monotone additional-energy lower bound
            min_extra_energy = agent.soc_beta * self.request.load * carried_distance
            if min_extra_energy > terminal_soc_slack + self.tol:
                return None, True  # stop scan

        # get actual energy consumption for reaching the delivery
        energy_to_delivery = energy_consmp(
            self.ctx,
            agent,
            current.node,
            self.delivery_node,
            current.z_dep,
        )
        delivery_soc = current.phi_dep - energy_to_delivery
        if self.extended_soc and delivery_soc < agent.soc_min_operational - self.tol:
            return None, False

        # insert delivery state, propagate it, and check request (u) ride-time
        delivery_state = journey._make_update_node(current, self.delivery_node)
        _collect_stats(self.stats, "delivery_node_propagations")
        direct_time = self.ctx.t_travel(self.pickup_node, self.delivery_node)
        ride_time = (
            delivery_state.t_start
            - scan.pickup_start
            - self.ctx.node_t_service[self.pickup_node]
        )
        if ride_time > self.ctx.rho * direct_time + self.tol:
            return None, True  # stop scan

        # check suffix propagation result
        delivery_obj = _node_objective(
            self.ctx,
            self.delivery_node,
            delivery_state.t_start,
            self.soft_tw,
        )
        if b_idx == len(route) - 2:
            suffix, delta_obj = self._terminal_suffix(
                scan,
                b_idx,
                delivery_state,
                energy_to_delivery,
                delivery_obj,
                terminal_soc_slack,
                base_travel_time,
            )
            if suffix is None:
                return None, False
        else:
            suffix_result, delta_obj = self._route_suffix(
                scan,
                b_idx,
                delivery_state,
                energy_to_delivery,
                delivery_obj,
                terminal_soc_slack,
                base_travel_time,
            )
            if suffix_result is None:
                return None, False
            suffix = suffix_result.states

        # compile the full candidate journey
        candidate_states = scan.tentative_prefix.copy()
        candidate_states.append(delivery_state)
        candidate_states.extend(suffix)
        candidate_route = [state.node for state in candidate_states]
        candidate_journey = ModemsJourney._from_propagated(
            self.ctx,
            journey.agent_name,
            candidate_route,
            candidate_states,
        )
        return InsertionCandidate(candidate_journey, delta_obj), False

    def _terminal_suffix(
        self,
        scan: _PickupInsertionScan,
        b_idx: int,
        delivery_state: NodeState,
        energy_to_delivery: float,
        delivery_obj: float,
        terminal_soc_slack: float,
        base_t_travel: float,
    ) -> tuple[list[NodeState] | None, float]:
        """Check and construct a candidate whose delivery is last (with depot re-opt)"""
        journey = scan.journey
        route = journey.route
        states = journey.states
        agent = journey.agent
        final_depot = self.ctx.nearest_final_depot(self.delivery_node)

        if self.extended_soc:
            # check energy feasibility
            energy_to_depot = energy_consmp(
                self.ctx,
                agent,
                self.delivery_node,
                final_depot,
                delivery_state.z_dep,
            )
            energy_gain = energy_consmp(
                self.ctx,
                agent,
                route[b_idx],
                route[b_idx + 1],
                states[b_idx].z_dep,
            )
            chk_delta_energy = (
                scan.delta_energy + energy_to_delivery + energy_to_depot - energy_gain
            )
            terminal_soc = delivery_state.phi_arr - energy_to_depot
            if terminal_soc < agent.soc_min_operational - self.tol:
                return None, 0.0
            if terminal_soc_slack - chk_delta_energy < -self.tol:
                return None, 0.0
        else:
            # check time (energy-equivalent) feasibility
            chk_delta_t_travel = (
                scan.delta_t_travel
                + self.ctx.t_travel(scan.current_state.node, self.delivery_node)
                + self.ctx.t_travel(self.delivery_node, final_depot)
                - self.ctx.t_travel(route[b_idx], route[b_idx + 1])
            )
            if base_t_travel + chk_delta_t_travel > agent.duration_max + self.tol:
                return None, 0.0

        # re-opt final depot and return objective difference
        state_h_f = journey._make_update_node(delivery_state, final_depot)
        t_completion = delivery_state.t_dep if self.ctx.is_open() else state_h_f.t_arr
        delta_t_mission = max(0.0, t_completion - self.base_t_mission)
        delta_obj = (
            -self.rejection_penalty
            + scan.pickup_obj
            + delivery_obj
            + scan.segment_delta_obj
            + delta_t_mission
        )
        return [state_h_f], delta_obj

    def _route_suffix(
        self,
        scan: _PickupInsertionScan,
        b_idx: int,
        delivery_state: NodeState,
        energy_to_delivery: float,
        delivery_obj: float,
        terminal_soc_slack: float,
        base_t_travel: float,
    ) -> tuple[_PropagatedSuffix | None, float]:
        """Check and propagate the original route (stale suffix) after the delivery"""
        journey = scan.journey
        route = journey.route
        states = journey.states
        agent = journey.agent
        next_node = route[b_idx + 1]

        if self.extended_soc:
            # check energy feasibility
            energy_to_next = energy_consmp(
                self.ctx,
                agent,
                self.delivery_node,
                next_node,
                delivery_state.z_dep,
            )
            energy_gain = energy_consmp(
                self.ctx,
                agent,
                route[b_idx],
                next_node,
                states[b_idx].z_dep,
            )
            chk_delta_energy = (
                scan.delta_energy + energy_to_delivery + energy_to_next - energy_gain
            )
            if terminal_soc_slack - chk_delta_energy < -self.tol:
                return None, 0.0
        else:
            # check time (energy-equivalent) feasibility
            chk_delta_t_travel = (
                scan.delta_t_travel
                + self.ctx.t_travel(scan.current_state.node, self.delivery_node)
                + self.ctx.t_travel(self.delivery_node, next_node)
                - self.ctx.t_travel(route[b_idx], next_node)
            )
            if base_t_travel + chk_delta_t_travel > agent.duration_max + self.tol:
                return None, 0.0

        # update/propagate suffix, validate it, and return objective difference
        suffix = _propagate_candidate_suffix(
            self.ctx,
            journey,
            b_idx,
            delivery_state,
            chk_delta_energy,
            self.soft_tw,
            self.base_t_mission,
            scan.pickup_t_start_of,
            self.tol,
            self.stats,
        )
        if suffix is None:
            return None, 0.0
        delta_obj = (
            -self.rejection_penalty
            + scan.pickup_obj
            + delivery_obj
            + scan.segment_delta_obj
            + suffix.delta_obj
        )
        return suffix, delta_obj


def alns_feasible_insertions(
    solution: ModemsSolution,
    unassigned_name: str,
    agents: list[str] | None = None,
    tol: float = FLOAT_TOL,
    stats: dict[str, int] | None = None,
) -> list[InsertionCandidate]:
    """
    Enumerate fully propagated insertion candidates following Algorithm 3. For each
    pickup position, one mutable tentative prefix is propagated incrementally as the
    delivery position moves forward. So, each original node in the modified segment is
    propagated once for that pickup position. A feasible candidate gets its objective
    difference and an independent copy (snapshot) of the tentative prefix, so that it
    has a complete and feasible propagated journey that can be adopted directly.

    stats, if given, is updated in place with counters instrumenting this call
    (pickup_scans_started/_rejected, position_pairs_considered/_pruned,
    candidates_accepted/_rejected, prefix/delivery/suffix_node_propagations) --
    used by the insertion-ablation harness (modems.insertion_ablation) to compare
    this implementation against less-optimized variants. Does not impact the results
    nor incurs cost when left as the default None

    Only valid for SolverStrategy.alns: the propagation encodes ALNS timing rules
    (no service before e^r, tardiness-only tau). Raises ValueError otherwise; use
    milp_feasible_insertions() for the MILP strategies
    """
    if solution.ctx.strategy != SolverStrategy.alns:
        raise ValueError(
            "alns_feasible_insertions() requires SolverStrategy.alns, got "
            f"{solution.ctx.strategy}"
        )
    return _InsertionEnumerator(
        solution,
        unassigned_name,
        agents,
        tol,
        stats,
    ).enumerate()


def _copy_suffix_with_soc_shift(
    states: list[NodeState],
    i_start: int,
    delta_energy: float,
) -> list[NodeState]:
    """Copy states[i_start:] and apply the constant SoC shift due to insertion"""
    suffix_states: list[NodeState] = [s.copy() for s in states[i_start:]]
    for s in suffix_states:
        s.phi_arr -= delta_energy
        s.phi_dep -= delta_energy
    return suffix_states


def _propagate_candidate_suffix(
    ctx: ProblemContext,
    journey: ModemsJourney,
    from_idx: int,
    from_state: NodeState,
    delta_energy: float,
    soft_tw: bool,
    base_t_mission: float,
    pickup_t_start_of_override: dict[str, float] | None = None,
    tol: float = FLOAT_TOL,
    stats: dict[str, int] | None = None,
) -> _PropagatedSuffix | None:
    """
    Efficiently propagate the original suffix after an inserted delivery: Time is
    propagated only until an existing pickup waiting absorbs the delay. Then, the
    remaining states keep their original temporal values. Loads only increase at
    the modified segment (not here); suffix loads are unchanged. SoC values decrease
    by pickup+carried distance; suffix SoC values get a constant energy shift.
    All returned states are updated copies owned by the candidate. Return None when
    an affected hard time window or ride-time constraint is infeasible (in suffix)
    """
    route = journey.route
    states = journey.states
    is_open = ctx.is_open()
    first_idx = from_idx + 1
    first_t_arr = from_state.t_dep + ctx.t_travel(from_state.node, route[first_idx])
    first_t_dly = first_t_arr - states[first_idx].t_arr
    if first_t_dly < -tol:  # sanity check
        raise ValueError("insertion unexpectedly advances the route suffix")
    curr_delay = max(0.0, first_t_dly)
    t_shift_active = curr_delay > tol
    if not t_shift_active:  # temporal shift absorbed, update SoC only
        return _PropagatedSuffix(
            _copy_suffix_with_soc_shift(states, first_idx, delta_energy),
            0.0,
        )

    # update state timing values until shift is absorbed
    delta_obj = 0.0
    candidate_states: list[NodeState] = []
    r_onboard = journey.onboard_at(from_idx)
    pickup_t_start_of_override = pickup_t_start_of_override or {}
    pickup_t_start_of = {
        r: pickup_t_start_of_override.get(r, states[journey.request_pickup[r]].t_start)
        for r in r_onboard
    }
    for idx in range(first_idx, len(route)):
        node = route[idx]
        curr_state = states[idx]
        cand_state = curr_state.copy()
        cand_state.phi_arr = curr_state.phi_arr - delta_energy
        cand_state.phi_dep = curr_state.phi_dep - delta_energy
        _collect_stats(stats, "suffix_node_propagations")

        if t_shift_active:
            cand_state.t_arr = curr_state.t_arr + curr_delay
            if ctx.is_pickup(node):
                # existing pickup waiting is a scheduling decision. A delay consumes
                # that waiting before it can change the service start
                cand_state.t_start = max(curr_state.t_start, cand_state.t_arr)
                cand_state.t_wait = cand_state.t_start - cand_state.t_arr
                cand_state.tau = max(0.0, cand_state.t_start - ctx.node_latest_p[node])
                cand_state.t_dep = cand_state.t_start + ctx.node_t_service[node]
                if not soft_tw and (cand_state.t_start > ctx.node_latest_p[node] + tol):
                    return None
                r_name = ctx.request_of(node)
                if r_name is None:  # sanity check
                    raise ValueError(f"station {node!r} unexpectedly has no request")
                pickup_t_start_of[r_name] = cand_state.t_start
            elif ctx.is_delivery(node):
                # service starts as soon as we arrive at delivery nodes
                cand_state.t_start = cand_state.t_arr
                cand_state.t_wait = 0.0
                cand_state.tau = 0.0
                cand_state.t_dep = cand_state.t_start + ctx.node_t_service[node]
                r_name = ctx.request_of(node)
                if r_name is None:  # sanity check
                    raise ValueError(f"station {node!r} unexpectedly has no request")
                elif r_name in pickup_t_start_of:
                    p_name = ctx.pickup_node[r_name]
                    t_ride = (
                        cand_state.t_start
                        - pickup_t_start_of[r_name]
                        - ctx.node_t_service[p_name]
                    )
                    t_direct = ctx.t_travel(p_name, ctx.delivery_node[r_name])
                    if t_ride > ctx.rho * t_direct + tol:
                        return None
            else:
                # service starts as soon as we arrive at final depot
                cand_state.t_start = cand_state.t_arr
                cand_state.t_wait = 0.0
                cand_state.tau = 0.0
                cand_state.t_dep = cand_state.t_arr

            if ctx.is_station(node):
                delta_obj += _node_objective(
                    ctx, node, cand_state.t_start, soft_tw
                ) - _node_objective(ctx, node, curr_state.t_start, soft_tw)
                curr_delay = max(0.0, cand_state.t_dep - curr_state.t_dep)
                t_shift_active = curr_delay > tol
            else:
                t_shift_active = False  # end of route, append and exit afterwards

        candidate_states.append(cand_state)

        if not t_shift_active and idx + 1 < len(route):  # copy remaining nodes
            candidate_states.extend(
                _copy_suffix_with_soc_shift(states, idx + 1, delta_energy)
            )
            return _PropagatedSuffix(candidate_states, delta_obj)

    # end of route reached, update objective
    if is_open:
        chk_t_completion = candidate_states[-2].t_dep
    else:
        chk_t_completion = candidate_states[-1].t_arr
    delta_t_mission = max(0.0, chk_t_completion - base_t_mission)
    return _PropagatedSuffix(candidate_states, delta_obj + delta_t_mission)
