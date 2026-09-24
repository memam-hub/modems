from __future__ import annotations

import time
from enum import StrEnum
from typing import Any

import pyomo.environ as pyo
from pyomo.opt import SolverFactory

from .algorithms import greedy_complete, preprocess
from .core import ModemsScenario, ProblemContext, ProblemType, SolverStrategy
from .solution import (
    DEFAULT_MILP_TIMELIMIT,
    DEFAULT_PARAMS_MILP,
    FLOAT_INF,
    FLOAT_TOL,
    ModemsInstance,
    ModemsJourney,
    ModemsSolution,
    ModemsSolutionInfo,
    SolutionStatus,
)


class SolverConfigType(StrEnum):
    """
    The option-key-naming convention/configuration a solver follows, decoupled from
    solver_name (the literal string passed to pyomo SolverFactory, such as: "cbc",
    "gurobi", "gurobi_persistent", "appsi_highs") since the same configuration can
    apply to multiple registered solver names

    cbc: pyomo classic SolverFactory("cbc"), time limit key "seconds" no threads
    gurobi: pyomo classic SolverFactory("gurobi") (or gurobi_persistent/gurobi_direct
        with the same option names), time limit key "timelimit", threads supported
    highs: pyomo appsi_highs interface, time limit key "time_limit", threads supported
    """

    cbc = "cbc"
    gurobi = "gurobi"
    highs = "highs"


# Generic option keys ModemsMilp.solve() accepts, mapped to the native key of
# each solver. Any option key NOT in this map is assumed solver-native and passed
# through unchanged (e.g., solver-specific tuning params like "mipgap")
_SOLVER_OPTION_KEY_MAP = {
    SolverConfigType.cbc: {"timelimit": "seconds"},
    SolverConfigType.gurobi: {"timelimit": "timelimit", "threads": "threads"},
    SolverConfigType.highs: {"timelimit": "time_limit", "threads": "threads"},
}

# default MILP solver data
DEFAULT_MILP_SOLVER_DATA: tuple[str, SolverConfigType] = ("cbc", SolverConfigType.cbc)


class MilpType(StrEnum):
    """MILP model variants, directly maps to MILP SolverStrategy"""

    milp1 = "milp1"
    milp2 = "milp2"
    milp3 = "milp3"

    @property
    def kappa(self) -> SolverStrategy:
        return {
            MilpType.milp1: SolverStrategy.milp1,
            MilpType.milp2: SolverStrategy.milp2,
            MilpType.milp3: SolverStrategy.milp3,
        }[self]


BIG_SOC_DEACTIVATE = 1.0  # sufficiently large constant to deactivate SoC propagation


class ModemsMilp:
    """Builds, warm-starts, solves, and post-processes the MODEMS MILP model"""

    ctx: ProblemContext
    milp_type: MilpType
    problem_type: ProblemType
    has_soft_tw: bool
    has_extended_soc: bool
    has_selectivity: bool
    milp_params: dict[str, Any]
    model: pyo.ConcreteModel
    solver_type: str
    solver_config_type: SolverConfigType
    solver_options: dict[str, Any]
    problem_done: bool = False

    def __init__(
        self,
        scenario: ModemsScenario,
        milp_type: MilpType | str,
        problem_type: ProblemType | str,
        milp_params: dict[str, Any] | None = None,
    ) -> None:
        """Build a model from the given scenario, MILP/problem type, and parameters"""
        milp_type = MilpType(milp_type)
        problem_type = ProblemType(problem_type)
        if milp_type == MilpType.milp1:
            if ProblemType.is_selective(problem_type):
                raise ValueError("MILP1 only supports non-selective routing problems")
        elif milp_type == MilpType.milp2 and not ProblemType.is_selective(problem_type):
            raise ValueError("MILP2 only supports selective routing problems")
        # MILP3 supports both
        milp_params = {**DEFAULT_PARAMS_MILP, **(milp_params or {})}
        big_m = milp_params["big_m"]
        if big_m <= 0:
            raise ValueError("big_m must be positive")
        self.milp_params = milp_params

        self.milp_type = milp_type
        self.problem_type = problem_type
        strategy = milp_type.kappa
        self.has_soft_tw = strategy.has_soft_tw()
        self.has_extended_soc = strategy.has_extended_soc()
        self.has_selectivity = ProblemType.is_selective(self.problem_type)

        # build context and model (if needed)
        self.ctx = ProblemContext(
            scenario.copy(),
            self.problem_type,
            strategy,
            milp_params,
        )
        self._build_model()

    @property
    def is_trivial(self) -> bool:
        """
        Flag for trivial problems: nothing for an optimizer to decide, skips
        building the model entirely and solve() short-circuits to _solve_trivial()
        """
        return not self.ctx.agent_names or not self.ctx.request_names

    # ---------------------------------------------------------------------------------
    # Model construction
    # ---------------------------------------------------------------------------------

    def _build_arc_index(self) -> list[tuple[str, str, str]]:
        """
        Structurally-legal (i, j, k) travel arcs:
          v^k -> S^p (agent leaves its own start only towards a pickup)
          S^p -> S' \\setminus {i}      (pickup to any other pickup/delivery)
          S^d -> (S' \\setminus {i}) union H_f (delivery to any station or a final hub)
        """
        ctx = self.ctx
        arcs = []
        for k in ctx.agent_names:
            v_k = ctx.agent_initial_node[k]
            for p in ctx.pickup_node.values():
                arcs.append((v_k, p, k))
            for i in list(ctx.pickup_node.values()) + list(ctx.delivery_node.values()):
                for j in list(ctx.pickup_node.values()) + list(
                    ctx.delivery_node.values()
                ):
                    if i != j:
                        arcs.append((i, j, k))
            for i in ctx.delivery_node.values():
                for h in ctx.final_depot_names:
                    arcs.append((i, h, k))
        return arcs

    def _define_sets(self) -> None:
        """Define model sets"""
        m = self.model
        ctx = self.ctx
        m.set_agents = pyo.Set(initialize=ctx.agent_names)
        m.set_requests = pyo.Set(initialize=ctx.request_names)
        m.set_hubs_initial = pyo.Set(initialize=list(ctx.agent_initial_node.values()))
        m.set_stations_pickup = pyo.Set(initialize=list(ctx.pickup_node.values()))
        m.set_stations_delivery = pyo.Set(initialize=list(ctx.delivery_node.values()))
        m.set_stations = m.set_stations_pickup | m.set_stations_delivery
        m.set_hubs_final = pyo.Set(initialize=ctx.final_depot_names)

        arc_list = self._build_arc_index()
        m.set_arcs = pyo.Set(initialize=arc_list, dimen=3)
        # python-level in/out adjacency for fast rule construction
        self._out_arcs = {}
        self._in_arcs = {}
        for i, j, k in arc_list:
            self._out_arcs.setdefault((i, k), []).append(j)
            self._in_arcs.setdefault((j, k), []).append(i)

    def _define_parameters(self) -> None:
        """Define model parameters"""
        m = self.model
        ctx = self.ctx

        m.param_agent_time_initial = pyo.Param(
            m.set_agents,
            initialize={k: ctx.agents[k].time_initial for k in ctx.agent_names},
            within=pyo.NonNegativeReals,
        )
        m.param_agent_load_max = pyo.Param(
            m.set_agents,
            initialize={k: ctx.agents[k].load_max for k in ctx.agent_names},
            within=pyo.NonNegativeIntegers,
        )
        max_load = max(ctx.agents[k].load_max for k in ctx.agent_names)
        m.param_load_max_fleet = pyo.Param(
            initialize=max_load, within=pyo.NonNegativeIntegers
        )

        if self.milp_type == MilpType.milp1:
            m.param_agent_duration_max = pyo.Param(
                m.set_agents,
                initialize={k: ctx.agents[k].duration_max for k in ctx.agent_names},
                within=pyo.NonNegativeReals,
            )
        if self.has_extended_soc:
            m.param_agent_soc_min = pyo.Param(
                m.set_agents,
                initialize={
                    k: ctx.agents[k].soc_min_operational for k in ctx.agent_names
                },
                within=pyo.PercentFraction,
            )
            m.param_agent_soc_initial = pyo.Param(
                m.set_agents,
                initialize={k: ctx.agents[k].soc_initial for k in ctx.agent_names},
                within=pyo.PercentFraction,
            )
            m.param_agent_alpha = pyo.Param(
                m.set_agents,
                initialize={k: ctx.agents[k].soc_alpha for k in ctx.agent_names},
                within=pyo.NonNegativeReals,
            )
            m.param_agent_beta = pyo.Param(
                m.set_agents,
                initialize={k: ctx.agents[k].soc_beta for k in ctx.agent_names},
                within=pyo.NonNegativeReals,
            )

        m.param_request_pickup = pyo.Param(
            m.set_requests, initialize=ctx.pickup_node, within=pyo.Any
        )
        m.param_request_delivery = pyo.Param(
            m.set_requests, initialize=ctx.delivery_node, within=pyo.Any
        )

        m.param_station_q = pyo.Param(
            m.set_stations, initialize=ctx.node_load, within=pyo.Integers
        )
        m.param_station_s = pyo.Param(
            m.set_stations, initialize=ctx.node_t_service, within=pyo.NonNegativeReals
        )
        m.param_pickup_e = pyo.Param(
            m.set_stations_pickup,
            initialize=ctx.node_earliest_p,
            within=pyo.NonNegativeReals,
        )
        m.param_pickup_l = pyo.Param(
            m.set_stations_pickup,
            initialize=ctx.node_latest_p,
            within=pyo.NonNegativeReals,
        )

        # store travel times only between relevant node pairs
        relevant_pairs = set()
        for i, j, _ in self.model.set_arcs:
            relevant_pairs.add((i, j))
        for r in ctx.request_names:
            relevant_pairs.add((ctx.pickup_node[r], ctx.delivery_node[r]))
        m.param_travel_time = pyo.Param(
            pyo.Set(initialize=list(relevant_pairs), dimen=2),
            initialize={(i, j): ctx.t_travel(i, j) for (i, j) in relevant_pairs},
            within=pyo.Any,
        )

        m.param_eps = pyo.Param(initialize=ctx.eps, within=pyo.NonNegativeReals)
        m.param_zeta = pyo.Param(initialize=ctx.zeta, within=pyo.NonNegativeReals)
        m.param_eta = pyo.Param(initialize=ctx.eta, within=pyo.NonNegativeReals)
        m.param_rho = pyo.Param(initialize=ctx.rho, within=pyo.NonNegativeReals)
        m.param_big_m = pyo.Param(
            initialize=ctx.model_params["big_m"], within=pyo.NonNegativeReals
        )

    def _define_variables(self) -> None:
        """Define model variables"""
        m = self.model
        m.var_x = pyo.Var(m.set_arcs, within=pyo.Binary)
        m.var_t = pyo.Var(m.set_stations, within=pyo.NonNegativeReals)
        m.var_T = pyo.Var(within=pyo.NonNegativeReals)
        m.var_z = pyo.Var(m.set_stations, within=pyo.NonNegativeIntegers)

        if self.has_soft_tw:
            m.var_tau = pyo.Var(m.set_stations_pickup, within=pyo.NonNegativeReals)
        if self.has_selectivity:
            m.var_y = pyo.Var(m.set_requests, within=pyo.Binary)
        if self.has_extended_soc:
            m.var_phi = pyo.Var(
                m.set_stations, bounds=(0.0, 1.0), within=pyo.PercentFraction
            )

    def _define_objective(self) -> None:
        """Define model objective function"""
        m = self.model

        def rule(m: Any) -> Any:
            obj = m.var_T
            obj += sum(
                m.param_eps
                * (
                    m.var_t[m.param_request_pickup[r]]
                    + m.var_t[m.param_request_delivery[r]]
                )
                for r in m.set_requests
            )
            if self.has_soft_tw:
                obj += sum(
                    m.param_zeta * m.var_tau[m.param_request_pickup[r]]
                    for r in m.set_requests
                )
            if self.has_selectivity:
                obj += sum(m.param_eta * (1 - m.var_y[r]) for r in m.set_requests)
            return obj

        m.objective = pyo.Objective(rule=rule, sense=pyo.minimize)

    def _x_in(self, j: str, k: str) -> list[str]:
        """All arcs of agent k entering j"""
        return self._in_arcs.get((j, k), [])

    def _x_out(self, i: str, k: str) -> list[str]:
        """All arcs of agent k exiting i"""
        return self._out_arcs.get((i, k), [])

    def _sum_x_in(self, m: Any, j: str, k: str | None = None) -> Any:
        """Sum of all arcs of agent k entering j. Defaults to sum over all agents"""
        if k is None:
            return sum(
                m.var_x[i, j, kk] for kk in m.set_agents for i in self._x_in(j, kk)
            )
        return sum(m.var_x[i, j, k] for i in self._x_in(j, k))

    def _sum_x_out(self, m: Any, i: str, k: str | None = None) -> Any:
        """Sum of all arcs of agent k exiting i. Defaults to sum over all agents"""
        if k is None:
            return sum(
                m.var_x[i, j, kk] for kk in m.set_agents for j in self._x_out(i, kk)
            )
        return sum(m.var_x[i, j, k] for j in self._x_out(i, k))

    def _define_constraints(self) -> None:
        """Define model constraints"""
        m = self.model
        ctx = self.ctx

        # --- request assignment / flow conservation ---
        if self.has_selectivity:

            def cstr_req_enter_pickup(m: Any, r: str) -> Any:
                return self._sum_x_in(m, m.param_request_pickup[r]) == m.var_y[r]

            m.cstr_req_enter_pickup = pyo.Constraint(
                m.set_requests, rule=cstr_req_enter_pickup
            )

            scheduled = [r for r in ctx.request_names if ctx.requests[r].is_scheduled()]
            if scheduled:

                def cstr_sched_accept(m: Any, r: str) -> Any:
                    return m.var_y[r] == 1

                m.cstr_sched_accept = pyo.Constraint(scheduled, rule=cstr_sched_accept)

            if self.milp_type == MilpType.milp3 and self.ctx.model_params.get(
                "force_non_selective", False
            ):
                new_reqs = [r for r in ctx.request_names if ctx.requests[r].is_new()]

                def cstr_non_selective(m: Any, r: str) -> Any:
                    return m.var_y[r] == 1

                m.cstr_non_selective = pyo.Constraint(new_reqs, rule=cstr_non_selective)
        else:

            def cstr_visit_pickup_once(m: Any, r: str) -> Any:
                return self._sum_x_in(m, m.param_request_pickup[r]) == 1

            m.cstr_visit_pickup_once = pyo.Constraint(
                m.set_requests, rule=cstr_visit_pickup_once
            )

        def cstr_same_agent(m: Any, r: str, k: str) -> Any:
            p, d = m.param_request_pickup[r], m.param_request_delivery[r]
            return self._sum_x_in(m, p, k) == self._sum_x_in(m, d, k)

        m.cstr_same_agent = pyo.Constraint(
            m.set_requests, m.set_agents, rule=cstr_same_agent
        )

        def cstr_flow_pickup(m: Any, r: str, k: str) -> Any:
            p = m.param_request_pickup[r]
            return self._sum_x_in(m, p, k) == self._sum_x_out(m, p, k)

        m.cstr_flow_pickup = pyo.Constraint(
            m.set_requests, m.set_agents, rule=cstr_flow_pickup
        )

        def cstr_flow_delivery(m: Any, r: str, k: str) -> Any:
            d = m.param_request_delivery[r]
            return self._sum_x_in(m, d, k) == self._sum_x_out(m, d, k)

        m.cstr_flow_delivery = pyo.Constraint(
            m.set_requests, m.set_agents, rule=cstr_flow_delivery
        )

        def cstr_agent_active_service(m: Any, k: str) -> Any:
            v_k = ctx.agent_initial_node[k]
            if not self._x_out(v_k, k):
                return pyo.Constraint.Skip
            return self._sum_x_out(m, v_k, k) <= 1

        m.cstr_agent_active_service = pyo.Constraint(
            m.set_agents, rule=cstr_agent_active_service
        )

        def cstr_agent_active_inactive(m: Any, k: str) -> Any:
            v_k = ctx.agent_initial_node[k]
            if not self._x_out(v_k, k):
                return pyo.Constraint.Skip
            n_stations = len(m.set_stations)
            lhs = sum(
                m.var_x[i, j, k]
                for i in m.set_stations
                for j in self._x_out(i, k)
                if j in m.set_stations
            )
            return lhs <= (n_stations - 1) * self._sum_x_out(m, v_k, k)

        m.cstr_agent_active_inactive = pyo.Constraint(
            m.set_agents, rule=cstr_agent_active_inactive
        )

        # --- temporal integrity/continuity ---
        def cstr_time_initial(m: Any, r: str) -> Any:
            p = m.param_request_pickup[r]
            return m.var_t[p] >= sum(
                (
                    m.param_agent_time_initial[k]
                    + m.param_travel_time[ctx.agent_initial_node[k], p]
                )
                * m.var_x[ctx.agent_initial_node[k], p, k]
                for k in m.set_agents
            )

        m.cstr_time_initial = pyo.Constraint(m.set_requests, rule=cstr_time_initial)

        def cstr_time_flow(m: Any, i: str, j: str) -> Any:
            sum_x = sum(
                m.var_x[i, j, k] for k in m.set_agents if j in self._x_out(i, k)
            )
            return m.var_t[j] >= m.var_t[i] + m.param_station_s[
                i
            ] + m.param_travel_time[i, j] - m.param_big_m * (1 - sum_x)

        station_pairs = [
            (i, j)
            for i in list(ctx.pickup_node.values()) + list(ctx.delivery_node.values())
            for j in list(ctx.pickup_node.values()) + list(ctx.delivery_node.values())
            if i != j
        ]
        m.cstr_time_flow = pyo.Constraint(station_pairs, rule=cstr_time_flow)

        is_open = ProblemType.is_open(self.problem_type)

        if self.has_selectivity:

            def cstr_time_mission(m: Any, r: str) -> Any:
                d = m.param_request_delivery[r]
                if is_open:
                    return m.var_T >= m.var_t[d] + m.param_station_s[
                        d
                    ] - m.param_big_m * (1 - m.var_y[r])
                hub_term = sum(
                    m.param_travel_time[d, h] * m.var_x[d, h, k]
                    for k in m.set_agents
                    for h in ctx.final_depot_names
                    if h in self._x_out(d, k)
                )
                return m.var_T >= m.var_t[d] + m.param_station_s[
                    d
                ] + hub_term - m.param_big_m * (1 - m.var_y[r])

            m.cstr_time_mission = pyo.Constraint(m.set_requests, rule=cstr_time_mission)

            def cstr_time_p_before_d(m: Any, r: str) -> Any:
                p, d = m.param_request_pickup[r], m.param_request_delivery[r]
                return m.var_t[d] - m.var_t[p] - m.param_station_s[
                    p
                ] >= m.param_travel_time[p, d] - m.param_big_m * (1 - m.var_y[r])

            m.cstr_time_p_before_d = pyo.Constraint(
                m.set_requests, rule=cstr_time_p_before_d
            )

            def cstr_time_max_ride(m: Any, r: str) -> Any:
                p, d = m.param_request_pickup[r], m.param_request_delivery[r]
                return m.var_t[d] - m.var_t[p] - m.param_station_s[
                    p
                ] <= m.param_rho * m.param_travel_time[p, d] + m.param_big_m * (
                    1 - m.var_y[r]
                )

            m.cstr_time_max_ride = pyo.Constraint(
                m.set_requests, rule=cstr_time_max_ride
            )
        else:

            def cstr_time_mission(m: Any, r: str) -> Any:
                d = m.param_request_delivery[r]
                if is_open:
                    return m.var_T >= m.var_t[d] + m.param_station_s[d]
                hub_term = sum(
                    m.param_travel_time[d, h] * m.var_x[d, h, k]
                    for k in m.set_agents
                    for h in ctx.final_depot_names
                    if h in self._x_out(d, k)
                )
                return m.var_T >= m.var_t[d] + m.param_station_s[d] + hub_term

            m.cstr_time_mission = pyo.Constraint(m.set_requests, rule=cstr_time_mission)

            def cstr_time_p_before_d(m: Any, r: str) -> Any:
                p, d = m.param_request_pickup[r], m.param_request_delivery[r]
                return (
                    m.var_t[d] - m.var_t[p] - m.param_station_s[p]
                    >= m.param_travel_time[p, d]
                )

            m.cstr_time_p_before_d = pyo.Constraint(
                m.set_requests, rule=cstr_time_p_before_d
            )

            def cstr_time_max_ride(m: Any, r: str) -> Any:
                p, d = m.param_request_pickup[r], m.param_request_delivery[r]
                return (
                    m.var_t[d] - m.var_t[p] - m.param_station_s[p]
                    <= m.param_rho * m.param_travel_time[p, d]
                )

            m.cstr_time_max_ride = pyo.Constraint(
                m.set_requests, rule=cstr_time_max_ride
            )

        # --- objective linearization for rejected requests ---
        if self.has_selectivity:

            def cstr_lin_t_p(m: Any, r: str) -> Any:
                return m.var_t[m.param_request_pickup[r]] <= m.param_big_m * m.var_y[r]

            m.cstr_lin_t_p = pyo.Constraint(m.set_requests, rule=cstr_lin_t_p)

            def cstr_lin_t_d(m: Any, r: str) -> Any:
                return (
                    m.var_t[m.param_request_delivery[r]] <= m.param_big_m * m.var_y[r]
                )

            m.cstr_lin_t_d = pyo.Constraint(m.set_requests, rule=cstr_lin_t_d)

            if self.has_soft_tw:

                def cstr_lin_tau(m: Any, r: str) -> Any:
                    return (
                        m.var_tau[m.param_request_pickup[r]]
                        <= m.param_big_m * m.var_y[r]
                    )

                m.cstr_lin_tau = pyo.Constraint(m.set_requests, rule=cstr_lin_tau)

        # --- time window restrictions ---
        new_reqs = [r for r in ctx.request_names if ctx.requests[r].is_new()]
        sch_reqs = [r for r in ctx.request_names if ctx.requests[r].is_scheduled()]

        if self.milp_type == MilpType.milp1:
            # soft TW, non-selective
            def cstr_tw_new_lb(m: Any, r: str) -> Any:
                p = m.param_request_pickup[r]
                return m.var_t[p] >= m.param_pickup_e[p] - m.var_tau[p]

            def cstr_tw_new_ub(m: Any, r: str) -> Any:
                p = m.param_request_pickup[r]
                return m.var_t[p] <= m.param_pickup_l[p] + m.var_tau[p]

            if new_reqs:
                m.cstr_tw_new_lb = pyo.Constraint(new_reqs, rule=cstr_tw_new_lb)
                m.cstr_tw_new_ub = pyo.Constraint(new_reqs, rule=cstr_tw_new_ub)

            def cstr_tw_sch_lb(m: Any, r: str) -> Any:
                p = m.param_request_pickup[r]
                return m.var_t[p] >= m.param_pickup_e[p]

            def cstr_tw_sch_ub(m: Any, r: str) -> Any:
                p = m.param_request_pickup[r]
                return m.var_t[p] <= m.param_pickup_l[p] + m.var_tau[p]

            if sch_reqs:
                m.cstr_tw_sch_lb = pyo.Constraint(sch_reqs, rule=cstr_tw_sch_lb)
                m.cstr_tw_sch_ub = pyo.Constraint(sch_reqs, rule=cstr_tw_sch_ub)
        elif self.milp_type == MilpType.milp2:
            # hard TW, selective, indiscriminate for new/scheduled
            def cstr_tw_hard_lb(m: Any, r: str) -> Any:
                p = m.param_request_pickup[r]
                return m.var_t[p] >= m.param_pickup_e[p] - m.param_big_m * (
                    1 - m.var_y[r]
                )

            def cstr_tw_hard_ub(m: Any, r: str) -> Any:
                p = m.param_request_pickup[r]
                return m.var_t[p] <= m.param_pickup_l[p] + m.param_big_m * (
                    1 - m.var_y[r]
                )

            m.cstr_tw_hard_lb = pyo.Constraint(m.set_requests, rule=cstr_tw_hard_lb)
            m.cstr_tw_hard_ub = pyo.Constraint(m.set_requests, rule=cstr_tw_hard_ub)
        else:
            # milp3: soft TW, selective
            def cstr_tw_new_lb(m: Any, r: str) -> Any:
                p = m.param_request_pickup[r]
                rhs = m.param_pickup_e[p] - m.var_tau[p]
                if self.has_selectivity:
                    rhs -= m.param_big_m * (1 - m.var_y[r])
                return m.var_t[p] >= rhs

            def cstr_tw_new_ub(m: Any, r: str) -> Any:
                p = m.param_request_pickup[r]
                rhs = m.param_pickup_l[p] + m.var_tau[p]
                if self.has_selectivity:
                    rhs += m.param_big_m * (1 - m.var_y[r])
                return m.var_t[p] <= rhs

            if new_reqs:
                m.cstr_tw_new_lb = pyo.Constraint(new_reqs, rule=cstr_tw_new_lb)
                m.cstr_tw_new_ub = pyo.Constraint(new_reqs, rule=cstr_tw_new_ub)

            def cstr_tw_sch_lb(m: Any, r: str) -> Any:
                p = m.param_request_pickup[r]
                rhs = m.param_pickup_e[p]
                if self.has_selectivity:
                    rhs -= m.param_big_m * (1 - m.var_y[r])
                return m.var_t[p] >= rhs

            def cstr_tw_sch_ub(m: Any, r: str) -> Any:
                p = m.param_request_pickup[r]
                rhs = m.param_pickup_l[p] + m.var_tau[p]
                if self.has_selectivity:
                    rhs += m.param_big_m * (1 - m.var_y[r])
                return m.var_t[p] <= rhs

            if sch_reqs:
                m.cstr_tw_sch_lb = pyo.Constraint(sch_reqs, rule=cstr_tw_sch_lb)
                m.cstr_tw_sch_ub = pyo.Constraint(sch_reqs, rule=cstr_tw_sch_ub)

        # --- special timing constraints for milp1 instead of energy ---
        if self.milp_type == MilpType.milp1:

            def cstr_duration_max(m: Any, k: str) -> Any:
                arcs_k = [(i, j) for (i, j, kk) in m.set_arcs if kk == k]
                if not arcs_k:
                    return pyo.Constraint.Skip
                return (
                    sum(
                        m.param_travel_time[i, j] * m.var_x[i, j, k]
                        for (i, j) in arcs_k
                    )
                    <= m.param_agent_duration_max[k]
                )

            m.cstr_duration_max = pyo.Constraint(m.set_agents, rule=cstr_duration_max)

        # --- passenger load ---
        def cstr_load_flow(m: Any, i: str, j: str) -> Any:
            sum_x = sum(
                m.var_x[i, j, k] for k in m.set_agents if j in self._x_out(i, k)
            )
            return m.var_z[j] >= m.var_z[i] + m.param_station_q[
                j
            ] - m.param_load_max_fleet * (1 - sum_x)

        m.cstr_load_flow = pyo.Constraint(station_pairs, rule=cstr_load_flow)

        if self.has_selectivity:

            def cstr_load_pickup_lb(m: Any, r: str) -> Any:
                p = m.param_request_pickup[r]
                return m.var_z[p] >= m.param_station_q[p] * m.var_y[r]

            def cstr_load_pickup_ub(m: Any, r: str) -> Any:
                p = m.param_request_pickup[r]
                return m.var_z[p] <= (
                    sum(
                        m.param_agent_load_max[k] * m.var_x[p, j, k]
                        for k in m.set_agents
                        for j in self._x_out(p, k)
                    )
                    + m.param_load_max_fleet * (1 - m.var_y[r])
                )

            m.cstr_load_pickup_lb = pyo.Constraint(
                m.set_requests, rule=cstr_load_pickup_lb
            )
            m.cstr_load_pickup_ub = pyo.Constraint(
                m.set_requests, rule=cstr_load_pickup_ub
            )
        else:

            def cstr_load_pickup_lb(m: Any, r: str) -> Any:
                p = m.param_request_pickup[r]
                return m.var_z[p] >= m.param_station_q[p]

            def cstr_load_pickup_ub(m: Any, r: str) -> Any:
                p = m.param_request_pickup[r]
                return m.var_z[p] <= sum(
                    m.param_agent_load_max[k] * m.var_x[p, j, k]
                    for k in m.set_agents
                    for j in self._x_out(p, k)
                )

            m.cstr_load_pickup_lb = pyo.Constraint(
                m.set_requests, rule=cstr_load_pickup_lb
            )
            m.cstr_load_pickup_ub = pyo.Constraint(
                m.set_requests, rule=cstr_load_pickup_ub
            )
        # z_i >= 0 for delivery nodes is implied by variable domain (NonNegativeIntegers)

        # --- extended SoC ---
        if self.has_extended_soc:

            def cstr_soc_initial(m: Any, r: str) -> Any:
                p = m.param_request_pickup[r]
                direct_sum = sum(
                    m.var_x[ctx.agent_initial_node[k], p, k] for k in m.set_agents
                )
                return m.var_phi[p] <= sum(
                    (
                        m.param_agent_soc_initial[k]
                        - m.param_agent_alpha[k]
                        * m.param_travel_time[ctx.agent_initial_node[k], p]
                    )
                    * m.var_x[ctx.agent_initial_node[k], p, k]
                    for k in m.set_agents
                ) + (1 - direct_sum)

            m.cstr_soc_initial = pyo.Constraint(m.set_requests, rule=cstr_soc_initial)

            soc_flow_index = [
                (i, j, k)
                for (i, j, k) in m.set_arcs
                if i in m.set_stations and j in m.set_stations
            ]

            def cstr_soc_flow(m: Any, i: str, j: str, k: str) -> Any:
                return m.var_phi[j] <= m.var_phi[i] - m.param_agent_alpha[
                    k
                ] * m.param_travel_time[i, j] - m.param_agent_beta[
                    k
                ] * m.param_travel_time[
                    i, j
                ] * m.var_z[
                    i
                ] + (
                    1 - m.var_x[i, j, k]
                )

            m.cstr_soc_flow = pyo.Constraint(soc_flow_index, rule=cstr_soc_flow)

            def cstr_soc_limits_lb(m: Any, j: str) -> Any:
                return m.var_phi[j] >= sum(
                    m.param_agent_soc_min[k] * m.var_x[i, j, k]
                    for k in m.set_agents
                    for i in self._x_in(j, k)
                )

            m.cstr_soc_limits_lb = pyo.Constraint(
                m.set_stations, rule=cstr_soc_limits_lb
            )
            # phi_j <= 1 is already implied by the variable's declared bounds

            def cstr_soc_hub(m: Any, i: str) -> Any:
                lhs = m.var_phi[i] - sum(
                    m.param_agent_alpha[k]
                    * m.param_travel_time[i, h]
                    * m.var_x[i, h, k]
                    for k in m.set_agents
                    for h in ctx.final_depot_names
                    if h in self._x_out(i, k)
                )
                rhs = sum(
                    m.param_agent_soc_min[k] * m.var_x[i, h, k]
                    for k in m.set_agents
                    for h in ctx.final_depot_names
                    if h in self._x_out(i, k)
                )
                return lhs >= rhs

            m.cstr_soc_hub = pyo.Constraint(m.set_stations_delivery, rule=cstr_soc_hub)

    def _build_model(self) -> None:
        """Build the complete MILP model"""
        if self.is_trivial:
            return
        self.model = pyo.ConcreteModel()
        self._define_sets()
        self._define_parameters()
        self._define_variables()
        self._define_objective()
        self._define_constraints()

    # ---------------------------------------------------------------------------------
    # Warm-starting and solution extraction (MILP <-> ModemsSolution converter)
    # ---------------------------------------------------------------------------------

    def _pyo_value(self, value: pyo.Any, default_value: float = 0.0) -> float:
        """Helper function to read pyomo values, defaults to default_value"""
        result = pyo.value(value, exception=False)
        return default_value if result is None else float(result)

    def _warm_start(self, solution: ModemsSolution) -> None:
        """
        Populate all decision variables from a (feasible, complete) ModemsSolution
        object, so the solver can be warm-started from a constructive/ALNS incumbent
        """
        m = self.model
        ctx = self.ctx

        # reset all values
        for val in m.var_x.values():
            val.value = 0
        for r in m.set_stations:
            m.var_t[r].value = 0.0
            m.var_z[r].value = 0
            if self.has_soft_tw and r in m.set_stations_pickup:
                m.var_tau[r].value = 0.0
            if self.has_extended_soc:
                m.var_phi[r].value = 1.0

        # parse given solution
        t_mission = 0.0
        is_open = ProblemType.is_open(self.problem_type)
        for k, journey in solution.journeys.items():
            route = journey.route
            for i, j in zip(route[:-1], route[1:]):
                if (i, j, k) in m.set_arcs:
                    m.var_x[i, j, k].value = 1
            for i, node in enumerate(route):
                state = journey.states[i]
                if node in m.set_stations:
                    m.var_t[node].value = state.t_start
                    m.var_z[node].value = state.z_dep
                    if self.has_soft_tw and node in m.set_stations_pickup:
                        request = ctx.requests[ctx.node_request[node]]
                        tau = max(0.0, state.t_start - ctx.node_latest_p[node])
                        if request.is_new():
                            tau = max(tau, ctx.node_earliest_p[node] - state.t_start)
                        # preserve an inherited MILP timing/slack decision, while
                        # defensively raising stale slack to snapshot min required tau
                        m.var_tau[node].value = max(state.tau, tau)
                    if self.has_extended_soc:
                        m.var_phi[node].value = max(0.0, min(1.0, state.phi_arr))
            if journey.is_active:
                if is_open:
                    t_mission = max(t_mission, journey.states[-2].t_dep)
                else:
                    t_mission = max(t_mission, journey.states[-1].t_arr)
        m.var_T.value = t_mission

        if self.has_selectivity:
            for r in m.set_requests:
                m.var_y[r].value = 1 if r in solution.accepted else 0

    def _extract_solution(self) -> ModemsSolution:
        """
        Post-process a solved MILP into the shared ModemsSolution representation.
        The decided route (agent assignment + visiting order from x), the request
        accept/reject decision (from y), service-start times (from t), and each pickup
        soft-TW slack (from tau) are taken from the MILP. Arrival, waiting, departure,
        load, and SoC are then reconstructed along that route, with waiting inferred
        as max(0, t_i - arrival_i) to preserve valid MILP waiting decisions
        """
        m = self.model
        sol = ModemsSolution(self.ctx)
        for k in self.ctx.agent_names:
            arcs = {
                i: j
                for (i, j, chk_k) in m.set_arcs
                if chk_k == k
                and self._pyo_value(m.var_x[i, j, chk_k]) is not None
                and self._pyo_value(m.var_x[i, j, chk_k]) > 0.5
            }
            v_k = self.ctx.agent_initial_node[k]
            route = [v_k]
            node_curr = v_k
            while node_curr in arcs:
                node_curr = arcs[node_curr]
                route.append(node_curr)
            if len(route) == 1:
                # idle agent: match ModemsJourney's own default idle representation
                route.append(self.ctx.nearest_final_depot(v_k))
            tau_of = (
                {
                    node: self._pyo_value(m.var_tau[node])
                    for node in route
                    if node in m.set_stations_pickup
                }
                if self.has_soft_tw
                else None
            )
            t_start_of = {
                node: self._pyo_value(m.var_t[node])
                for node in route
                if node in m.set_stations
            }
            # populate the agent journey
            sol.journeys[k] = ModemsJourney.from_route(
                self.ctx,
                k,
                route,
                tau_of=tau_of,
                t_start_of=t_start_of,
            )
        for r_name in self.ctx.request_names:
            accepted = True
            if self.has_selectivity:
                accepted = self._pyo_value(m.var_y[r_name]) > 0.5
            if accepted:
                sol.accepted.add(r_name)
            else:
                sol.rejected.add(r_name)
        r_routed = {
            r_name
            for journey in sol.journeys.values()
            for r_name in journey.request_pickup
        }
        if r_routed != sol.accepted:
            raise ValueError(
                "MILP route extraction is inconsistent with request acceptance"
            )
        if any(not journey._is_feasible() for journey in sol.journeys.values()):
            raise ValueError("MILP route extraction produced an infeasible journey")
        objective = self._pyo_value(self.model.objective)
        if abs(sol.objective() - objective) > FLOAT_TOL:
            raise ValueError("MILP solution and re-constructed objective mismatch")
        return sol

    # ---------------------------------------------------------------------------------
    # Solve / results
    # ---------------------------------------------------------------------------------

    def _check_solution_status(self) -> tuple[int, str]:
        """
        Returns (flag, message):
          0  = proven optimal
          1  = feasible incumbent found (time limit reached before proof)
         -1  = proven infeasible
         -2  = no incumbent found before the time limit (status genuinely
               unknown -- NOT the same as proven infeasible)
        """
        if not self.problem_done:
            raise NameError("Problem not solved. Solve problem first!")
        term_cond = self.results.solver.termination_condition

        if (self.results.solver.status == pyo.SolverStatus.ok) and (
            term_cond == pyo.TerminationCondition.optimal
        ):
            return 0, "Optimal solution found!"

        if term_cond == pyo.TerminationCondition.infeasible:
            return -1, "Proven infeasible."

        if term_cond == pyo.TerminationCondition.intermediateNonInteger:
            return (
                -2,
                "Time limit reached before any integer-feasible solution was found",
            )

        if term_cond in (
            pyo.TerminationCondition.maxTimeLimit,
            pyo.TerminationCondition.feasible,
        ) or (self.results.solver.status == pyo.SolverStatus.aborted):
            # An aborted/time-limited run can still have zero found incumbents
            # (rare, but cbc does strange things sometimes) and pyo.value() alone is
            # not indicative enough, so check the reported upper bound
            has_incumbent = True
            try:
                ub = self.results.problem.upper_bound
                has_incumbent = ub is not None and abs(ub) < FLOAT_INF
            except Exception:
                pass
            if not has_incumbent:
                return (
                    -2,
                    "Time limit reached before any integer-feasible solution was found",
                )
            try:
                pyo.value(self.model.objective)
                return 1, "Time limit reached / feasible solution."
            except Exception:
                return (
                    -2,
                    "Time limit reached before any integer-feasible solution was found",
                )

        return -2, f"Solver terminated without a proven result ({term_cond})"

    def solve(
        self,
        solver_name: str,
        solver_config_type: SolverConfigType | str,
        solver_options: dict[str, Any] | None = None,
        tee: bool = False,
        print_results: bool = False,
        warm_start_solution: ModemsSolution | None = None,
    ) -> Any | None:
        """
        Solve the MILP model

        Args:
            solver_name: The literal string passed to pyomo SolverFactory (e.g., "cbc",
                "gurobi", "gurobi_persistent", "appsi_highs")
            solver_config_type: The option-key-naming configuration that the solver
                follows (cbc/gurobi/highs), check SolverConfigType
            solver_options: Optional solver options using generic keys ("timelimit",
                "threads") mapped per solver_config_type, or any other solver-native
                key passed straight through (e.g., {"mipgap": 0.01})
            tee: Print solver output
            print_results: Print solver results
            warm_start_solution: Optional incumbent used to warm-start the solver
        """
        self.solver_type = solver_name
        self.solver_config_type = SolverConfigType(solver_config_type)
        self.solver_options = {
            "timelimit": DEFAULT_MILP_TIMELIMIT,
            **(solver_options or {}),
        }

        if self.is_trivial:
            self._solve_trivial()
            if print_results:
                print(
                    "Trivial instance (no agents or no requests); optimizer not invoked"
                )
            return None

        self.solver = SolverFactory(solver_name)
        if solver_name == "cbc" and not self.solver.available(exception_flag=False):
            # PuLP wheels bundle CBC but do not consistently expose it on PATH
            # for Pyomo. try a simple lookup
            try:
                import pulp

                cbc_path = pulp.PULP_CBC_CMD().path
            except (ImportError, AttributeError) as excp:
                raise RuntimeError(
                    "CBC is unavailable: install the project's pulp[cbc] "
                    "dependency or provide another configured solver"
                ) from excp
            self.solver = SolverFactory("cbc", executable=cbc_path)
        key_map = _SOLVER_OPTION_KEY_MAP[self.solver_config_type]
        generic_keys = {"timelimit", "threads"}
        options: dict[str, Any] = {}
        for key, value in self.solver_options.items():
            if key in generic_keys:
                mapped_key = key_map.get(key)
                if mapped_key is not None:
                    options[mapped_key] = value
                else:
                    # this solver_config_type does not support this generic option
                    # such as "cbc" + "threads", silently drop it
                    pass
            else:
                options[key] = value  # already solver-native, pass through
        self.solver.options.update(options)

        use_warmstart = warm_start_solution is not None
        if use_warmstart:
            self._warm_start(warm_start_solution)

        t_beg = time.perf_counter()
        self.results = self.solver.solve(self.model, tee=tee, warmstart=use_warmstart)
        t_exe = time.perf_counter() - t_beg
        self.problem_done = True
        self._build_instance(t_exe)
        if print_results:
            print("Solver results:")
            print(self.results)
        return self.results

    def _solve_trivial(self) -> None:
        """
        Build the instance directly (no optimizer call) for a scenario with
        no agents and/or no requests -- nothing is actually being decided,
        so the result is exact by construction, not a heuristic approximation.
        """
        t_beg = time.perf_counter()
        base_plan, r_unassigned = preprocess(
            self.ctx,
        )
        ok = base_plan is not None
        if base_plan is not None and r_unassigned:
            base_plan, _, ok = greedy_complete(base_plan, r_unassigned)
        if ok and base_plan is not None:
            solution = base_plan
            status = SolutionStatus.optimal
            objective = solution.objective()
        else:
            solution = ModemsSolution(self.ctx)
            status = SolutionStatus.infeasible
            objective = None
        t_exe = time.perf_counter() - t_beg

        solution_info = ModemsSolutionInfo(
            status=status,
            solution_time=t_exe,
            objective=objective if objective is not None else FLOAT_INF,
            solver_name=self.solver_type,
            solver_options=self.solver_options,
            solver_diagnostics={"trivial_solution": 1},
        )
        self.instance = ModemsInstance(
            self.ctx,
            solution,
            solution_info,
        )
        self.problem_done = True

    def _loaded_solution_is_valid(self, tol: float = FLOAT_TOL) -> bool:
        """
        Check that the currently loaded variable values actually satisfy every active
        constraint. CBC's warm-started, time-limited solves may sometimes load an
        internally-inconsistent solution (e.g., t_i violating its own soft-TW lower
        bound) despite reporting a plausible-looking status and bounds. This is a
        solver/interface reliability issue, not something we can fix upstream, so we
        validate defensively before trusting any non-proven-optimal incumbent
        """
        for conditions in self.model.component_objects(pyo.Constraint, active=True):
            for idx in conditions:
                condition = conditions[idx]
                try:
                    lb, body, ub = condition.to_bounded_expression()
                    body_val = pyo.value(body)
                    if lb is not None and body_val < pyo.value(lb) - tol:
                        return False
                    if ub is not None and body_val > pyo.value(ub) + tol:
                        return False
                except Exception:
                    return False
        return True

    def _build_instance(self, solution_time: float) -> None:
        """Build the canonical (scenario, solution, solver-details) ModemsInstance"""
        flag, _ = self._check_solution_status()

        if flag >= 0 and not self._loaded_solution_is_valid():
            # The solver reported an incumbent, but the loaded variable values do NOT
            # actually satisfy the model constraints, treat as no incumbent found
            flag = -2

        status = {
            0: SolutionStatus.optimal,
            1: SolutionStatus.feasible,
            -1: SolutionStatus.infeasible,
            # Timed out before a valid incumbent; not proven infeasible
            -2: SolutionStatus.unknown,
        }[flag]

        lower_bound = upper_bound = objective = None
        if flag >= 0:
            objective = pyo.value(self.model.objective)
        if flag == 0:
            lower_bound = objective
            upper_bound = objective
        else:
            try:
                lower_bound = self.results.problem.lower_bound
            except Exception:
                pass
            try:
                upper_bound = self.results.problem.upper_bound
            except Exception:
                pass

        solution = self._extract_solution() if flag >= 0 else ModemsSolution(self.ctx)

        solution_info = ModemsSolutionInfo(
            status=status,
            solution_time=solution_time,
            objective=objective if objective is not None else FLOAT_INF,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            solver_name=self.solver_type,
            solver_options=self.solver_options,
            solver_diagnostics={
                "nr_constraints": getattr(
                    self.results.problem,
                    "number_of_constraints",
                    None,
                ),
                "nr_variables": getattr(
                    self.results.problem,
                    "number_of_variables",
                    None,
                ),
            },
        )
        self.instance = ModemsInstance(
            self.ctx,
            solution,
            solution_info,
        )
