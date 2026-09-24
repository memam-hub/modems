from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, cast

import numpy as np
from alns import ALNS
from alns import Result as AlnsResult
from alns.accept import SimulatedAnnealing
from alns.select import RouletteWheel
from alns.stop import NoImprovement

from .algorithms import (
    InsertionCandidate,
    alns_feasible_insertions,
    greedy_complete,
    preprocess,
)
from .core import ModemsScenario, ProblemContext, ProblemType, SolverStrategy
from .network import NetworkNodeName
from .solution import (
    DEFAULT_PARAMS_ALNS,
    DEFAULT_PARAMS_RD,
    DEFAULT_PARAMS_RW,
    DEFAULT_PARAMS_SA,
    ModemsInstance,
    ModemsSolution,
    ModemsSolutionInfo,
    SolutionStatus,
)


class ModemsAlns:
    """Adaptive Large Neighborhood Search wrapper to use the alns package"""

    problem_type: ProblemType
    ctx: ProblemContext
    best_solution: ModemsSolution | None
    result: AlnsResult | None
    alns_params: dict[str, Any]
    _destroy_min_pct: float = DEFAULT_PARAMS_RD["min_pct"]
    _destroy_max_pct: float = DEFAULT_PARAMS_RD["max_pct"]

    def __init__(
        self,
        scenario: ModemsScenario,
        problem_type: ProblemType = ProblemType.closed_selective,
        model_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize with a problem context and a dummy solution. If not given,
        model_params defaults to DEFAULT_PARAMS_OBJ
        """
        self.problem_type = ProblemType(problem_type)
        self.ctx = ProblemContext(
            scenario, self.problem_type, SolverStrategy.alns, model_params
        )
        self.alns_params = {}
        self.best_solution = None
        self.result = None

    # ----------------------------------------------------------------------------------
    # Initial solution construction
    # ----------------------------------------------------------------------------------

    def _create_initial_solution(
        self, partial_plan: ModemsSolution | None = None
    ) -> ModemsSolution:
        """[Alg.1+2] Create an initial solution using preprocess+greedy insertion"""
        base_plan, r_unassigned = preprocess(self.ctx, partial_plan=partial_plan)
        if base_plan is None:
            raise RuntimeError(
                "Infeasible base plan: scheduled requests or initial agent "
                "states cannot be satisfied."
            )
        solution, _, ok = greedy_complete(base_plan, r_unassigned)
        if not ok:
            raise RuntimeError(
                "No feasible non-selective initial solution could be constructed."
            )
        return solution

    # ----------------------------------------------------------------------------------
    # Shared destroy helpers
    # ----------------------------------------------------------------------------------

    def _destroy_size(self, state: ModemsSolution, rng: np.random.Generator) -> int:
        """Number of requests to destroy/remove"""
        nr_accepted = len(state.accepted)
        if nr_accepted == 0:
            return 0
        mu = rng.uniform(self._destroy_min_pct, self._destroy_max_pct)
        return min(nr_accepted, max(1, int(np.floor(mu * nr_accepted))))

    def _remove_requests(
        self, state: ModemsSolution, requests: list[str]
    ) -> ModemsSolution:
        """Remove requests sequentially, updating states/sets after every removal"""
        state = state.copy()
        for r in requests:
            k = state.agent_of(r)
            if k is not None:  # just for sanity
                state.journeys[k].remove_request(r)
            state.accepted.discard(r)
            state.rejected.add(r)
        return state

    # ----------------------------------------------------------------------------------
    # OD1-OD6: destroy operators
    # ----------------------------------------------------------------------------------

    def destroy_random(
        self, state: ModemsSolution, rng: np.random.Generator
    ) -> ModemsSolution:
        """OD1: Random -- select requests uniformly at random"""
        nr_to_del = self._destroy_size(state, rng)
        if nr_to_del == 0:
            return state.copy()
        selected = rng.choice(sorted(state.accepted), size=nr_to_del, replace=False)
        return self._remove_requests(state, [str(r) for r in selected])

    def destroy_random_zone(
        self, state: ModemsSolution, rng: np.random.Generator
    ) -> ModemsSolution:
        """OD2: Random-Zone -- remove requests sharing a randomly selected station"""
        nr_to_del = self._destroy_size(state, rng)
        if nr_to_del == 0:
            return state.copy()
        ctx = self.ctx

        def station_name(node: str) -> int:
            """Get corresponding physical node index of the virtual node name"""
            return NetworkNodeName.get_node_index(node)

        r_remaining = set(state.accepted)
        r_to_del = []
        nr_attempts = 0  # safeguard against infinite loops
        while len(r_to_del) < nr_to_del and r_remaining and nr_attempts < 100:
            nr_attempts += 1
            r = str(rng.choice(sorted(r_remaining)))
            node = str(rng.choice([ctx.pickup_node[r], ctx.delivery_node[r]]))
            zone = station_name(node)
            zone_requests = [
                r
                for r in sorted(r_remaining)
                if station_name(ctx.pickup_node[r]) == zone
                or station_name(ctx.delivery_node[r]) == zone
            ]
            rng.shuffle(zone_requests)
            for r in zone_requests:
                if len(r_to_del) >= nr_to_del:
                    break
                r_to_del.append(r)
                r_remaining.discard(r)
        return self._remove_requests(state, r_to_del)

    def destroy_lowest_demand(
        self, state: ModemsSolution, rng: np.random.Generator
    ) -> ModemsSolution:
        """
        OD3: Lowest-Demand -- remove requests in increasing order of load q^r,
        randomly breaking ties
        """
        nr_to_del = self._destroy_size(state, rng)
        if nr_to_del == 0:
            return state.copy()
        # tie-break on request name: load alone has a small integer range and ties
        # are common. So, without a total-order key, stability of sorted() silently
        # falls back to state.accepted's own (hash-randomized) set iteration order
        # for tied requests. Instead, this gives consistent sampling results
        r_to_del = sorted(state.accepted, key=lambda r: (self.ctx.requests[r].load, r))
        return self._remove_requests(state, r_to_del[:nr_to_del])

    def destroy_worst_cost(
        self, state: ModemsSolution, rng: np.random.Generator
    ) -> ModemsSolution:
        """OD4: Worst-Cost -- remove requests with the largest objective saving"""
        nr_to_del = self._destroy_size(state, rng)
        if nr_to_del == 0:
            return state.copy()
        base_obj = state.objective(include_rejection=False)
        obj_savings: list[tuple[float, str]] = []
        for r in sorted(state.accepted):
            sol_temp = state.copy()
            k = sol_temp.agent_of(r)
            sol_temp.journeys[k].remove_request(r)  # type: ignore : r is accepted
            sol_temp.accepted.discard(r)
            obj_temp = sol_temp.objective(include_rejection=False)
            obj_savings.append((base_obj - obj_temp, r))
        obj_savings.sort(key=lambda x: x[0], reverse=True)
        r_to_del = [r for _, r in obj_savings[:nr_to_del]]
        return self._remove_requests(state, r_to_del)

    def destroy_worst_waiting(
        self, state: ModemsSolution, rng: np.random.Generator
    ) -> ModemsSolution:
        """
        OD5: Worst-Waiting -- remove requests with the largest pickup waiting time,
        breaking ties randomly
        """
        size = self._destroy_size(state, rng)
        if size == 0:
            return state.copy()

        def wait_of(r: str) -> float:
            """Get waiting time of the given request"""
            k = state.agent_of(r)
            journey = state.journeys[k]  # type: ignore : r is accepted
            p_idx = journey.request_pickup[r]
            return journey.states[p_idx].t_wait

        # tie-break on request name: wait time is exactly 0.0 for well-scheduled
        # requests. So, without a total-order key, stability of sorted() silently
        # falls back to state.accepted's own (hash-randomized) set iteration order
        # for tied requests. Instead, this gives consistent sampling results
        r_to_del = sorted(state.accepted, key=lambda r: (wait_of(r), r), reverse=True)
        return self._remove_requests(state, r_to_del[:size])

    def destroy_worst_energy(
        self, state: ModemsSolution, rng: np.random.Generator
    ) -> ModemsSolution:
        """OD6: Worst-Energy -- remove requests with the largest marginal energy draw"""
        nr_to_del = self._destroy_size(state, rng)
        if nr_to_del == 0:
            return state.copy()
        base_obj = state.total_energy()
        obj_savings: list[tuple[float, str]] = []
        for r in sorted(state.accepted):
            sol_temp = state.copy()
            k = sol_temp.agent_of(r)
            sol_temp.journeys[k].remove_request(r)  # type: ignore : r is accepted
            sol_temp.accepted.discard(r)
            obj_savings.append((base_obj - sol_temp.total_energy(), r))
        obj_savings.sort(key=lambda x: x[0], reverse=True)
        r_to_del = [r for _, r in obj_savings[:nr_to_del]]
        return self._remove_requests(state, r_to_del)

    # ----------------------------------------------------------------------------------
    # Shared repair helpers
    # ----------------------------------------------------------------------------------

    def _best_candidate(
        self, state: ModemsSolution, request_name: str
    ) -> InsertionCandidate | None:
        """
        Compile all feasible candidates (journeys) for the given request (if any) and
        greedily return the one with min objective increase. None if no candidates exist
        """
        candidates = alns_feasible_insertions(state, request_name)
        if not candidates:
            return None
        return min(candidates, key=lambda c: c.delta_obj)

    def _commit(
        self,
        state: ModemsSolution,
        request_name: str,
        candidate: InsertionCandidate,
    ) -> None:
        """Adopt a feasibility-proven replacement journey in constant time"""
        state.journeys[candidate.journey.agent_name] = candidate.journey
        state.accepted.add(request_name)
        state.rejected.discard(request_name)

    def _repair_common(
        self,
        state: ModemsSolution,
        select_next: Callable[
            [ModemsSolution, set[str]], tuple[str, InsertionCandidate | None] | None
        ],
    ) -> ModemsSolution:
        """
        Shared repair loop: scheduled requests (R_s subset R_u) are restored first,
        then the (new) pending requests. If any scheduled cannot be restored, the
        candidate is discarded (its objective is forced to +inf, rendered infeasible).
        select_next(state, pending) calls the repair operator and returns a selected
        request alongside its candidate journey (already evaluated, following [Alg.3]).
        A request paired with None is discarded; a None selection means no pending
        request is insertable.
        """
        state = state.copy()
        if state._infeasible:
            state.recompute_rejected()
            return state
        r_pending = state.pending()
        r_pending_scheduled = set(
            r for r in r_pending if self.ctx.requests[r].is_scheduled()
        )
        while r_pending_scheduled:
            selection = select_next(state, r_pending_scheduled)
            if selection is not None:
                r, candidate = selection
                if candidate is not None:
                    self._commit(state, r, candidate)
                    r_pending_scheduled.discard(r)
                    r_pending.discard(r)
                    continue
            state._infeasible = True
            return state

        while r_pending:
            selection = select_next(state, r_pending)
            if selection is None:
                break
            r, candidate = selection
            if candidate is None:
                r_pending.discard(r)
                continue
            self._commit(state, r, candidate)
            r_pending.discard(r)

        if not self.ctx.is_selective() and state.rejected:
            state._infeasible = True
        return state

    # ----------------------------------------------------------------------------------
    # OR1-OR5: repair operators
    # ----------------------------------------------------------------------------------

    def repair_random(
        self, state: ModemsSolution, rng: np.random.Generator
    ) -> ModemsSolution:
        """OR1: Random -- insert requests in random order at minimum cost"""

        def select_next(
            state: ModemsSolution, pending: set[str]
        ) -> tuple[str, InsertionCandidate | None] | None:
            """Randomly select a pending request, return its best candidate"""
            if not pending:
                return None
            r_name = str(rng.choice(sorted(pending)))
            return (r_name, self._best_candidate(state, r_name))

        return self._repair_common(state, select_next)

    def repair_greedy(
        self, state: ModemsSolution, rng: np.random.Generator
    ) -> ModemsSolution:
        """OR2: Greedy -- select the request/position with smallest cost increase"""

        def select_next(
            state: ModemsSolution, pending: set[str]
        ) -> tuple[str, InsertionCandidate] | None:
            """Greedily select the best overall request+candidate position"""
            selection_best = None
            obj_best = None
            for r_name in sorted(pending):
                candidate = self._best_candidate(state, r_name)
                if candidate is None:
                    continue
                if obj_best is None or candidate.delta_obj < obj_best:
                    obj_best = candidate.delta_obj
                    selection_best = (r_name, candidate)
            return selection_best

        return self._repair_common(state, select_next)

    def repair_regret_2(
        self, state: ModemsSolution, rng: np.random.Generator
    ) -> ModemsSolution:
        """OR3: Regret-2 -- prioritize requests whose later (2) insertion is costlier"""

        def select_next(
            state: ModemsSolution, pending: set[str]
        ) -> tuple[str, InsertionCandidate] | None:
            """
            Compare the regret of best - 2nd best insertion for all candidates,
            prioritizing requests with a single candidate
            """
            selection_best = None
            regret_best = None
            for r in sorted(pending):
                candidates = alns_feasible_insertions(state, r)
                if not candidates:
                    continue
                candidates.sort(key=lambda c: c.delta_obj)
                if len(candidates) == 1:
                    return r, candidates[0]  # immediate priority
                regret = candidates[1].delta_obj - candidates[0].delta_obj
                if regret_best is None or regret > regret_best:
                    regret_best = regret
                    selection_best = (r, candidates[0])
            return selection_best

        return self._repair_common(state, select_next)

    def repair_longest_trip(
        self, state: ModemsSolution, rng: np.random.Generator
    ) -> ModemsSolution:
        """OR4: Longest-Trip -- insert longest direct trips first"""

        def select_next(
            state: ModemsSolution, pending: set[str]
        ) -> tuple[str, InsertionCandidate | None] | None:
            """Select longest direct trips (max detour support) then greedy feasible"""
            if not pending:
                return None

            def sort_key(r: str) -> tuple[float, float]:
                """Sort based on trip length"""
                request = self.ctx.requests[r]
                t_direct_trip = self.ctx.t_travel(
                    self.ctx.pickup_node[r], self.ctx.delivery_node[r]
                )
                return (-t_direct_trip, request.latest_pickup)

            r_name = min(sorted(pending), key=sort_key)
            return (r_name, self._best_candidate(state, r_name))

        return self._repair_common(state, select_next)

    def repair_most_constrained(
        self, state: ModemsSolution, rng: np.random.Generator
    ) -> ModemsSolution:
        """OR5: Most-Constrained -- insert requests with fewest candidates first"""

        def select_next(
            state: ModemsSolution, pending: set[str]
        ) -> tuple[str, InsertionCandidate | None] | None:
            """sort based on number of feasible insertion candidates"""
            selection_best = None
            nr_insert_best = None
            for r in sorted(pending):
                cands = alns_feasible_insertions(state, r)
                if not cands:
                    continue
                best_candidate = min(cands, key=lambda c: c.delta_obj)
                key = (len(cands), best_candidate.delta_obj)
                if nr_insert_best is None or key < nr_insert_best:
                    nr_insert_best = key
                    selection_best = (r, best_candidate)
            return selection_best

        return self._repair_common(state, select_next)

    # ----------------------------------------------------------------------------------
    # Main search/ALNS solve method
    # ----------------------------------------------------------------------------------

    def solve(
        self,
        seed: int,
        max_iter: int = DEFAULT_PARAMS_ALNS["max_iter"],
        params_rw: dict[str, Any] | None = None,
        params_sa: dict[str, Any] | None = None,
        params_rd: dict[str, Any] | None = None,
        partial_plan: ModemsSolution | None = None,
    ) -> Any | None:
        """
        Run the ALNS search

        Args:
            params_rw: RouletteWheel parameters, defaults to DEFAULT_PARAMS_RW
                        must contain {"scores": [s1,s2,s3,s4], "decay": float}
            params_sa: SimulatedAnnealing parameters, defaults to DEFAULT_PARAMS_SA
                        must contain {"start_temp", "end_temp", "cooling_rate"}
            params_rd: bounds for request destroy, defaults to DEFAULT_PARAMS_RD
            seed: RNG seed
            max_iter: maximum number of non-improving ALNS iterations
            partial_plan: Updated ModemsSolution excerpt from the previous plan.
                Required whenever the scenario has scheduled requests
        """
        rng = np.random.default_rng(seed)
        initial_solution = self._create_initial_solution(partial_plan=partial_plan)
        params_rw = {**DEFAULT_PARAMS_RW, **(params_rw or {})}
        params_sa = {**DEFAULT_PARAMS_SA, **(params_sa or {})}
        params_rd = {**DEFAULT_PARAMS_RD, **(params_rd or {})}
        if not 0.0 < params_rd["min_pct"] < params_rd["max_pct"] < 1.0:
            raise ValueError("request destroy percentages must be 0 < min < max < 1")
        if max_iter <= 0.0:
            raise ValueError("ALNS iterations must be >= 1")
        self.alns_params = {
            **DEFAULT_PARAMS_ALNS,
            "seed": seed,
            "max_iter": max_iter,
            "obj": self.ctx.model_params,
            "params_rw": params_rw,
            "params_sa": params_sa,
            "params_rd": params_rd,
        }
        self._destroy_min_pct = self.alns_params["params_rd"]["min_pct"]
        self._destroy_max_pct = self.alns_params["params_rd"]["max_pct"]

        if not self.ctx.scenario.agents or not self.ctx.scenario.requests:
            # do not run the search, the base solution is sufficient
            self.best_solution = initial_solution
            solution_info = ModemsSolutionInfo(
                status=SolutionStatus.optimal,
                solution_time=0.0,
                objective=initial_solution.objective(),
                solver_name=self.ctx.strategy.value,
                solver_options=self.alns_params,
                solver_diagnostics={"iterations": 0},
            )
            self.instance = ModemsInstance(
                self.ctx,
                self.best_solution,
                solution_info,
            )
            return None

        alns = ALNS(rng)
        destroy_ops: list[Callable] = [
            self.destroy_random,
            self.destroy_random_zone,
            self.destroy_lowest_demand,
            self.destroy_worst_cost,
            self.destroy_worst_waiting,
            self.destroy_worst_energy,
        ]
        repair_ops: list[Callable] = [
            self.repair_random,
            self.repair_greedy,
            self.repair_regret_2,
            self.repair_longest_trip,
            self.repair_most_constrained,
        ]
        for op in destroy_ops:
            alns.add_destroy_operator(op)
        for op in repair_ops:
            alns.add_repair_operator(op)

        select = RouletteWheel(
            scores=self.alns_params["params_rw"]["scores"],
            decay=self.alns_params["params_rw"]["decay"],
            num_destroy=len(destroy_ops),
            num_repair=len(repair_ops),
        )
        accept = SimulatedAnnealing(
            start_temperature=self.alns_params["params_sa"]["start_temp"],
            end_temperature=self.alns_params["params_sa"]["end_temp"],
            step=self.alns_params["params_sa"]["cooling_rate"],
            method="exponential",
        )
        stop = NoImprovement(max_iter)

        t_beg = time.perf_counter()
        self.result = alns.iterate(initial_solution, select, accept, stop)
        t_exe = time.perf_counter() - t_beg
        self.best_solution = cast(ModemsSolution, self.result.best_state)

        solution_info = ModemsSolutionInfo(
            status=SolutionStatus.feasible,
            solution_time=t_exe,
            objective=self.best_solution.objective(),
            solver_name=self.ctx.strategy.value,
            solver_options=self.alns_params,
            solver_diagnostics={"iterations": len(self.result.statistics.runtimes)},
        )
        self.instance = ModemsInstance(
            self.ctx,
            self.best_solution,
            solution_info,
        )
        return self.result

    # ----------------------------------------------------------------------------------
    # Reporting helpers
    # ----------------------------------------------------------------------------------

    def plot_routes(
        self,
        legend: bool = True,
        show: bool = True,
        outfile: str = "",
    ) -> None:
        if self.best_solution is None:
            raise NameError("Not solved yet. Call solve() first!")
        is_open = self.ctx.is_open()
        routes = {}
        for k, journey in self.best_solution.journeys.items():
            if not journey.is_active:
                continue
            routes[k] = journey.route[:-1] if is_open else journey.route
        self.ctx.scenario.network.plot_routes(
            routes, legend=legend, show=show, outfile=outfile
        )

    def plot_metrics(self, outdir: str, prefix: str) -> None:
        """Plot the ALNS convergence (objective) and operator-performance charts"""
        if self.result is None:
            raise NameError("Not solved yet. Call solve() first!")
        import os

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        fig.set_size_inches(18.5, 10.5, forward=True)
        self.result.plot_objectives(ax=ax)
        fig.savefig(
            os.path.join(outdir, f"{prefix}_objectives.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig)

        fig, ax = plt.subplots()
        fig.set_size_inches(18.5, 10.5, forward=True)
        self.result.plot_operator_counts(fig=fig)
        fig.savefig(
            os.path.join(outdir, f"{prefix}_operators.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig)


def plot_alns_metrics_from_stats(
    stats: dict[str, Any], outdir: str, prefix: str
) -> None:
    """
    Reconstruct the same two charts as ModemsAlns.plot_metrics() -- objective
    convergence and destroy/repair operator counts -- from the plain
    JSON-serializable data benchmark.py's _solve_one() persists alongside a
    solved ALNS result (stats["objectives"]/["destroy_operator_counts"]/
    ["repair_operator_counts"]), rather than from the live alns.Result object
    plot_metrics() itself requires. This is what lets a later, read-only
    "build" phase produce ALNS-specific plots without re-solving anything --
    the rendering logic below mirrors alns.Result.plot_objectives()/
    plot_operator_counts()/_plot_op_counts() exactly, since those methods
    only ever touch this same data internally.
    """
    import os

    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(outdir, exist_ok=True)

    objectives = stats.get("objectives") or []
    if objectives:
        fig, ax = plt.subplots()
        fig.set_size_inches(18.5, 10.5, forward=True)
        ax.plot(objectives)
        ax.plot(np.minimum.accumulate(objectives))
        ax.set_title("Objective value at each iteration")
        ax.set_ylabel("Objective value")
        ax.set_xlabel("Iteration (#)")
        ax.legend(["Current", "Best"], loc="upper right")
        fig.savefig(
            os.path.join(outdir, f"{prefix}_objectives.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig)

    destroy_counts = stats.get("destroy_operator_counts") or {}
    repair_counts = stats.get("repair_operator_counts") or {}
    if destroy_counts or repair_counts:
        fig, (d_ax, r_ax) = plt.subplots(nrows=2)
        fig.set_size_inches(18.5, 10.5, forward=True)
        fig.subplots_adjust(hspace=0.7, bottom=0.2)
        legend = ["Best", "Better", "Accepted", "Rejected"]
        _plot_operator_counts(d_ax, destroy_counts, "Destroy operators", len(legend))
        _plot_operator_counts(r_ax, repair_counts, "Repair operators", len(legend))
        fig.legend(legend, ncol=len(legend), loc="lower center")
        fig.savefig(
            os.path.join(outdir, f"{prefix}_operators.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig)


def _plot_operator_counts(
    ax: Any, operator_counts: dict[str, list[float]], title: str, nr_types: int
) -> None:
    """Render one (destroy or repair) operator-counts horizontal bar chart --
    mirrors alns.Result._plot_op_counts()'s own rendering exactly."""
    import numpy as np

    if not operator_counts:
        ax.set_title(title)
        return
    operator_names = list(
        str(o).removeprefix("destroy_").removeprefix("repair_")
        for o in operator_counts.keys()
    )
    counts = np.array(list(operator_counts.values()))
    cumulative_counts = counts[:, :nr_types].cumsum(axis=1)

    ax.set_xlim(right=cumulative_counts[:, -1].max())
    for idx in range(nr_types):
        widths = counts[:, idx]
        starts = cumulative_counts[:, idx] - widths
        ax.barh(operator_names, widths, left=starts, height=0.5)
        for y, (x, label) in enumerate(zip(starts + widths / 2, widths)):
            ax.text(x, y, str(label), ha="center", va="center")

    ax.set_title(title)
    ax.set_xlabel("Iterations where operator resulted in this outcome (#)")
    ax.set_ylabel("Operator name")
