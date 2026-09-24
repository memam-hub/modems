"""
insertion_ablation: a staged ablation comparing alns_feasible_insertions ([V3], the
complete Algorithm 3 production implementation) against three progressively-improving
alternative implementations of the same feasible-insertion-candidate search, so its
runtime advantage can be attributed to specific elements rather than reported as one
generic "ours is faster" number.

  V0 (naive):           try every precedence-feasible (pickup, delivery) position pair
                        (only that a<=b); propagate the modified journey [a:m_k] after
                        each insertion, check feasibility, and return only the feasible
                        candidate. No optimization of any kind across (similar) pairs.
  V1 (pre-filtered):    includes V0, plus a pair is skipped without calling insert_at()
                        if a cheap capacity check against the original route load
                        profile already proves it infeasible.
  V2 (+ incremental     includes V1, plus the propagated prefix for a fixed pickup
      prefix reuse):    position is built once and incrementally extended as the
                        delivery position advances (mimics _PickupInsertionScan),
                        instead of being re-propagated from scratch per pair. Differs
                        from V3 where it has NO monotone early exit and NO lazy suffix
                        propagation: every surviving pair still gets a full,
                        unconditional delivery-to-end propagation.
  V3 (production):      Our fully optimized approach alns_feasible_insertions, called
                        with the stats hook enabled to collect data without affecting
                        production. Includes prefix/delivery/suffix node propagations,
                        and monotone pruning.

None of V0/V1/V2 should be used for actual (production) work; they exist solely to be
timed against V3 on identical (solution, request) inputs, thus are intentionally NOT
exported from modems.__init__
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import time
from enum import StrEnum
from typing import Any, Callable

from .algorithms import (
    InsertionCandidate,
    _sort_key,
    alns_feasible_insertions,
    greedy_complete,
    preprocess,
)
from .benchmark import (
    MAX_ROWS_PER_BLOCK,
    ManifestPhase,
    ManifestStatus,
    ProgressCallback,
    _latex_begin_doc_block,
    _latex_end_doc_block,
    _latex_fmt,
    stable_seed,
)
from .core import (
    DEFAULT_BASE_SEED,
    DEFAULT_PARAMS_OBJ,
    ProblemContext,
    ProblemType,
    ScenarioTiming,
    ScenarioType,
    SolverStrategy,
)
from .generator import ModemsScenarioGenerator, SocRangeSpec
from .solution import ModemsJourney, ModemsSolution


class InsertionVariant(StrEnum):
    """Utility class for the insertion variant names"""

    v0 = "V0_naive"
    v1 = "V1_prefiltered"
    v2 = "V2_incremental"
    v3 = "V3_production"


class InsertionAblationMetric(StrEnum):
    """Utility class for the insertion ablation metric names"""

    t_exe_V0 = "t_exe_V0"
    t_exe_V1 = "t_exe_V1"
    t_exe_V2 = "t_exe_V2"
    t_exe_V3 = "t_exe_V3"
    speedup_V0_V3 = "speedup_V0_V3"
    speedup_V2_V3 = "speedup_V2_V3"


# prefix for the insertion ablation benchmarking result
BNCH_ABLATION_PFX = "IA_"


# --------------------------------------------------------------------------------------
# V0: naive, every precedence-feasible pair, insert_at() from scratch each time
# --------------------------------------------------------------------------------------


def variant_v0_naive(
    solution: ModemsSolution,
    request_name: str,
    agents: list[str] | None = None,
    stats: dict[str, int] | None = None,
) -> list[InsertionCandidate]:
    """Every (a, b) precedence pair, each fully re-propagated via insert_at()"""
    ctx = solution.ctx
    request = ctx.requests[request_name]
    agent_names = ctx.agent_names if agents is None else agents
    base_obj = solution.objective()
    candidates: list[InsertionCandidate] = []

    for a_name in agent_names:
        journey = solution.journeys[a_name]
        if request.load > journey.agent.load_max:
            continue
        for a_idx in range(len(journey.route) - 1):
            for b_idx in range(a_idx, len(journey.route) - 1):
                if stats is not None:
                    stats["position_pairs_considered"] = (
                        stats.get("position_pairs_considered", 0) + 1
                    )
                chk_sol = solution.copy()
                try:
                    chk_sol.journeys[a_name].insert_at(request_name, a_idx, b_idx)
                except ValueError:
                    if stats is not None:
                        stats["candidates_rejected"] = (
                            stats.get("candidates_rejected", 0) + 1
                        )
                    continue
                chk_sol.accepted.add(request_name)
                candidates.append(
                    InsertionCandidate(
                        chk_sol.journeys[a_name], chk_sol.objective() - base_obj
                    )
                )
                if stats is not None:
                    stats["candidates_accepted"] = (
                        stats.get("candidates_accepted", 0) + 1
                    )
    return candidates


# --------------------------------------------------------------------------------------
# V1: naive + cheap capacity pre-filter before insert_at()
# --------------------------------------------------------------------------------------


def variant_v1_prefiltered(
    solution: ModemsSolution,
    request_name: str,
    agents: list[str] | None = None,
    stats: dict[str, int] | None = None,
) -> list[InsertionCandidate]:
    """
    Includes V0, plus skip pairs when adding the request load to the original load
    profile would exceed the agent max capacity
    """
    ctx = solution.ctx
    request = ctx.requests[request_name]
    agent_names = ctx.agent_names if agents is None else agents
    base_obj = solution.objective()
    candidates: list[InsertionCandidate] = []

    for a_name in agent_names:
        journey = solution.journeys[a_name]
        if request.load > journey.agent.load_max:
            continue
        z_dep = [state.z_dep for state in journey.states]
        for a_idx in range(len(journey.route) - 1):
            for b_idx in range(a_idx, len(journey.route) - 1):
                segment = z_dep[a_idx + 1 : b_idx + 1]
                if (
                    max(segment) if segment else 0
                ) + request.load > journey.agent.load_max:
                    if stats is not None:
                        stats["position_pairs_pruned"] = (
                            stats.get("position_pairs_pruned", 0) + 1
                        )
                    continue
                if stats is not None:
                    stats["position_pairs_considered"] = (
                        stats.get("position_pairs_considered", 0) + 1
                    )
                chk_sol = solution.copy()
                try:
                    chk_sol.journeys[a_name].insert_at(request_name, a_idx, b_idx)
                except ValueError:
                    if stats is not None:
                        stats["candidates_rejected"] = (
                            stats.get("candidates_rejected", 0) + 1
                        )
                    continue
                chk_sol.accepted.add(request_name)
                candidates.append(
                    InsertionCandidate(
                        chk_sol.journeys[a_name], chk_sol.objective() - base_obj
                    )
                )
                if stats is not None:
                    stats["candidates_accepted"] = (
                        stats.get("candidates_accepted", 0) + 1
                    )
    return candidates


# --------------------------------------------------------------------------------------
# V2: V1 + incremental prefix reuse, still no monotone pruning or lazy suffix
# --------------------------------------------------------------------------------------


def variant_v2_incremental(
    solution: ModemsSolution,
    request_name: str,
    agents: list[str] | None = None,
    stats: dict[str, int] | None = None,
) -> list[InsertionCandidate]:
    """
    Includes V1 capacity pre-filter, plus shares the incrementally propagated prefix
    for a fixed pickup position across advancing delivery positions. No monotone
    early exits and no lazy suffix propagation (full walk of the remaining route)
    """

    def _collect_stats(key: str, nr_key: int = 1) -> None:
        if stats is not None:
            stats[key] = stats.get(key, 0) + nr_key

    ctx = solution.ctx
    request = ctx.requests[request_name]
    pickup_node = ctx.pickup_node[request_name]
    delivery_node = ctx.delivery_node[request_name]
    agent_names = ctx.agent_names if agents is None else agents
    base_obj = solution.objective()
    candidates: list[InsertionCandidate] = []

    for a_name in agent_names:
        journey = solution.journeys[a_name]
        if request.load > journey.agent.load_max:
            continue
        route = journey.route
        states = journey.states

        for a_idx in range(len(route) - 1):
            pickup_state = journey._make_update_node(states[a_idx], pickup_node)
            _collect_stats("prefix_node_propagations")
            if pickup_state.z_dep > journey.agent.load_max:
                continue
            prefix_states = [*states[: a_idx + 1], pickup_state]
            current = pickup_state

            for b_idx in range(a_idx, len(route) - 1):
                if b_idx > a_idx:
                    node_b = route[b_idx]
                    state_b = journey._make_update_node(current, node_b)
                    _collect_stats("prefix_node_propagations")
                    prefix_states.append(state_b)
                    current = state_b

                _collect_stats("position_pairs_considered")
                if current.z_dep > journey.agent.load_max:
                    _collect_stats("candidates_rejected")
                    continue

                delivery_state = journey._make_update_node(current, delivery_node)
                _collect_stats("delivery_node_propagations")
                candidate_states = [*prefix_states, delivery_state]

                if b_idx == len(route) - 2:
                    # re-opt final depot
                    final_depot = ctx.nearest_final_depot(delivery_node)
                    depot_state = journey._make_update_node(delivery_state, final_depot)
                    _collect_stats("suffix_node_propagations")
                    candidate_states.append(depot_state)
                else:
                    prev_state = delivery_state
                    for idx in range(b_idx + 1, len(route)):
                        state = journey._make_update_node(prev_state, route[idx])
                        _collect_stats("suffix_node_propagations")
                        candidate_states.append(state)
                        prev_state = state

                chk_journey = ModemsJourney._from_propagated(
                    ctx,
                    a_name,
                    [state.node for state in candidate_states],
                    candidate_states,
                )
                if not chk_journey._is_alns_feasible():
                    _collect_stats("candidates_rejected")
                    continue

                chk_sol = solution.copy()
                chk_sol.journeys[a_name] = chk_journey
                chk_sol.accepted.add(request_name)
                candidates.append(
                    InsertionCandidate(chk_journey, chk_sol.objective() - base_obj)
                )
                _collect_stats("candidates_accepted")
    return candidates


# --------------------------------------------------------------------------------------
# V3: the production implementation, called with its own stats hook
# --------------------------------------------------------------------------------------


def variant_v3_production(
    solution: ModemsSolution,
    request_name: str,
    agents: list[str] | None = None,
    stats: dict[str, int] | None = None,
) -> list[InsertionCandidate]:
    """alns_feasible_insertions(), with stats from its own instrumentation"""
    return alns_feasible_insertions(solution, request_name, agents, stats=stats)


# --------------------------------------------------------------------------------------
# Measurement primitives
# --------------------------------------------------------------------------------------


INSERTION_VARIANT_FCNS: dict[
    InsertionVariant, Callable[..., list[InsertionCandidate]]
] = {
    InsertionVariant.v0: variant_v0_naive,
    InsertionVariant.v1: variant_v1_prefiltered,
    InsertionVariant.v2: variant_v2_incremental,
    InsertionVariant.v3: variant_v3_production,
}

# V0/V1 are O(pairs) x O(route length) per call, so this value is kept as a (safety)
# reminder when extending this ablation beyond the current <=50-request range
DEFAULT_MAX_REQUESTS_FOR_NAIVE_VARIANTS = 200


def build_measurement_scenario(
    seed: int,
    nr_agents: int,
    nr_requests: int,
    scenario_type: ScenarioType,
    scenario_timing: ScenarioTiming,
    model_params: dict[str, Any],
    nr_probes: int = 3,
    soc_range: SocRangeSpec = SocRangeSpec.normal(),
) -> tuple[ModemsSolution, list[str]] | tuple[None, None]:
    """
    Build a scenario and populate it with preprocess()+greedy_complete(), deliberately
    hold/postpone up to nr_probes requests to later use as insertion probes against a
    realistically full route. Return (None, None) if the base_plan or the fill itself
    is infeasible. Uses generate_random_scenario(), passing desired SoC ranges
    """
    generator = ModemsScenarioGenerator(seed=seed)
    scenario = generator.generate_random_scenario(
        nr_agents=nr_agents,
        nr_requests=nr_requests,
        scenario_type=scenario_type,
        scenario_timing=scenario_timing,
        soc_lb=soc_range.lb,
        soc_ub=soc_range.ub,
    )
    ctx = ProblemContext(
        scenario, ProblemType.closed_selective, SolverStrategy.alns, model_params
    )
    base_plan, r_unassigned = preprocess(ctx)
    if base_plan is None:
        return None, None
    ctx = base_plan.ctx
    # deliberately hold some requests, sorted in the same manner as greedy_complete
    r_ordered = sorted(r_unassigned, key=lambda r: _sort_key(ctx, r))
    nr_probes = min(nr_probes, max(0, len(r_ordered) - 1))
    nr_seeds = len(r_ordered) - nr_probes
    if (nr_seeds <= 0) or not r_ordered[nr_seeds:]:
        return None, None
    solution, _, _ = greedy_complete(base_plan, set(r_ordered[:nr_seeds]))
    return (solution, r_ordered[nr_seeds:])


def measure_variant(
    solution: ModemsSolution,
    request_name: str,
    insert_variant: InsertionVariant,
    nr_repeats: int = 5,
) -> dict[str, Any]:
    """
    Run one variant against one (solution, request) probe nr_repeats times (the search
    itself is deterministic, nr_repeats purely stabilize wall-clock reading agains
    system noise); return its timing plus whatever stats that variant populated
    """
    insert_variant_fcn = INSERTION_VARIANT_FCNS[insert_variant]
    stats: dict[str, int] = {}
    nr_candidates = 0
    t_beg = time.perf_counter()
    for _ in range(nr_repeats):
        stats = {}
        candidates = insert_variant_fcn(solution, request_name, stats=stats)
        nr_candidates = len(candidates)
    t_exe = (time.perf_counter() - t_beg) / nr_repeats
    return {
        "variant_name": insert_variant.value,
        "nr_candidates": nr_candidates,
        "t_exe": t_exe,
        **stats,
    }


# --------------------------------------------------------------------------------------
# Manifest construction (a single JSON object, keyed by point_name, rewritten in full
# on every update), same mechanics as modems.benchmark/workday_benchmark
# --------------------------------------------------------------------------------------


def _manifest_path(outdir: str) -> str:
    """Return the manifest.json path for the given output directory"""
    return os.path.join(outdir, "manifest.json")


def _load_manifest_dict(outdir: str) -> dict[str, dict[str, Any]]:
    """Read manifest.json, or {} if it does not exist yet"""
    path = _manifest_path(outdir)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _update_manifest_row(outdir: str, row: dict[str, Any]) -> None:
    """Read-modify-write manifest.json, setting/replacing this point/row entry"""
    os.makedirs(outdir, exist_ok=True)
    manifest = _load_manifest_dict(outdir)
    manifest[row["point_name"]] = row
    with open(_manifest_path(outdir), "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def read_ablation_manifest(outdir: str) -> dict[str, dict[str, Any]]:
    """Return {point_name: manifest row}, or {} if it does not exist yet"""
    return _load_manifest_dict(outdir)


# --------------------------------------------------------------------------------------
# Phase 1: populate manifest with feasible candidate seed point names, set as pending
# --------------------------------------------------------------------------------------


def generate_ablation_suite(
    outdir: str,
    request_counts: list[int],
    agent_counts: list[int],
    scenario_types: list[ScenarioType],
    scenario_timings: list[ScenarioTiming],
    nr_repeats: int = 1,
    base_seed: int = DEFAULT_BASE_SEED,
    model_params: dict[str, Any] | None = None,
    nr_probes: int = 3,
    soc_range: SocRangeSpec = SocRangeSpec.normal(),
    max_feasibility_attempts: int = 5,
    on_progress: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """
    Screen every (nr_requests, nr_agents, sc_type, sc_timing) combination , repetition)
    combination for a fillable scenario with at least one insertion probe
    (build_measurement_scenario succeeds) and write one "pending" manifest row per
    combination. Safe to re-run: points with existing manifest rows are skipped.
    soc_range is baked into each point_name (e.g., "..._soc40-60") so a normal-SoC and
    a SoC-stress corpus can coexist in the same --outdir/manifest without collisions
    """
    model_params = model_params or DEFAULT_PARAMS_OBJ
    points_dict = read_ablation_manifest(outdir)

    combos = [
        (nr_requests, nr_agents, sc_type, sc_timing, i_rep)
        for nr_requests in request_counts
        for nr_agents in agent_counts
        for sc_type in scenario_types
        for sc_timing in scenario_timings
        for i_rep in range(nr_repeats)
    ]
    nr_total = len(combos)
    summary: list[dict[str, str]] = []
    for idx, (nr_requests, nr_agents, sc_type, sc_timing, i_rep) in enumerate(
        combos, start=1
    ):
        point_name = (
            f"{BNCH_ABLATION_PFX}"
            f"a{nr_agents}r{nr_requests}{sc_type}{sc_timing}{i_rep}"
            f"{soc_range.suffix}"
        )
        if on_progress:
            on_progress(
                {
                    "phase": ManifestPhase.generate,
                    "index": idx,
                    "total": nr_total,
                    "scenario_name": point_name,
                    "status": ManifestStatus.starting,
                }
            )
        # skip existing points
        if point_name in points_dict:
            status = ManifestStatus.existing
            summary.append({"point_name": point_name, "status": status})
            if on_progress:
                on_progress(
                    {
                        "phase": ManifestPhase.generate,
                        "index": idx,
                        "total": nr_total,
                        "scenario_name": point_name,
                        "status": status,
                    }
                )
            continue

        # try to get a valid candidate seed within max_feasibility_attempts
        seed = None
        for attempt in range(max_feasibility_attempts):
            candidate_seed = stable_seed(
                base_seed, nr_requests, nr_agents, sc_type, sc_timing, i_rep, attempt
            )
            solution, _ = build_measurement_scenario(
                candidate_seed,
                nr_agents,
                nr_requests,
                sc_type,
                sc_timing,
                model_params,
                nr_probes,
                soc_range=soc_range,
            )
            if solution is not None:
                seed = candidate_seed
                break

        if seed is None:
            status = ManifestStatus.infeasible
            summary.append({"point_name": point_name, "status": status})
            if on_progress:
                on_progress(
                    {
                        "phase": ManifestPhase.generate,
                        "index": idx,
                        "total": nr_total,
                        "scenario_name": point_name,
                        "status": status,
                    }
                )
            continue

        # update manifest and stats
        _update_manifest_row(
            outdir,
            {
                "point_name": point_name,
                "nr_requests": nr_requests,
                "nr_agents": nr_agents,
                "scenario_type": sc_type,
                "scenario_timing": sc_timing,
                "repetition": i_rep,
                "seed": seed,
                "nr_probes": nr_probes,
                "soc_range": soc_range.to_dict(),
                "status": ManifestStatus.pending,
                "result_file": None,
                "timestamp": None,
            },
        )
        status = ManifestStatus.created
        summary.append({"point_name": point_name, "status": status})
        if on_progress:
            on_progress(
                {
                    "phase": ManifestPhase.generate,
                    "index": idx,
                    "total": nr_total,
                    "scenario_name": point_name,
                    "status": status,
                }
            )
    return summary


# --------------------------------------------------------------------------------------
# Phase 2: solving (measuring every variant against every probe request)
# --------------------------------------------------------------------------------------


def _solve_one_point(
    row: dict[str, Any],
    insert_variants: list[InsertionVariant],
    max_requests_for_naive: int,
    nr_repeats: int,
    model_params: dict[str, Any],
) -> dict[str, Any]:
    """
    Rebuild one measurement point from its manifest row and run every variant
    against every (postponed) probe request, aggregating per-variant statistics
    """
    solution, r_probes = build_measurement_scenario(
        row["seed"],
        row["nr_agents"],
        row["nr_requests"],
        row["scenario_type"],
        row["scenario_timing"],
        model_params,
        row["nr_probes"],
        soc_range=SocRangeSpec.from_dict(
            row.get("soc_range", SocRangeSpec.normal().to_dict())
        ),
    )
    if solution is None:
        raise RuntimeError("manifest row no longer reproduces a fillable scenario")
    if r_probes is None:
        raise RuntimeError("manifest row produces fillable scenario but bad r_probes")

    # try all variant for the given point
    route_lengths = [len(j.route) for j in solution.journeys.values()]
    variant_results: dict[str, Any] = {}
    for insert_variant in insert_variants:
        is_naive = insert_variant in (InsertionVariant.v0, InsertionVariant.v1)
        variant_name = insert_variant.value
        if is_naive and row["nr_requests"] > max_requests_for_naive:
            variant_results[variant_name] = {
                "skipped": True,
                "reason": f"nr_requests > {max_requests_for_naive} (naive variant)",
            }
            continue
        probe_results = [
            measure_variant(
                solution, request_name, insert_variant, nr_repeats=nr_repeats
            )
            for request_name in r_probes
        ]
        numeric_keys = {
            key
            for res in probe_results
            for key, value in res.items()
            if isinstance(value, (int, float)) and key != "nr_candidates"
        }
        variant_results[variant_name] = {
            "skipped": False,
            "nr_probes": len(probe_results),
            "total_t_exe": sum(p["t_exe"] for p in probe_results),
            "mean_t_exe": statistics.mean(p["t_exe"] for p in probe_results),
            "mean_nr_candidates": statistics.mean(
                p["nr_candidates"] for p in probe_results
            ),
            **{
                f"mean_{key}": statistics.mean(p.get(key, 0) for p in probe_results)
                for key in numeric_keys
                if key != "t_exe"
            },
            "probe_results": probe_results,
        }

    return {
        "point_name": row["point_name"],
        "nr_agents": row["nr_agents"],
        "nr_requests": row["nr_requests"],
        "scenario_type": row["scenario_type"],
        "scenario_timing": row["scenario_timing"],
        "repetition": row["repetition"],
        "soc_range": row.get("soc_range", SocRangeSpec.normal().to_dict()),
        "nr_accepted": len(solution.accepted),
        "nr_probes": len(r_probes),
        "nr_measure_repeats": nr_repeats,
        "mean_route_length": statistics.mean(route_lengths),
        "max_route_length": max(route_lengths),
        "variant_results": variant_results,
    }


def solve_ablation_suite(
    outdir: str,
    insert_variants: list[InsertionVariant] | None = None,
    max_requests_for_naive: int = DEFAULT_MAX_REQUESTS_FOR_NAIVE_VARIANTS,
    nr_measure_repeats: int = 3,
    model_params: dict[str, Any] | None = None,
    retry_failed: bool = False,
    on_progress: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """
    Read the manifest, measure every variant (V0-V3) against every probe request for
    each row not already "done" (retrying "failed" rows if retry_failed=True), write
    each result JSON, and update that point entry in manifest to "done"/"failed". V0/V1
    are skipped beyond max_requests_for_naive by default. Safe to interrupt and re-run
    """
    model_params = model_params or DEFAULT_PARAMS_OBJ
    variants: list[InsertionVariant] = insert_variants or [n for n in InsertionVariant]
    results_dir = os.path.join(outdir, "results")
    os.makedirs(results_dir, exist_ok=True)

    manifest = read_ablation_manifest(outdir)
    pending_rows = [
        row
        for row in manifest.values()
        if (status := ManifestStatus(row["status"])) == ManifestStatus.pending
        or (retry_failed and status == ManifestStatus.failed)
    ]

    outcomes = []
    nr_total = len(pending_rows)
    for idx, row in enumerate(pending_rows, start=1):
        point_name = row["point_name"]
        if on_progress:
            on_progress(
                {
                    "phase": ManifestPhase.solve,
                    "index": idx,
                    "total": nr_total,
                    "scenario_name": point_name,
                    "status": ManifestStatus.starting,
                }
            )
        try:
            # try to solve the point, update its outcome
            t_beg = time.perf_counter()
            result = _solve_one_point(
                row, variants, max_requests_for_naive, nr_measure_repeats, model_params
            )
            t_exe = time.perf_counter() - t_beg
            result_file = os.path.join(results_dir, f"{point_name}.json")
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2, default=str)
            status = ManifestStatus.done
            _update_manifest_row(
                outdir,
                {
                    **row,
                    "status": status,
                    "result_file": result_file,
                    "timestamp": time.time(),
                    "t_exe": t_exe,
                },
            )
            outcomes.append({"point_name": point_name, "status": status})
            if on_progress:
                on_progress(
                    {
                        "phase": ManifestPhase.solve,
                        "index": idx,
                        "total": nr_total,
                        "scenario_name": point_name,
                        "status": status,
                        "t_exe": t_exe,
                    }
                )
        except Exception as excp:
            # point failed
            status = ManifestStatus.failed
            _update_manifest_row(
                outdir,
                {
                    **row,
                    "status": status,
                    "timestamp": time.time(),
                    "error": f"{type(excp).__name__}: {excp}",
                },
            )
            outcomes.append(
                {
                    "point_name": point_name,
                    "status": status,
                    "error": f"{type(excp).__name__}: {excp}",
                }
            )
            if on_progress:
                on_progress(
                    {
                        "phase": ManifestPhase.solve,
                        "index": idx,
                        "total": nr_total,
                        "scenario_name": point_name,
                        "status": status,
                    }
                )
    return outcomes


# --------------------------------------------------------------------------------------
# Phase 3: table building (reads manifest + results, does not re-solve)
# --------------------------------------------------------------------------------------


def _speedup(slow: float | None, fast: float | None) -> float | None:
    """Return slow/fast; None if either side is missing or fast is ~0"""
    if slow is None or fast is None or fast <= 0:
        return None
    return slow / fast


def build_ablation_table(
    outdir: str,
    table_name: str = "ablation_table",
    rows_per_block: int = MAX_ROWS_PER_BLOCK,
) -> list[dict[str, Any]]:
    """
    Build the V0-V3 comparison table from the manifest + result files; write a
    {table_name}.csv, {table_name}.json, and {table_name}.tex to outdir. Each row
    per measurement point (nr_requests, nr_agents, spatial_type, timing, repetition)
    includes the mean per-probe wall-clock time for each variant and V3's own internal
    breakdown (position pairs pruned, prefix/suffix node propagations)
    """
    manifest = read_ablation_manifest(outdir)
    rows: list[dict[str, Any]] = []
    for point_name, row in sorted(manifest.items()):
        base = {
            "point_name": point_name,
            "status": row["status"],
        }
        if ManifestStatus(row["status"]) != ManifestStatus.done:
            rows.append(
                {
                    **base,
                    "nr_probes": None,
                    "nr_accepted": None,
                    "mean_route_length": None,
                    **{m.value: None for m in InsertionAblationMetric},
                    "V3_pairs_pruned": None,
                    "V3_prefix_propagations": None,
                    "V3_suffix_propagations": None,
                }
            )
            continue
        with open(row["result_file"]) as f:
            data = json.load(f)
        variant_results = data["variant_results"]

        def variant_t_exe(insert_variant: InsertionVariant) -> float | None:
            val = variant_results.get(insert_variant.value)
            return None if val is None or val.get("skipped") else val["mean_t_exe"]

        def v3_stat(key: str) -> float | None:
            v3_val = variant_results.get(InsertionVariant.v3.value)
            return None if v3_val is None or v3_val.get("skipped") else v3_val.get(key)

        t_exe_variants = (
            variant_t_exe(InsertionVariant.v0),
            variant_t_exe(InsertionVariant.v1),
            variant_t_exe(InsertionVariant.v2),
            variant_t_exe(InsertionVariant.v3),
        )
        metric = InsertionAblationMetric
        rows.append(
            {
                **base,
                "nr_probes": data.get("nr_probes"),
                "nr_accepted": data.get("nr_accepted"),
                "mean_route_length": data.get("mean_route_length"),
                metric.t_exe_V0: t_exe_variants[0],
                metric.t_exe_V1: t_exe_variants[1],
                metric.t_exe_V2: t_exe_variants[2],
                metric.t_exe_V3: t_exe_variants[3],
                metric.speedup_V0_V3: _speedup(t_exe_variants[0], t_exe_variants[3]),
                metric.speedup_V2_V3: _speedup(t_exe_variants[2], t_exe_variants[3]),
                "V3_pairs_pruned": v3_stat("mean_position_pairs_pruned"),
                "V3_prefix_propagations": v3_stat("mean_prefix_node_propagations"),
                "V3_suffix_propagations": v3_stat("mean_suffix_node_propagations"),
            }
        )

    os.makedirs(outdir, exist_ok=True)
    fieldnames = (
        [
            "point_name",
            "status",
            "nr_probes",
            "nr_accepted",
            "mean_route_length",
        ]
        + [m.value for m in InsertionAblationMetric]
        + [
            "V3_pairs_pruned",
            "V3_prefix_propagations",
            "V3_suffix_propagations",
        ]
    )
    # export/write data files
    with open(os.path.join(outdir, f"{table_name}.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    with open(os.path.join(outdir, f"{table_name}.json"), "w") as f:
        json.dump(rows, f, indent=2, default=str)

    _write_ablation_latex_table(
        rows, os.path.join(outdir, f"{table_name}.tex"), rows_per_block
    )
    return rows


def _write_ablation_latex_table(
    rows: list[dict[str, Any]], tex_path: str, rows_per_block: int
) -> None:
    """Write the V0-V3 comparison latex file, splitting into table* blocks"""
    blocks = [
        rows[i : i + rows_per_block] for i in range(0, len(rows), rows_per_block)
    ] or [[]]
    parts = []
    parts.append(_latex_begin_doc_block())
    for b, block in enumerate(blocks, start=1):
        cont = f" (cont. {b}/{len(blocks)})" if len(blocks) > 1 else ""
        parts.append(_latex_table_block(block, cont))
    parts.append(_latex_end_doc_block())
    with open(tex_path, "w") as f:
        f.write("\n\n".join(parts) + "\n")


def _latex_table_block(rows: list[dict[str, Any]], continuation_note: str = "") -> str:
    """Render one table* block (booktabs/siunitx, landscape) for a slice of rows"""
    header = (
        r"\begin{landscape}"
        "\n"
        r"\begin{table*}[t]"
        "\n"
        r"\scriptsize"
        "\n"
        r"\centering"
        "\n"
        r"\begin{tabular}{l c c "
        r"c c c c "
        r"c c"
        r"c c c}"
        "\n"
        r"\hline"
        "\n"
        r"Key & {$|R_a|$} & Route & "
        r"{V0 (s)} & {V1 (s)} & {V2 (s)} & {V3 (s)} & "
        r"{V0/V3} & {V2/V3} &"
        r"Pruned & Prefix & Suffix\\"
        "\n"
        r"\hline"
    )
    body_lines = [
        " & ".join(
            [
                str(r["point_name"]).replace("_", r"\_"),
                _latex_fmt(r["nr_accepted"], "{:.0f}"),
                _latex_fmt(r["mean_route_length"], "{:.0f}"),
                _latex_fmt(r[InsertionAblationMetric.t_exe_V0.value], "{:.3f}"),
                _latex_fmt(r[InsertionAblationMetric.t_exe_V1.value], "{:.3f}"),
                _latex_fmt(r[InsertionAblationMetric.t_exe_V2.value], "{:.3f}"),
                _latex_fmt(r[InsertionAblationMetric.t_exe_V3.value], "{:.3f}"),
                _latex_fmt(r[InsertionAblationMetric.speedup_V0_V3.value], "{:.1f}"),
                _latex_fmt(r[InsertionAblationMetric.speedup_V2_V3.value], "{:.1f}"),
                _latex_fmt(r["V3_pairs_pruned"], "{:.0f}"),
                _latex_fmt(r["V3_prefix_propagations"], "{:.0f}"),
                _latex_fmt(r["V3_suffix_propagations"], "{:.0f}"),
            ]
        )
        + r" \\"
        for r in rows
    ]
    footer = (
        r"\hline"
        "\n"
        r"\end{tabular}"
        "\n"
        r"\caption{Comparing different approaches to identify feasible insertion "
        r"candidates" + continuation_note + r". "
        r"V0=naive (every precedence-feasible pair, full re-propagation), "
        r"V1=+capacity pre-filter, V2=+incremental prefix reuse, "
        r"V3=production (\texttt{alns\_feasible\_insertions}). Route is the "
        r"mean number of route nodes (across agents) after insertion. Runtimes are "
        r"averaged over the number of request probes and measurement repetitions. "
        r"V0/V1 are skipped for requests over the naive-variant size "
        r"threshold. V0/V3 and V2/V3 are speedup ratios. "
        r"V3 metrics: Pruned is the mean number of pruned position pairs, "
        r"Prefix and Suffix denote the mean number of prefix and suffix node "
        r"propagation counts, respectively.}"
        "\n"
        r"\end{table*}"
        "\n"
        r"\end{landscape}"
    )
    return "\n".join([header] + body_lines + [footer])


# --------------------------------------------------------------------------------------
# Cross-corpus comparison: [A] normal-SoC corpus vs. [B] SoC-stress corpus
# --------------------------------------------------------------------------------------


def compare_route_length_buckets(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    bucket_size: int = 10,
    metric: InsertionAblationMetric | str = InsertionAblationMetric.speedup_V2_V3,
) -> list[dict[str, Any]]:
    """
    Bucket two build_ablation_table()-style row lists into mean_route_length-based,
    bucket_size-wide bins, comparing 'metric' mean within each bucket, and using only
    "done" rows with a non-None mean_route_length. Return one row per bucket with data
    on either side (nr_a/nr_b is 0 if A corpus has none), each with both A and B means
    and the B/A ratio (None wherever either side is empty or A has 0 mean).
    We bucket here by mean_route_length not by nr_requests, since both have different
    meanings under a tight SoC budget: here, doing many short trips (many requests) is
    not really comparable to doing fewer longer trips (few requests)
    """
    metric = InsertionAblationMetric(metric)

    def _bucketed(rows: list[dict[str, Any]]) -> dict[int, list[float]]:
        buckets: dict[int, list[float]] = {}
        for row in rows:
            if ManifestStatus(row.get("status")) != ManifestStatus.done:
                continue
            route_length = row.get("mean_route_length")
            value = row.get(metric)
            if route_length is None or value is None:
                continue
            bucket = int(route_length // bucket_size) * bucket_size
            buckets.setdefault(bucket, []).append(float(value))
        return buckets

    a_buckets = _bucketed(rows_a)
    b_buckets = _bucketed(rows_b)

    result = []
    for bucket in sorted(set(a_buckets) | set(b_buckets)):
        a_values = a_buckets.get(bucket, [])
        b_values = b_buckets.get(bucket, [])
        a_mean = statistics.mean(a_values) if a_values else None
        b_mean = statistics.mean(b_values) if b_values else None
        b_to_a = (
            b_mean / a_mean
            if a_mean is not None and b_mean is not None and a_mean != 0
            else None
        )
        result.append(
            {
                "route_bucket_start": bucket,
                "route_bucket_end": bucket + bucket_size,
                "metric": metric,
                "nr_a": len(a_values),
                "mean_a": a_mean,
                "nr_b": len(b_values),
                "mean_b": b_mean,
                "ratio_b_to_a": b_to_a,
            }
        )
    return result


def write_corpus_comparison(
    outdir_a: str,
    outdir_b: str,
    outdir: str,
    bucket_size: int = 10,
    metric: InsertionAblationMetric | str = InsertionAblationMetric.speedup_V2_V3,
    table_name: str = "corpus_comparison",
) -> list[dict[str, Any]]:
    """
    Read both corpora's manifests/results directly (do not re-solve), bucket results
    per achieved route length, and write {table_name}.{csv,json} to outdir. Use the
    row shape from build_ablation_table() to match the per-corpus tables
    """
    rows_a = build_ablation_table(outdir_a)
    rows_b = build_ablation_table(outdir_b)
    comparison = compare_route_length_buckets(rows_a, rows_b, bucket_size, metric)

    # export/write data files
    os.makedirs(outdir, exist_ok=True)
    fieldnames = [
        "route_bucket_start",
        "route_bucket_end",
        "metric",
        "nr_a",
        "mean_a",
        "nr_b",
        "mean_b",
        "ratio_b_to_a",
    ]
    with open(os.path.join(outdir, f"{table_name}.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in comparison:
            writer.writerow(row)
    with open(os.path.join(outdir, f"{table_name}.json"), "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    _write_corpus_comparison_latex(
        comparison, os.path.join(outdir, f"{table_name}.tex"), metric
    )
    return comparison


def _write_corpus_comparison_latex(
    comparison: list[dict[str, Any]],
    tex_path: str,
    metric: InsertionAblationMetric | str,
) -> None:
    """Write the corpus-comparison table as a single table"""
    header = (
        r"\begin{table}[t]"
        "\n"
        r"\centering"
        "\n"
        r"\begin{tabular}{c c c c c c}"
        "\n"
        r"\hline"
        "\n"
        r"Route range & {$|X|$} & {Mean X} & {$|Y|$} & {Mean Y} & {Y/X} \\"
        "\n"
        r"\hline"
    )
    body_lines = [
        " & ".join(
            [
                f"{row['route_bucket_start']}--{row['route_bucket_end']}",
                _latex_fmt(row["nr_a"], "{:.0f}"),
                _latex_fmt(row["mean_a"], "{:.3f}"),
                _latex_fmt(row["nr_b"], "{:.0f}"),
                _latex_fmt(row["mean_b"], "{:.3f}"),
                _latex_fmt(row["ratio_b_to_a"], "{:.2f}"),
            ]
        )
        + r" \\"
        for row in comparison
    ]
    footer = (
        r"\hline"
        "\n"
        r"\end{tabular}"
        "\n"
        r"\caption{Corpus X against Y: Comparison of "
        + str(metric).replace("_", r"\_")
        + r", bucketed by achieved (mean) route length. X/Y $>$ 1 favors corpus Y.}"
        "\n"
        r"\end{table}"
    )
    body = "\n".join([header, *body_lines, footer])
    with open(tex_path, "w") as f:
        f.write(
            _latex_begin_doc_block()
            + "\n\n"
            + body
            + "\n\n"
            + _latex_end_doc_block()
            + "\n"
        )
