from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Iterable
from enum import StrEnum
from typing import Any, cast

from .algorithms import greedy_complete, preprocess
from .alns import ModemsAlns
from .core import (
    DEFAULT_BASE_SEED,
    ModemsScenario,
    ObjectiveType,
    ProblemContext,
    ScenarioSize,
    ScenarioTiming,
    ScenarioType,
    SolverStrategy,
    resolve_problem_type,
    scenario_bucket,
)
from .generator import ModemsScenarioGenerator, SocRangeSpec
from .milp import DEFAULT_MILP_SOLVER_DATA, MilpType, ModemsMilp, SolverConfigType
from .solution import (
    DEFAULT_MILP_TIMELIMIT,
    DEFAULT_PARAMS_ALNS,
    DEFAULT_PARAMS_MILP,
    FLOAT_INF,
    ModemsSolution,
    ModemsSolutionInfo,
    SolutionStatus,
)

# Hook fcn invoked with a small event dict when each suite item starts/finishes, so
# that the caller (benchmarks/ CLI script) can print live progress. Common event fields:
# "phase" ("generate"/"solve"), "index" (1-based), "total", "status" ("starting"/a
# terminal status), phase-specific identifiers ("scenario_name", "solver_strategy"),
# and, "t_exe" on completion
ProgressCallback = Callable[[dict[str, Any]], None]


def stable_seed(base_seed: int, *parts: object) -> int:
    """Deterministic seed derivation (stable across processes/reruns)"""
    key = f"{base_seed}:" + ":".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode()).hexdigest()
    return int(digest[:8], 16)


class ExportFormat(StrEnum):
    """File formats written by the table builders"""

    csv = "csv"
    json = "json"
    tex = "tex"


DEFAULT_EXPORT_FORMATS = frozenset({ExportFormat.csv, ExportFormat.json})


def resolve_export_formats(
    formats: str | Iterable[str] | None = None,
) -> frozenset[ExportFormat]:
    """
    Normalize an export-format selection: None -> DEFAULT_EXPORT_FORMATS; otherwise a
    comma-separated string or an iterable of names; "all" selects every format
    """
    if formats is None:
        return DEFAULT_EXPORT_FORMATS
    names = formats.split(",") if isinstance(formats, str) else list(formats)
    names = [str(n).strip().lower() for n in names if str(n).strip()]
    if "all" in names:
        return frozenset(ExportFormat)
    return frozenset(ExportFormat(n) for n in names)


def write_table_files(
    rows: list[dict[str, Any]],
    outdir: str,
    table_name: str,
    fieldnames: list[str],
    formats: str | Iterable[str] | None = None,
    write_tex: Callable[[str], None] | None = None,
    csv_rows: list[dict[str, Any]] | None = None,
) -> list[str]:
    """
    Write rows to {outdir}/{table_name}.{csv,json,tex} for the selected formats (check
    resolve_export_formats). The tex file is produced by write_tex(path), since every
    table has its own LaTeX layout; csv_rows optionally replaces rows in the csv only
    (e.g., display-formatted values). Return the written paths
    """
    selected = resolve_export_formats(formats)
    os.makedirs(outdir, exist_ok=True)
    paths = []
    if ExportFormat.csv in selected:
        path = os.path.join(outdir, f"{table_name}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows if csv_rows is None else csv_rows)
        paths.append(path)
    if ExportFormat.json in selected:
        path = os.path.join(outdir, f"{table_name}.json")
        with open(path, "w") as f:
            json.dump(rows, f, indent=2, default=str)
        paths.append(path)
    if ExportFormat.tex in selected and write_tex is not None:
        path = os.path.join(outdir, f"{table_name}.tex")
        write_tex(path)
        paths.append(path)
    return paths


class ManifestStatus(StrEnum):
    """Utility class for the manifest row status values"""

    starting = "starting"
    created = "created"
    existing = "existing"
    infeasible = "infeasible"
    pending = "pending"
    failed = "failed"
    done = "done"

    @staticmethod
    def is_skipped(status: ManifestStatus) -> bool:
        return status == ManifestStatus.existing

    @staticmethod
    def is_infeasible(status: ManifestStatus) -> bool:
        return status == ManifestStatus.infeasible


class ManifestPhase(StrEnum):
    """Utility class for the manifest generation/solution phases"""

    generate = "generate"
    solve = "solve"
    build = "build"
    all = "all"


# prefix for a (single) static scenario benchmarking result
BNCH_SCENARIO_PFX = "S"

# prefix for a workday benchmarking result
BNCH_WORKDAY_PFX = "W"

# --------------------------------------------------------------------------------------
# ModemsBenchmarkResult
# --------------------------------------------------------------------------------------


class ModemsBenchmarkResult:
    """
    One (scenario, solver) benchmark outcome: the scenario/problem setup (shared),
    an optional constructive-heuristic baseline for that solver strategy, and
    the solver final result. When available, the baseline determines improvement pct
    """

    ctx: ProblemContext
    solver_strategy: SolverStrategy
    baseline_solution: ModemsSolution | None
    baseline_info: ModemsSolutionInfo | None
    final_solution: ModemsSolution
    final_info: ModemsSolutionInfo

    def __init__(
        self,
        ctx: ProblemContext,
        solver_strategy: SolverStrategy | str,
        baseline_solution: ModemsSolution | None,
        baseline_info: ModemsSolutionInfo | None,
        final_solution: ModemsSolution,
        final_info: ModemsSolutionInfo,
    ) -> None:
        self.ctx = ctx
        self.solver_strategy = SolverStrategy(solver_strategy)
        self.baseline_solution = baseline_solution
        self.baseline_info = baseline_info
        self.final_solution = final_solution
        self.final_info = final_info

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_context": self.ctx.to_dict(),
            "solver_strategy": self.solver_strategy,
            "baseline": (
                {
                    "solution": self.baseline_solution.to_dict(),
                    "solution_info": self.baseline_info.to_dict(),
                }
                if self.baseline_solution is not None and self.baseline_info is not None
                else None
            ),
            "final": {
                "solution": self.final_solution.to_dict(),
                "solution_info": self.final_info.to_dict(),
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModemsBenchmarkResult:
        ctx = ProblemContext.from_dict(data["problem_context"])
        solver_strategy = data["solver_strategy"]
        baseline = data.get("baseline")
        baseline_solution = (
            ModemsSolution.from_dict(ctx, baseline["solution"])
            if baseline is not None
            else None
        )
        baseline_info = (
            ModemsSolutionInfo.from_dict(baseline["solution_info"])
            if baseline is not None
            else None
        )
        final_solution = ModemsSolution.from_dict(ctx, data["final"]["solution"])
        final_info = ModemsSolutionInfo.from_dict(data["final"]["solution_info"])
        return cls(
            ctx,
            solver_strategy,
            baseline_solution,
            baseline_info,
            final_solution,
            final_info,
        )

    def to_json(self, file_path: str) -> str:
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4, default=str)
        return str(file_path)

    @classmethod
    def from_json(cls, file_path: str) -> ModemsBenchmarkResult:
        with open(file_path) as f:
            return cls.from_dict(json.load(f))

    def improvement_pct(self, ndigits: int | None = None) -> float | None:
        """100 * (baseline - final) / baseline, negative if final is worse"""
        if self.baseline_info is None:
            return None
        base_obj = self.baseline_info.objective
        final_obj = self.final_info.objective
        if not base_obj or base_obj == FLOAT_INF or not math.isfinite(base_obj):
            return None
        if final_obj is None or final_obj == FLOAT_INF or not math.isfinite(final_obj):
            return None
        value = 100.0 * (base_obj - final_obj) / base_obj
        return value if ndigits is None else round(value, ndigits)

    def gap_pct(self, ndigits: int | None = None) -> float | None:
        """duality gap from upper and lower bounds"""
        info = self.final_info
        if (
            info.lower_bound is None
            or info.upper_bound is None
            or info.upper_bound == 0
            or info.lower_bound == FLOAT_INF
            or info.upper_bound == FLOAT_INF
            or not math.isfinite(info.lower_bound)
            or not math.isfinite(info.upper_bound)
        ):
            return None
        value = 100.0 * (info.upper_bound - info.lower_bound) / info.upper_bound
        return value if ndigits is None else round(value, ndigits)


# --------------------------------------------------------------------------------------
# Manifest (a single JSON object, keyed by "{scenario_name}__{solver_strategy}")
# --------------------------------------------------------------------------------------


def _manifest_path(outdir: str) -> str:
    """
    Path to the static suite manifest.json: a single JSON object, keyed by
    {scenario_name}__{solver_strategy}, rewritten in full on every update
    """
    return os.path.join(outdir, "manifest.json")


def _load_manifest_dict(outdir: str) -> dict[str, dict[str, Any]]:
    """Read manifest.json, or {} if it does not exist yet"""
    path = _manifest_path(outdir)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _update_manifest_row(outdir: str, row: dict[str, Any]) -> None:
    """Read-modify-write manifest.json, setting/replacing this row entry"""
    os.makedirs(outdir, exist_ok=True)
    manifest = _load_manifest_dict(outdir)
    manifest[f"{row['scenario_name']}__{row['solver_strategy']}"] = row
    with open(_manifest_path(outdir), "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def read_manifest(outdir: str) -> dict[tuple[str, str], dict[str, Any]]:
    """Returns dict (scenario_name, solver_strategy) -> manifest row"""
    return {
        (row["scenario_name"], row["solver_strategy"]): row
        for row in _load_manifest_dict(outdir).values()
    }


# --------------------------------------------------------------------------------------
# Phase 1: generation
# --------------------------------------------------------------------------------------


def generate_benchmark_suite(
    outdir: str,
    scenario_sizes: list[ScenarioSize | str] = [n for n in ScenarioSize],
    scenario_types: list[ScenarioType | str] = [n for n in ScenarioType],
    scenario_timings: list[ScenarioTiming | str] = [n for n in ScenarioTiming],
    nr_repeats: int = 1,
    base_seed: int = DEFAULT_BASE_SEED,
    solver_strategies: list[SolverStrategy | str] = [n for n in SolverStrategy],
    model_params: dict[str, Any] | None = None,
    scenario_soc_ranges: list[SocRangeSpec] = [
        SocRangeSpec.normal(),
        SocRangeSpec.stress(),
    ],
    max_feasibility_attempts: int = 5,
    objective: ObjectiveType | str = ObjectiveType.closed,
    on_progress: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """
    Generate scenarios for every combo of (size, type, timing, soc_range) nr_repeats
    times, ensure base-plan feasibility for each (scenario, solver), then write a
    "pending" manifest row entry. Safe to re-run: scenarios with existing manifest
    rows are skipped. on_progress, if given, is called once before and once after
    each combo is screened, check ProgressCallback
    """
    scenario_sizes = [ScenarioSize(n) for n in scenario_sizes]
    scenario_types = [ScenarioType(n) for n in scenario_types]
    scenario_timings = [ScenarioTiming(n) for n in scenario_timings]
    solver_strategies = [SolverStrategy(n) for n in solver_strategies]
    objective = ObjectiveType(objective)
    scenarios_dir = os.path.join(outdir, "scenarios")
    os.makedirs(scenarios_dir, exist_ok=True)
    model_params = {**DEFAULT_PARAMS_MILP, **(model_params or {})}
    manifest_data = read_manifest(outdir)

    summary = []
    combos = [
        (sc_size, sc_type, sc_timing, sc_soc, i_rep)
        for sc_size in scenario_sizes
        for sc_type in scenario_types
        for sc_timing in scenario_timings
        for sc_soc in scenario_soc_ranges
        for i_rep in range(nr_repeats)
    ]
    nr_total = len(combos)
    for index, (sc_size, sc_type, sc_timing, sc_soc, i_rep) in enumerate(
        combos, start=1
    ):
        nr_agents, (min_requests, max_requests) = scenario_bucket(sc_size)
        scenario_name = (
            f"{BNCH_SCENARIO_PFX}{objective.letter}{i_rep}_"
            f"{sc_size.letter}{sc_type.letter}{sc_timing.letter}{sc_soc.suffix}"
        )
        if on_progress:
            on_progress(
                {
                    "phase": ManifestPhase.generate,
                    "index": index,
                    "total": nr_total,
                    "scenario_name": scenario_name,
                    "status": ManifestStatus.starting,
                }
            )
        status = ManifestStatus.existing
        if all((scenario_name, s) in manifest_data for s in solver_strategies):
            summary.append({"scenario_name": scenario_name, "status": status})
            if on_progress:
                on_progress(
                    {
                        "phase": ManifestPhase.generate,
                        "index": index,
                        "total": nr_total,
                        "scenario_name": scenario_name,
                        "status": status,
                    }
                )
            continue

        scenario = None
        for attempt in range(max_feasibility_attempts):
            seed = stable_seed(base_seed, sc_size, sc_type, sc_timing, i_rep, attempt)
            generator = ModemsScenarioGenerator(seed=seed)
            nr_requests = int(generator.rng.integers(min_requests, max_requests + 1))
            candidate = generator.generate_random_scenario(
                nr_agents=nr_agents,
                nr_requests=nr_requests,
                scenario_type=sc_type,
                scenario_timing=sc_timing,
                soc_lb=sc_soc.lb,
                soc_ub=sc_soc.ub,
            )
            # select any solver strategy, base_plan is never completed
            ctx = ProblemContext(
                candidate,
                resolve_problem_type(SolverStrategy.alns, objective),
                SolverStrategy.alns,
                model_params,
            )
            base_plan, _ = preprocess(ctx)
            if base_plan is not None:
                scenario = candidate
                break

        status = ManifestStatus.infeasible
        if scenario is None:
            summary.append(
                {
                    "scenario_name": scenario_name,
                    "status": status,
                }
            )
            if on_progress:
                on_progress(
                    {
                        "phase": "generate",
                        "index": index,
                        "total": nr_total,
                        "scenario_name": scenario_name,
                        "status": status,
                    }
                )
            continue

        full_name = scenario.make_scenario_name(i_rep=i_rep)
        scenario.to_json(os.path.join(scenarios_dir, f"{scenario_name}.json"))
        for solver_strategy in solver_strategies:
            if (scenario_name, solver_strategy) in manifest_data:
                continue
            _update_manifest_row(
                outdir,
                {
                    "scenario_name": scenario_name,
                    "full_name": full_name,
                    "objective_type": objective,
                    "size": sc_size,
                    "type": sc_type,
                    "timing": sc_timing,
                    "repetition": i_rep,
                    "seed": seed,
                    "nr_agents": nr_agents,
                    "nr_requests": nr_requests,
                    "solver_strategy": solver_strategy,
                    "status": ManifestStatus.pending,
                    "result_file": None,
                    "timestamp": None,
                },
            )
        status = ManifestStatus.created
        summary.append({"scenario_name": scenario_name, "status": status})
        if on_progress:
            on_progress(
                {
                    "phase": ManifestPhase.generate,
                    "index": index,
                    "total": nr_total,
                    "scenario_name": scenario_name,
                    "status": status,
                }
            )
    return summary


# --------------------------------------------------------------------------------------
# Phase 2: solving
# --------------------------------------------------------------------------------------


def _compute_baseline(
    ctx: ProblemContext,
) -> tuple[ModemsSolution | None, ModemsSolutionInfo | None]:
    t_beg = time.perf_counter()
    base_plan, r_unassigned = preprocess(ctx)
    if base_plan is None:
        return None, None
    baseline, _, completed = greedy_complete(base_plan, r_unassigned)
    if not completed:
        return None, None
    t_exe = time.perf_counter() - t_beg
    info = ModemsSolutionInfo(
        status=SolutionStatus.feasible,
        solution_time=t_exe,
        objective=baseline.objective(),
        solver_name="constructive",
    )
    return baseline, info


def _solve_one(
    scenario: ModemsScenario,
    solver_strategy: SolverStrategy,
    model_params: dict[str, Any],
    milp_timelimit: float,
    alns_max_iter: int,
    seed: int,
    solver_name: str = DEFAULT_MILP_SOLVER_DATA[0],
    solver_config_type: SolverConfigType = DEFAULT_MILP_SOLVER_DATA[1],
    alns_stats_out: dict[str, Any] | None = None,
    objective: ObjectiveType | str = ObjectiveType.closed,
) -> ModemsBenchmarkResult | None:
    """
    Solve one (scenario, solver) combo. If solver_strategy is alns and alns_stats_out
    is given, it is populated in place with convergence/operator-performance data from
    the alns package: objectives per iteration and destroy/repair operator counts, as
    plain JSON-serializable data rather than the live alns.Result object to enable
    calling plot_metrics() in a later build phase (separate from solve)
    """
    problem_type = resolve_problem_type(solver_strategy, objective)
    ctx = ProblemContext(scenario, problem_type, solver_strategy, model_params)
    baseline, baseline_info = _compute_baseline(ctx)

    if solver_strategy == SolverStrategy.alns:
        # ALNS cannot start without a feasible incumbent
        if baseline is None or baseline_info is None:
            return None
        alns_model = ModemsAlns(scenario, problem_type, model_params)
        alns_model.solve(seed=seed, max_iter=alns_max_iter)
        if alns_stats_out is not None and alns_model.result is not None:
            stats = alns_model.result.statistics
            alns_stats_out["objectives"] = list(stats.objectives)
            alns_stats_out["destroy_operator_counts"] = dict(
                stats.destroy_operator_counts
            )
            alns_stats_out["repair_operator_counts"] = dict(
                stats.repair_operator_counts
            )
        return ModemsBenchmarkResult(
            alns_model.ctx,
            solver_strategy,
            baseline,
            baseline_info,
            cast(ModemsSolution, alns_model.best_solution),
            alns_model.instance.solution_info,
        )
    else:
        milp_type = MilpType(solver_strategy)
        milp_model = ModemsMilp(scenario, milp_type, problem_type, model_params)
        milp_model.solve(
            solver_name=solver_name,
            solver_config_type=solver_config_type,
            solver_options={"timelimit": milp_timelimit},
            warm_start_solution=baseline,
        )
        return ModemsBenchmarkResult(
            milp_model.ctx,
            solver_strategy,
            baseline,
            baseline_info,
            milp_model.instance.solution,
            milp_model.instance.solution_info,
        )


def _scenario_seed(row: dict[str, Any]) -> int:
    """The generation seed stored with a manifest row, defaults to DEFAULT_BASE_SEED"""

    return row["seed"] if row.get("seed") is not None else DEFAULT_BASE_SEED


def solve_benchmark_suite(
    outdir: str,
    model_params: dict[str, Any] | None = None,
    milp_timelimit: float = DEFAULT_MILP_TIMELIMIT,
    alns_max_iter: int = DEFAULT_PARAMS_ALNS["max_iter"],
    retry_failed: bool = False,
    solver_name: str = DEFAULT_MILP_SOLVER_DATA[0],
    solver_config_type: SolverConfigType = DEFAULT_MILP_SOLVER_DATA[1],
    on_progress: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """
    Read the manifest, solve every row not already "done" (retrying "failed" rows
    if retry_failed=True), write each result JSON, and update that manifest entry to
    "done"/"failed". Safe to interrupt and re-run. If given, on_progress is called
    once before and after each (scenario, solver) row is solved, check ProgressCallback
    """
    scenarios_dir = os.path.join(outdir, "scenarios")
    results_dir = os.path.join(outdir, "results")
    os.makedirs(results_dir, exist_ok=True)

    manifest = read_manifest(outdir)
    pending_rows = [
        row
        for row in manifest.values()
        if (status := ManifestStatus(row["status"])) == ManifestStatus.pending
        or (retry_failed and status == ManifestStatus.failed)
    ]

    outcomes = []
    nr_total = len(pending_rows)
    model_params = {**DEFAULT_PARAMS_MILP, **(model_params or {})}
    for index, row in enumerate(pending_rows, start=1):
        scenario_name, solver_strategy = row["scenario_name"], row["solver_strategy"]
        solver_strategy = SolverStrategy(solver_strategy)
        if on_progress:
            on_progress(
                {
                    "phase": ManifestPhase.solve,
                    "index": index,
                    "total": nr_total,
                    "scenario_name": scenario_name,
                    "solver_strategy": solver_strategy,
                    "status": ManifestStatus.starting,
                }
            )
        scenario_path = os.path.join(scenarios_dir, f"{scenario_name}.json")
        try:
            objective = row["objective_type"]
            scenario = ModemsScenario.from_json(scenario_path)
            t_beg = time.perf_counter()
            alns_stats: dict[str, Any] = {}
            result = _solve_one(
                scenario,
                solver_strategy,
                model_params=model_params,
                milp_timelimit=milp_timelimit,
                alns_max_iter=alns_max_iter,
                seed=_scenario_seed(row),
                solver_name=solver_name,
                solver_config_type=solver_config_type,
                alns_stats_out=alns_stats,
                objective=objective,
            )
            t_exe = time.perf_counter() - t_beg
            if result is None:
                status = ManifestStatus.failed
                _update_manifest_row(
                    outdir, {**row, "status": status, "timestamp": time.time()}
                )
                outcomes.append(
                    {
                        "scenario_name": scenario_name,
                        "solver_strategy": solver_strategy,
                        "status": status,
                    }
                )
                if on_progress:
                    on_progress(
                        {
                            "phase": ManifestPhase.solve,
                            "index": index,
                            "total": nr_total,
                            "scenario_name": scenario_name,
                            "solver_strategy": solver_strategy,
                            "status": status,
                            "t_exe": t_exe,
                        }
                    )
                continue
            result_file = os.path.join(
                results_dir, f"{scenario_name}_{solver_strategy}.json"
            )
            result.to_json(result_file)
            if solver_strategy == SolverStrategy.alns and alns_stats:
                alns_stats_file = os.path.join(
                    results_dir, f"{scenario_name}_{solver_strategy}_alns_stats.json"
                )
                with open(alns_stats_file, "w") as f:
                    json.dump(alns_stats, f)
            status = ManifestStatus.done
            _update_manifest_row(
                outdir,
                {
                    **row,
                    "status": status,
                    "result_file": result_file,
                    "timestamp": time.time(),
                    "t_exe": t_exe,
                    "final_status": result.final_info.status,
                    "final_objective": result.final_info.objective,
                    "baseline_objective": (
                        result.baseline_info.objective
                        if result.baseline_info is not None
                        else None
                    ),
                    "baseline_time": (
                        result.baseline_info.solution_time
                        if result.baseline_info is not None
                        else None
                    ),
                },
            )
            outcomes.append(
                {
                    "scenario_name": scenario_name,
                    "solver_strategy": solver_strategy,
                    "status": status,
                }
            )
            if on_progress:
                on_progress(
                    {
                        "phase": ManifestPhase.solve,
                        "index": index,
                        "total": nr_total,
                        "scenario_name": scenario_name,
                        "solver_strategy": solver_strategy,
                        "status": status,
                        "t_exe": t_exe,
                    }
                )
        except Exception as excp:
            status = ManifestStatus.failed
            _update_manifest_row(
                outdir,
                {
                    **row,
                    "status": status,
                    "timestamp": time.time(),
                    "error": str(excp),
                },
            )
            outcomes.append(
                {
                    "scenario_name": scenario_name,
                    "solver_strategy": solver_strategy,
                    "status": status,
                    "error": str(excp),
                }
            )
            if on_progress:
                on_progress(
                    {
                        "phase": ManifestPhase.solve,
                        "index": index,
                        "total": nr_total,
                        "scenario_name": scenario_name,
                        "solver_strategy": solver_strategy,
                        "status": status,
                    }
                )
    return outcomes


# --------------------------------------------------------------------------------------
# Phase 3: table building (parses manifest + results, no re-solving done)
# --------------------------------------------------------------------------------------

STATUS_SYMBOL = {
    SolutionStatus.optimal: r"$\star$",
    SolutionStatus.feasible: r"$\dagger$",
    SolutionStatus.infeasible: r"$\times$",
    SolutionStatus.unknown: "?",
}

MAX_ROWS_PER_BLOCK = 36


def _latex_fmt_is_nan(value: Any) -> str | None:
    if (
        not isinstance(value, (int, float))
        or value == FLOAT_INF
        or not math.isfinite(value)
    ):
        return r"$\mathrm{-}$"
    return None


def _latex_fmt(value: Any, spec: str = "{:.2f}") -> str:
    chk_value = _latex_fmt_is_nan(value)
    if chk_value is not None:
        return chk_value
    text = spec.format(value)
    # values rounding to zero (e.g., -0.04 with 1 decimal) print as 0, not -0
    return spec.format(0.0) if float(text) == 0.0 else text


def _latex_fmt_pct(value: Any, spec: str = "{:.1f}") -> str:
    chk_value = _latex_fmt_is_nan(value)
    return chk_value if chk_value is not None else spec.format(max(0.0, value * 100.0))


def _latex_fmt_min(value: Any, spec: str = "{:.1f}") -> str:
    chk_value = _latex_fmt_is_nan(value)
    return chk_value if chk_value is not None else spec.format(max(0.0, value))


def _latex_fmt_s_to_ms(value: Any, spec: str = "{:.2f}") -> str:
    chk_value = _latex_fmt_is_nan(value)
    return chk_value if chk_value is not None else spec.format(max(0.0, value * 1_000))


def benchmark_table_rows(outdir: str) -> list[dict[str, Any]]:
    """
    One row per (scenario, solver) from the manifest + result files, without writing
    anything; rows that are not done keep their status with empty metrics
    """
    manifest = read_manifest(outdir)
    rows = []
    for (scenario_name, solver_strategy), row in sorted(manifest.items()):
        solver_strategy = SolverStrategy(solver_strategy)
        status = ManifestStatus(row["status"])
        if status != ManifestStatus.done:
            rows.append(
                {
                    "scenario": scenario_name,
                    "size": row["size"],
                    "objective_type": row["objective_type"],
                    "nr_agents": None,
                    "nr_requests": None,
                    "solver": solver_strategy,
                    "status": status,
                    "status_symbol": "$-$",
                    "nr_accepted": None,
                    "baseline_objective": None,
                    "objective": None,
                    "improvement_pct": None,
                    "lower_bound": None,
                    "upper_bound": None,
                    "gap_pct": None,
                    "baseline_time": None,
                    "time": None,
                }
            )
            continue
        result = ModemsBenchmarkResult.from_json(row["result_file"])
        info = result.final_info
        sol_status = SolutionStatus(info.status)
        has_incumbent = sol_status in (
            SolutionStatus.optimal,
            SolutionStatus.feasible,
        )
        rows.append(
            {
                "scenario": scenario_name,
                "size": row["size"],
                "objective_type": row["objective_type"],
                "nr_agents": len(result.final_solution.ctx.agents),
                "nr_requests": len(result.final_solution.ctx.requests),
                "solver": solver_strategy,
                "status": sol_status,
                "status_symbol": STATUS_SYMBOL.get(sol_status, "?"),
                "nr_accepted": (
                    len(result.final_solution.accepted) if has_incumbent else None
                ),
                "baseline_objective": (
                    result.baseline_info.objective
                    if result.baseline_info is not None
                    else None
                ),
                "objective": info.objective if info.objective != FLOAT_INF else None,
                "improvement_pct": result.improvement_pct(2),
                "lower_bound": info.lower_bound,
                "upper_bound": info.upper_bound,
                "gap_pct": result.gap_pct(2),
                "baseline_time": (
                    result.baseline_info.solution_time
                    if result.baseline_info is not None
                    else None
                ),
                "time": info.solution_time,
            }
        )

    return rows


def build_benchmark_table(
    outdir: str,
    table_name: str = "benchmark_table",
    rows_per_block: int = MAX_ROWS_PER_BLOCK,
    formats: str | Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Build the benchmark summary table from the manifest + result files, write
    {table_name}.{csv,json,tex} (the selected formats, check resolve_export_formats)
    to outdir. The tex is a
    landscape, booktabs/siunitx LaTeX fragment, split into multiple table* blocks if
    the entries are more than nr_rows_per_block. Return the list of row dicts
    """
    rows = benchmark_table_rows(outdir)
    fieldnames = [
        "scenario",
        "size",
        "objective_type",
        "nr_agents",
        "nr_requests",
        "solver",
        "status",
        "status_symbol",
        "nr_accepted",
        "baseline_objective",
        "objective",
        "improvement_pct",
        "lower_bound",
        "upper_bound",
        "gap_pct",
        "baseline_time",
        "time",
    ]
    write_table_files(
        rows,
        outdir,
        table_name,
        fieldnames,
        formats,
        write_tex=lambda path: _write_latex_table(
            rows, path, nr_rows_per_block=rows_per_block
        ),
    )
    return rows


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def gap_to_best(
    rows: list[dict[str, Any]], key: str = "objective"
) -> dict[tuple[str, str], float]:
    """
    Per (scenario, solver): 100 * (row[key] - best) / best, where best is the lowest
    objective any solver found on that scenario (raw objectives are not comparable
    across scenarios, gaps to the per-scenario best are). key="baseline_objective"
    measures the constructive baseline against that same best, which the baselines
    themselves never enter
    """
    best: dict[str, float] = {}
    for row in rows:
        if row.get("objective") is not None:
            best[row["scenario"]] = min(
                best.get(row["scenario"], FLOAT_INF), row["objective"]
            )
    return {
        (row["scenario"], row["solver"]): round(
            100.0 * (row[key] - best[row["scenario"]]) / best[row["scenario"]],
            2,
        )
        for row in rows
        if row.get(key) is not None
        and row[key] != FLOAT_INF
        and best.get(row["scenario"], 0.0) > 0
    }


def plot_benchmark_figure(rows: list[dict[str, Any]], path: str) -> str:
    """
    One figure summarizing a (static) benchmark suite, by solver and size bucket:
    gap to the per-scenario best objective, improvement over the constructive
    baseline, solve time (log scale), and, for the MILPs only, the duality gap
    100 x (UB - LB) / UB (gap_pct), all as horizontal boxes. The gap and solve-time
    panels also show the solver's constructive baseline as a hollow box above each
    (colored) solver box, its gap taken to the same best. ALNS has no bounds (it
    never proves optimality), so it is left out of the gap panel; so is any MILP run
    missing a bound (no incumbent, or a solver such as CBC that reports no lower bound
    when stopped by the time limit), which the per-box n= counts reveal
    """
    from matplotlib.patches import Patch

    from .plotting import draw_hboxes, save_figure, setup_fig_grid

    solvers = [s for s in SolverStrategy if any(row["solver"] == s for row in rows)]
    sizes = [s for s in ScenarioSize if any(row.get("size") == s for row in rows)]
    size_color = dict(zip(ScenarioSize, ("tab:blue", "tab:orange", "tab:green")))
    groups = [(s, w) for s in solvers for w in sizes]
    labels = [f"{s.upper()} · {w}" for s, w in groups]
    colors = [size_color[w] for _, w in groups]
    milp_groups = [(s, w) for s, w in groups if s != SolverStrategy.alns]
    gaps = gap_to_best(rows)
    baseline_gaps = gap_to_best(rows, key="baseline_objective")

    def select(value, selected=groups) -> list[list[float]]:
        return [
            [
                v
                for r in rows
                if r["solver"] == s and r.get("size") == w
                for v in [value(r)]
                if v is not None
            ]
            for s, w in selected
        ]

    fig, axes = setup_fig_grid(2, 2, 18.0, max(8.0, 0.5 * len(groups) * 2 + 4))
    # (axis, title, solver value, constructive baseline value or None)
    panels = [
        (
            axes[0, 0],
            "Gap to best found (%)",
            lambda r: gaps.get((r["scenario"], r["solver"])),
            lambda r: baseline_gaps.get((r["scenario"], r["solver"])),
        ),
        (
            axes[0, 1],
            "Improvement over constructive (%)",
            lambda r: r.get("improvement_pct"),
            None,
        ),
        (
            axes[1, 0],
            "Solve time (s)",
            lambda r: r.get("time"),
            lambda r: r.get("baseline_time"),
        ),
    ]
    legend_handles = [
        Patch(facecolor="none", edgecolor="k", linewidth=2, label="Constructive"),
        Patch(facecolor="lightgray", edgecolor="k", label="Solver"),
    ]
    for ax, title, value, baseline_value in panels:
        hollow = None if baseline_value is None else select(baseline_value)
        draw_hboxes(ax, select(value), labels, colors, hollow=hollow)
        ax.set_title(title)
    axes[1, 0].set_xscale("log")

    ax = axes[1, 1]
    draw_hboxes(
        ax,
        select(lambda r: r.get("gap_pct"), milp_groups),
        [f"{s.upper()} · {w}" for s, w in milp_groups],
        [size_color[w] for _, w in milp_groups],
    )
    ax.set_title("MILP duality gap (%)")
    fig.tight_layout()
    # below the grid, so it never covers a box; savefig's tight bbox keeps it
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        fontsize=14,
    )
    return save_figure(fig, path)


def _write_latex_table(
    rows: list[dict[str, Any]], tex_path: str, nr_rows_per_block: int
) -> str:
    blocks = [
        rows[i : i + nr_rows_per_block] for i in range(0, len(rows), nr_rows_per_block)
    ] or [[]]
    nr_blocks = len(blocks)

    parts = []
    parts.append(_latex_begin_doc_block())
    for idx, block in enumerate(blocks, start=1):
        cont = f" (cont. {idx}/{nr_blocks})" if nr_blocks > 1 else ""
        parts.append(_latex_table_block(block, cont))
    parts.append(_latex_end_doc_block())
    with open(tex_path, "w") as f:
        f.write("\n\n".join(parts) + "\n")
    return tex_path


def _latex_begin_doc_block() -> str:
    """Begin latex document + preamble"""
    header = (
        r"\documentclass{article}"
        "\n"
        r"\usepackage[T1]{fontenc}"
        "\n"
        r"\usepackage[utf8]{inputenc}"
        "\n"
        r"\usepackage{lmodern}"
        "\n"
        r"\usepackage{amsmath}"
        "\n"
        r"\usepackage{amssymb}"
        "\n"
        r"\usepackage{pdflscape}"
        "\n"
        r"\usepackage{siunitx}"
        "\n"
        r"\begin{document}"
        "\n"
    )
    return header


def _latex_end_doc_block() -> str:
    """End latex document"""
    return r"\end{document}" "\n"


def _latex_table_block(rows: list[dict[str, Any]], continuation_note: str = "") -> str:
    """Write a latex table block, including the continuation_note if given"""
    header = (
        r"\begin{landscape}"
        "\n"
        r"\begin{table*}[t]"
        "\n"
        r"\centering"
        "\n"
        r"\scriptsize"
        "\n"
        r"\begin{tabular}{l c c "
        r"c c c c c c "
        r"c c c "
        r"c c}"
        "\n"
        r"\hline"
        "\n"
        r"Key & ${|K|}$ & ${|R|}$ & "
        r"Solver & Status& ${|R_a|}$ & \multicolumn{3}{c}{Objective} & "
        r"{LB} & {UB} & {Gap (\%)} & "
        r"\multicolumn{2}{c}{Time} \\"
        "\n"
        r" &  &  & "
        r" & & & Base & Solver & {$\Delta$(\%)} & "
        r" & & & "
        r" Base (ms) & Solver (s) \\"
        "\n"
        r"\hline"
    )
    body_lines = []
    for r in rows:
        body_lines.append(
            " & ".join(
                [
                    r["scenario"].replace("_", r"\_"),
                    _latex_fmt(r["nr_agents"], "{:.0f}"),
                    _latex_fmt(r["nr_requests"], "{:.0f}"),
                    r["solver"],
                    r.get("status_symbol", "?"),
                    _latex_fmt(r["nr_accepted"], "{:.0f}"),
                    _latex_fmt(r["baseline_objective"], "{:.1f}"),
                    _latex_fmt(r["objective"], "{:.1f}"),
                    _latex_fmt(r["improvement_pct"], "{:.1f}"),
                    _latex_fmt(r["lower_bound"], "{:.1f}"),
                    _latex_fmt(r["upper_bound"], "{:.1f}"),
                    _latex_fmt(r["gap_pct"], "{:.1f}"),
                    _latex_fmt_s_to_ms(r["baseline_time"]),
                    _latex_fmt(r["time"]),
                ]
            )
            + r" \\"
        )
    footer = (
        r"\hline"
        "\n"
        r"\end{tabular}"
        "\n"
        r"\caption{Benchmark results" + continuation_note + r". "
        r"Status: $\star$ denotes a proven optimal solution while $\dagger$ is a "
        r"feasible solution without an optimality certificate. "
        r"Constructive-heuristic time is displayed in milliseconds, while solver "
        r"runtime is in seconds. Improvement $\Delta$ is measured relative to the "
        r"base heuristic.}"
        "\n"
        r"\end{table*}"
        "\n"
        r"\end{landscape}"
    )
    return "\n".join([header] + body_lines + [footer])
