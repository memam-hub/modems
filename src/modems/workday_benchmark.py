from __future__ import annotations

import csv
import json
import math
import os
import time
from enum import StrEnum
from typing import Any

from modems.network import RoadNetwork
from modems.solution import (
    DEFAULT_MILP_TIMELIMIT,
    DEFAULT_PARAMS_ALNS,
    DEFAULT_PARAMS_MILP,
    FLOAT_INF,
)

from .benchmark import (
    BNCH_WORKDAY_PFX,
    MAX_ROWS_PER_BLOCK,
    ManifestPhase,
    ManifestStatus,
    ProgressCallback,
    _latex_begin_doc_block,
    _latex_end_doc_block,
    _latex_fmt,
    _latex_fmt_min,
    _latex_fmt_pct,
    stable_seed,
)
from .core import (
    DEFAULT_BASE_SEED,
    ModemsAgent,
    ModemsRequest,
    ScenarioSize,
    ScenarioTiming,
    ScenarioType,
    SolverStrategy,
    scenario_bucket,
)
from .generator import (
    DEFAULT_BASE_RATE_PER_HOUR,
    DEFAULT_WORKDAY_BUFFER,
    DEFAULT_WORKDAY_LENGTH,
    ModemsScenarioGenerator,
)
from .milp import DEFAULT_MILP_SOLVER_DATA, SolverConfigType
from .rolling_horizon import (
    RequestOutcome,
    RollingHorizonSimulator,
    WorkdayLog,
    compare_solvers_one_workday,
)


class WorkdayStartTime(StrEnum):
    """Whether all fleet agents start at t=0, or staggered throughout the day"""

    normal = "N"
    staggered = "S"


# Staggered agent i (0-indexed) starts at time = i * (workday_length / STAGGER_DIVISOR)
STAGGER_DIVISOR = 5.0

# ScenarioTiming to surge parameters conversion:
# loose: small, spread-out bursts; tight: larger, more concentrated bursts
TIMING_SURGE_PARAMS: dict[ScenarioTiming, dict[str, Any]] = {
    ScenarioTiming.loose: {
        "nr_surges": 2,
        "surge_size_range": (3, 6),
        "surge_spread_min": 10.0,
    },
    ScenarioTiming.tight: {
        "nr_surges": 4,
        "surge_size_range": (8, 14),
        "surge_spread_min": 3.0,
    },
}

# Use a simplified name for readable reporting when working fixed fleet, mark as TODO
FIXED_FLEET_NAMING = True


def _build_fleet(
    generator: ModemsScenarioGenerator,
    nr_agents: int,
    start_time: WorkdayStartTime = WorkdayStartTime.normal,
    workday_length: float = DEFAULT_WORKDAY_LENGTH,
) -> list[ModemsAgent]:
    """
    Generate nr_agents using generate_random_agent, all starting at full SoC (1.0),
    with an additional delay when start_time is WorkdayStart.staggered
    """

    agents: list[ModemsAgent] = []
    for i in range(nr_agents):
        delay = (
            i * workday_length / STAGGER_DIVISOR
            if start_time == WorkdayStartTime.staggered
            else 0.0
        )  # i * (workday_length / STAGGER_DIVISOR)
        agents.append(
            generator.generate_random_agent(
                node_lb=0,
                node_ub=generator.network.nr_nodes,
                time_lb=delay,
                time_ub=delay,
                soc_lb=1.0,
                soc_ub=1.0,
            )
        )
    return agents


def _build_workday(
    seed: int,
    nr_agents: int,
    scenario_type: ScenarioType,
    scenario_timing: ScenarioTiming,
    workday_length: float,
    start_time: WorkdayStartTime = WorkdayStartTime.normal,
    base_rate_per_hour: float = DEFAULT_BASE_RATE_PER_HOUR,
) -> tuple[
    ModemsScenarioGenerator, list[ModemsAgent], list[tuple[float, ModemsRequest]]
]:
    """Deterministically rebuild the fleet and workday requests from a given seed"""
    generator = ModemsScenarioGenerator(seed=seed)
    agents = _build_fleet(generator, nr_agents, start_time, workday_length)
    request_submissions = generator.generate_workday_requests(
        workday_length=workday_length,
        scenario_type=scenario_type,
        base_rate_per_hour=base_rate_per_hour,
        **TIMING_SURGE_PARAMS[ScenarioTiming(scenario_timing)],
    )
    return generator, agents, request_submissions


# --------------------------------------------------------------------------------------
# Manifest construction (a single JSON object, keyed by workday_name, rewritten in full
# on every update), same mechanics as modems.benchmark
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
    manifest[row["workday_name"]] = row
    with open(_manifest_path(outdir), "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def read_workday_manifest(outdir: str) -> dict[str, dict[str, Any]]:
    """Return {workday_name: manifest row}, or {} if it does not exist yet"""
    return _load_manifest_dict(outdir)


# --------------------------------------------------------------------------------------
# Phase 1: populate manifest with feasible (candidate) seed points, where the actual
# (fleet, request_submissions) reproduce deterministically from seed at solve time
# --------------------------------------------------------------------------------------


def generate_workday_suite(
    outdir: str,
    scenario_sizes: list[ScenarioSize] = [n for n in ScenarioSize],
    scenario_types: list[ScenarioType] = [n for n in ScenarioType],
    scenario_timings: list[ScenarioTiming] = [n for n in ScenarioTiming],
    start_times: list[WorkdayStartTime] = [n for n in WorkdayStartTime],
    base_rates: list[float] = [DEFAULT_BASE_RATE_PER_HOUR],
    nr_repeats: int = 1,
    base_seed: int = DEFAULT_BASE_SEED,
    workday_length: float = DEFAULT_WORKDAY_LENGTH,
    model_params: dict[str, Any] | None = None,
    max_feasibility_attempts: int = 5,
    on_progress: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """
    Screen a workday for a feasible initial fleet, i.e., all fleet agents can reach
    a final hub, same check as RollingHorizonSimulator.__init__; write one "pending"
    manifest row per workday, skipping any existing manifest workday rows. When given,
    on_progress is called once before and once after each combo/point is screened
    """
    model_params = {**DEFAULT_PARAMS_MILP, **(model_params or {})}
    manifest_data = read_workday_manifest(outdir)

    summary = []
    combos = [
        (sc_size, sc_type, sc_timing, start_time, base_rate, i_rep)
        for sc_size in scenario_sizes
        for sc_type in scenario_types
        for sc_timing in scenario_timings
        for start_time in start_times
        for base_rate in base_rates
        for i_rep in range(nr_repeats)
    ]
    nr_total = len(combos)
    for index, (
        sc_size,
        sc_type,
        sc_timing,
        start_time,
        base_rate,
        i_rep,
    ) in enumerate(combos, start=1):
        nr_agents, (_, _) = scenario_bucket(sc_size)
        if FIXED_FLEET_NAMING:
            workday_name = (
                f"{BNCH_WORKDAY_PFX}br{int(base_rate)}{sc_timing}{start_time}{i_rep}"
            )
        else:
            workday_name = (
                f"{BNCH_WORKDAY_PFX}{sc_size}{sc_type}{sc_timing}{start_time}"
                f"br{int(base_rate)}{i_rep}"
            )
        if on_progress:
            on_progress(
                {
                    "phase": ManifestPhase.generate,
                    "index": index,
                    "total": nr_total,
                    "workday_name": workday_name,
                    "status": ManifestStatus.starting,
                }
            )
        if workday_name in manifest_data:
            status = ManifestStatus.existing
            summary.append({"workday_name": workday_name, "status": status})
            if on_progress:
                on_progress(
                    {
                        "phase": ManifestPhase.generate,
                        "index": index,
                        "total": nr_total,
                        "workday_name": workday_name,
                        "status": status,
                    }
                )
            continue

        seed = None
        for attempt in range(max_feasibility_attempts):
            candidate_seed = stable_seed(
                base_seed,
                sc_size,
                sc_type,
                sc_timing,
                start_time,
                base_rate,
                i_rep,
                attempt,
            )
            generator = ModemsScenarioGenerator(seed=candidate_seed)
            agents = _build_fleet(generator, nr_agents, start_time, workday_length)
            try:
                RollingHorizonSimulator(
                    generator.network, agents, model_params=model_params
                )
            except ValueError:
                continue
            # feasible seed found, continue
            seed = candidate_seed
            break

        if seed is None:
            status = ManifestStatus.infeasible
            summary.append({"workday_name": workday_name, "status": status})
            if on_progress:
                on_progress(
                    {
                        "phase": ManifestPhase.generate,
                        "index": index,
                        "total": nr_total,
                        "workday_name": workday_name,
                        "status": status,
                    }
                )
            continue

        _update_manifest_row(
            outdir,
            {
                "workday_name": workday_name,
                "size": sc_size,
                "type": sc_type,
                "timing": sc_timing,
                "start_time": start_time,
                "base_rate": base_rate,
                "repetition": i_rep,
                "nr_agents": nr_agents,
                "seed": seed,
                "workday_length": workday_length,
                "status": ManifestStatus.pending,
                "result_file": None,
                "timestamp": None,
            },
        )
        status = ManifestStatus.created
        summary.append({"workday_name": workday_name, "status": status})
        if on_progress:
            on_progress(
                {
                    "phase": ManifestPhase.generate,
                    "index": index,
                    "total": nr_total,
                    "workday_name": workday_name,
                    "status": status,
                }
            )
    return summary


# --------------------------------------------------------------------------------------
# Phase 2: reproduce from seed + solve MILP3 vs ALNS (via compare_solvers_one_workday)
# --------------------------------------------------------------------------------------


def _solve_one_workday(
    row: dict[str, Any],
    workday_buffer: float,
    model_params: dict[str, Any],
    milp_timelimit: float,
    alns_max_iter: int,
    solver_name: str,
    solver_config_type: SolverConfigType,
    workday_logs_out: dict[SolverStrategy, WorkdayLog] | None = None,
) -> dict[str, Any]:
    """
    Rebuild one workday from its manifest row (seed) and run the MILP3-vs-ALNS
    comparison, returning the results. If given, workday_logs_out is populated
    in place with {solver_strategy: WorkdayLog} -- see
    compare_solvers_one_workday's own workday_logs_out for what it holds.
    """
    workday_length = row["workday_length"]
    generator, agents, request_submissions = _build_workday(
        row["seed"],
        row["nr_agents"],
        row["type"],
        row["timing"],
        workday_length,
        start_time=WorkdayStartTime(row.get("start_time", WorkdayStartTime.normal)),
        base_rate_per_hour=row.get("base_rate", DEFAULT_BASE_RATE_PER_HOUR),
    )
    summaries = compare_solvers_one_workday(
        generator.network,
        agents,
        request_submissions,
        workday_length=workday_length + workday_buffer,
        model_params=model_params,
        milp_timelimit=milp_timelimit,
        alns_max_iter=alns_max_iter,
        solver_name=solver_name,
        solver_config_type=solver_config_type,
        workday_logs_out=workday_logs_out,
    )
    return {
        "workday_name": row["workday_name"],
        "size": row["size"],
        "type": row["type"],
        "timing": row["timing"],
        "repetition": row["repetition"],
        "nr_agents": row["nr_agents"],
        "seed": row["seed"],
        "workday_length": workday_length,
        "workday_buffer": workday_buffer,
        "nr_submissions": len(request_submissions),
        "milp3": summaries["milp3"],
        "alns": summaries["alns"],
    }


def solve_workday_suite(
    outdir: str,
    model_params: dict[str, Any] | None = None,
    milp_timelimit: float = DEFAULT_MILP_TIMELIMIT,
    alns_max_iter: int = DEFAULT_PARAMS_ALNS["max_iter"],
    workday_buffer: float = DEFAULT_WORKDAY_BUFFER,
    retry_failed: bool = False,
    solver_name: str = DEFAULT_MILP_SOLVER_DATA[0],
    solver_config_type: SolverConfigType = DEFAULT_MILP_SOLVER_DATA[1],
    on_progress: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """
    Read the manifest, run compare_solvers_one_workday() on every row not already
    "done", retrying "failed" rows if retry_failed=True, then write each paired-summary
    result JSON, and update that workday manifest entry to "done"/"failed". A full
    workday simulation drives many MILP/ALNS solves (one pair per replanning epoch), so
    this is considerably slower per row than a single static suite solve. The budgets
    milp_timelimit/alns_max_iter are per-epoch, exactly as RollingHorizonSimulator,
    not a whole-workday budget. If given, on_progress is called once before and after
    each workday is solved, check ProgressCallback. Safe to interrupt and re-run
    """
    results_dir = os.path.join(outdir, "results")
    os.makedirs(results_dir, exist_ok=True)
    model_params = {**DEFAULT_PARAMS_MILP, **(model_params or {})}

    manifest = read_workday_manifest(outdir)
    pending_rows = [
        row
        for row in manifest.values()
        if (status := ManifestStatus(row["status"])) == ManifestStatus.pending
        or (retry_failed and status == ManifestStatus.failed)
    ]

    outcomes = []
    nr_total = len(pending_rows)
    for index, row in enumerate(pending_rows, start=1):
        workday_name = row["workday_name"]
        if on_progress:
            on_progress(
                {
                    "phase": ManifestPhase.solve,
                    "index": index,
                    "total": nr_total,
                    "workday_name": workday_name,
                    "status": ManifestStatus.starting,
                }
            )
        try:
            t_beg = time.perf_counter()
            workday_logs: dict[SolverStrategy, WorkdayLog] = {}
            result = _solve_one_workday(
                row,
                workday_buffer=workday_buffer,
                model_params=model_params,
                milp_timelimit=milp_timelimit,
                alns_max_iter=alns_max_iter,
                solver_name=solver_name,
                solver_config_type=solver_config_type,
                workday_logs_out=workday_logs,
            )
            t_exe = time.perf_counter() - t_beg
            result_file = os.path.join(results_dir, f"{workday_name}.json")
            with open(result_file, "w") as f:
                json.dump(result, f, indent=4, default=str)
            workday_log_file = None
            if workday_logs:
                workday_log_file = os.path.join(
                    results_dir, f"{workday_name}_workday_log.json"
                )
                with open(workday_log_file, "w") as f:
                    json.dump(
                        {k: v.to_dict() for k, v in workday_logs.items()},
                        f,
                        default=str,
                    )
            status = ManifestStatus.done
            _update_manifest_row(
                outdir,
                {
                    **row,
                    "status": status,
                    "result_file": result_file,
                    "workday_log_file": workday_log_file,
                    "timestamp": time.time(),
                    "t_exe": t_exe,
                    "accept_rate_milp3": result["milp3"]["acceptance_rate"],
                    "accept_rate_alns": result["alns"]["acceptance_rate"],
                },
            )
            outcomes.append({"workday_name": workday_name, "status": status})
            if on_progress:
                on_progress(
                    {
                        "phase": ManifestPhase.solve,
                        "index": index,
                        "total": nr_total,
                        "workday_name": workday_name,
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
                    "error": f"{type(excp).__name__}: {excp}",
                },
            )
            outcomes.append(
                {
                    "workday_name": workday_name,
                    "status": status,
                    "error": f"{type(excp).__name__}: {excp}",
                }
            )
            if on_progress:
                on_progress(
                    {
                        "phase": ManifestPhase.solve,
                        "index": index,
                        "total": nr_total,
                        "workday_name": workday_name,
                        "status": status,
                    }
                )
    return outcomes


# --------------------------------------------------------------------------------------
# Phase 3: table building (reads manifest + results, does not re-solve)
# --------------------------------------------------------------------------------------


def _delta_pp(alns_value: float | None, milp3_value: float | None) -> float | None:
    """Return (alns - milp3) in percentage points, or None if either is missing"""
    if alns_value is None or milp3_value is None:
        return None
    return 100.0 * (alns_value - milp3_value)


def build_workday_summary_table(
    outdir: str,
    table_name: str = "summary_table",
    rows_per_block: int = MAX_ROWS_PER_BLOCK,
) -> list[dict[str, Any]]:
    """
    Build the paired MILP3-vs-ALNS workday comparison table from the manifest + result
    files; write {table_name}.csv, {table_name}.json, and {table_name}.tex to outdir.
    The tex file is landscape, booktabs/siunitx, split into multiple table* blocks
    beyond past rows_per_block rows) to outdir. Return the list of row dicts, one row
    per workday, with side-by-side MILP3 and ALNS metrics
    """
    manifest = read_workday_manifest(outdir)
    rows: list[dict[str, Any]] = []
    for workday_name, row in sorted(manifest.items()):
        status = ManifestStatus(row["status"])
        if status != ManifestStatus.done:
            rows.append(
                {
                    "workday": workday_name,
                    "size": row.get("size"),
                    "type": row.get("type"),
                    "timing": row.get("timing"),
                    "start_time": row.get("start_time"),
                    "base_rate": row.get("base_rate"),
                    "nr_agents": row.get("nr_agents"),
                    "status": status,
                    "nr_submissions": None,
                    "accept_milp3": None,
                    "accept_alns": None,
                    "accept_delta_pp": None,
                    "tardiness_milp3": None,
                    "tardiness_alns": None,
                    "excess_ride_time_milp3": None,
                    "excess_ride_time_alns": None,
                    "energy_consumed_milp3": None,
                    "energy_consumed_alns": None,
                    "energy_recovered_milp3": None,
                    "energy_recovered_alns": None,
                    "solve_time_milp3": None,
                    "solve_time_alns": None,
                }
            )
            continue
        with open(row["result_file"]) as f:
            data = json.load(f)
        milp3, alns = data["milp3"], data["alns"]
        rows.append(
            {
                "workday": workday_name,
                "size": row["size"],
                "type": row["type"],
                "timing": row["timing"],
                "start_time": row.get("start_time"),
                "base_rate": row.get("base_rate"),
                "nr_agents": row["nr_agents"],
                "status": status,
                "nr_submissions": data["nr_submissions"],
                "accept_milp3": milp3["acceptance_rate"],
                "accept_alns": alns["acceptance_rate"],
                "accept_delta_pp": _delta_pp(
                    alns["acceptance_rate"], milp3["acceptance_rate"]
                ),
                "tardiness_milp3": milp3["mean_delay_time"],
                "tardiness_alns": alns["mean_delay_time"],
                "excess_ride_time_milp3": milp3["mean_excess_ride_time"],
                "excess_ride_time_alns": alns["mean_excess_ride_time"],
                "energy_consumed_milp3": milp3["total_energy_consumed"],
                "energy_consumed_alns": alns["total_energy_consumed"],
                "energy_recovered_milp3": milp3["total_soc_gained_from_charging"],
                "energy_recovered_alns": alns["total_soc_gained_from_charging"],
                "solve_time_milp3": milp3["mean_solve_time_per_epoch"],
                "solve_time_alns": alns["mean_solve_time_per_epoch"],
            }
        )

    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"{table_name}.csv")
    json_path = os.path.join(outdir, f"{table_name}.json")
    tex_path = os.path.join(outdir, f"{table_name}.tex")

    fieldnames = [
        "workday",
        "size",
        "type",
        "timing",
        "start_time",
        "base_rate",
        "nr_agents",
        "status",
        "nr_submissions",
        "accept_milp3",
        "accept_alns",
        "accept_delta_pp",
        "tardiness_milp3",
        "tardiness_alns",
        "excess_ride_time_milp3",
        "excess_ride_time_alns",
        "energy_consumed_milp3",
        "energy_consumed_alns",
        "energy_recovered_milp3",
        "energy_recovered_alns",
        "solve_time_milp3",
        "solve_time_alns",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    with open(json_path, "w") as f:
        json.dump(rows, f, indent=4, default=str)

    _write_latex_file(rows, tex_path, rows_per_block=rows_per_block)
    return rows


def _format_clock(minutes: float | None, display_start: str | None) -> str:
    """
    Format a workday-relative offset (minutes since clock_start=0.0) for
    display: "HH:MM" wall-clock if display_start (an "HH:MM" anchor, e.g.
    "08:00") is given, else the plain float. "--" for None (not accepted /
    no metric). Display-only -- never applied to what gets written to JSON.
    """
    if minutes is None:
        return "--"
    if display_start is None:
        return f"{minutes:.1f}"
    start_h, start_m = (int(x) for x in display_start.split(":"))
    total_minutes = round(start_h * 60 + start_m + minutes) % (24 * 60)
    h, m = divmod(total_minutes, 60)
    return f"{h:02d}:{m:02d}"


def build_workday_requests_table(
    outdir: str,
    workday_name: str,
    table_name: str = "requests_table",
    rows_per_block: int = MAX_ROWS_PER_BLOCK,
    clock_display_start: str | None = None,
) -> list[dict[str, Any]]:
    """
    Build the per-request table for one workday (one row per submitted request,
    MILP3 vs ALNS outcome/timing side by side, both against the identical request
    stream) from the workday_name WorkdayLog. Write {outdir}/requests_tables/
    {workday_name}_{table_name}.{csv,json,tex} and return the row list with plain
    float times, matching the exports. If given, clock_display_start (e.g., "08:00")
    formats every clock-time column (submission/pickup/delivery) as an "HH:MM"
    wall-clock string in the .csv/.tex output specifically, while the .json output
    and this function return value always keep raw workday-relative float offsets;
    a display anchor is a presentation choice, not actual/relevant data
    """
    manifest = read_workday_manifest(outdir)
    row = manifest.get(workday_name)
    if row is None or row.get("workday_log_file") is None:
        raise ValueError(
            f"no solved workday_log for {workday_name!r} in {outdir!r} -- "
            "run the solve phase first"
        )
    with open(row["workday_log_file"]) as f:
        raw = json.load(f)
    logs = {SolverStrategy(k): WorkdayLog.from_dict(v) for k, v in raw.items()}
    milp3_log, alns_log = logs[SolverStrategy.milp3], logs[SolverStrategy.alns]
    alns_by_id = {r.request.request_id: r for r in alns_log.requests}

    rows: list[dict[str, Any]] = []
    for o_milp in milp3_log.requests:
        o_alns = alns_by_id.get(o_milp.request.request_id)
        rows.append(
            {
                "request_id": o_milp.request.request_id,
                "submission_time": o_milp.submission_time,
                "pickup_node": o_milp.request.node_pickup_index,
                "delivery_node": o_milp.request.node_delivery_index,
                "load": o_milp.request.load,
                "outcome_milp3": o_milp.outcome,
                "outcome_alns": o_alns.outcome if o_alns else None,
                "pickup_time_milp3": o_milp.pickup_time,
                "pickup_time_alns": o_alns.pickup_time if o_alns else None,
                "delivery_time_milp3": o_milp.delivery_time,
                "delivery_time_alns": o_alns.delivery_time if o_alns else None,
                "delay_milp3": o_milp.delay,
                "delay_alns": o_alns.delay if o_alns else None,
                "excess_ride_time_milp3": o_milp.excess_ride_time,
                "excess_ride_time_alns": o_alns.excess_ride_time if o_alns else None,
            }
        )

    tables_dir = os.path.join(outdir, "requests_tables")
    os.makedirs(tables_dir, exist_ok=True)
    base_name = f"{workday_name}_{table_name}"
    csv_path = os.path.join(tables_dir, f"{base_name}.csv")
    json_path = os.path.join(tables_dir, f"{base_name}.json")
    tex_path = os.path.join(tables_dir, f"{base_name}.tex")

    fieldnames = [
        "request_id",
        "submission_time",
        "pickup_node",
        "delivery_node",
        "load",
        "outcome_milp3",
        "outcome_alns",
        "pickup_time_milp3",
        "pickup_time_alns",
        "delivery_time_milp3",
        "delivery_time_alns",
        "delay_milp3",
        "delay_alns",
        "excess_ride_time_milp3",
        "excess_ride_time_alns",
    ]
    time_fields = {
        "submission_time",
        "pickup_time_milp3",
        "pickup_time_alns",
        "delivery_time_milp3",
        "delivery_time_alns",
    }
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            display_row = {
                k: (_format_clock(v, clock_display_start) if k in time_fields else v)
                for k, v in r.items()
            }
            writer.writerow(display_row)

    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)

    _write_requests_latex_file(rows, tex_path, rows_per_block, clock_display_start)
    return rows


def _write_requests_latex_file(
    rows: list[dict[str, Any]],
    tex_path: str,
    nr_rows_per_block: int,
    clock_display_start: str | None,
) -> str:
    """Write the per-workday requests table latex file, split into table* blocks"""
    blocks = [
        rows[i : i + nr_rows_per_block] for i in range(0, len(rows), nr_rows_per_block)
    ] or [[]]
    n_blocks = len(blocks)
    parts = [_latex_begin_doc_block()]
    for b, block in enumerate(blocks, start=1):
        cont = f" (cont. {b}/{n_blocks})" if n_blocks > 1 else ""
        parts.append(_requests_latex_table_block(block, cont, clock_display_start))
    parts.append(_latex_end_doc_block())
    with open(tex_path, "w") as f:
        f.write("\n\n".join(parts) + "\n")
    return tex_path


def _requests_latex_table_block(
    rows: list[dict[str, Any]],
    continuation_note: str,
    clock_display_start: str | None,
) -> str:
    """
    Render one table* block (booktabs, landscape, \\scriptsize) for a slice of
    per-request rows of one workday
    """
    header = (
        r"\begin{landscape}"
        "\n"
        r"\begin{table*}[t]"
        "\n"
        r"\scriptsize"
        "\n"
        r"\centering"
        "\n"
        r"\begin{tabular}{l c c c c c c c c c c c c c c c c}"
        "\n"
        r"\hline"
        "\n"
        r"Key & Submitted & $p^r$ & $d^r$ & $q^r$ & "
        r"\multicolumn{2}{c}{Accepted?} & "
        r"\multicolumn{2}{c}{Pickup time} & "
        r"\multicolumn{2}{c}{Delivery time} & "
        r"\multicolumn{2}{c}{Delay (min)} & "
        r"\multicolumn{2}{c}{Excess ride (min)} \\"
        "\n"
        r" & & & & & "
        r"{MILP3} & {ALNS} & "
        r"{MILP3} & {ALNS} & "
        r"{MILP3} & {ALNS} & "
        r"{MILP3} & {ALNS} & "
        r"{MILP3} & {ALNS} \\"
        "\n"
        r"\hline"
    )
    body_lines = []
    for row in rows:

        def fmt_outcome(val: str) -> str:
            if val == RequestOutcome.accepted:
                return r"$\checkmark$"
            return r"$\times$"

        body_lines.append(
            " & ".join(
                [
                    r"RQ\_" + str(row["request_id"])[:4].replace("_", r"\_"),
                    _format_clock(row["submission_time"], clock_display_start),
                    str(row["pickup_node"]),
                    str(row["delivery_node"]),
                    str(row["load"]),
                    fmt_outcome(row["outcome_milp3"]),
                    fmt_outcome(row["outcome_alns"]),
                    _format_clock(row["pickup_time_milp3"], clock_display_start),
                    _format_clock(row["pickup_time_alns"], clock_display_start),
                    _format_clock(row["delivery_time_milp3"], clock_display_start),
                    _format_clock(row["delivery_time_alns"], clock_display_start),
                    _latex_fmt_min(row["delay_milp3"]),
                    _latex_fmt_min(row["delay_alns"]),
                    _latex_fmt_min(row["excess_ride_time_milp3"]),
                    _latex_fmt_min(row["excess_ride_time_alns"]),
                ]
            )
            + r" \\"
        )
    footer = (
        r"\hline"
        "\n"
        r"\end{tabular}"
        "\n"
        r"\caption{Request-level rolling-horizon results for a simulated workday"
        + continuation_note
        + r". Pickup and delivery times represent the realized service-start times. "
        r"Delay denotes pickup tardiness, while excess ride time is measured relative "
        r"to the direct trip duration. Accepted and rejected requests are denoted by "
        r"$\checkmark$ and $\times$, respectively.}"
        "\n"
        r"\end{table*}"
        "\n"
        r"\end{landscape}"
    )
    return "\n".join([header] + body_lines + [footer])


def plot_workday_soc_acceptance(
    workday_log: WorkdayLog,
    outdir: str,
    prefix: str,
    legend: bool = True,
    show: bool = False,
) -> str:
    """
    Dual-axis plot for one (workday, solver) WorkdayLog: x-axis is clock time; left
    y-axis is the cumulative accepted% and rejected% for cumulative submitted requests;
    right y-axis is agent SoC over time (one line per agent, overlaid), constructed
    from their full-day node-by-node walk. Visually emphasizes when rejections are due
    to SoC considerations rather than objective-sensitive decisions. Plot is saved to
    {outdir}/{prefix}_soc_acceptance.png and its path is returned
    """
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)

    # prepare figure
    ylabel = "Cumulative % of submitted requests"
    fig, ax1 = RoadNetwork._setup_fig_axis("Time (min)", ylabel)
    ax1.set_ylim(-2, 102)
    t_final = workday_log.clock_end
    ax1.set_xlim(workday_log.clock_start - 0.2, t_final)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Agent SoC")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_zorder(1)
    ax1.set_zorder(2)

    soc_handles = []
    # compile and plot agent data
    for idx, (_, node_visits) in enumerate(workday_log.agent_node_visits.items()):
        # agent_id is not human-friendly, compile a numerical 1-based tag instead
        agent_name = ModemsAgent.make_agent_name(idx + 1)
        t_soc: list[float] = []
        soc: list[float] = []
        for node_visit in node_visits:
            t_soc.append(node_visit.arrival_time)
            soc.append(node_visit.soc_arrival)
        # agent stopped, extend flat to t_final
        t_soc.append(t_final)
        soc.append(soc[-1])
        (line,) = ax2.plot(
            t_soc,
            soc,
            linestyle="--",
            linewidth=3,
            color=RoadNetwork._agent_color(idx),
            label=f"{agent_name} SoC",
        )
        soc_handles.append(line)

    # compile and plot request data
    submission_records = sorted(workday_log.requests, key=lambda r: r.submission_time)
    t_submissions: list[float] = []
    r_accepted_pct: list[float] = []
    r_rejected_pct: list[float] = []
    nr_accepted = nr_rejected = 0
    for idx, r_record in enumerate(submission_records, start=1):
        if r_record.outcome == RequestOutcome.accepted:
            nr_accepted += 1
        elif r_record.outcome == RequestOutcome.rejected:
            nr_rejected += 1
        t_submissions.append(r_record.submission_time)
        r_accepted_pct.append(100.0 * nr_accepted / idx)
        r_rejected_pct.append(100.0 * nr_rejected / idx)
    # extend flat to t_final
    t_submissions.append(t_final)
    r_accepted_pct.append(r_accepted_pct[-1])
    r_rejected_pct.append(r_rejected_pct[-1])
    (line_accept,) = ax1.plot(
        t_submissions,
        r_accepted_pct,
        linewidth=4,
        color="tab:green",
        label="Accepted %",
    )
    (line_reject,) = ax1.plot(
        t_submissions,
        r_rejected_pct,
        linewidth=4,
        color="maroon",
        label="Rejected %",
    )

    if legend:
        ax1.legend(
            handles=[line_accept, line_reject, *soc_handles],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=3,
        )

    outfile = os.path.join(outdir, f"{prefix}_soc_acceptance.png")
    plt.savefig(outfile, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    plt.close(fig)
    return outfile


def _write_latex_file(
    rows: list[dict[str, Any]], tex_path: str, rows_per_block: int
) -> str:
    """Write the workday-comparison latex file, split into table* blocks"""
    blocks = [
        rows[i : i + rows_per_block] for i in range(0, len(rows), rows_per_block)
    ] or [[]]
    n_blocks = len(blocks)

    parts = []
    parts.append(_latex_begin_doc_block())
    for b, block in enumerate(blocks, start=1):
        cont = f" (cont. {b}/{n_blocks})" if n_blocks > 1 else ""
        parts.append(_latex_table_block(block, cont))
    parts.append(_latex_end_doc_block())
    with open(tex_path, "w") as f:
        f.write("\n\n".join(parts) + "\n")
    return tex_path


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
        r"\begin{tabular}{l c "
        r"c c "
        r"c c "
        r"c c "
        r"c c "
        r"c c}"
        "\n"
        r"\hline"
        "\n"
        r"Key & $|R|$ & "
        r"\multicolumn{2}{c}{{Acceptance\%}} & "
        r"\multicolumn{2}{c}{{Delay (min)}} & "
        r"\multicolumn{2}{c}{{Excess ride (min)}} & "
        r"\multicolumn{2}{c}{{Consumed Energy}} &"
        r"\multicolumn{2}{c}{{Recovered Energy}} \\"
        "\n"
        r" & & "
        r"{MILP3} & {ALNS} & "
        r"{MILP3} & {ALNS} & "
        r"{MILP3} & {ALNS} & "
        r"{MILP3} & {ALNS} & "
        r"{MILP3} & {ALNS} \\"
        "\n"
        r"\hline"
    )
    body_lines = []
    for row in rows:
        body_lines.append(
            " & ".join(
                [
                    str(row["workday"]).replace("_", r"\_"),
                    (
                        str(row["nr_submissions"])
                        if row["nr_submissions"] is not None
                        else "--"
                    ),
                    _latex_fmt_pct(row["accept_milp3"]),
                    _latex_fmt_pct(row["accept_alns"]),
                    _latex_fmt_min(row["tardiness_milp3"], "{:.1f}"),
                    _latex_fmt_min(row["tardiness_alns"], "{:.1f}"),
                    _latex_fmt_min(row["excess_ride_time_milp3"], "{:.1f}"),
                    _latex_fmt_min(row["excess_ride_time_alns"], "{:.1f}"),
                    _latex_fmt(row["energy_consumed_milp3"], "{:.2f}"),
                    _latex_fmt(row["energy_consumed_alns"], "{:.2f}"),
                    _latex_fmt(row["energy_recovered_milp3"], "{:.2f}"),
                    _latex_fmt(row["energy_recovered_alns"], "{:.2f}"),
                ]
            )
            + r" \\"
        )
    footer = (
        r"\hline"
        "\n"
        r"\end{tabular}"
        "\n"
        r"\caption{Aggregated rolling-horizon results" + continuation_note + r". "
        r"Each row contains one simulated workday, which was solved end-to-end "
        r"independently using MILP3 and ALNS. Acceptance is the total percentage "
        r"of completed requests at the end of the workday. Delay and excess ride are "
        r"per-request means, while consumed and recovered energy denote cumulative "
        r"SoC changes over the workday.}"
        "\n"
        r"\end{table*}"
        "\n"
        r"\end{landscape}"
    )
    return "\n".join([header] + body_lines + [footer])
