"""
Run the complete MODEMS workday (dynamic rolling-horizon) benchmark suite in one
command: by default, (i) generates a fixed 3-agent, mixed-spatial-type fleet across all
8 combinations of start_time (N: all agents start together at t=0; S: staggered,
agent i starts at i * workday/5) x base_rate (5, 8 requests/hour) x timing (uniform,
peaks: the shape of the earliest-pickup times), each with --nr-surges surges on top;
(ii) solve every pending workday: a full simulated
day is solved twice against the identical request submission/arrival stream using
compare_solvers_one_workday (MILP3 vs. ALNS); (iii) build the paired comparison table
(summary_table.{csv,json,tex}). Prints live progress during operation.

Safe to interrupt (Ctrl-C) and re-run: already-generated workdays and already-solved
rows are skipped. Use --phase to run just one part instead of the full pipeline. Call
--smoke for a small scale full pipeline to finish in a couple of minutes.

Usage:
    python3 run_workday_suite.py
    python3 run_workday_suite.py --smoke
    python3 run_workday_suite.py --phase solve --retry-failed
    python3 run_workday_suite.py --objective open
    python3 run_workday_suite.py --solver-name appsi_highs --solver-config-type highs
"""

import argparse
import json
import os

from _cli_common import (
    ParserArgumentName,
    add_cli_arguments,
    make_progress_printer,
    resolve_cli_phase,
    resolve_cli_sizes,
    resolve_cli_timings,
    resolve_cli_types,
)

from modems import (
    SolverConfigType,
    WorkdayLog,
    build_workday_summary_table,
    generate_workday_suite,
    plot_workday_soc_acceptance,
    read_workday_manifest,
    solve_workday_suite,
)
from modems.benchmark import ManifestPhase, ManifestStatus
from modems.core import ObjectiveType, ScenarioSize, ScenarioTiming, ScenarioType
from modems.generator import (
    DEFAULT_BASE_RATE_PER_HOUR,
    DEFAULT_WORKDAY_LENGTH,
)
from modems.workday_benchmark import WorkdayStartTime, build_workday_requests_table

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    shared_args = [
        ParserArgumentName.outdir,
        ParserArgumentName.smoke,
        ParserArgumentName.seed,
        ParserArgumentName.phase,
        ParserArgumentName.sizes,
        ParserArgumentName.types,
        ParserArgumentName.timings,
        ParserArgumentName.retry_failed,
        ParserArgumentName.nr_repeats,
        ParserArgumentName.solver_name,
        ParserArgumentName.solver_config_type,
        ParserArgumentName.milp_timelimit,
        ParserArgumentName.alns_max_iter,
        ParserArgumentName.latex_rows_per_block,
        ParserArgumentName.objective,
    ]
    add_cli_arguments(parser, shared_args)
    parser.set_defaults(outdir=os.path.join(os.path.dirname(__file__), "workday_suite"))
    parser.set_defaults(sizes=[ScenarioSize.large.value])
    parser.set_defaults(types=[ScenarioType.mixed.value])
    parser.set_defaults(nr_repeats=1)

    parser.add_argument(
        "--start-times",
        type=WorkdayStartTime,
        nargs="+",
        default=list(WorkdayStartTime),
        choices=list(WorkdayStartTime),
        metavar="{normal,staggered}",
        help=(
            "normal: all agents start at t=0. staggered: agent i starts at "
            "i*workday/5. Case-insensitive, the letter is accepted as an alias"
        ),
    )
    parser.add_argument(
        "--base-rates",
        type=int,
        nargs="+",
        default=[DEFAULT_BASE_RATE_PER_HOUR, 8],
        help="Baseline request arrivals per hour (before surges), an integer",
    )
    parser.add_argument(
        "--nr-surges",
        type=int,
        default=3,
        help=(
            "Number of surges per workday: 30-minute windows that each add one hour "
            "worth of base-rate requests, never overlapping each other or the peaks, "
            "kept 30 (uniform) or 15 (peaks) minutes apart. Raises if they do not fit, "
            "e.g., at most 8 (uniform) or 5 (peaks) in a 480-minute workday"
        ),
    )
    parser.add_argument(
        "--workday",
        type=float,
        default=DEFAULT_WORKDAY_LENGTH,
        help=(
            "Workday length in minutes over which requests are drawn; the simulation "
            "then drains until every accepted request is delivered"
        ),
    )
    parser.add_argument(
        "--table-name",
        default="summary_table",
        help="Output file basename for the workday summary",
    )
    parser.add_argument(
        "--requests-table-name",
        default="requests_table",
        help="Output file basename for the compiled workday requests data",
    )
    parser.add_argument(
        "--clock-display-start",
        default="08:00",
        help="Workday starting clock time for formatting clock-time columns",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help=(
            "Generate the dual-axis (cumulative accept/reject rate vs. agent SoC) "
            "plot for every solved workday, one per (workday, solver), as part "
            "of the build phase"
        ),
    )
    args = parser.parse_args()

    if args.smoke:
        args.sizes = [ScenarioSize.large]
        args.types = [ScenarioType.mixed]
        args.timings = [ScenarioTiming.uniform]
        args.start_times = [WorkdayStartTime.normal, WorkdayStartTime.staggered]
        args.base_rates = [DEFAULT_BASE_RATE_PER_HOUR]
        args.nr_repeats = 1
        args.nr_surges = 0
        args.workday = 90.0
        args.milp_timelimit = 8.0
        args.alns_max_iter = 100
    else:
        args.sizes = resolve_cli_sizes(args.sizes)
        args.types = resolve_cli_types(args.types)
        args.timings = resolve_cli_timings(args.timings)
        args.start_times = [WorkdayStartTime(s) for s in args.start_times]
    args.phases = resolve_cli_phase(args.phase)
    args.objective = ObjectiveType(args.objective)

    on_progress = make_progress_printer()

    if ManifestPhase.generate in args.phases:
        summary = generate_workday_suite(
            outdir=args.outdir,
            scenario_sizes=args.sizes,
            scenario_types=args.types,
            scenario_timings=args.timings,
            start_times=args.start_times,
            base_rates=args.base_rates,
            nr_surges=args.nr_surges,
            nr_repeats=args.nr_repeats,
            base_seed=args.seed,
            workday_length=args.workday,
            objective=args.objective,
            on_progress=on_progress,
        )
        nr_generated = nr_skipped = nr_failed = 0
        for row in summary:
            status = ManifestStatus(row["status"])
            if status == ManifestStatus.created:
                nr_generated += 1
            elif ManifestStatus.is_skipped(status):
                nr_skipped += 1
            elif ManifestStatus.is_infeasible(status):
                nr_failed += 1
        print(
            f"\n{nr_generated} generated, {nr_skipped} exist (skipped), "
            f"{nr_failed} infeasible (ignored)\n"
        )

    if ManifestPhase.solve in args.phases:
        outcomes = solve_workday_suite(
            outdir=args.outdir,
            milp_timelimit=args.milp_timelimit,
            alns_max_iter=args.alns_max_iter,
            retry_failed=args.retry_failed,
            solver_name=args.solver_name,
            solver_config_type=SolverConfigType(args.solver_config_type),
            on_progress=on_progress,
        )
        nr_done = nr_failed = 0
        for row in outcomes:
            status = ManifestStatus(row["status"])
            if status == ManifestStatus.done:
                nr_done += 1
            elif status == ManifestStatus.failed:
                nr_failed += 1
        print(
            f"\n{nr_done} solved, {nr_failed} failed. "
            "Run again to resume any that were interrupted.\n"
        )

    if ManifestPhase.build in args.phases:
        rows = build_workday_summary_table(
            args.outdir, table_name=args.table_name, rows_per_block=args.rows_per_block
        )
        print(
            f"{len(rows)} rows written to "
            f"{args.outdir}/{args.table_name}.{{csv,json,tex}}"
        )
        manifest = read_workday_manifest(args.outdir)
        for workday_name, row in sorted(manifest.items()):
            status = ManifestStatus(row["status"])
            if status != ManifestStatus.done:
                continue
            rows = build_workday_requests_table(
                args.outdir,
                workday_name,
                table_name=args.requests_table_name,
                rows_per_block=args.rows_per_block,
                clock_display_start=args.clock_display_start,
            )
            print(
                f"{len(rows)} rows written to {args.outdir}/requests_tables/"
                f"{workday_name}_{args.requests_table_name}.{{csv,json,tex}}"
            )

        if args.plots:
            plots_dir = os.path.join(args.outdir, "plots")
            manifest = read_workday_manifest(args.outdir)
            done_rows = [
                row
                for row in manifest.values()
                if ManifestStatus(row["status"]) == ManifestStatus.done
                and row.get("workday_log_file")
            ]
            nr_done_rows = len(done_rows)
            for idx, row in enumerate(done_rows, start=1):
                workday_name = row["workday_name"]
                if on_progress:
                    on_progress(
                        {
                            "phase": ManifestPhase.build,
                            "index": idx,
                            "total": nr_done_rows,
                            "workday_name": workday_name,
                            "status": ManifestStatus.starting,
                        }
                    )
                with open(row["workday_log_file"]) as f:
                    raw = json.load(f)
                for solver_key, log_dict in raw.items():
                    wlog = WorkdayLog.from_dict(log_dict)
                    plot_workday_soc_acceptance(
                        wlog, plots_dir, f"{workday_name}_{solver_key}"
                    )
                if on_progress:
                    on_progress(
                        {
                            "phase": ManifestPhase.build,
                            "index": idx,
                            "total": nr_done_rows,
                            "workday_name": workday_name,
                            "status": ManifestStatus.done,
                        }
                    )
            print(f"plots for {nr_done_rows} workdays written to {plots_dir}/")
