"""
Run the complete MODEMS static benchmark suite in one command: (i) generate scenarios
(size buckets x spatial types x timings x SoC ranges x nr_repeats), (ii) solve every
pending (scenario, solver) row with ALNS and MILP1/2/3: ALNS is initialized and each
MILP is warm-started with its own strategy-matched constructive baseline, (iii) build
the results tables/plots/files. Prints live progress during operation.

Safe to interrupt (Ctrl-C) and re-run: already-generated scenarios and already-solved
rows are skipped. Use --phase to run just one part instead of the full pipeline.

Usage:
    python3 run_suite.py --smoke
    python3 run_suite.py --phase solve --retry-failed
    python3 run_suite.py --solver-name appsi_highs --solver-config-type highs
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
    resolve_cli_soc,
    resolve_cli_timings,
    resolve_cli_types,
)

from modems import (
    ModemsInstance,
    SolverConfigType,
    build_benchmark_table,
    generate_benchmark_suite,
    plot_alns_metrics_from_stats,
    solve_benchmark_suite,
)
from modems.benchmark import (
    ManifestPhase,
    ManifestStatus,
    ModemsBenchmarkResult,
    SolverStrategy,
    read_manifest,
)
from modems.core import ScenarioSize, ScenarioTiming, ScenarioType
from modems.generator import (
    NORMAL_SOC_RANGE,
    STRESS_SOC_RANGE,
    ScenarioSocRange,
    SocRangeSpec,
)

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
    ]
    add_cli_arguments(parser, shared_args)
    parser.set_defaults(outdir=os.path.join(os.path.dirname(__file__), "results"))

    parser.add_argument(
        "--soc-test",
        default=ScenarioSocRange.both,
        choices=[t for t in ScenarioSocRange],
        help=(
            f"Which range(s) to use for the inital agent SoC. 'normal' is "
            f"{NORMAL_SOC_RANGE}, 'stress' is {STRESS_SOC_RANGE}, 'custom' uses "
            "--soc-custom LB UB, 'both' (default) includes normal and stress "
            "in the generation/solution/build"
        ),
    )
    parser.add_argument(
        "--soc-custom",
        type=float,
        nargs=2,
        metavar=("LB", "UB"),
        default=None,
        help="Custom initial-SoC bounds (0 < LB <= UB <= 1.0), with --soc-test custom",
    )

    parser.add_argument("--table-name", default="benchmark_table")
    parser.add_argument(
        "--plots",
        action="store_true",
        help=(
            "Generate instance.plot() (network/routes/timing/SoC/load) for every "
            "solved result, plus ALNS convergence/operator-count charts for ALNS "
            "results, as part of the build phase"
        ),
    )
    args = parser.parse_args()

    if args.smoke:
        args.sizes = [ScenarioSize.small, ScenarioSize.medium]
        args.types = [ScenarioType.random, ScenarioType.clustered]
        args.timings = [ScenarioTiming.loose]
        args.nr_repeats = 2
        args.milp_timelimit = 8.0
        args.alns_max_iter = 100
        args.soc_test = [SocRangeSpec.normal()]
    else:
        args.sizes = resolve_cli_sizes(args.sizes)
        args.types = resolve_cli_types(args.types)
        args.timings = resolve_cli_timings(args.timings)
        try:
            args.soc_test = resolve_cli_soc(
                args.soc_test, custom_bounds=args.soc_custom
            )
        except ValueError as excp:
            parser.error(str(excp))
    args.phases = resolve_cli_phase(args.phase)

    on_progress = make_progress_printer()

    if ManifestPhase.generate in args.phases:
        summary = generate_benchmark_suite(
            outdir=args.outdir,
            scenario_sizes=args.sizes,
            scenario_types=args.types,
            scenario_timings=args.timings,
            nr_repeats=args.nr_repeats,
            base_seed=args.seed,
            scenario_soc_ranges=args.soc_test,
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
        outcomes = solve_benchmark_suite(
            outdir=args.outdir,
            seed=args.seed,
            alns_max_iter=args.alns_max_iter,
            milp_timelimit=args.milp_timelimit,
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
        rows = build_benchmark_table(
            args.outdir, table_name=args.table_name, rows_per_block=args.rows_per_block
        )
        print(
            f"{len(rows)} rows written to "
            f"{args.outdir}/{args.table_name}.{{csv,json,tex}}"
        )

        if args.plots:
            plots_dir = os.path.join(args.outdir, "plots")
            results_dir = os.path.join(args.outdir, "results")
            manifest = read_manifest(args.outdir)
            done_rows = [
                row
                for row in manifest.values()
                if ManifestStatus(row["status"]) == ManifestStatus.done
            ]
            nr_done_rows = len(done_rows)
            for idx, row in enumerate(done_rows):
                scenario_name = row["scenario_name"]
                if on_progress:
                    on_progress(
                        {
                            "phase": ManifestPhase.build,
                            "index": idx,
                            "total": nr_done_rows,
                            "scenario_name": scenario_name,
                            "status": ManifestStatus.starting,
                        }
                    )
                solver_strategy = SolverStrategy(row["solver_strategy"])
                prefix = f"{scenario_name}_{solver_strategy}"
                result = ModemsBenchmarkResult.from_json(row["result_file"])
                instance = ModemsInstance(
                    result.ctx, result.final_solution, result.final_info
                )
                instance.plot(outdir=plots_dir, name=prefix, show=False)
                if solver_strategy == SolverStrategy.alns:
                    alns_stats_file = os.path.join(
                        results_dir, f"{prefix}_alns_stats.json"
                    )
                    if os.path.exists(alns_stats_file):
                        with open(alns_stats_file) as f:
                            alns_stats = json.load(f)
                        plot_alns_metrics_from_stats(alns_stats, plots_dir, prefix)
                if on_progress:
                    on_progress(
                        {
                            "phase": ManifestPhase.build,
                            "index": idx,
                            "total": nr_done_rows,
                            "scenario_name": scenario_name,
                            "status": ManifestStatus.done,
                        }
                    )
            print(f"plots for {nr_done_rows} results written to {plots_dir}/")
