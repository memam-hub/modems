"""
Run the Algorithm 3 insertion-candidate ablation (check modems.insertion_ablation):
by default, generates + solves + builds both corpora (normal-SoC and SoC-stress)
and their comparison.

Full default run: ~25 minutes. Use --smoke for a small (~1-2 minute) sanity run
that exercises the exact same pipeline (both corpora, both tables, the comparison)
at a small enough scale to finish quickly, including exporting/writing every file.

--corpus {both,normal,stress} (default both) selects which corpus/corpora to
generate/solve/build; the comparison step only runs when both are present. Everything
else (--phase, --request-counts, --variants, ...) exists to let you develop/iterate on
one piece of the pipeline without paying for the rest, the default (no flags) is the
one that matters for the paper.

Safe to interrupt (Ctrl-C) and re-run: already-generated points and already-measured
rows are skipped. Use --phase to run just one part instead of the full pipeline.

Usage:
    python3 run_ablation_suite.py
    python3 run_ablation_suite.py --smoke
    python3 run_ablation_suite.py --corpus stress --phase solve --retry-failed
"""

import argparse
import os

from _cli_common import (
    ParserArgumentName,
    add_cli_arguments,
    make_progress_printer,
    resolve_cli_phase,
    resolve_cli_soc,
    resolve_cli_timings,
    resolve_cli_types,
)

from modems.benchmark import ManifestPhase
from modems.core import ScenarioTiming, ScenarioType
from modems.generator import ScenarioSocRange
from modems.insertion_ablation import (
    DEFAULT_MAX_REQUESTS_FOR_NAIVE_VARIANTS,
    INSERTION_VARIANT_FCNS,
    ManifestStatus,
    build_ablation_table,
    generate_ablation_suite,
    solve_ablation_suite,
    write_corpus_comparison,
)

DEFAULT_REQUEST_COUNTS = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
DEFAULT_AGENT_COUNTS = [1, 2]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    shared_args = [
        ParserArgumentName.outdir,
        ParserArgumentName.types,
        ParserArgumentName.timings,
        ParserArgumentName.nr_repeats,
        ParserArgumentName.phase,
        ParserArgumentName.seed,
        ParserArgumentName.smoke,
        ParserArgumentName.retry_failed,
        ParserArgumentName.latex_rows_per_block,
        ParserArgumentName.ablation_bucket_size,
        ParserArgumentName.ablation_comparison_metric,
    ]
    add_cli_arguments(parser, shared_args)
    parser.set_defaults(
        outdir=os.path.join(os.path.dirname(__file__), "ablation_suite")
    )

    parser.add_argument(
        "--corpus",
        default=ScenarioSocRange.both,
        choices=[t for t in ScenarioSocRange],
        help=(
            "Which corpus/corpora to generate/solve/build. 'both' (default) "
            "also builds the comparison; 'normal'/'stress'/'custom' alone build "
            "just that one corpus's own table, nothing to compare it against"
        ),
    )
    parser.add_argument(
        "--soc-custom",
        type=float,
        nargs=2,
        metavar=("LB", "UB"),
        default=None,
        help="Custom initial-SoC bounds (0 < LB <= UB <= 1.0), with --corpus custom",
    )
    parser.add_argument(
        "--request-counts", type=int, nargs="+", default=DEFAULT_REQUEST_COUNTS
    )
    parser.add_argument(
        "--agent-counts", type=int, nargs="+", default=DEFAULT_AGENT_COUNTS
    )
    parser.add_argument(
        "--nr-probes",
        type=int,
        default=3,
        help="Number of held requests to act as insertion probes",
    )
    types = list(INSERTION_VARIANT_FCNS.keys())
    parser.add_argument(
        "--variants",
        nargs="+",
        default=types,
        choices=types,
        help="Which insertion variants to measure (default: all four)",
    )
    parser.add_argument(
        "--nr-measure-repeats",
        type=int,
        default=5,
        help="Repeated timed calls per (point, probe, variant), averaged for stability",
    )
    parser.add_argument(
        "--max-requests-for-naive",
        type=int,
        default=DEFAULT_MAX_REQUESTS_FOR_NAIVE_VARIANTS,
        help="Max number before skipping V0/V1 (quadratic-ish, very expensive)",
    )
    parser.add_argument("--table-name", default="ablation_table")
    parser.add_argument(
        "--comparison-table-name",
        default="corpus_comparison",
        help="Basename for the normal-vs-stress comparison table under /comparison/",
    )
    args = parser.parse_args()

    if args.smoke:
        args.request_counts = [10, 20, 30, 50]
        args.agent_counts = [1, 2]
        args.types = [ScenarioType.random]
        args.timings = [ScenarioTiming.uniform]
        args.nr_repeats = 1
        args.nr_measure_repeats = 5
        args.corpus = "both"
    else:
        args.types = resolve_cli_types(args.types)
        args.timings = resolve_cli_timings(args.timings)
    args.phases = resolve_cli_phase(args.phase)
    # both_requested is checked against the raw --corpus choice before it is
    # resolved into the per-corpus SocRangeSpec list below
    both_requested = ScenarioSocRange(args.corpus) == ScenarioSocRange.both
    try:
        args.corpus = resolve_cli_soc(args.corpus, custom_bounds=args.soc_custom)
    except ValueError as excp:
        parser.error(str(excp))

    on_progress = make_progress_printer()

    for soc_spec in args.corpus:
        corpus_outdir = os.path.join(args.outdir, soc_spec.label)
        print(f"\n=== {soc_spec.label} corpus (soc_range={soc_spec.bounds}) ===")

        if ManifestPhase.generate in args.phases:
            summary = generate_ablation_suite(
                outdir=corpus_outdir,
                request_counts=args.request_counts,
                agent_counts=args.agent_counts,
                scenario_types=args.types,
                scenario_timings=args.timings,
                nr_repeats=args.nr_repeats,
                base_seed=args.seed,
                nr_probes=args.nr_probes,
                soc_range=soc_spec,
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
            outcomes = solve_ablation_suite(
                outdir=corpus_outdir,
                insert_variants=args.variants,
                max_requests_for_naive=args.max_requests_for_naive,
                nr_measure_repeats=args.nr_measure_repeats,
                retry_failed=args.retry_failed,
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
                f"\n{nr_done} measured, {nr_failed} failed. "
                "Run again to resume any that were interrupted.\n"
            )

        if ManifestPhase.build in args.phases:
            rows = build_ablation_table(
                corpus_outdir,
                table_name=args.table_name,
                rows_per_block=args.rows_per_block,
            )
            print(
                f"{len(rows)} rows written to "
                f"{corpus_outdir}/{args.table_name}.{{csv,json,tex}}"
            )

    if ManifestPhase.build in args.phases and both_requested:
        comparison_outdir = os.path.join(args.outdir, "comparison")
        comparison_rows = write_corpus_comparison(
            outdir_a=os.path.join(args.outdir, "normal"),
            outdir_b=os.path.join(args.outdir, "stress"),
            outdir=comparison_outdir,
            bucket_size=args.bucket_size,
            metric=args.metric,
            table_name=args.comparison_table_name,
        )
        print(
            f"\n{len(comparison_rows)} buckets written to "
            f"{comparison_outdir}/{args.comparison_table_name}.{{csv,json,tex}}"
        )
