"""
Compare two workday suites after run_workday_suite: one solved with --objective closed
and one with --objective open, otherwise generated with identical arguments and --seed
(so both contain the same workdays and demand). Writes a combined manifest and summary
table (each workday's closed row directly followed by its open row) and three figures:
decision effects with 95% CIs, outcomes against demand load, and the four runs of the
representative workday (highest demand load among workdays with surges).
Compares only, both suites must already be generated/solved.

Usage:
    python3 compare_workday_objectives.py --closed results_workday_closed
                                          --open results_workday_open
    python3 compare_workday_objectives.py --closed C --open O --outdir out
                                          --all-workdays --window 45
"""

import argparse
import os

from _cli_common import ParserArgumentName, add_cli_arguments

from modems.workday_comparison import (
    DEFAULT_ROLLING_WINDOW,
    METRICS,
    compare_workday_objectives,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_cli_arguments(parser, [ParserArgumentName.outdir])
    parser.set_defaults(
        outdir=os.path.join(os.path.dirname(__file__), "results_workday_comparison"),
    )
    parser.add_argument("--closed", required=True, help="Outdir of the closed suite")
    parser.add_argument("--open", required=True, help="Outdir of the open suite")
    parser.add_argument(
        "--table-name",
        default="combined_summary",
        help="Basename for the combined summary table",
    )
    parser.add_argument(
        "--all-workdays",
        action="store_true",
        help="Also plot the four runs of every paired workday into <outdir>/workdays",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=DEFAULT_ROLLING_WINDOW,
        help="Minutes of submissions behind each rolling acceptance point",
    )
    args = parser.parse_args()

    result = compare_workday_objectives(
        closed_dir=args.closed,
        open_dir=args.open,
        outdir=args.outdir,
        table_name=args.table_name,
        all_workdays=args.all_workdays,
        window=args.window,
    )

    print(f"{result['nr_paired']} paired workdays")
    if result["unpaired"]:
        print(f"unpaired (tables only): {', '.join(result['unpaired'])}")
    labels = {m.key: m.label for m in METRICS}
    print(f"\n{'outcome':<24} {'decision':<11} {'group':<10} {'n':>4}  mean [95% CI]")
    for e in result["estimates"]:
        if e["mean"] is None:
            text = "--"
        else:
            text = f"{e['mean']:+.3f} [{e['mean'] - e['ci']:+.3f}, {e['mean'] + e['ci']:+.3f}]"
        print(
            f"{labels[e['metric']]:<24} {e['decision']:<11} {e['group']:<10} "
            f"{e['n']:>4}  {text}"
        )
    print(
        f"\n{len(result['rows'])} rows written to {args.outdir}/{args.table_name}.{{csv,json,tex}}"
    )
    for name, path in result["figures"].items():
        print(f"{name} figure: {path}")
