"""
Compare two insertion-ablation corpora after run_ablation_suite: [A] normal-SoC corpus
with default SoC ranges and [B] SoC-stress corpus with lower SoC ranges. Bucket results
using traversed mean_route_length (not nr_requests, it is nonrepresentative once
soc_range(s) differ), then select a metric to report/compare for each bucket plus the
B/A ratio. Compares only, both corpora must already be generated/solved.

Usage:
    python3 compare_ablation_corpora.py --corpus-a results_ablation
                                        --corpus-b results_ablation_soc
    python3 compare_ablation_corpora.py --corpus-a A --corpus-b B
                                        --metric speedup_V0_V3 --bucket-size 5
"""

import argparse
import os

from _cli_common import ParserArgumentName, add_cli_arguments

from modems.insertion_ablation import write_corpus_comparison

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    shared_args = [
        ParserArgumentName.outdir,
        ParserArgumentName.ablation_bucket_size,
        ParserArgumentName.ablation_comparison_metric,
    ]
    add_cli_arguments(parser, shared_args)
    parser.set_defaults(
        outdir=os.path.join(os.path.dirname(__file__), "results_ablation_comparison"),
    )

    parser.add_argument
    parser.add_argument("--corpus-a", required=True, help="Outdir of first corpus A")
    parser.add_argument("--corpus-b", required=True, help="Outdir of second corpus B")
    parser.add_argument(
        "--table-name",
        default="corpus_comparison",
        help="Basename for the normal-vs-stress comparison table",
    )
    args = parser.parse_args()

    rows = write_corpus_comparison(
        outdir_a=args.corpus_a,
        outdir_b=args.corpus_b,
        outdir=args.outdir,
        bucket_size=args.bucket_size,
        metric=args.metric,
        table_name=args.table_name,
    )

    print(f"{'route bucket':>14}  {'A: n, mean':>16}  {'B: n, mean':>16}  {'B/A':>6}")
    for row in rows:
        a_str = f"{row['nr_a']}, {row['mean_a']:.1f}" if row["nr_a"] else "--"
        b_str = f"{row['nr_b']}, {row['mean_b']:.1f}" if row["nr_b"] else "--"
        ratio_str = f"{row['ratio_b_to_a']:.1f}" if row["ratio_b_to_a"] else "--"
        bucket_str = f"{row['route_bucket_start']}-{row['route_bucket_end']}"
        print(f"{bucket_str:>14}  {a_str:>16}  {b_str:>16}  {ratio_str:>6}")

    print(
        f"\n{len(rows)} buckets written to {args.outdir}/{args.table_name}.{{csv,json,tex}}"
    )
