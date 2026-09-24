"""
Shared CLI helpers for the benchmarks/ scripts, with a console progress-printing
callback (current item name, solver, nr/total).
"""

from __future__ import annotations

import argparse
import os
from enum import StrEnum
from typing import Any, Callable

from modems.benchmark import MAX_ROWS_PER_BLOCK, ManifestPhase
from modems.core import DEFAULT_BASE_SEED, ScenarioSize, ScenarioTiming, ScenarioType
from modems.generator import ScenarioSocRange, SocRangeSpec
from modems.insertion_ablation import InsertionAblationMetric
from modems.milp import DEFAULT_MILP_SOLVER_DATA, SolverConfigType
from modems.solution import DEFAULT_MILP_TIMELIMIT, DEFAULT_PARAMS_ALNS


class ParserArgumentName(StrEnum):
    """Supported parser argument names, check add_cli_arguments for specific (help)"""

    outdir = "outdir"
    sizes = "sizes"
    types = "types"
    timings = "timings"
    nr_repeats = "nr_repeats"
    phase = "phase"
    seed = "seed"
    smoke = "smoke"
    retry_failed = "retry_failed"
    solver_name = "solver_name"
    solver_config_type = "solver_config_type"
    milp_timelimit = "milp_timelimit"
    alns_max_iter = "alns_max_iter"
    latex_rows_per_block = "rows_per_block"
    ablation_bucket_size = "bucket_size"
    ablation_comparison_metric = "metric"


def add_cli_arguments(
    parser: argparse.ArgumentParser, argument_names: list[ParserArgumentName]
) -> None:
    """Augment the parser with the desired argument_names"""
    if ParserArgumentName.outdir in argument_names:
        parser.add_argument(
            "--outdir",
            default=os.path.join(os.path.dirname(__file__), "output"),
            help="desired output directory",
        )
    if ParserArgumentName.sizes in argument_names:
        types = [t for t in ScenarioSize]
        parser.add_argument(
            "--sizes",
            nargs="+",
            default=types,
            choices=types,
            help="size buckets for number of agents/requests",
        )
    if ParserArgumentName.types in argument_names:
        types = [t.value for t in ScenarioType]
        parser.add_argument(
            "--types",
            nargs="+",
            default=types,
            choices=types,
            help="spatial types, how the requests are distributed over the network",
        )
    if ParserArgumentName.timings in argument_names:
        timings = [t.value for t in ScenarioTiming]
        parser.add_argument(
            "--timings",
            nargs="+",
            default=timings,
            choices=timings,
            help="request timings, the spread between request submission/arrival times",
        )
    if ParserArgumentName.nr_repeats in argument_names:
        parser.add_argument(
            "--nr-repeats",
            type=int,
            default=2,
            help="Number of repetitions for a scenario combination",
        )
    if ParserArgumentName.phase in argument_names:
        parser.add_argument(
            "--phase",
            default=ManifestPhase.all,
            choices=[n for n in ManifestPhase],
            help=(
                "Which phase(s) to run. 'all' (default) runs the complete "
                "pipeline -- generate, solve, build table -- in one command."
            ),
        )
    if ParserArgumentName.seed in argument_names:
        parser.add_argument(
            "--seed", type=int, default=DEFAULT_BASE_SEED, help="base rng seed"
        )
    if ParserArgumentName.smoke in argument_names:
        parser.add_argument(
            "--smoke",
            action="store_true",
            help=(
                "Activate to run a quick smoke-test (suite) to verify that"
                "instrumentation works, should run on normal hardware in <=2 (min)"
            ),
        )
    if ParserArgumentName.retry_failed in argument_names:
        parser.add_argument(
            "--retry-failed",
            action="store_true",
            help="retry solving previously failed points",
        )
    if ParserArgumentName.solver_name in argument_names:
        parser.add_argument(
            "--solver-name",
            default=DEFAULT_MILP_SOLVER_DATA[0],
            help=(
                "The literal string passed to pyomo's SolverFactory for MILPs, "
                "such as: cbc, appsi_highs, gurobi. Must be correctly configured with "
                "solver-config-type. Does not impact ALNS"
            ),
        )
    if ParserArgumentName.solver_config_type in argument_names:
        parser.add_argument(
            "--solver-config-type",
            default=DEFAULT_MILP_SOLVER_DATA[1].value,
            choices=[c.value for c in SolverConfigType],
            help=(
                "The solver option-key-naming convention (check SolverConfigType), "
                "must be correctly configured with solver-name. Does not impact ALNS"
            ),
        )
    if ParserArgumentName.milp_timelimit in argument_names:
        parser.add_argument(
            "--milp-timelimit",
            type=float,
            default=DEFAULT_MILP_TIMELIMIT,
            help="Per-epoch MILP timelimit in seconds",
        )
    if ParserArgumentName.alns_max_iter in argument_names:
        parser.add_argument(
            "--alns-max-iter",
            type=int,
            default=DEFAULT_PARAMS_ALNS["max_iter"],
            help="Per-epoch ALNS iteration budget",
        )
    if ParserArgumentName.latex_rows_per_block in argument_names:
        parser.add_argument("--rows-per-block", type=int, default=MAX_ROWS_PER_BLOCK)
    if ParserArgumentName.ablation_bucket_size in argument_names:
        parser.add_argument(
            "--bucket-size",
            type=int,
            default=10,
            help="Comparison route-length bucket width",
        )
    if ParserArgumentName.ablation_comparison_metric in argument_names:
        parser.add_argument(
            "--metric",
            default=InsertionAblationMetric.speedup_V2_V3.value,
            choices=[m.value for m in InsertionAblationMetric],
            help="Comparison metric averaged per route-length bucket",
        )


def resolve_cli_phase(arg_phase: Any | None = None) -> list[ManifestPhase]:
    """Expand --phase into the concrete set of {'generate','solve','build'} to run"""
    arg_phase = ManifestPhase.all if arg_phase is None else ManifestPhase(arg_phase)
    if arg_phase == ManifestPhase.all:
        return [ManifestPhase.generate, ManifestPhase.solve, ManifestPhase.build]
    return [arg_phase]


def resolve_cli_sizes(arg_sizes: Any | None = None) -> list[ScenarioSize]:
    if arg_sizes is None:
        return [t for t in ScenarioSize]
    return [ScenarioSize(t) for t in arg_sizes]


def resolve_cli_types(arg_types: Any | None = None) -> list[ScenarioType]:
    if arg_types is None:
        return [t for t in ScenarioType]
    return [ScenarioType(t) for t in arg_types]


def resolve_cli_timings(arg_timings: Any | None = None) -> list[ScenarioTiming]:
    if arg_timings is None:
        return [t for t in ScenarioTiming]
    return [ScenarioTiming(t) for t in arg_timings]


def resolve_cli_soc(
    arg_soc: Any | None = None,
    custom_bounds: tuple[float, float] | None = None,
) -> list[SocRangeSpec]:
    """
    arg_soc: the --soc-test choice (ScenarioSocRange member or its string value).
    custom_bounds: the --soc-custom LB UB pair, required (and only meaningful)
    when arg_soc == "custom". Raises ValueError on a missing/invalid combination
    """
    arg_soc = ScenarioSocRange.both if arg_soc is None else ScenarioSocRange(arg_soc)
    if arg_soc == ScenarioSocRange.custom:
        if custom_bounds is None:
            raise ValueError("--soc-test custom requires --soc-custom LB UB")
        return [SocRangeSpec.custom(*custom_bounds)]
    if arg_soc == ScenarioSocRange.both:
        return [SocRangeSpec.normal(), SocRangeSpec.stress()]
    if arg_soc == ScenarioSocRange.normal:
        return [SocRangeSpec.normal()]
    return [SocRangeSpec.stress()]


def make_progress_printer() -> Callable[[dict[str, Any]], None]:
    """
    Return an on_progress callback (see ProgressCallback in modems.benchmark) that
    prints one line per lifecycle event, e.g.:
        [3/12] solve SMC0 milp2 starting
        [3/12] solve SMC0 milp2 done (12.3s)
    """

    def _printer(event: dict[str, Any]) -> None:
        index = event.get("index")
        total = event.get("total")
        counter = (
            f"[{index}/{total}]" if index is not None and total is not None else ""
        )
        phase = event.get("phase", "")
        # static suite events carry scenario_name; workday suite events carry
        # workday_name -- only one of the two is ever present on a given event
        name = event.get("scenario_name") or event.get("workday_name") or ""
        solver = event.get("solver_strategy", "")
        status = event.get("status", "")
        line = " ".join(
            str(part) for part in (counter, phase, name, solver, status) if part
        )
        t_elapsed = event.get("t_exe")
        if t_elapsed is not None:
            line += f" ({t_elapsed:.1f}s)"
        print(line, flush=True)

    return _printer
