"""
Tests for modems.benchmark: result bookkeeping, and the resumable generate and/or solve
and/or build pipeline at benchmarks/run_suite.py. Phase 1 generation tests do not solve;
Phase 2 runs real solvers on the smallest (smoke) instances
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

import modems.benchmark as benchmark_module
from modems.benchmark import (
    ExportFormat,
    ManifestPhase,
    ManifestStatus,
    ModemsBenchmarkResult,
    _latex_fmt,
    _solve_one,
    build_benchmark_table,
    gap_to_best,
    generate_benchmark_suite,
    plot_benchmark_figure,
    read_manifest,
    resolve_export_formats,
    solve_benchmark_suite,
    stable_seed,
    write_table_files,
)
from modems.core import (
    DEFAULT_BASE_SEED,
    ModemsAgent,
    ModemsRequest,
    ScenarioSize,
    ScenarioType,
)
from modems.generator import SocRangeSpec
from modems.solution import (
    FLOAT_INF,
    ModemsSolution,
    ModemsSolutionInfo,
    SolutionStatus,
)

from .builders import (
    duration_limited_scenario,
    greedy_solution,
    line_scenario,
    make_ctx,
)

# plain strings, letters, and enum members are all accepted
SMALLEST = dict(
    scenario_sizes=["s"],
    scenario_types=[ScenarioType.random],
    scenario_timings=["Uniform"],
    scenario_soc_ranges=[SocRangeSpec.normal()],
)


def generate(outdir, **kwargs: Any) -> list[dict[str, Any]]:
    return generate_benchmark_suite(str(outdir), **{**SMALLEST, **kwargs})


# --------------------------------------------------------------------------------------
# stable_seed / ModemsBenchmarkResult
# --------------------------------------------------------------------------------------


def test_stable_seed_is_a_deterministic_32_bit_function_of_its_parts() -> None:
    assert stable_seed(42, "a", 1) == stable_seed(42, "a", 1)
    assert (
        len({stable_seed(42, "a", 1), stable_seed(42, "a", 2), stable_seed(43, "a", 1)})
        == 3
    )
    assert 0 <= stable_seed(0) < 2**32


def _result(
    baseline: float | None, final: float, lb=None, ub=None
) -> ModemsBenchmarkResult:
    ctx = make_ctx(line_scenario([ModemsRequest(1, 2)]), "milp3")
    solution = ModemsSolution(ctx)
    base_info = None if baseline is None else ModemsSolutionInfo(objective=baseline)
    return ModemsBenchmarkResult(
        ctx,
        "milp3",
        None if baseline is None else solution,
        base_info,
        solution,
        ModemsSolutionInfo(objective=final, lower_bound=lb, upper_bound=ub),
    )


@pytest.mark.parametrize(
    "baseline, final, expected",
    [
        (100.0, 80.0, 20.0),
        (100.0, 120.0, -20.0),
        (None, 80.0, None),
        (0.0, 80.0, None),
        (FLOAT_INF, 80.0, None),
        (100.0, FLOAT_INF, None),
        (100.0, None, None),
    ],
)
def test_improvement_is_a_signed_percentage_of_the_baseline(
    baseline, final, expected
) -> None:
    assert _result(baseline, final).improvement_pct() == pytest.approx(expected)


@pytest.mark.parametrize(
    "value, spec, text",
    [
        (12.345, "{:.1f}", "12.3"),
        (-12.345, "{:.1f}", "-12.3"),
        (-0.04, "{:.1f}", "0.0"),
        (-0.0, "{:.1f}", "0.0"),
        (0.04, "{:.1f}", "0.0"),
        (-0.4, "{:.0f}", "0"),
        (None, "{:.1f}", r"$\mathrm{-}$"),
        (FLOAT_INF, "{:.1f}", r"$\mathrm{-}$"),
    ],
)
def test_latex_numbers_keep_their_sign_but_never_print_negative_zero(
    value: float | None, spec: str, text: str
) -> None:
    assert _latex_fmt(value, spec) == text


@pytest.mark.parametrize(
    "lb, ub, expected",
    [
        (75.0, 100.0, 25.0),
        (None, 100.0, None),
        (75.0, 0.0, None),
        (75.0, FLOAT_INF, None),
    ],
)
def test_gap_is_a_percentage_of_the_upper_bound(lb, ub, expected) -> None:
    assert _result(None, 100.0, lb, ub).gap_pct() == pytest.approx(expected)


@pytest.mark.parametrize("with_baseline", [True, False])
def test_result_json_round_trip(tmp_path, with_baseline: bool) -> None:
    ctx = make_ctx(line_scenario([ModemsRequest(1, 2, earliest_pickup=0.0)]))
    solution = greedy_solution(ctx)
    info = ModemsSolutionInfo(status="feasible", objective=solution.objective())
    result = ModemsBenchmarkResult(
        ctx,
        "alns",
        solution if with_baseline else None,
        info if with_baseline else None,
        solution,
        info,
    )
    restored = ModemsBenchmarkResult.from_json(result.to_json(str(tmp_path / "r.json")))
    assert restored.to_dict() == result.to_dict()
    assert (restored.baseline_solution is None) == (not with_baseline)


# --------------------------------------------------------------------------------------
# _solve_one when the constructive baseline fails
# --------------------------------------------------------------------------------------


@pytest.mark.integration
def test_milp_is_still_solved_without_a_constructive_baseline() -> None:
    """Non-selective greedy exceeds the driving budget: MILP1 proves infeasibility"""
    result = _solve_one(
        duration_limited_scenario(),
        "milp1",
        {},
        milp_timelimit=10.0,
        alns_max_iter=1,
        seed=1,
    )
    assert result is not None
    assert result.baseline_solution is None and result.improvement_pct() is None
    assert result.final_info.status is SolutionStatus.infeasible


def test_alns_is_skipped_without_a_feasible_base_plan() -> None:
    agent = ModemsAgent("s", 6, soc_initial=0.26, soc_min_operational=0.25)
    scenario = line_scenario([ModemsRequest(1, 2)], agents=[agent])
    assert (
        _solve_one(scenario, "alns", {}, milp_timelimit=1.0, alns_max_iter=1, seed=1)
        is None
    )


# --------------------------------------------------------------------------------------
# Phase 1: generation (no solving)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("objective, prefix", [("closed", "SC"), ("open", "SO")])
def test_generation_writes_one_scenario_and_one_pending_row_per_solver(
    tmp_path, objective: str, prefix: str
) -> None:
    summary = generate(
        tmp_path, nr_repeats=2, objective=objective, solver_strategies=["milp3", "alns"]
    )
    names = [f"{prefix}0_SRU80_100", f"{prefix}1_SRU80_100"]
    assert summary == [
        {"scenario_name": n, "status": ManifestStatus.created} for n in names
    ]
    assert sorted(os.listdir(tmp_path / "scenarios")) == [f"{n}.json" for n in names]
    manifest = read_manifest(str(tmp_path))
    assert set(manifest) == {(n, s) for n in names for s in ("milp3", "alns")}
    for row in manifest.values():
        assert row["status"] == "pending" and row["objective_type"] == objective
        assert row["nr_agents"] == 1 and 4 <= row["nr_requests"] <= 8
    raw = json.loads((tmp_path / "manifest.json").read_text())
    assert isinstance(raw, dict) and len(raw) == 4


def test_generation_is_reproducible_and_resumable(tmp_path) -> None:
    generate(tmp_path / "a", base_seed=5)
    generate(tmp_path / "b", base_seed=5)
    scenario_a = (tmp_path / "a" / "scenarios" / "SC0_SRU80_100.json").read_text()
    assert (
        scenario_a == (tmp_path / "b" / "scenarios" / "SC0_SRU80_100.json").read_text()
    )

    manifest_before = (tmp_path / "a" / "manifest.json").read_text()
    summary = generate(tmp_path / "a", base_seed=99)  # existing rows are never rebuilt
    assert summary == [
        {"scenario_name": "SC0_SRU80_100", "status": ManifestStatus.existing}
    ]
    assert (tmp_path / "a" / "manifest.json").read_text() == manifest_before


def test_generation_skips_combinations_without_a_feasible_base_plan(tmp_path) -> None:
    """Agents starting at the operational SoC floor cannot all reach a hub"""
    summary = generate(
        tmp_path,
        scenario_sizes=[ScenarioSize.large],
        scenario_soc_ranges=[SocRangeSpec.custom(0.25, 0.25)],
        max_feasibility_attempts=2,
    )
    assert summary == [
        {"scenario_name": "SC0_LRU25_25", "status": ManifestStatus.infeasible}
    ]
    assert read_manifest(str(tmp_path)) == {}
    assert os.listdir(tmp_path / "scenarios") == []


def test_generation_reports_progress_before_and_after_each_combination(
    tmp_path,
) -> None:
    events: list[dict] = []
    generate(tmp_path, nr_repeats=2, on_progress=events.append)
    assert [(e["phase"], e["index"], e["total"], e["status"]) for e in events] == [
        (ManifestPhase.generate, 1, 2, ManifestStatus.starting),
        (ManifestPhase.generate, 1, 2, ManifestStatus.created),
        (ManifestPhase.generate, 2, 2, ManifestStatus.starting),
        (ManifestPhase.generate, 2, 2, ManifestStatus.created),
    ]


# --------------------------------------------------------------------------------------
# Phases 2 and 3: solve and build (integration)
# --------------------------------------------------------------------------------------


QUICK_SEED = 2  # its smallest-bucket scenario (4 requests) solves to optimality in <1s


def solve(outdir, **kwargs: Any) -> list[dict[str, Any]]:
    return solve_benchmark_suite(
        str(outdir), milp_timelimit=10.0, alns_max_iter=20, **kwargs
    )


@pytest.mark.integration
@pytest.mark.parametrize("objective", ["closed", "open"])
def test_pipeline_solves_every_strategy_and_builds_the_table(
    tmp_path, objective: str
) -> None:
    generate(tmp_path, objective=objective, base_seed=QUICK_SEED)
    events: list[dict] = []
    outcomes = solve(tmp_path, on_progress=events.append)

    assert sorted(o["solver_strategy"] for o in outcomes) == [
        "alns",
        "milp1",
        "milp2",
        "milp3",
    ]
    assert all(o["status"] == ManifestStatus.done for o in outcomes)
    assert len(events) == 8 and {e["phase"] for e in events} == {ManifestPhase.solve}
    for row in read_manifest(str(tmp_path)).values():
        result = ModemsBenchmarkResult.from_json(row["result_file"])
        assert result.ctx.is_open() == (objective == "open")
        assert result.final_info.status in (
            SolutionStatus.optimal,
            SolutionStatus.feasible,
        )
        assert result.final_info.objective <= result.baseline_info.objective + 1e-6
    results = os.listdir(tmp_path / "results")
    assert sum(f.endswith("_alns_stats.json") for f in results) == 1

    rows = build_benchmark_table(str(tmp_path), rows_per_block=2, formats="all")
    figure = plot_benchmark_figure(rows, str(tmp_path / "solver_comparison.png"))
    assert os.path.getsize(figure) > 0
    assert {r["size"] for r in rows} == {"small"}
    assert [r["solver"] for r in rows] == ["alns", "milp1", "milp2", "milp3"]
    assert all(r["status_symbol"] in (r"$\star$", r"$\dagger$") for r in rows)
    tex = (tmp_path / "benchmark_table.tex").read_text()
    assert tex.count(r"\begin{table*}") == 2
    assert len((tmp_path / "benchmark_table.csv").read_text().splitlines()) == 5
    assert (
        json.loads((tmp_path / "benchmark_table.json").read_text())[0]["scenario"]
        == f"S{objective[0].upper()}0_SRU80_100"
    )

    assert solve(tmp_path) == []  # nothing left to solve


@pytest.mark.integration
def test_failed_rows_are_kept_and_only_retried_on_request(tmp_path) -> None:
    generate(tmp_path, solver_strategies=["alns"], base_seed=QUICK_SEED)
    scenario_file = tmp_path / "scenarios" / "SC0_SRU80_100.json"
    content = scenario_file.read_text()
    scenario_file.unlink()

    (outcome,) = solve(tmp_path)
    assert outcome["status"] == ManifestStatus.failed and outcome["error"]
    assert solve(tmp_path) == []
    rows = build_benchmark_table(str(tmp_path))
    assert rows[0]["status"] == ManifestStatus.failed and rows[0]["objective"] is None

    scenario_file.write_text(content)
    (outcome,) = solve(tmp_path, retry_failed=True)
    assert outcome["status"] == ManifestStatus.done


# --------------------------------------------------------------------------------------
# Export formats and the summary figure
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "selection, expected",
    [
        (None, {"csv", "json"}),
        ("csv,json", {"csv", "json"}),
        (" TEX , csv", {"csv", "tex"}),
        (["json"], {"json"}),
        ("all", {"csv", "json", "tex"}),
        ("csv,all", {"csv", "json", "tex"}),
    ],
)
def test_export_format_selection(selection, expected: set) -> None:
    assert resolve_export_formats(selection) == {ExportFormat(f) for f in expected}


def test_export_format_selection_rejects_unknown_formats() -> None:
    with pytest.raises(ValueError):
        resolve_export_formats("csv,xlsx")


@pytest.mark.parametrize("selection", ["csv", "json,tex", "all"])
def test_table_files_are_written_only_for_the_selected_formats(
    tmp_path, selection: str
) -> None:
    rows = [{"a": 1, "b": "x", "ignored": 0}]
    written = write_table_files(
        rows,
        str(tmp_path),
        "t",
        ["a", "b"],
        selection,
        write_tex=lambda path: open(path, "w").write("tex"),
        csv_rows=[{"a": 1, "b": "display"}],
    )
    expected = {f"t.{f}" for f in resolve_export_formats(selection)}
    assert {os.path.basename(p) for p in written} == expected
    assert sorted(os.listdir(tmp_path)) == sorted(expected)
    if "t.csv" in expected:
        assert (tmp_path / "t.csv").read_text().splitlines() == ["a,b", "1,display"]
    if "t.json" in expected:
        assert json.loads((tmp_path / "t.json").read_text()) == rows


def test_gap_to_best_is_relative_to_each_scenarios_best_objective() -> None:
    rows = [
        {"scenario": "A", "solver": "alns", "objective": 100.0},
        {"scenario": "A", "solver": "milp1", "objective": 125.0},
        {"scenario": "A", "solver": "milp2", "objective": None},
        {"scenario": "B", "solver": "alns", "objective": 10.0},
        {"scenario": "B", "solver": "milp1", "objective": 10.0},
    ]
    assert gap_to_best(rows) == {
        ("A", "alns"): 0.0,
        ("A", "milp1"): 25.0,
        ("B", "alns"): 0.0,
        ("B", "milp1"): 0.0,
    }


def test_alns_seed_is_the_scenario_seed_shared_by_every_decision(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scenario name encodes the objective, so it must not seed the search"""
    for objective in ("closed", "open"):
        generate(tmp_path, objective=objective, solver_strategies=["milp3", "alns"])
    manifest = read_manifest(str(tmp_path))
    seeds = {row["seed"] for row in manifest.values()}
    assert len(seeds) == 1  # closed and open rows of one scenario share its seed

    received: list[int] = []

    def fake_solve_one(*args: Any, seed: int, **kwargs: Any) -> None:
        received.append(seed)
        raise RuntimeError("stop after recording the seed")

    monkeypatch.setattr(benchmark_module, "_solve_one", fake_solve_one)
    solve_benchmark_suite(str(tmp_path))
    assert len(received) == 4 and set(received) == seeds


def test_rows_without_a_stored_seed_fall_back_to_the_default_seed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generate(tmp_path, solver_strategies=["alns"])
    path = tmp_path / "manifest.json"
    data = json.loads(path.read_text())
    for row in data.values():
        row.pop("seed")
    path.write_text(json.dumps(data))
    received: list[int] = []

    def fake_solve_one(*args: Any, seed: int, **kwargs: Any) -> None:
        received.append(seed)
        raise RuntimeError("stop after recording the seed")

    monkeypatch.setattr(benchmark_module, "_solve_one", fake_solve_one)
    solve_benchmark_suite(str(tmp_path))
    assert received == [DEFAULT_BASE_SEED]
