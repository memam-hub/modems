from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from modems.algorithms import greedy_complete, preprocess
from modems.benchmark import (
    ManifestStatus,
    ModemsBenchmarkResult,
    build_benchmark_table,
    generate_benchmark_suite,
    read_manifest,
    solve_benchmark_suite,
    stable_seed,
)
from modems.core import (
    ModemsScenario,
    ProblemContext,
    ProblemType,
    ScenarioSize,
    ScenarioTiming,
    ScenarioType,
    SolverStrategy,
)
from modems.generator import SocRangeSpec
from modems.milp import DEFAULT_MILP_SOLVER_DATA, MilpType
from modems.solution import FLOAT_INF, ModemsSolutionInfo, SolutionStatus

# --------------------------------------------------------------------------------------
# stable_seed
# --------------------------------------------------------------------------------------


def test_stable_seed_is_deterministic() -> None:
    """stable_seed() returns the same value for the same inputs every call"""
    assert stable_seed(42, "S", "R", 0) == stable_seed(42, "S", "R", 0)
    assert stable_seed(42, "S", "R", "T", 1) == stable_seed(42, "S", "R", "T", 1)


def test_stable_seed_differs_for_different_inputs() -> None:
    """stable_seed() returns a different value when any input part changes"""
    assert stable_seed(42, "S", "R", 0) != stable_seed(42, "S", "R", 1)
    assert stable_seed(42, "S", "R", 0) != stable_seed(42, "M", "R", 0)


# --------------------------------------------------------------------------------------
# ModemsBenchmarkResult
# --------------------------------------------------------------------------------------


def _make_result(tiny_scenario: ModemsScenario) -> ModemsBenchmarkResult:
    """Build a ModemsBenchmarkResult from a constructive baseline (no real solve)"""
    params = {"eps": 0.01, "zeta": 1.0, "eta": 100.0, "rho": 2.0}
    ctx = ProblemContext(
        tiny_scenario, ProblemType.closed_selective, SolverStrategy.alns, params
    )
    base_plan, r_unassigned = preprocess(ctx)
    assert base_plan is not None
    sol, _, _ = greedy_complete(base_plan, r_unassigned)
    sol_info = ModemsSolutionInfo(
        status=SolutionStatus.feasible,
        objective=sol.objective(),
        solver_name="baseline",
    )
    final_info = ModemsSolutionInfo(
        status=SolutionStatus.optimal,
        objective=sol.objective() * 0.9,
        lower_bound=sol.objective() * 0.9,
        upper_bound=sol.objective() * 0.9,
        solver_name=DEFAULT_MILP_SOLVER_DATA[0],
    )
    return ModemsBenchmarkResult(
        ctx,
        "milp3",
        sol,
        sol_info,
        sol,
        final_info,
    )


def test_benchmark_result_to_dict_from_dict_round_trip(
    tiny_scenario: ModemsScenario,
) -> None:
    """to_dict()/from_dict() (JSON round-trip) preserve solver_strategy and objective"""
    result = _make_result(tiny_scenario)
    d = result.to_dict()
    result2 = ModemsBenchmarkResult.from_dict(json.loads(json.dumps(d)))
    assert result2.solver_strategy == "milp3"
    assert result2.final_solution.objective() == pytest.approx(
        result.final_solution.objective()
    )


def test_benchmark_result_improvement_and_gap_pct(
    tiny_scenario: ModemsScenario,
) -> None:
    """improvement_pct()/gap_pct() compute the expected percentages from the fixture"""
    result = _make_result(tiny_scenario)
    assert result.improvement_pct() == pytest.approx(0.10, abs=1e-6)
    assert result.gap_pct() == pytest.approx(0.0, abs=1e-6)


def test_benchmark_result_improvement_pct_none_when_baseline_missing_info(
    tiny_scenario: ModemsScenario,
) -> None:
    """improvement_pct() returns None when the baseline objective is non-finite"""
    result = _make_result(tiny_scenario)
    result.baseline_info.objective = FLOAT_INF
    assert result.improvement_pct() is None


def test_benchmark_result_round_trip_without_constructive(
    tiny_scenario: ModemsScenario,
) -> None:
    """A result with no baseline round-trips with baseline fields staying None"""
    result = _make_result(tiny_scenario)
    result.baseline_solution = None
    result.baseline_info = None

    restored = ModemsBenchmarkResult.from_dict(json.loads(json.dumps(result.to_dict())))

    assert restored.baseline_solution is None
    assert restored.baseline_info is None
    assert restored.improvement_pct() is None


def test_benchmark_gap_ignores_nonfinite_bounds(tiny_scenario: ModemsScenario) -> None:
    """gap_pct() returns None when the upper bound is non-finite"""
    result = _make_result(tiny_scenario)
    result.final_info.upper_bound = FLOAT_INF
    assert result.gap_pct() is None


def test_milp_still_solves_when_constructive_fails(
    tiny_scenario: ModemsScenario,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_solve_one() still solves via a (faked) MILP even if _compute_baseline() fails"""
    import modems.benchmark as benchmark

    class FakeMilp:
        """Stand-in for ModemsMilp that skips real solver, asserting no warm start"""

        def __init__(
            self,
            scenario: ModemsScenario,
            milp_type: MilpType,
            problem_type: ProblemType,
            model_params: dict,
        ) -> None:
            """Build a trivial idle-fleet instance instead of a real Pyomo model"""
            ctx = benchmark.ProblemContext(
                scenario,
                problem_type,
                SolverStrategy.milp3,
                model_params,
            )
            self.ctx = ctx
            self.milp_type = milp_type
            self.problem_type = problem_type
            solution = benchmark.ModemsSolution(ctx)
            self.instance = SimpleNamespace(
                solution=solution,
                solution_info=ModemsSolutionInfo(
                    status=SolutionStatus.feasible,
                    objective=solution.objective(),
                ),
            )

        def solve(self, **kwargs: Any) -> None:
            """Assert the caller never passes a warm start when baseline failed"""
            assert kwargs["warm_start_solution"] is None

    monkeypatch.setattr(benchmark, "_compute_baseline", lambda *args: (None, None))
    monkeypatch.setattr(benchmark, "ModemsMilp", FakeMilp)

    result = benchmark._solve_one(
        tiny_scenario,
        SolverStrategy("milp3"),
        {"eps": 0.01, "zeta": 1.0, "eta": 100.0, "rho": 2.5},
        milp_timelimit=1.0,
        alns_max_iter=1,
        seed=1,
    )

    assert result is not None
    assert result.baseline_solution is None


# --------------------------------------------------------------------------------------
# generate_benchmark_suite / manifest resumability (no solving, fast)
# --------------------------------------------------------------------------------------


def test_generate_suite_creates_scenarios_and_manifest_rows(tmp_path: Any) -> None:
    """generate_benchmark_suite() writes one manifest row per (scenario, solver)"""
    summary = generate_benchmark_suite(
        outdir=str(tmp_path),
        scenario_sizes=[ScenarioSize.small],
        scenario_types=[ScenarioType.random],
        scenario_timings=[ScenarioTiming.loose],
        scenario_soc_ranges=[SocRangeSpec.normal()],
        nr_repeats=1,
        base_seed=1,
    )
    assert any(row["status"] == ManifestStatus.created for row in summary)
    manifest = read_manifest(str(tmp_path))
    assert len(manifest) == 4  # one row per solver (alns, milp1, milp2, milp3)


def test_generate_suite_is_resumable(tmp_path: Any) -> None:
    """Re-running generate_benchmark_suite() skips already-generated scenarios"""
    generate_benchmark_suite(
        outdir=str(tmp_path),
        scenario_sizes=[ScenarioSize.small],
        scenario_types=[ScenarioType.random],
        scenario_timings=[ScenarioTiming.loose],
        scenario_soc_ranges=[SocRangeSpec.normal()],
        nr_repeats=1,
        base_seed=1,
    )
    summary2 = generate_benchmark_suite(
        outdir=str(tmp_path),
        scenario_sizes=[ScenarioSize.small],
        scenario_types=[ScenarioType.random],
        scenario_timings=[ScenarioTiming.loose],
        scenario_soc_ranges=[SocRangeSpec.normal()],
        nr_repeats=1,
        base_seed=1,
    )
    assert all(row["status"] == ManifestStatus.existing for row in summary2)


def test_manifest_is_a_streamlined_json_object(tmp_path: Any) -> None:
    """manifest.json is one JSON object keyed by {scenario__solver}"""
    generate_benchmark_suite(
        outdir=str(tmp_path),
        scenario_sizes=[ScenarioSize.small],
        scenario_types=[ScenarioType.random],
        scenario_timings=[ScenarioTiming.loose],
        scenario_soc_ranges=[SocRangeSpec.normal()],
        nr_repeats=1,
        base_seed=1,
    )
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    with open(manifest_path) as f:
        on_disk = json.load(f)  # raises if it is NDJSON rather than one JSON document
    assert set(on_disk.keys()) == {
        "DS_SRL0_soc80-100__alns",
        "DS_SRL0_soc80-100__milp1",
        "DS_SRL0_soc80-100__milp2",
        "DS_SRL0_soc80-100__milp3",
    }
    assert on_disk["DS_SRL0_soc80-100__alns"]["scenario_name"] == "DS_SRL0_soc80-100"
    assert on_disk["DS_SRL0_soc80-100__alns"]["solver_strategy"] == "alns"


def test_generate_suite_reports_progress(tmp_path: Any) -> None:
    """on_progress fires a start and an end event per (size, type, repetition) combo"""
    events: list[dict[str, Any]] = []
    generate_benchmark_suite(
        outdir=str(tmp_path),
        scenario_sizes=[ScenarioSize.small],
        scenario_types=[ScenarioType.random, ScenarioType.clustered],
        scenario_timings=[ScenarioTiming.loose],
        scenario_soc_ranges=[SocRangeSpec.normal()],
        nr_repeats=1,
        base_seed=1,
        on_progress=events.append,
    )
    assert len(events) == 4  # 2 combos x (starting, terminal status)
    assert {e["status"] for e in events} == {"starting", "created"}
    assert all(e["phase"] == "generate" and e["total"] == 2 for e in events)
    assert {e["scenario_name"] for e in events} == {
        "DS_SRL0_soc80-100",
        "DS_SCL0_soc80-100",
    }


def test_solve_suite_reports_progress(tmp_path: Any) -> None:
    """on_progress fires a start and a done event per (scenario, solver) row"""
    generate_benchmark_suite(
        outdir=str(tmp_path),
        scenario_sizes=[ScenarioSize.small],
        scenario_types=[ScenarioType.random],
        scenario_timings=[ScenarioTiming.loose],
        scenario_soc_ranges=[SocRangeSpec.normal()],
        nr_repeats=1,
        base_seed=1,
        solver_strategies=[SolverStrategy("alns")],
    )
    events: list[dict[str, Any]] = []
    solve_benchmark_suite(
        outdir=str(tmp_path), alns_max_iter=20, on_progress=events.append
    )
    assert len(events) == 2  # one row x (starting, done)
    assert [e["status"] for e in events] == ["starting", "done"]
    assert all(
        e["phase"] == "solve"
        and e["scenario_name"] == "DS_SRL0_soc80-100"
        and e["solver_strategy"] == "alns"
        for e in events
    )
    assert events[1]["t_exe"] >= 0.0


# --------------------------------------------------------------------------------------
# solve_benchmark_suite / table building (real CBC + real alns solver calls)
# --------------------------------------------------------------------------------------


@pytest.mark.integration
def test_solve_and_build_table_end_to_end(tmp_path: Any) -> None:
    """generate -> solve (real CBC/alns) -> build_table runs and is resumable"""
    generate_benchmark_suite(
        outdir=str(tmp_path),
        scenario_sizes=[ScenarioSize.small],
        scenario_types=[ScenarioType.random],
        scenario_timings=[ScenarioTiming.loose],
        scenario_soc_ranges=[SocRangeSpec.normal()],
        nr_repeats=1,
        base_seed=1,
    )
    outcomes = solve_benchmark_suite(
        outdir=str(tmp_path), milp_timelimit=10.0, alns_max_iter=100, seed=1
    )
    assert len(outcomes) == 4
    assert all(o["status"] == "done" for o in outcomes)

    # resumability: rerun should find nothing pending
    outcomes2 = solve_benchmark_suite(
        outdir=str(tmp_path), milp_timelimit=10.0, alns_max_iter=100, seed=1
    )
    assert outcomes2 == []

    rows = build_benchmark_table(str(tmp_path))
    assert len(rows) == 4
    for row in rows:
        assert row["status_symbol"] in {
            r"$\star$",
            r"$\dagger$",
            r"$\times$",
            "?",
            "--",
        }
