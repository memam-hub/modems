"""
Tests for modems.workday_comparison: statistics, pairing, impacts-estimates on
synthetic runs with hand-computable differences, and the full closed-vs-open
comparison on two tiny real suites (integration)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from modems.benchmark import ManifestStatus
from modems.core import ModemsRequest, ObjectiveType, SolverStrategy
from modems.rolling_horizon import RequestOutcome, WorkdayLog
from modems.workday_benchmark import generate_workday_suite, solve_workday_suite
from modems.workday_comparison import (
    WorkdayRun,
    _check_pairs,
    _load_suite,
    compare_workday_objectives,
    impact_estimates,
    mean_ci,
    pair_name,
    representative_pair,
    rolling_acceptance,
    t_critical,
    welch_ci,
)

# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "df, value", [(1, 12.706), (4, 2.776), (30, 2.042), (60, 2.000), (1e9, 1.960)]
)
def test_t_critical_values(df: float, value: float) -> None:
    assert t_critical(df) == pytest.approx(value, abs=2e-3)


def test_mean_ci_matches_the_t_interval() -> None:
    mean, half = mean_ci([1.0, 2.0, 3.0, 4.0])
    assert mean == 2.5
    assert half == pytest.approx(3.182 * math.sqrt(5 / 3) / 2, rel=1e-4)
    assert mean_ci([]) is None
    assert mean_ci([5.0]) == (5.0, math.inf)


def test_welch_ci_is_the_difference_b_minus_a() -> None:
    diff, half = welch_ci([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    assert diff == 3.0
    assert half == pytest.approx(2.776 * math.sqrt(2 / 3), rel=1e-3)  # df = 4
    assert welch_ci([1.0], [2.0, 3.0]) is None
    assert welch_ci([1.0, 1.0], [2.0, 2.0]) == (1.0, 0.0)


# --------------------------------------------------------------------------------------
# Pairing and synthetic Figure 1 estimates
# --------------------------------------------------------------------------------------


def test_pair_name_drops_only_the_objective_letter() -> None:
    assert pair_name("WC0_SRUN5_0") == pair_name("WO0_SRUN5_0") == "W0_SRUN5_0"


def _result(acceptance: float, delay: float, served: int = 4) -> dict:
    return {
        "acceptance_rate": acceptance,
        "mean_delay_time": delay,
        "mean_excess_ride_time": 1.0,
        "total_agent_travel_time": 10.0 * served,
        "total_energy_consumed": 0.02 * served,
        "nr_requests_completed": served,
    }


def _run(
    objective: str, rep: int, start: str, milp3: dict, alns: dict, surges: int = 0
) -> WorkdayRun:
    letter = objective[0].upper()
    return WorkdayRun(
        name=f"W{letter}{rep}_SRU{start[0].upper()}5_{surges}",
        objective=ObjectiveType(objective),
        row={
            "seed": rep,
            "start_time": start,
            "nr_surges": surges,
            "workday_length": 60.0,
            "nr_agents": 1,
        },
        results={SolverStrategy.milp3: milp3, SolverStrategy.alns: alns},
        nr_submissions=5 + rep,
    )


def _suites() -> tuple[dict, dict]:
    """Two workdays per start time; open adds +1 min delay (MILP3) / +2 (ALNS)"""
    closed, opened = {}, {}
    for rep, start in [
        (0, "normal"),
        (1, "normal"),
        (2, "staggered"),
        (3, "staggered"),
    ]:
        delay = 1.0 + rep
        c = _run("closed", rep, start, _result(0.8, delay), _result(0.9, delay + 0.5))
        o = _run(
            "open", rep, start, _result(0.8, delay + 1.0), _result(1.0, delay + 2.5)
        )
        closed[c.pair], opened[o.pair] = c, o
    return closed, opened


def _estimate(estimates: list, metric: str, decision: str, group: str) -> dict:
    (match,) = [
        e
        for e in estimates
        if (e["metric"], e["decision"], e["group"]) == (metric, decision, group)
    ]
    return match


def test_effects_are_paired_differences_with_the_right_groups() -> None:
    estimates = impact_estimates(*_suites())
    objective_milp3 = _estimate(estimates, "delay", "objective", "MILP3")
    assert (objective_milp3["mean"], objective_milp3["ci"], objective_milp3["n"]) == (
        1.0,
        0.0,
        4,
    )
    assert _estimate(estimates, "delay", "objective", "ALNS")["mean"] == 2.0
    assert _estimate(estimates, "acceptance", "objective", "ALNS")[
        "mean"
    ] == pytest.approx(10.0)
    assert _estimate(estimates, "delay", "solver", "closed")["mean"] == 0.5
    assert _estimate(estimates, "delay", "solver", "open")["mean"] == 1.5
    # start time: staggered (reps 2, 3) - normal (reps 0, 1) of per-workday averages
    start = _estimate(estimates, "delay", "start_time", "MILP3")
    assert start["mean"] == pytest.approx(2.0) and start["n"] == 4
    travel = _estimate(estimates, "travel", "objective", "MILP3")
    assert travel["mean"] == 0.0  # 10 min per served request in every run


def test_metrics_without_served_requests_are_skipped() -> None:
    closed, opened = _suites()
    opened["W0_SRUN5_0"].results[SolverStrategy.milp3] = _result(0.0, 0.0, served=0)
    estimates = impact_estimates(closed, opened)
    assert _estimate(estimates, "delay", "objective", "MILP3")["n"] == 3
    assert _estimate(estimates, "acceptance", "objective", "MILP3")["n"] == 4


def test_representative_pair_prefers_the_busiest_workday_with_surges() -> None:
    closed, opened = _suites()
    assert representative_pair(closed, opened) == "W3_SRUS5_0"  # busiest overall
    busy = _run("closed", 1, "normal", _result(1, 1), _result(1, 1), surges=2)
    closed[busy.pair] = busy
    opened[busy.pair] = _run(
        "open", 1, "normal", _result(1, 1), _result(1, 1), surges=2
    )
    assert representative_pair(closed, opened) == busy.pair
    assert representative_pair({}, {}) is None


def test_pairs_must_share_their_seed() -> None:
    closed, opened = _suites()
    _check_pairs(closed, opened)
    opened["W0_SRUN5_0"].row["seed"] = 99
    with pytest.raises(ValueError, match="different seeds"):
        _check_pairs(closed, opened)


def test_rolling_acceptance_uses_the_trailing_window() -> None:
    def record(t: float, outcome: str) -> dict:
        request = ModemsRequest(1, 2, earliest_pickup=t + 20, request_id=f"r{t}")
        return {
            "request": request.to_dict(),
            "submission_time": t,
            "earliest_pickup": t + 20,
            "outcome": outcome,
        }

    log = WorkdayLog.from_dict(
        {
            "clock_start": 0.0,
            "clock_end": 100.0,
            "agent_node_visits": {},
            "requests": [
                record(0.0, RequestOutcome.accepted),
                record(10.0, RequestOutcome.rejected),
                record(20.0, RequestOutcome.unresolved),
                record(50.0, RequestOutcome.accepted),
            ],
        }
    )
    times, rates = rolling_acceptance(log, window=30.0)
    assert times == [0.0, 10.0, 50.0]
    assert rates == [100.0, 50.0, 100.0]  # the rejection at t=10 left the window


# --------------------------------------------------------------------------------------
# Integration: two real suites
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def suites(tmp_path_factory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("suites")
    for objective in ("closed", "open"):
        generate_workday_suite(
            str(root / objective),
            scenario_sizes=["s"],
            scenario_types=["random"],
            scenario_timings=["uniform"],
            start_times=["normal", "staggered"],
            base_rates=[5],
            nr_surges=0,
            workday_length=40.0,
            nr_repeats=2,
            objective=objective,
        )
        solve_workday_suite(str(root / objective), milp_timelimit=2.0, alns_max_iter=10)
    return root / "closed", root / "open"


@pytest.mark.integration
def test_comparison_end_to_end(tmp_path: Path, suites: tuple[Path, Path]) -> None:
    closed_dir, open_dir = suites
    result = compare_workday_objectives(
        str(closed_dir), str(open_dir), str(tmp_path), all_workdays=True
    )
    assert result["nr_paired"] == 4 and result["unpaired"] == []

    rows = json.loads((tmp_path / "combined_summary.json").read_text())
    assert [r["objective_type"] for r in rows] == ["closed", "open"] * 4
    assert [pair_name(r["workday"]) for r in rows[::2]] == [
        pair_name(r["workday"]) for r in rows[1::2]
    ]
    assert all(
        r["nr_submissions"] == n
        for r, n in zip(rows[1::2], [r["nr_submissions"] for r in rows[::2]])
    )
    assert sum(r["nr_served_alns"] for r in rows) > 0
    for r in rows:
        served = r["nr_served_alns"]
        expected = r["travel_time_alns"] / served if served else None
        assert r["travel_time_per_served_alns"] == pytest.approx(expected)
    manifest = json.loads((tmp_path / "combined_manifest.json").read_text())
    assert [m["workday_name"] for m in manifest] == [r["workday"] for r in rows]
    assert (tmp_path / "combined_summary.tex").read_text().count("WO") == 4

    assert set(result["figures"]) == {"impacts", "demand", "workday"}
    assert all(Path(p).stat().st_size > 0 for p in result["figures"].values())
    assert len(list((tmp_path / "workdays").glob("*.png"))) == 4
    assert {e["decision"] for e in result["estimates"]} == {
        "objective",
        "solver",
        "start_time",
    }


@pytest.mark.integration
def test_suites_are_validated(tmp_path: Path, suites: tuple[Path, Path]) -> None:
    closed_dir, open_dir = suites
    with pytest.raises(ValueError, match="must only contain closed"):
        compare_workday_objectives(str(open_dir), str(closed_dir), str(tmp_path))
    runs = _load_suite(str(closed_dir), ObjectiveType.closed)
    assert all(
        ManifestStatus(r.row["status"]) == ManifestStatus.done for r in runs.values()
    )
