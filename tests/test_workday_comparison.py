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
from modems.workday_benchmark import (
    WorkdayStartTime,
    generate_workday_suite,
    solve_workday_suite,
)
from modems.workday_comparison import (
    WorkdayRun,
    _check_pairs,
    _load_suite,
    compare_workday_objectives,
    impact_estimates,
    mean_ci,
    pair_name,
    representative_group,
    rolling_acceptance,
    start_time_pairs,
    t_critical,
    workday_group_name,
    workday_groups,
)

# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "df, value", [(1, 12.706), (4, 2.776), (30, 2.042), (31, 1.96), (10**6, 1.96)]
)
def test_t_critical_values(df: int, value: float) -> None:
    assert t_critical(df) == pytest.approx(value)
    assert t_critical(0) == math.inf


def test_mean_ci_matches_the_t_interval() -> None:
    mean, half = mean_ci([1.0, 2.0, 3.0, 4.0])
    assert mean == 2.5
    assert half == pytest.approx(3.182 * math.sqrt(5 / 3) / 2, rel=1e-4)
    assert mean_ci([]) is None
    assert mean_ci([5.0]) == (5.0, math.inf)


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
            "seed": rep,  # the seed depends on the demand only
            "size": "small",
            "type": "random",
            "timing": "uniform",
            "base_rate": 5,
            "repetition": rep,
            "start_time": start,
            "nr_surges": surges,
            "workday_length": 60.0,
            "nr_agents": 1,
        },
        results={SolverStrategy.milp3: milp3, SolverStrategy.alns: alns},
        nr_submissions=5 + rep,
    )


def _suites() -> tuple[dict, dict]:
    """
    Two demands, each run with both start times: staggered adds +2 min delay, open
    adds +1 min (MILP3) / +2 (ALNS), ALNS adds +0.5 over MILP3 (closed)
    """
    closed, opened = {}, {}
    for rep, start in [
        (0, "normal"),
        (0, "staggered"),
        (1, "normal"),
        (1, "staggered"),
    ]:
        delay = 1.0 + rep + (2.0 if start == "staggered" else 0.0)
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


def test_each_decision_is_split_by_the_other_two() -> None:
    estimates = impact_estimates(*_suites())
    groups = {
        decision: [
            e["group"]
            for e in estimates
            if e["metric"] == "delay" and e["decision"] == decision
        ]
        for decision in ("objective", "solver", "start_time")
    }
    assert groups == {
        "objective": [
            "MILP3 · normal",
            "MILP3 · staggered",
            "ALNS · normal",
            "ALNS · staggered",
        ],
        "solver": [
            "closed · normal",
            "closed · staggered",
            "open · normal",
            "open · staggered",
        ],
        "start_time": [
            "MILP3 · closed",
            "MILP3 · open",
            "ALNS · closed",
            "ALNS · open",
        ],
    }
    levels = [
        e["levels"]
        for e in estimates
        if e["metric"] == "delay" and e["decision"] == "objective"
    ]
    assert levels == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_effects_are_first_level_minus_second_level() -> None:
    """closed - open, MILP3 - ALNS, normal - staggered, on paired workdays"""
    estimates = impact_estimates(*_suites())
    for start in ("normal", "staggered"):
        # one workday pair per demand (rep 0 and 1) with this start time
        milp3 = _estimate(estimates, "delay", "objective", f"MILP3 · {start}")
        assert (milp3["mean"], milp3["ci"], milp3["n"]) == (-1.0, 0.0, 2)
        assert _estimate(estimates, "delay", "objective", f"ALNS · {start}")[
            "mean"
        ] == pytest.approx(-2.0)
        assert _estimate(estimates, "delay", "solver", f"closed · {start}")[
            "mean"
        ] == pytest.approx(-0.5)
        assert _estimate(estimates, "delay", "solver", f"open · {start}")[
            "mean"
        ] == pytest.approx(-1.5)
    assert _estimate(estimates, "acceptance", "objective", "ALNS · normal")[
        "mean"
    ] == pytest.approx(-10.0)
    for group in ("MILP3 · closed", "MILP3 · open", "ALNS · closed", "ALNS · open"):
        start = _estimate(estimates, "delay", "start_time", group)
        assert (start["mean"], start["ci"], start["n"]) == (-2.0, 0.0, 2)
    travel = _estimate(estimates, "travel", "objective", "MILP3 · normal")
    assert travel["mean"] == 0.0  # 10 min per served request in every run


def test_metrics_without_served_requests_are_skipped() -> None:
    closed, opened = _suites()
    opened["W0_SRUN5_0"].results[SolverStrategy.milp3] = _result(0.0, 0.0, served=0)
    estimates = impact_estimates(closed, opened)
    assert _estimate(estimates, "delay", "objective", "MILP3 · normal")["n"] == 1
    assert _estimate(estimates, "delay", "objective", "MILP3 · staggered")["n"] == 2
    assert _estimate(estimates, "acceptance", "objective", "MILP3 · normal")["n"] == 2


def test_group_name_drops_the_objective_and_start_time_letters() -> None:
    closed, opened = _suites()
    names = {workday_group_name(r) for r in [*closed.values(), *opened.values()]}
    assert names == {"W0_SRU5_0", "W1_SRU5_0"}
    assert workday_group_name(_run("open", 3, "staggered", {}, {}, surges=4)) == (
        "W3_SRU5_4"
    )


def test_workday_groups_gather_every_run_of_one_demand() -> None:
    closed, opened = _suites()
    groups = workday_groups(closed, opened)
    assert list(groups) == ["W0_SRU5_0", "W1_SRU5_0"]
    assert set(groups["W0_SRU5_0"]) == {
        (ObjectiveType(o), WorkdayStartTime(t))
        for o in ("closed", "open")
        for t in ("normal", "staggered")
    }
    assert all(run.row["repetition"] == 0 for run in groups["W0_SRU5_0"].values())
    # a different seed is a different demand, and one objective alone is no group
    for run in (closed["W0_SRUS5_0"], opened["W0_SRUS5_0"]):
        run.row["seed"] = 99
    del opened["W1_SRUN5_0"], opened["W1_SRUS5_0"]
    groups = workday_groups(closed, opened)
    assert {name: len(group) for name, group in groups.items()} == {"W0_SRU5_0": 2}
    assert workday_groups({}, {}) == {}


def test_representative_group_prefers_complete_busy_groups_with_surges() -> None:
    closed, opened = _suites()
    groups = workday_groups(closed, opened)
    assert representative_group(groups) == "W1_SRU5_0"  # busiest overall
    assert representative_group({}) is None

    def add(objective: str, start: str, surges: int) -> None:
        run = _run(objective, 0, start, _result(1, 1), _result(1, 1), surges=surges)
        (closed if objective == "closed" else opened)[run.pair] = run

    # with surges but only one start time: complete groups come first
    for objective in ("closed", "open"):
        add(objective, "normal", surges=2)
    assert representative_group(workday_groups(closed, opened)) == "W1_SRU5_0"
    # complete and with surges: preferred over a busier group without surges
    for objective in ("closed", "open"):
        add(objective, "staggered", surges=2)
    assert representative_group(workday_groups(closed, opened)) == "W0_SRU5_2"


def test_start_times_pair_only_on_identical_demand() -> None:
    closed, opened = _suites()
    assert len(start_time_pairs(closed, opened)) == 2
    # suites generated with the start time in the seed: no identical demand
    for runs in (closed, opened):
        for run in runs.values():
            if run.row["start_time"] == "staggered":
                run.row["seed"] += 100
    assert start_time_pairs(closed, opened) == []
    estimates = impact_estimates(closed, opened)
    start = _estimate(estimates, "delay", "start_time", "ALNS · closed")
    assert (start["mean"], start["n"]) == (None, 0)


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
            base_rates=[8],
            nr_surges=0,
            workday_length=45.0,
            nr_repeats=2,
            objective=objective,
        )
        solve_workday_suite(str(root / objective), milp_timelimit=2.0, alns_max_iter=10)
    return root / "closed", root / "open"


@pytest.mark.integration
def test_comparison_end_to_end(tmp_path: Path, suites: tuple[Path, Path]) -> None:
    closed_dir, open_dir = suites
    result = compare_workday_objectives(
        str(closed_dir), str(open_dir), str(tmp_path), all_workdays=True, formats="all"
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

    assert set(result["figures"]) == {
        "impacts",
        "demand",
        "impact_boxes",
        "demand_boxes",
        "workday",
    }
    assert result["nr_start_time_pairs"] == 2
    assert all(Path(p).stat().st_size > 0 for p in result["figures"].values())
    # one figure per demand: 2 repetitions, each with both start times
    assert sorted(p.name for p in (tmp_path / "workdays").glob("*.png")) == [
        "W0_SRU8_0.png",
        "W1_SRU8_0.png",
    ]
    assert Path(result["figures"]["workday"]).name == (
        f"most_demand_{result['representative']}.png"
    )
    assert {e["decision"] for e in result["estimates"]} == {
        "objective",
        "solver",
        "start_time",
    }
    # at most one workday pair per demand; acceptance is defined for every run
    assert all(e["n"] <= 2 for e in result["estimates"])
    assert all(e["n"] == 2 for e in result["estimates"] if e["metric"] == "acceptance")


@pytest.mark.integration
def test_suites_are_validated(tmp_path: Path, suites: tuple[Path, Path]) -> None:
    closed_dir, open_dir = suites
    with pytest.raises(ValueError, match="must only contain closed"):
        compare_workday_objectives(str(open_dir), str(closed_dir), str(tmp_path))
    runs = _load_suite(str(closed_dir), ObjectiveType.closed)
    assert all(
        ManifestStatus(r.row["status"]) == ManifestStatus.done for r in runs.values()
    )
