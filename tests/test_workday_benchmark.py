from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import modems.workday_benchmark as workday_benchmark
from modems.benchmark import ManifestStatus
from modems.core import (
    ModemsRequest,
    ObjectiveType,
    ScenarioSize,
    ScenarioTiming,
    ScenarioType,
    SolverStrategy,
)
from modems.generator import ModemsScenarioGenerator
from modems.rolling_horizon import (
    AgentNodeVisit,
    RequestOutcome,
    RequestRecord,
    WorkdayLog,
)
from modems.workday_benchmark import (
    WorkdayStartTime,
    build_workday_requests_table,
    build_workday_summary_table,
    generate_workday_suite,
    plot_workday_soc_acceptance,
    read_workday_manifest,
    solve_workday_suite,
)


def _request(request_id: str, pickup: int = 1, delivery: int = 2) -> ModemsRequest:
    """Build a minimal request with a stable identifier for report tests"""
    return ModemsRequest(
        node_pickup_index=pickup,
        node_delivery_index=delivery,
        load=2,
        service_time=0.5,
        earliest_pickup=10.0,
        request_id=request_id,
    )


def _workday_log(*records: RequestRecord) -> WorkdayLog:
    """Build a small full-day log containing one agent SoC walk"""
    return WorkdayLog(
        clock_start=0.0,
        clock_end=120.0,
        requests=list(records),
        agent_node_visits={
            "agent_1": [
                AgentNodeVisit("h_1", 0.0, 0.0, 1.0, 1.0),
                AgentNodeVisit("s_2", 25.0, 25.5, 0.9, 0.9),
            ]
        },
    )


def _summary(acceptance_rate: float) -> dict[str, float]:
    """Return the subset of workday summary fields consumed by table building"""
    return {
        "acceptance_rate": acceptance_rate,
        "mean_delay_time": 1.5,
        "mean_excess_ride_time": 2.0,
        "total_energy_consumed": 0.25,
        "total_soc_gained_from_charging": 0.1,
        "mean_solve_time_per_epoch": 0.1,
        "nr_requests_completed": 4,
        "total_agent_travel_time": 30.0,
    }


def _manifest_row(workday_name: str = "WC0_SRUN5_1") -> dict[str, Any]:
    """Return a valid pending manifest row for solve-phase tests"""
    return {
        "workday_name": workday_name,
        "objective_type": ObjectiveType.closed,
        "size": ScenarioSize.small,
        "type": ScenarioType.random,
        "timing": ScenarioTiming.uniform,
        "start_time": WorkdayStartTime.normal,
        "base_rate": 5,
        "nr_surges": 1,
        "repetition": 0,
        "nr_agents": 1,
        "seed": 123,
        "workday_length": 60.0,
        "status": ManifestStatus.pending,
        "result_file": None,
        "timestamp": None,
    }


def _write_manifest(outdir: Path, row: dict[str, Any]) -> None:
    """Write one manifest row in the same JSON form as the production code"""
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "manifest.json").open("w") as file:
        json.dump({row["workday_name"]: row}, file, default=str)


# --------------------------------------------------------------------------------------
# Deterministic input construction
# --------------------------------------------------------------------------------------


def test_build_fleet_applies_staggered_start_and_full_soc() -> None:
    """Staggered fleets start one-fifth of a workday apart at full charge"""
    generator = ModemsScenarioGenerator(seed=7)

    fleet = workday_benchmark._build_fleet(
        generator,
        nr_agents=3,
        start_time=WorkdayStartTime.staggered,
        workday_length=500.0,
    )

    assert [agent.time_initial for agent in fleet] == pytest.approx([0.0, 100.0, 200.0])
    assert all(agent.soc_initial == 1.0 for agent in fleet)


def test_build_workday_is_reproducible_from_seed() -> None:
    """The manifest seed reproduces both the fleet and arrival stream exactly"""
    first = workday_benchmark._build_workday(
        seed=19,
        nr_agents=2,
        scenario_type=ScenarioType.mixed,
        scenario_timing=ScenarioTiming.peaks,
        workday_length=240.0,
        base_rate_per_hour=1,
        nr_surges=2,
    )
    second = workday_benchmark._build_workday(
        seed=19,
        nr_agents=2,
        scenario_type=ScenarioType.mixed,
        scenario_timing=ScenarioTiming.peaks,
        workday_length=240.0,
        base_rate_per_hour=1,
        nr_surges=2,
    )

    _, first_agents, first_submissions = first
    _, second_agents, second_submissions = second
    assert [agent.to_dict() for agent in first_agents] == [
        agent.to_dict() for agent in second_agents
    ]
    assert [(time, request.to_dict()) for time, request in first_submissions] == [
        (time, request.to_dict()) for time, request in second_submissions
    ]


# --------------------------------------------------------------------------------------
# Generate and solve phases
# --------------------------------------------------------------------------------------


def test_generate_suite_creates_manifest_and_is_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Generation persists one row, and a second run reports it as existing"""
    monkeypatch.setattr(
        workday_benchmark,
        "RollingHorizonSimulator",
        lambda *args, **kwargs: object(),
    )
    kwargs = {
        "outdir": str(tmp_path),
        "scenario_sizes": [ScenarioSize.small],
        "scenario_types": [ScenarioType.random],
        "scenario_timings": [ScenarioTiming.uniform],
        "start_times": [WorkdayStartTime.normal],
        "base_rates": [5],
        "nr_surges": 1,
        "nr_repeats": 1,
        "base_seed": 11,
        "workday_length": 60.0,
    }

    first = generate_workday_suite(**kwargs)
    second = generate_workday_suite(**kwargs)
    manifest = read_workday_manifest(str(tmp_path))

    assert first == [{"workday_name": "WC0_SRUN5_1", "status": "created"}]
    assert second == [{"workday_name": "WC0_SRUN5_1", "status": "existing"}]
    assert set(manifest) == {"WC0_SRUN5_1"}
    assert manifest["WC0_SRUN5_1"]["status"] == "pending"


def test_generate_suite_bakes_objective_into_name_and_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An open workday gets the O letter, stores its objective, and screens with it"""
    screened: list[Any] = []
    monkeypatch.setattr(
        workday_benchmark,
        "RollingHorizonSimulator",
        lambda *args, **kwargs: screened.append(kwargs["objective"]),
    )
    generate_workday_suite(
        outdir=str(tmp_path),
        scenario_sizes=[ScenarioSize.small],
        scenario_types=[ScenarioType.random],
        scenario_timings=[ScenarioTiming.uniform],
        start_times=[WorkdayStartTime.normal],
        base_rates=[5],
        nr_surges=1,
        base_seed=11,
        workday_length=60.0,
        objective="open",
    )
    manifest = read_workday_manifest(str(tmp_path))
    assert set(manifest) == {"WO0_SRUN5_1"}
    assert (
        ObjectiveType(manifest["WO0_SRUN5_1"]["objective_type"]) == ObjectiveType.open
    )
    assert screened == [ObjectiveType.open]


def test_generate_suite_retries_infeasible_fleets_and_reports_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rejected fleet is retried with the next deterministic attempt seed"""
    attempts = 0

    def reject_first_fleet(*args: Any, **kwargs: Any) -> object:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("infeasible idle route")
        return object()

    events: list[dict[str, Any]] = []
    monkeypatch.setattr(
        workday_benchmark, "RollingHorizonSimulator", reject_first_fleet
    )

    result = generate_workday_suite(
        outdir=str(tmp_path),
        scenario_sizes=[ScenarioSize.small],
        scenario_types=[ScenarioType.random],
        scenario_timings=[ScenarioTiming.uniform],
        start_times=[WorkdayStartTime.normal],
        base_rates=[5],
        max_feasibility_attempts=2,
        on_progress=events.append,
    )

    assert attempts == 2
    assert result[0]["status"] == ManifestStatus.created
    assert [event["status"] for event in events] == [
        ManifestStatus.starting,
        ManifestStatus.created,
    ]
    assert all(event["phase"] == "generate" for event in events)


def test_generate_suite_does_not_persist_an_infeasible_combination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exhausting feasibility attempts returns infeasible without a manifest row"""

    def reject_fleet(*args: Any, **kwargs: Any) -> None:
        raise ValueError("infeasible idle route")

    monkeypatch.setattr(workday_benchmark, "RollingHorizonSimulator", reject_fleet)

    result = generate_workday_suite(
        outdir=str(tmp_path),
        scenario_sizes=[ScenarioSize.small],
        scenario_types=[ScenarioType.random],
        scenario_timings=[ScenarioTiming.uniform],
        start_times=[WorkdayStartTime.normal],
        max_feasibility_attempts=2,
    )

    assert result[0]["status"] == ManifestStatus.infeasible
    assert read_workday_manifest(str(tmp_path)) == {}


def test_solve_one_workday_forwards_manifest_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The solve adapter rebuilds once and forwards per-epoch solver budgets"""
    row = _manifest_row()
    generator = SimpleNamespace(network=object())
    agents = [object()]
    submissions = [(1.0, _request("request-a"))]
    observed: dict[str, Any] = {}

    monkeypatch.setattr(
        workday_benchmark,
        "_build_workday",
        lambda *args, **kwargs: (generator, agents, submissions),
    )

    def fake_compare(*args: Any, **kwargs: Any) -> dict[str, dict[str, float]]:
        observed["args"] = args
        observed["kwargs"] = kwargs
        return {"milp3": _summary(0.75), "alns": _summary(0.80)}

    monkeypatch.setattr(workday_benchmark, "compare_solvers_one_workday", fake_compare)
    logs: dict[SolverStrategy, WorkdayLog] = {}

    result = workday_benchmark._solve_one_workday(
        row,
        model_params={"rho": 2.5},
        milp_timelimit=2.0,
        alns_max_iter=40,
        solver_name="cbc",
        solver_config_type=workday_benchmark.SolverConfigType.cbc,
        workday_logs_out=logs,
    )

    assert observed["args"] == (generator.network, agents, submissions)
    assert observed["kwargs"]["workday_length"] == 60.0  # no buffer, drained instead
    assert observed["kwargs"]["milp_timelimit"] == 2.0
    assert observed["kwargs"]["alns_max_iter"] == 40
    assert observed["kwargs"]["workday_logs_out"] is logs
    assert observed["kwargs"]["objective"] == ObjectiveType.closed
    assert result["objective_type"] == ObjectiveType.closed
    assert result["nr_submissions"] == 1
    assert result["milp3"]["acceptance_rate"] == 0.75


def test_solve_suite_persists_result_log_and_done_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful solve atomically leaves report inputs and a done manifest row"""
    row = _manifest_row()
    _write_manifest(tmp_path, row)
    record = RequestRecord(
        _request("request-a"),
        submission_time=1.0,
        earliest_pickup=2.0,
        outcome=RequestOutcome.accepted,
        pickup_time=12.0,
        delivery_time=20.0,
        wait=1.0,
        delay=0.0,
        excess_ride_time=2.0,
    )

    def fake_solve(
        manifest_row: dict[str, Any],
        *args: Any,
        workday_logs_out: dict[SolverStrategy, WorkdayLog],
        **kwargs: Any,
    ) -> dict[str, Any]:
        workday_logs_out[SolverStrategy.milp3] = _workday_log(record)
        workday_logs_out[SolverStrategy.alns] = _workday_log(record)
        return {
            "workday_name": manifest_row["workday_name"],
            "nr_submissions": 1,
            "milp3": _summary(0.75),
            "alns": _summary(0.80),
        }

    monkeypatch.setattr(workday_benchmark, "_solve_one_workday", fake_solve)
    events: list[dict[str, Any]] = []

    outcomes = solve_workday_suite(str(tmp_path), on_progress=events.append)
    manifest = read_workday_manifest(str(tmp_path))[row["workday_name"]]

    assert outcomes == [{"workday_name": row["workday_name"], "status": "done"}]
    assert Path(manifest["result_file"]).is_file()
    assert Path(manifest["workday_log_file"]).is_file()
    assert manifest["accept_rate_milp3"] == 0.75
    assert manifest["accept_rate_alns"] == 0.80
    assert [event["status"] for event in events] == ["starting", "done"]
    assert solve_workday_suite(str(tmp_path)) == []


def test_solve_suite_records_failure_and_retry_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failures remain inspectable and are retried only when explicitly requested"""
    row = _manifest_row()
    _write_manifest(tmp_path, row)
    calls = 0

    def fail_solve(*args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("solver unavailable")

    monkeypatch.setattr(workday_benchmark, "_solve_one_workday", fail_solve)

    first = solve_workday_suite(str(tmp_path))
    skipped = solve_workday_suite(str(tmp_path))
    retried = solve_workday_suite(str(tmp_path), retry_failed=True)
    manifest = read_workday_manifest(str(tmp_path))[row["workday_name"]]

    assert first[0]["status"] == ManifestStatus.failed
    assert first[0]["error"] == "RuntimeError: solver unavailable"
    assert skipped == []
    assert retried[0]["status"] == ManifestStatus.failed
    assert calls == 2
    assert manifest["error"] == "RuntimeError: solver unavailable"


# --------------------------------------------------------------------------------------
# Build phase and report formatting
# --------------------------------------------------------------------------------------


def test_delta_percentage_points_handles_values_and_missing_data() -> None:
    """Acceptance deltas are fractions converted to percentage points"""
    assert workday_benchmark._delta_pp(0.8, 0.65) == pytest.approx(15.0)
    assert workday_benchmark._delta_pp(None, 0.65) is None
    assert workday_benchmark._delta_pp(0.8, None) is None


@pytest.mark.parametrize(
    ("minutes", "anchor", "expected"),
    [
        (None, "08:00", "--"),
        (12.25, None, "12.2"),
        (90.0, "08:00", "09:30"),
        (120.0, "23:30", "01:30"),
    ],
)
def test_format_clock(minutes: float | None, anchor: str | None, expected: str) -> None:
    """Clock display covers raw offsets, missing values, and midnight rollover"""
    assert workday_benchmark._format_clock(minutes, anchor) == expected


def test_build_workday_table_writes_all_formats_and_pending_rows(
    tmp_path: Path,
) -> None:
    """Build is read-only with respect to solving and retains unfinished rows"""
    done = _manifest_row("W_done")
    pending = _manifest_row("W_pending")
    pending["status"] = ManifestStatus.pending
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "nr_submissions": 8,
                "milp3": _summary(0.625),
                "alns": _summary(0.75),
            }
        )
    )
    done["status"] = ManifestStatus.done
    done["result_file"] = str(result_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    with (tmp_path / "manifest.json").open("w") as file:
        json.dump({"W_done": done, "W_pending": pending}, file, default=str)

    rows = build_workday_summary_table(str(tmp_path), rows_per_block=1)

    by_name = {row["workday"]: row for row in rows}
    assert by_name["W_done"]["accept_delta_pp"] == pytest.approx(12.5)
    assert by_name["W_pending"]["nr_submissions"] is None
    for suffix in ("csv", "json", "tex"):
        assert (tmp_path / f"summary_table.{suffix}").is_file()
    assert (tmp_path / "summary_table.tex").read_text().count(r"\begin{table*}") == 2


def test_build_requests_table_requires_a_solved_log(tmp_path: Path) -> None:
    """The request report fails clearly if a solved log is missing"""
    _write_manifest(tmp_path, _manifest_row())

    with pytest.raises(ValueError, match="run the solve phase first"):
        build_workday_requests_table(str(tmp_path), "WC0_SRUN5_1")


def test_build_requests_table_joins_by_id_and_preserves_raw_json_times(
    tmp_path: Path,
) -> None:
    """Paired request rows join on persistent IDs, not list position or node name"""
    accepted = RequestRecord(
        _request("request-a"),
        submission_time=5.0,
        earliest_pickup=10.0,
        outcome=RequestOutcome.accepted,
        pickup_time=15.0,
        delivery_time=25.0,
        wait=2.0,
        delay=1.0,
        excess_ride_time=3.0,
    )
    rejected = RequestRecord(
        _request("request-b", pickup=2, delivery=3),
        submission_time=10.0,
        earliest_pickup=20.0,
        outcome=RequestOutcome.rejected,
    )
    alns_accepted = RequestRecord(
        _request("request-a"),
        submission_time=5.0,
        earliest_pickup=12.0,
        outcome=RequestOutcome.accepted,
        pickup_time=14.0,
        delivery_time=24.0,
        wait=1.0,
        delay=0.0,
        excess_ride_time=2.0,
    )
    sidecar = tmp_path / "workday_log.json"
    sidecar.write_text(
        json.dumps(
            {
                "milp3": _workday_log(accepted, rejected).to_dict(),
                "alns": _workday_log(alns_accepted).to_dict(),
            },
            default=str,
        )
    )
    manifest_row = _manifest_row()
    manifest_row["status"] = ManifestStatus.done
    manifest_row["workday_log_file"] = str(sidecar)
    _write_manifest(tmp_path, manifest_row)

    rows = build_workday_requests_table(
        str(tmp_path),
        manifest_row["workday_name"],
        rows_per_block=1,
        clock_display_start="08:00",
    )

    assert [row["request_id"] for row in rows] == ["request-a", "request-b"]
    assert rows[0]["pickup_time_alns"] == 14.0
    assert rows[1]["outcome_alns"] is None
    json_rows = json.loads(
        (tmp_path / "requests_tables" / "WC0_SRUN5_1_requests_table.json").read_text()
    )
    assert json_rows[0]["pickup_time_milp3"] == 15.0
    with (tmp_path / "requests_tables" / "WC0_SRUN5_1_requests_table.csv").open(
        newline=""
    ) as file:
        csv_rows = list(csv.DictReader(file))
    assert csv_rows[0]["submission_time"] == "08:05"
    assert csv_rows[0]["earliest_pickup"] == "08:10"
    assert csv_rows[0]["pickup_time_milp3"] == "08:15"
    latex = (
        tmp_path / "requests_tables" / "WC0_SRUN5_1_requests_table.tex"
    ).read_text()
    assert latex.count(r"\begin{table*}") == 2


def test_plot_workday_soc_acceptance_creates_nonempty_png(tmp_path: Path) -> None:
    """The plotting helper supports a minimal log and closes after saving"""
    log = _workday_log(
        RequestRecord(
            _request("request-a"),
            submission_time=5.0,
            earliest_pickup=10.0,
            outcome=RequestOutcome.accepted,
        ),
        RequestRecord(
            _request("request-b", pickup=2, delivery=3),
            submission_time=10.0,
            earliest_pickup=15.0,
            outcome=RequestOutcome.rejected,
        ),
    )

    output = Path(plot_workday_soc_acceptance(log, str(tmp_path), "sample"))

    assert output == tmp_path / "sample_soc_acceptance.png"
    assert output.stat().st_size > 0


# --------------------------------------------------------------------------------------
# End to end: generate / solve (real MILP3 and ALNS) / build / plot
# --------------------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("objective", [ObjectiveType.closed, ObjectiveType.open])
def test_workday_pipeline_end_to_end(tmp_path: Path, objective: ObjectiveType) -> None:
    outdir = str(tmp_path)
    (created,) = generate_workday_suite(
        outdir,
        scenario_sizes=[ScenarioSize.small],
        scenario_types=[ScenarioType.random],
        scenario_timings=[ScenarioTiming.uniform],
        start_times=[WorkdayStartTime.staggered],
        base_rates=[8],
        nr_surges=0,
        workday_length=45.0,
        objective=objective,
    )
    assert created["status"] == ManifestStatus.created
    name = created["workday_name"]

    (solved,) = solve_workday_suite(outdir, milp_timelimit=5.0, alns_max_iter=10)
    assert solved == {"workday_name": name, "status": ManifestStatus.done}
    row = read_workday_manifest(outdir)[name]
    assert 0.0 <= row["accept_rate_milp3"] <= 1.0

    (summary,) = build_workday_summary_table(outdir)
    assert (summary["workday"], summary["objective_type"]) == (name, objective)
    assert summary["status"] == ManifestStatus.done
    assert summary["accept_delta_pp"] == pytest.approx(
        100.0 * (summary["accept_alns"] - summary["accept_milp3"])
    )
    requests = build_workday_requests_table(outdir, name, clock_display_start="08:00")
    with open(row["workday_log_file"]) as f:
        logs = {k: WorkdayLog.from_dict(v) for k, v in json.load(f).items()}
    assert set(logs) == {"milp3", "alns"}
    assert len(requests) == len(logs["alns"].requests) == len(logs["milp3"].requests)
    for solver_key, log in logs.items():
        outcomes = {record.outcome for record in log.requests}
        assert RequestOutcome.unresolved not in outcomes
        path = plot_workday_soc_acceptance(
            log, str(tmp_path / "plots"), f"{name}_{solver_key}"
        )
        assert Path(path).stat().st_size > 0
    assert solve_workday_suite(outdir) == []
