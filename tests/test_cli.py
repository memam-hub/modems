"""
End-to-end runs of <benchmarks/> command-line scripts on the smallest (--smoke) configs,
with the full generate -> solve -> build pipeline; checks argument wiring, exit codes,
and generated files. Each run takes a few seconds (typically less than 2 minutes)
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"

pytestmark = pytest.mark.integration


def run(script: str, *args: str) -> str:
    result = subprocess.run(
        [sys.executable, str(BENCHMARKS / script), *args],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_run_suite(tmp_path: Path) -> None:
    common = [
        "--outdir",
        str(tmp_path),
        "--sizes",
        "s",
        "--types",
        "R",
        "--timings",
        "u",
    ]
    common += [
        "--nr-repeats",
        "1",
        "--seed",
        "2",
        "--soc-test",
        "normal",
        "--objective",
        "o",
    ]
    run("run_suite.py", *common, "--phase", "generate")
    assert os.listdir(tmp_path / "scenarios") == ["SO0_SRU80_100.json"]
    output = run(
        "run_suite.py",
        *common,
        *("--milp-timelimit", "5", "--alns-max-iter", "10"),
        *("--export-formats", "csv,tex"),
    )  # phase "all" resumes: generation is skipped, solving and building run
    assert "1 exist (skipped)" in output and "4 solved, 0 failed" in output
    with open(tmp_path / "benchmark_table.csv") as f:
        rows = list(csv.DictReader(f))
    assert [row["solver"] for row in rows] == ["alns", "milp1", "milp2", "milp3"]
    assert {row["objective_type"] for row in rows} == {"open"}
    assert (tmp_path / "benchmark_table.tex").exists()
    assert not (tmp_path / "benchmark_table.json").exists()
    assert (tmp_path / "solver_comparison.png").stat().st_size > 0


def test_run_ablation_suite_and_compare_corpora(tmp_path: Path) -> None:
    run(
        "run_ablation_suite.py",
        *("--outdir", str(tmp_path), "--types", "random", "--timings", "uniform"),
        *("--request-counts", "6", "--agent-counts", "1", "--nr-measure-repeats", "1"),
        *("--nr-repeats", "1", "--corpus", "both", "--bucket-size", "100"),
        *("--export-formats", "all"),
    )
    for corpus in ("normal", "stress"):
        rows = json.loads((tmp_path / corpus / "ablation_table.json").read_text())
        assert [row["status"] for row in rows] == ["done"]
    assert (tmp_path / "comparison" / "corpus_comparison.tex").exists()
    assert (tmp_path / "comparison" / "normal_vs_stress.png").stat().st_size > 0
    for corpus in ("normal", "stress"):
        assert (tmp_path / corpus / "insertion_variants.png").stat().st_size > 0

    run(
        "compare_ablation_corpora.py",
        *(
            "--corpus-a",
            str(tmp_path / "normal"),
            "--corpus-b",
            str(tmp_path / "stress"),
        ),
        *("--outdir", str(tmp_path / "again"), "--bucket-size", "100"),
    )
    assert json.loads(
        (tmp_path / "again" / "corpus_comparison.json").read_text()
    ) == json.loads((tmp_path / "comparison" / "corpus_comparison.json").read_text())


def test_run_workday_suite(tmp_path: Path) -> None:
    run(
        "run_workday_suite.py",
        *("--outdir", str(tmp_path), "--sizes", "small", "--types", "random"),
        *("--timings", "uniform", "--start-times", "normal", "--base-rates", "5"),
        *("--nr-surges", "0", "--workday", "30", "--milp-timelimit", "5"),
        *("--alns-max-iter", "10", "--clock-display-start", "08:00", "--plots"),
    )
    (row,) = json.loads((tmp_path / "summary_table.json").read_text())
    assert row["status"] == "done"
    name = row["workday"]
    assert (tmp_path / "requests_tables" / f"{name}_requests_table.csv").exists()
    assert sorted(os.listdir(tmp_path / "plots")) == [
        f"{name}_alns_soc_acceptance.png",
        f"{name}_milp3_soc_acceptance.png",
    ]


@pytest.mark.parametrize(
    "script, flag",
    [("run_suite.py", "--soc-test"), ("run_ablation_suite.py", "--corpus")],
)
def test_scripts_reject_invalid_custom_soc_bounds(
    tmp_path: Path, script: str, flag: str
) -> None:
    """argparse-level rejection: exit code 2, nothing written"""
    result = subprocess.run(
        [sys.executable, str(BENCHMARKS / script), "--outdir", str(tmp_path)]
        + [flag, "custom", "--soc-custom", "0.9", "0.5"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 2
    assert "0 < lb <= ub <= 1.0" in result.stderr
    assert not (tmp_path / "manifest.json").exists()


def test_compare_workday_objectives(tmp_path: Path) -> None:
    common = ["--sizes", "small", "--types", "random", "--timings", "uniform"]
    common += ["--start-times", "normal", "staggered", "--base-rates", "5"]
    common += ["--nr-surges", "0", "--workday", "30", "--milp-timelimit", "2"]
    common += ["--alns-max-iter", "10"]
    for objective in ("closed", "open"):
        run(
            "run_workday_suite.py",
            *("--outdir", str(tmp_path / objective), "--objective", objective),
            *common,
        )
    output = run(
        "compare_workday_objectives.py",
        *("--closed", str(tmp_path / "closed"), "--open", str(tmp_path / "open")),
        *("--outdir", str(tmp_path / "comparison")),
    )
    assert "2 workdays paired across objectives" in output
    assert "1 workdays paired across start times" in output
    rows = json.loads((tmp_path / "comparison" / "combined_summary.json").read_text())
    assert [r["objective_type"] for r in rows] == ["closed", "open", "closed", "open"]
    figures = sorted(p.name for p in (tmp_path / "comparison").glob("*.png"))
    most_demand = sorted(
        p.name for p in (tmp_path / "comparison" / "most_demand").iterdir()
    )
    assert "demand_vs_accept_delay.png" in figures
    assert "var_decision_impact.png" in figures
    assert "var_decision_boxes.png" in figures
    assert any(f.startswith("most_requests_") for f in most_demand)
    assert any(f.startswith("most_passengers_") for f in most_demand)
    assert any(f.endswith("_requests_table.tex") for f in most_demand)


@pytest.mark.parametrize("script", ["run_suite.py", "run_workday_suite.py"])
def test_scripts_reject_unknown_export_formats(tmp_path: Path, script: str) -> None:
    result = subprocess.run(
        [sys.executable, str(BENCHMARKS / script), "--outdir", str(tmp_path)]
        + ["--export-formats", "csv,xlsx"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 2 and "choose from csv,json,tex or all" in result.stderr
