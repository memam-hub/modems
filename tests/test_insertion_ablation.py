"""
Tests for modems.insertion_ablation: the V0-V3 insertion variants must be
interchangeable (identical candidates), and the resumable generate / solve / build /
compare pipeline behind benchmarks/run_ablation_suite.py must work end-to-end
"""

from __future__ import annotations

import csv
import json
import math
from typing import Any

import pytest

from modems.algorithms import _sort_key, alns_feasible_insertions
from modems.benchmark import ManifestPhase, ManifestStatus
from modems.core import DEFAULT_PARAMS_OBJ, ScenarioTiming, ScenarioType
from modems.generator import SocRangeSpec
from modems.insertion_ablation import (
    INSERTION_VARIANT_FCNS,
    InsertionAblationMetric,
    InsertionVariant,
    build_ablation_table,
    build_measurement_scenario,
    compare_route_length_buckets,
    generate_ablation_suite,
    measure_variant,
    read_ablation_manifest,
    solve_ablation_suite,
    write_corpus_comparison,
)

UNIFORM_RANDOM = dict(
    scenario_types=[ScenarioType.random], scenario_timings=[ScenarioTiming.uniform]
)


def point(
    seed: int,
    nr_agents: int,
    nr_requests: int,
    soc=None,
    sc_type="random",
    timing="uniform",
):
    solution, probes = build_measurement_scenario(
        seed,
        nr_agents,
        nr_requests,
        ScenarioType(sc_type),
        ScenarioTiming(timing),
        DEFAULT_PARAMS_OBJ,
        nr_probes=3,
        soc_range=soc or SocRangeSpec.normal(),
    )
    assert solution is not None
    return solution, probes


def signature(candidates) -> dict:
    return {
        (c.journey.agent_name, tuple(c.journey.route)): round(c.delta_obj, 7)
        for c in candidates
    }


# --------------------------------------------------------------------------------------
# Variants are interchangeable
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seed, nr_agents, nr_requests, soc, sc_type, timing",
    [
        (1, 1, 10, None, "random", "uniform"),
        (2, 2, 14, None, "clustered", "peaks"),
        (3, 3, 18, None, "mixed", "uniform"),
        (4, 1, 12, SocRangeSpec.stress(), "random", "peaks"),
        (5, 2, 12, SocRangeSpec.stress(), "clustered", "uniform"),
    ],
)
def test_all_variants_return_identical_candidates(
    seed, nr_agents, nr_requests, soc, sc_type, timing
) -> None:
    solution, probes = point(seed, nr_agents, nr_requests, soc, sc_type, timing)
    for request_name in probes:
        results = {
            v: signature(INSERTION_VARIANT_FCNS[v](solution, request_name))
            for v in InsertionVariant
        }
        assert results[InsertionVariant.v0] == results[InsertionVariant.v1]
        assert results[InsertionVariant.v0] == results[InsertionVariant.v2]
        assert results[InsertionVariant.v0] == results[InsertionVariant.v3]


@pytest.mark.parametrize("variant", list(InsertionVariant))
def test_variant_stats_count_every_returned_candidate(
    variant: InsertionVariant,
) -> None:
    solution, probes = point(2, 2, 12)
    for request_name in probes:
        stats: dict[str, int] = {}
        candidates = INSERTION_VARIANT_FCNS[variant](
            solution, request_name, stats=stats
        )
        assert stats.get("candidates_accepted", 0) == len(candidates)
        considered = stats.get("position_pairs_considered", 0)
        assert considered == stats.get("candidates_accepted", 0) + stats.get(
            "candidates_rejected", 0
        )


def test_production_variant_is_algorithm3_with_propagation_counters() -> None:
    solution, probes = point(3, 1, 12)
    stats: dict[str, int] = {}
    candidates = INSERTION_VARIANT_FCNS[InsertionVariant.v3](
        solution, probes[0], stats=stats
    )
    assert signature(candidates) == signature(
        alns_feasible_insertions(solution, probes[0])
    )
    assert (
        stats["prefix_node_propagations"] > 0
        and stats["delivery_node_propagations"] > 0
    )
    assert stats["pickup_scans_started"] > 0


# --------------------------------------------------------------------------------------
# Measurement points
# --------------------------------------------------------------------------------------


def test_measurement_point_holds_out_the_last_greedy_requests_as_probes() -> None:
    solution, probes = point(1, 1, 8)
    ctx = solution.ctx
    assert len(probes) == 3
    assert probes == sorted(ctx.request_names, key=lambda r: _sort_key(ctx, r))[-3:]
    assert solution.accepted.isdisjoint(probes)
    assert solution.accepted  # the rest was offered to greedy_complete


def test_measurement_point_keeps_at_least_one_seed_request() -> None:
    solution, probes = point(1, 1, 2)
    assert len(probes) == 1


def test_measurement_point_is_none_without_a_feasible_base_plan() -> None:
    result = build_measurement_scenario(
        1, 3, 5, ScenarioType.random, ScenarioTiming.uniform, DEFAULT_PARAMS_OBJ,
        soc_range=SocRangeSpec.custom(0.25, 0.25),
    )  # fmt: skip
    assert result == (None, None)


def test_measure_variant_reports_time_candidates_and_stats() -> None:
    solution, probes = point(2, 2, 10)
    measured = measure_variant(solution, probes[0], InsertionVariant.v3, nr_repeats=3)
    expected = len(alns_feasible_insertions(solution, probes[0]))
    assert measured["variant_name"] == "V3_production"
    assert measured["nr_candidates"] == measured["candidates_accepted"] == expected
    assert measured["t_exe"] > 0.0


# --------------------------------------------------------------------------------------
# Pipeline: generate / solve / build
# --------------------------------------------------------------------------------------


def generate(outdir, **kwargs: Any) -> list[dict]:
    arguments = dict(request_counts=[6], agent_counts=[1], **UNIFORM_RANDOM)
    return generate_ablation_suite(str(outdir), **{**arguments, **kwargs})


def test_generation_names_points_and_is_resumable(tmp_path) -> None:
    events: list[dict] = []
    summary = generate(
        tmp_path, request_counts=[6, 8], nr_repeats=2, on_progress=events.append
    )
    names = ["I0_a1r6RU80_100", "I1_a1r6RU80_100", "I0_a1r8RU80_100", "I1_a1r8RU80_100"]
    assert summary == [
        {"point_name": n, "status": ManifestStatus.created} for n in names
    ]
    assert len(events) == 8 and events[0]["phase"] == ManifestPhase.generate
    manifest = read_ablation_manifest(str(tmp_path))
    assert set(manifest) == set(names)
    assert all(
        row["status"] == "pending" and row["soc_range"] == {"kind": "normal"}
        for row in manifest.values()
    )

    before = (tmp_path / "manifest.json").read_text()
    assert {
        s["status"] for s in generate(tmp_path, request_counts=[6, 8], nr_repeats=2)
    } == {ManifestStatus.existing}
    assert (tmp_path / "manifest.json").read_text() == before


def test_corpora_with_different_soc_ranges_share_one_manifest(tmp_path) -> None:
    generate(tmp_path)
    generate(tmp_path, soc_range=SocRangeSpec.stress())
    manifest = read_ablation_manifest(str(tmp_path))
    assert set(manifest) == {"I0_a1r6RU80_100", "I0_a1r6RU50_70"}
    assert manifest["I0_a1r6RU50_70"]["soc_range"] == {"kind": "stress"}


def test_generation_marks_unfillable_points_infeasible(tmp_path) -> None:
    summary = generate(
        tmp_path,
        agent_counts=[3],
        soc_range=SocRangeSpec.custom(0.25, 0.25),
        max_feasibility_attempts=2,
    )
    assert summary == [
        {"point_name": "I0_a3r6RU25_25", "status": ManifestStatus.infeasible}
    ]
    assert read_ablation_manifest(str(tmp_path)) == {}


def test_pipeline_measures_every_variant_and_builds_the_table(tmp_path) -> None:
    generate(tmp_path, request_counts=[6, 12])
    events: list[dict] = []
    outcomes = solve_ablation_suite(
        str(tmp_path),
        max_requests_for_naive=8,
        nr_measure_repeats=1,
        on_progress=events.append,
    )
    assert [o["status"] for o in outcomes] == [ManifestStatus.done] * 2
    assert len(events) == 4
    assert solve_ablation_suite(str(tmp_path)) == []  # resumable: nothing left

    manifest = read_ablation_manifest(str(tmp_path))
    small = json.loads(open(manifest["I0_a1r6RU80_100"]["result_file"]).read())
    large = json.loads(open(manifest["I0_a1r12RU80_100"]["result_file"]).read())
    assert all(
        not small["variant_results"][v.value]["skipped"] for v in InsertionVariant
    )
    assert large["variant_results"]["V0_naive"]["skipped"]
    assert large["variant_results"]["V1_prefiltered"]["skipped"]
    assert not large["variant_results"]["V3_production"]["skipped"]
    means = {
        small["variant_results"][v.value]["mean_nr_candidates"]
        for v in InsertionVariant
    }
    assert len(means) == 1  # every variant finds the same candidates

    rows = {
        r["point_name"]: r
        for r in build_ablation_table(str(tmp_path), rows_per_block=1)
    }
    small_row, large_row = rows["I0_a1r6RU80_100"], rows["I0_a1r12RU80_100"]
    assert small_row[InsertionAblationMetric.speedup_V0_V3] == pytest.approx(
        small_row[InsertionAblationMetric.t_exe_V0]
        / small_row[InsertionAblationMetric.t_exe_V3]
    )
    assert large_row[InsertionAblationMetric.t_exe_V0] is None
    assert large_row[InsertionAblationMetric.speedup_V0_V3] is None
    assert large_row[InsertionAblationMetric.speedup_V2_V3] is not None
    assert (tmp_path / "ablation_table.tex").read_text().count(r"\begin{table") == 2
    with open(tmp_path / "ablation_table.csv") as f:
        assert len(list(csv.DictReader(f))) == 2


def test_table_lists_pending_points_without_metrics(tmp_path) -> None:
    generate(tmp_path)
    (row,) = build_ablation_table(str(tmp_path))
    assert row["status"] == "pending"
    assert all(row[m.value] is None for m in InsertionAblationMetric)


# --------------------------------------------------------------------------------------
# Corpus comparison
# --------------------------------------------------------------------------------------


def _row(
    route_length: float | None, speedup: float | None, status: str = "done"
) -> dict:
    return {
        "status": status,
        "mean_route_length": route_length,
        "speedup_V2_V3": speedup,
        "t_exe_V3": 1.0,
    }


def test_route_length_buckets_compare_means_side_by_side() -> None:
    rows_a = [
        _row(3.0, 2.0),
        _row(8.0, 4.0),
        _row(12.0, 5.0),
        _row(15.0, 1.0, "pending"),
        _row(None, 9.0),
    ]
    rows_b = [_row(5.0, 9.0), _row(25.0, 3.0), _row(26.0, None)]
    result = compare_route_length_buckets(rows_a, rows_b, bucket_size=10)
    assert result == [
        {
            "route_bucket_start": 0,
            "route_bucket_end": 10,
            "metric": "speedup_V2_V3",
            "nr_a": 2,
            "mean_a": 3.0,
            "nr_b": 1,
            "mean_b": 9.0,
            "ratio_b_to_a": 3.0,
        },
        {
            "route_bucket_start": 10,
            "route_bucket_end": 20,
            "metric": "speedup_V2_V3",
            "nr_a": 1,
            "mean_a": 5.0,
            "nr_b": 0,
            "mean_b": None,
            "ratio_b_to_a": None,
        },
        {
            "route_bucket_start": 20,
            "route_bucket_end": 30,
            "metric": "speedup_V2_V3",
            "nr_a": 0,
            "mean_a": None,
            "nr_b": 1,
            "mean_b": 3.0,
            "ratio_b_to_a": None,
        },
    ]


def test_route_length_buckets_use_the_requested_metric() -> None:
    (bucket,) = compare_route_length_buckets(
        [_row(3.0, 2.0)], [_row(4.0, 7.0)], metric="t_exe_V3"
    )
    assert (bucket["metric"], bucket["mean_a"], bucket["ratio_b_to_a"]) == (
        "t_exe_V3",
        1.0,
        1.0,
    )


def test_corpus_comparison_end_to_end(tmp_path) -> None:
    for name, soc in (
        ("normal", SocRangeSpec.normal()),
        ("stress", SocRangeSpec.stress()),
    ):
        generate(tmp_path / name, soc_range=soc)
        solve_ablation_suite(str(tmp_path / name), nr_measure_repeats=1)
    comparison = write_corpus_comparison(
        str(tmp_path / "normal"),
        str(tmp_path / "stress"),
        str(tmp_path / "cmp"),
        bucket_size=100,
    )
    (bucket,) = comparison
    assert bucket["nr_a"] == bucket["nr_b"] == 1
    assert math.isclose(bucket["ratio_b_to_a"], bucket["mean_b"] / bucket["mean_a"])
    assert json.loads(
        (tmp_path / "cmp" / "corpus_comparison.json").read_text()
    ) == json.loads(json.dumps(comparison, default=str))
    assert (tmp_path / "cmp" / "corpus_comparison.csv").exists()
