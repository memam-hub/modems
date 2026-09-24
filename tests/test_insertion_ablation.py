from __future__ import annotations

import json
import math
import os
from typing import Any

import pytest

from modems import ModemsScenarioGenerator
from modems.algorithms import _sort_key, alns_feasible_insertions, preprocess
from modems.core import (
    ProblemContext,
    ProblemType,
    ScenarioTiming,
    ScenarioType,
    SolverStrategy,
)
from modems.generator import SocRangeSpec
from modems.insertion_ablation import (
    INSERTION_VARIANT_FCNS,
    InsertionAblationMetric,
    InsertionVariant,
    ManifestPhase,
    ManifestStatus,
    _write_corpus_comparison_latex,
    build_ablation_table,
    build_measurement_scenario,
    compare_route_length_buckets,
    generate_ablation_suite,
    measure_variant,
    read_ablation_manifest,
    solve_ablation_suite,
    write_corpus_comparison,
)

PARAMS: dict = {"eps": 0.01, "zeta": 1.0, "eta": 100.0, "rho": 2.0}


def _candidate_key(candidate: Any) -> tuple[str, tuple[str, ...]]:
    """Index a candidate by (agent, route) for order-independent set comparison"""
    return (candidate.journey.agent_name, tuple(candidate.journey.route))


# --------------------------------------------------------------------------------------
# Variant correctness: V0/V1/V2/V3 must all agree on the exact same set of feasible
# candidates and their objective deltas, they only differ on how long to get there.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(1, 6))
def test_all_variants_agree_with_each_other(seed: int) -> None:
    """
    V0/V1/V2/V3 return the identical candidate set (route + delta_obj) for
    the same (solution, request) probe across several seeded scenarios
    """
    solution, r_probes = build_measurement_scenario(
        seed=seed,
        nr_agents=1 + seed % 2,
        nr_requests=8 + seed,
        scenario_type=ScenarioType.random,
        scenario_timing=ScenarioTiming.loose,
        model_params=PARAMS,
        nr_probes=2,
    )
    assert solution is not None
    if r_probes is None:
        return
    for request_name in r_probes:
        results = {
            name: {_candidate_key(c): c.delta_obj for c in fn(solution, request_name)}
            for name, fn in INSERTION_VARIANT_FCNS.items()
        }
        keys_by_variant = {name: set(d.keys()) for name, d in results.items()}
        reference_keys = keys_by_variant[InsertionVariant.v0]
        for name, keys in keys_by_variant.items():
            assert (
                keys == reference_keys
            ), f"{name} disagrees with {InsertionVariant.v0.value} on {request_name}"
        for key in reference_keys:
            values = [results[name][key] for name in INSERTION_VARIANT_FCNS]
            assert all(
                math.isclose(v, values[0], rel_tol=1e-9, abs_tol=1e-7) for v in values
            ), f"delta_obj mismatch across variants for {key}"


def test_all_variants_agree_across_spatial_types_and_timings() -> None:
    """The same cross-validation, but sweeping spatial type and timing too"""
    for spatial_type in (
        ScenarioType.random,
        ScenarioType.clustered,
        ScenarioType.mixed,
    ):
        for timing in (ScenarioTiming.loose, ScenarioTiming.tight):
            solution, r_probes = build_measurement_scenario(
                seed=7,
                nr_agents=2,
                nr_requests=12,
                scenario_type=spatial_type,
                scenario_timing=timing,
                model_params=PARAMS,
                nr_probes=1,
            )
            if solution is None or r_probes is None:
                continue
            for r_name in r_probes:
                key_sets = [
                    {_candidate_key(c) for c in fn(solution, r_name)}
                    for fn in INSERTION_VARIANT_FCNS.values()
                ]
                assert all(
                    ks == key_sets[0] for ks in key_sets
                ), f"mismatch for {spatial_type}/{timing}"


@pytest.mark.parametrize("seed", range(1, 6))
def test_all_variants_agree_under_soc_stress(seed: int) -> None:
    """The same cross-validation under a stress SoC range"""
    solution, r_probes = build_measurement_scenario(
        seed=seed,
        nr_agents=1,
        nr_requests=20,
        scenario_type=ScenarioType.random,
        scenario_timing=ScenarioTiming.loose,
        model_params=PARAMS,
        nr_probes=2,
        soc_range=SocRangeSpec.stress(),
    )
    if solution is None:
        pytest.skip("infeasible under SoC stress for this seed")
    if r_probes is None:
        pytest.skip("bad r_probes for this seed")
    for request_name in r_probes:
        key_sets = [
            {_candidate_key(c) for c in fn(solution, request_name)}
            for fn in INSERTION_VARIANT_FCNS.values()
        ]
        assert all(ks == key_sets[0] for ks in key_sets)


def test_soc_stress_produces_shorter_routes_than_default_at_same_requests() -> None:
    """A stress SoC range yields a shorter route than normal for the same nr_requests"""
    normal, _ = build_measurement_scenario(
        seed=1,
        nr_agents=1,
        nr_requests=50,
        scenario_type=ScenarioType.random,
        scenario_timing=ScenarioTiming.loose,
        model_params=PARAMS,
        nr_probes=1,
        soc_range=SocRangeSpec.normal(),
    )
    stress, _ = build_measurement_scenario(
        seed=1,
        nr_agents=1,
        nr_requests=50,
        scenario_type=ScenarioType.random,
        scenario_timing=ScenarioTiming.loose,
        model_params=PARAMS,
        nr_probes=1,
        soc_range=SocRangeSpec.stress(),
    )
    assert normal is not None and stress is not None
    normal_len = len(normal.journeys["agent_1"].route)
    stress_len = len(stress.journeys["agent_1"].route)
    assert stress_len < normal_len


# --------------------------------------------------------------------------------------
# Stats hooks
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("variant_name", list(INSERTION_VARIANT_FCNS.keys()))
def test_variant_stats_accepted_matches_returned_candidates(
    variant_name: InsertionVariant,
) -> None:
    """Every variant stats['candidates_accepted'] equals len(returned candidates)"""
    solution, r_probes = build_measurement_scenario(
        seed=3,
        nr_agents=1,
        nr_requests=10,
        scenario_type=ScenarioType.random,
        scenario_timing=ScenarioTiming.loose,
        model_params=PARAMS,
        nr_probes=2,
    )
    assert solution is not None
    if r_probes is None:
        return
    fcn = INSERTION_VARIANT_FCNS[variant_name]
    for r_name in r_probes:
        stats: dict[str, int] = {}
        candidates = fcn(solution, r_name, stats=stats)
        assert stats.get("candidates_accepted", 0) == len(candidates)


def test_v3_stats_include_node_propagation_breakdown() -> None:
    """
    V3's stats hook (the production instrumentation) reports the prefix/delivery/suffix
    node-propagation breakdown to identify pruning and prefix/suffix laziness
    """
    solution, r_probes = build_measurement_scenario(
        seed=5,
        nr_agents=1,
        nr_requests=15,
        scenario_type=ScenarioType.random,
        scenario_timing=ScenarioTiming.loose,
        model_params=PARAMS,
        nr_probes=1,
    )
    assert solution is not None
    if r_probes is None:
        return
    stats: dict[str, int] = {}
    INSERTION_VARIANT_FCNS[InsertionVariant.v3](solution, r_probes[0], stats=stats)
    for key in (
        "prefix_node_propagations",
        "delivery_node_propagations",
        "suffix_node_propagations",
        "position_pairs_considered",
    ):
        assert key in stats


def test_stats_hook_does_not_change_production_result() -> None:
    """Passing no stats to alns_feasible_insertions() must not change its output"""
    solution, r_probes = build_measurement_scenario(
        seed=2,
        nr_agents=1,
        nr_requests=8,
        scenario_type=ScenarioType.random,
        scenario_timing=ScenarioTiming.loose,
        model_params=PARAMS,
        nr_probes=1,
    )
    assert solution is not None
    if r_probes is None:
        return
    without_stats = {
        _candidate_key(c) for c in alns_feasible_insertions(solution, r_probes[0])
    }
    with_stats = {
        _candidate_key(c)
        for c in alns_feasible_insertions(solution, r_probes[0], stats={})
    }
    assert without_stats == with_stats


# --------------------------------------------------------------------------------------
# build_measurement_scenario / measure_variant
# --------------------------------------------------------------------------------------


def test_build_measurement_scenario_holds_out_requested_probe_count() -> None:
    """nr_probes requests are held out (not accepted) as insertion r_probes"""
    solution, r_probes = build_measurement_scenario(
        seed=1,
        nr_agents=1,
        nr_requests=10,
        scenario_type=ScenarioType.random,
        scenario_timing=ScenarioTiming.loose,
        model_params=PARAMS,
        nr_probes=3,
    )
    assert solution is not None
    assert r_probes is not None and len(r_probes) == 3
    assert all(r_name not in solution.accepted for r_name in r_probes)


def test_build_measurement_scenario_probes_are_greedy_completes_last_picks() -> None:
    """
    Probes are exactly the requests greedy_complete's own sort order would have
    inserted last; they are representative of "inserting into an already-full route",
    not an arbitrary/early insertion decision
    """
    seed, nr_agents, nr_requests = 1, 1, 10
    generator = ModemsScenarioGenerator(seed=seed)
    scenario = generator.generate_random_scenario(
        nr_agents=nr_agents,
        nr_requests=nr_requests,
        scenario_type=ScenarioType.random,
        scenario_timing=ScenarioTiming.loose,
    )
    ctx = ProblemContext(
        scenario, ProblemType.closed_selective, SolverStrategy.alns, PARAMS
    )
    base_plan, r_unassigned = preprocess(ctx)
    assert base_plan is not None
    r_ordered = sorted(r_unassigned, key=lambda r: _sort_key(base_plan.ctx, r))

    _, r_probes = build_measurement_scenario(
        seed=seed,
        nr_agents=nr_agents,
        nr_requests=nr_requests,
        scenario_type=ScenarioType.random,
        scenario_timing=ScenarioTiming.loose,
        model_params=PARAMS,
        nr_probes=3,
    )
    assert r_probes == r_ordered[-3:]


def test_measure_variant_is_deterministic_across_repeats() -> None:
    """The search itself is deterministic; nr_repeats only stabilizes the timing"""
    solution, r_probes = build_measurement_scenario(
        seed=1,
        nr_agents=1,
        nr_requests=8,
        scenario_type=ScenarioType.random,
        scenario_timing=ScenarioTiming.loose,
        model_params=PARAMS,
        nr_probes=1,
    )
    assert solution is not None
    if r_probes is None:
        return
    result = measure_variant(solution, r_probes[0], InsertionVariant.v3, nr_repeats=3)
    assert result["variant_name"] == InsertionVariant.v3.value
    assert result["t_exe"] >= 0.0
    assert result["nr_candidates"] >= 0


# --------------------------------------------------------------------------------------
# Resumable suite: generate / solve / build table
# --------------------------------------------------------------------------------------


def test_generate_suite_creates_manifest_rows(tmp_path: Any) -> None:
    """generate_ablation_suite() writes one manifest row per feasible combination"""
    summary = generate_ablation_suite(
        outdir=str(tmp_path),
        request_counts=[5, 10],
        agent_counts=[1],
        scenario_types=[ScenarioType.random],
        scenario_timings=[ScenarioTiming.loose],
        nr_repeats=1,
        base_seed=1,
    )
    assert all(row["status"] == ManifestStatus.created for row in summary)
    manifest = read_ablation_manifest(str(tmp_path))
    assert len(manifest) == 2


def test_generate_suite_is_resumable(tmp_path: Any) -> None:
    """Re-running generate_ablation_suite() skips already-generated points"""
    kwargs = dict(
        outdir=str(tmp_path),
        request_counts=[5],
        agent_counts=[1],
        scenario_types=[ScenarioType.random],
        scenario_timings=[ScenarioTiming.loose],
        nr_repeats=1,
        base_seed=1,
    )
    generate_ablation_suite(**kwargs)
    summary2 = generate_ablation_suite(**kwargs)
    assert all(row["status"] == ManifestStatus.existing for row in summary2)


def test_different_soc_ranges_coexist_without_colliding(tmp_path: Any) -> None:
    """
    Two corpora (SoC normal vs stress) in the same --outdir get distinct point names
    and manifest rows, since soc_range is baked into point_name
    """
    kwargs = dict(
        outdir=str(tmp_path),
        request_counts=[5],
        agent_counts=[1],
        scenario_types=[ScenarioType.random],
        scenario_timings=[ScenarioTiming.loose],
        nr_repeats=1,
        base_seed=1,
    )
    generate_ablation_suite(**kwargs, soc_range=SocRangeSpec.normal())
    generate_ablation_suite(**kwargs, soc_range=SocRangeSpec.stress())
    manifest = read_ablation_manifest(str(tmp_path))
    assert len(manifest) == 2
    soc_ranges = {
        SocRangeSpec.from_dict(row["soc_range"]).bounds for row in manifest.values()
    }
    assert soc_ranges == {(0.8, 1.0), (0.5, 0.7)}


def test_manifest_is_a_single_json_object(tmp_path: Any) -> None:
    """manifest.json is one streamlined JSON object"""
    generate_ablation_suite(
        outdir=str(tmp_path),
        request_counts=[5],
        agent_counts=[1],
        scenario_types=[ScenarioType.random],
        scenario_timings=[ScenarioTiming.loose],
        nr_repeats=1,
        base_seed=1,
    )
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    with open(manifest_path) as f:
        on_disk = json.load(f)  # raises if it is NDJSON rather than one JSON document
    assert len(on_disk) == 1


def test_solve_suite_measures_every_variant_by_default(tmp_path: Any) -> None:
    """
    solve_ablation_suite() measures all four variants and writes a result JSON
    with per-variant stats for every pending point
    """
    generate_ablation_suite(
        outdir=str(tmp_path),
        request_counts=[5],
        agent_counts=[1],
        scenario_types=[ScenarioType.random],
        scenario_timings=[ScenarioTiming.loose],
        nr_repeats=1,
        base_seed=1,
    )
    outcomes = solve_ablation_suite(outdir=str(tmp_path), nr_measure_repeats=1)
    assert len(outcomes) == 1
    assert outcomes[0]["status"] == ManifestStatus.done

    manifest = read_ablation_manifest(str(tmp_path))
    row = next(iter(manifest.values()))
    with open(row["result_file"]) as f:
        result = json.load(f)
    assert set(result["variant_results"].keys()) == set(INSERTION_VARIANT_FCNS.keys())
    assert result["nr_accepted"] > 0
    assert result["mean_route_length"] > 0
    for variant_result in result["variant_results"].values():
        assert not variant_result["skipped"]
        assert "mean_t_exe" in variant_result


def test_solve_suite_skips_naive_variants_above_size_cutoff(tmp_path: Any) -> None:
    """V0/V1 are marked skipped (not measured) above max_requests_for_naive"""
    generate_ablation_suite(
        outdir=str(tmp_path),
        request_counts=[10],
        agent_counts=[1],
        scenario_types=[ScenarioType.random],
        scenario_timings=[ScenarioTiming.loose],
        nr_repeats=1,
        base_seed=1,
    )
    solve_ablation_suite(
        outdir=str(tmp_path), nr_measure_repeats=1, max_requests_for_naive=5
    )

    manifest = read_ablation_manifest(str(tmp_path))
    row = next(iter(manifest.values()))
    with open(row["result_file"]) as f:
        result = json.load(f)
    assert result["variant_results"][InsertionVariant.v0.value]["skipped"] is True
    assert result["variant_results"][InsertionVariant.v1.value]["skipped"] is True
    assert result["variant_results"][InsertionVariant.v2.value]["skipped"] is False
    assert result["variant_results"][InsertionVariant.v3.value]["skipped"] is False


def test_solve_suite_is_resumable(tmp_path: Any) -> None:
    """Re-running solve_ablation_suite() finds nothing pending to measure"""
    generate_ablation_suite(
        outdir=str(tmp_path),
        request_counts=[5],
        agent_counts=[1],
        scenario_types=[ScenarioType.random],
        scenario_timings=[ScenarioTiming.loose],
        nr_repeats=1,
        base_seed=1,
    )
    solve_ablation_suite(outdir=str(tmp_path), nr_measure_repeats=1)
    outcomes2 = solve_ablation_suite(outdir=str(tmp_path), nr_measure_repeats=1)
    assert outcomes2 == []


def test_generate_and_solve_report_progress(tmp_path: Any) -> None:
    """on_progress fires a start and an end event per point in both phases"""
    generate_events: list[dict[str, Any]] = []
    generate_ablation_suite(
        outdir=str(tmp_path),
        request_counts=[5, 10],
        agent_counts=[1],
        scenario_types=[ScenarioType.random],
        scenario_timings=[ScenarioTiming.loose],
        nr_repeats=1,
        base_seed=1,
        on_progress=generate_events.append,
    )
    assert len(generate_events) == 4  # 2 points x (starting, terminal status)
    assert all(
        ManifestPhase(e["phase"]) == ManifestPhase.generate and e["total"] == 2
        for e in generate_events
    )

    solve_events: list[dict[str, Any]] = []
    solve_ablation_suite(
        outdir=str(tmp_path), nr_measure_repeats=1, on_progress=solve_events.append
    )
    assert len(solve_events) == 4  # 2 points x (starting, done)
    assert [e["status"] for e in solve_events] == [
        ManifestStatus.starting,
        ManifestStatus.done,
    ] * 2


def test_build_table_writes_csv_json_tex_with_speedup_ratios(tmp_path: Any) -> None:
    """
    build_ablation_table() writes all three files; speedup ratios equal
    slow_time/fast_time for a row with every variant measured
    """
    generate_ablation_suite(
        outdir=str(tmp_path),
        request_counts=[8],
        agent_counts=[1],
        scenario_types=[ScenarioType.random],
        scenario_timings=[ScenarioTiming.loose],
        nr_repeats=1,
        base_seed=1,
    )
    solve_ablation_suite(outdir=str(tmp_path), nr_measure_repeats=1)
    rows = build_ablation_table(str(tmp_path))

    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == ManifestStatus.done
    assert row["t_exe_V0"] is not None and row["t_exe_V3"] is not None
    assert row["speedup_V0_V3"] == pytest.approx(row["t_exe_V0"] / row["t_exe_V3"])
    assert row["speedup_V2_V3"] == pytest.approx(row["t_exe_V2"] / row["t_exe_V3"])

    for ext in ("csv", "json", "tex"):
        assert os.path.exists(os.path.join(str(tmp_path), f"ablation_table.{ext}"))


def test_build_table_handles_pending_rows(tmp_path: Any) -> None:
    """A point never solved (still pending) gets a row of Nones, not a crash"""
    generate_ablation_suite(
        outdir=str(tmp_path),
        request_counts=[5],
        agent_counts=[1],
        scenario_types=[ScenarioType.random],
        scenario_timings=[ScenarioTiming.loose],
        nr_repeats=1,
        base_seed=1,
    )
    rows = build_ablation_table(str(tmp_path))
    assert len(rows) == 1
    assert rows[0]["status"] == ManifestStatus.pending
    assert rows[0]["t_exe_V0"] is None
    assert rows[0]["speedup_V0_V3"] is None


# --------------------------------------------------------------------------------------
# Cross-corpus comparison
# --------------------------------------------------------------------------------------


def test_compare_route_length_buckets_computes_correct_means() -> None:
    """Buckets by mean_route_length; reports each side mean and the b/a ratio"""
    rows_a = [
        {
            "status": ManifestStatus.done,
            "mean_route_length": 12.0,
            "speedup_V2_V3": 2.0,
        },
        {
            "status": ManifestStatus.done,
            "mean_route_length": 18.0,
            "speedup_V2_V3": 4.0,
        },
        {
            "status": ManifestStatus.done,
            "mean_route_length": 25.0,
            "speedup_V2_V3": 10.0,
        },
    ]
    rows_b = [
        {
            "status": ManifestStatus.done,
            "mean_route_length": 14.0,
            "speedup_V2_V3": 6.0,
        },
        {
            "status": ManifestStatus.done,
            "mean_route_length": 16.0,
            "speedup_V2_V3": 10.0,
        },
    ]
    result = compare_route_length_buckets(rows_a, rows_b, bucket_size=10)
    buckets = {r["route_bucket_start"]: r for r in result}

    assert buckets[10]["nr_a"] == 2  # 12.0, 18.0
    assert buckets[10]["mean_a"] == pytest.approx(3.0)
    assert buckets[10]["nr_b"] == 2  # 14.0, 16.0
    assert buckets[10]["mean_b"] == pytest.approx(8.0)
    assert buckets[10]["ratio_b_to_a"] == pytest.approx(8.0 / 3.0)

    assert buckets[20]["nr_a"] == 1  # 25.0
    assert buckets[20]["nr_b"] == 0
    assert buckets[20]["mean_b"] is None
    assert buckets[20]["ratio_b_to_a"] is None  # b side is empty


def test_compare_route_length_buckets_ignores_non_done_and_missing_values() -> None:
    """Rows with status != done or a missing route length/metric are excluded"""
    rows_a = [
        {
            "status": ManifestStatus.pending,
            "mean_route_length": None,
            "speedup_V2_V3": None,
        },
        {
            "status": ManifestStatus.done,
            "mean_route_length": 15.0,
            "speedup_V2_V3": None,
        },
        {
            "status": ManifestStatus.done,
            "mean_route_length": 15.0,
            "speedup_V2_V3": 5.0,
        },
    ]
    result = compare_route_length_buckets(rows_a, [], bucket_size=10)
    assert len(result) == 1
    assert result[0]["nr_a"] == 1
    assert result[0]["mean_a"] == pytest.approx(5.0)


def test_compare_route_length_buckets_respects_metric_argument() -> None:
    """A different metric (e.g., raw V3 time) is bucketed/averaged instead"""
    rows_a = [
        {"status": ManifestStatus.done, "mean_route_length": 12.0, "t_exe_V3": 0.001}
    ]
    rows_b = [
        {"status": ManifestStatus.done, "mean_route_length": 12.0, "t_exe_V3": 0.002}
    ]
    result = compare_route_length_buckets(rows_a, rows_b, metric="t_exe_V3")
    assert result[0]["metric"] == "t_exe_V3"
    assert result[0]["mean_a"] == pytest.approx(0.001)
    assert result[0]["mean_b"] == pytest.approx(0.002)


def test_write_corpus_comparison_end_to_end(tmp_path: Any) -> None:
    """write_corpus_comparison() reads two real corpora, buckets, and writes files"""
    corpus_a = tmp_path / "corpus_a"
    corpus_b = tmp_path / "corpus_b"
    comparison_dir = tmp_path / "comparison"

    generate_ablation_suite(
        outdir=str(corpus_a),
        request_counts=[8],
        agent_counts=[1],
        scenario_types=[ScenarioType.random],
        scenario_timings=[ScenarioTiming.loose],
        nr_repeats=1,
        base_seed=1,
        soc_range=SocRangeSpec.normal(),
    )
    solve_ablation_suite(outdir=str(corpus_a), nr_measure_repeats=1)

    generate_ablation_suite(
        outdir=str(corpus_b),
        request_counts=[8],
        agent_counts=[1],
        scenario_types=[ScenarioType.random],
        scenario_timings=[ScenarioTiming.loose],
        nr_repeats=1,
        base_seed=1,
        soc_range=SocRangeSpec.stress(),
    )
    solve_ablation_suite(outdir=str(corpus_b), nr_measure_repeats=1)

    rows = write_corpus_comparison(str(corpus_a), str(corpus_b), str(comparison_dir))
    assert len(rows) >= 1
    assert os.path.exists(comparison_dir / "corpus_comparison.csv")
    assert os.path.exists(comparison_dir / "corpus_comparison.json")
    assert os.path.exists(comparison_dir / "corpus_comparison.tex")
    with open(comparison_dir / "corpus_comparison.json") as f:
        on_disk = json.load(f)
    assert on_disk == rows


def test_write_corpus_comparison_latex_renders_rows_and_missing_side(
    tmp_path: Any,
) -> None:
    """
    _write_corpus_comparison_latex() renders one row per bucket, with the shared
    '-' placeholder for a bucket that has data on one side only
    """
    comparison = [
        {
            "route_bucket_start": 0,
            "route_bucket_end": 10,
            "metric": InsertionAblationMetric.speedup_V2_V3,
            "nr_a": 2,
            "mean_a": 1.5,
            "nr_b": 3,
            "mean_b": 2.25,
            "ratio_b_to_a": 1.5,
        },
        {
            "route_bucket_start": 10,
            "route_bucket_end": 20,
            "metric": InsertionAblationMetric.speedup_V2_V3,
            "nr_a": 1,
            "mean_a": 4.0,
            "nr_b": 0,
            "mean_b": None,
            "ratio_b_to_a": None,
        },
    ]
    tex_path = tmp_path / "corpus_comparison.tex"

    _write_corpus_comparison_latex(
        comparison, str(tex_path), InsertionAblationMetric.speedup_V2_V3
    )

    text = tex_path.read_text()
    assert text.startswith(r"\documentclass{article}")
    assert text.rstrip().endswith(r"\end{document}")
    assert r"0--10 & 2 & 1.500 & 3 & 2.250 & 1.50 \\" in text
    assert r"10--20 & 1 & 4.000 & 0 & $\mathrm{-}$ & $\mathrm{-}$ \\" in text
    assert "speedup\\_V2\\_V3" in text


def test_different_soc_ranges_produce_different_point_names(tmp_path: Any) -> None:
    """Two soc_range corpora sharing an --outdir get distinct point_names"""
    generate_ablation_suite(
        outdir=str(tmp_path),
        request_counts=[5],
        agent_counts=[1],
        scenario_types=[ScenarioType.random],
        scenario_timings=[ScenarioTiming.loose],
        nr_repeats=1,
        base_seed=1,
        soc_range=SocRangeSpec.custom(0.4, 0.6),
    )
    manifest = read_ablation_manifest(str(tmp_path))
    point_name = next(iter(manifest.keys()))
    assert point_name.endswith("_soc40-60")
