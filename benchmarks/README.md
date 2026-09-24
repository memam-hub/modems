# Benchmark suite

Four core scripts, backed by `modems.benchmark`, `modems.workday_benchmark`, and
`modems.insertion_ablation`. `run_*.py` each run generate + solve + build-table in
one command by default (`--phase` isolates one part), printing live progress, and
are safe to interrupt/resume.

```bash
python3 run_suite.py            # static: one scenario, all 4 solvers
python3 run_workday_suite.py    # dynamic: full simulated day, MILP3 vs ALNS
python3 run_ablation_suite.py   # V0-V3 insertion ablation, two SoC corpora + comparison
python3 compare_ablation_corpora.py --corpus-a A --corpus-b B  # compare two runs
```

`_cli_common.py` is utility to handle shared arguments and progress printing.

## run_suite.py

Generates scenarios (S/M/L size buckets x R/C/M spatial types x L/T timings), solves
every pending `(scenario, solver)` row (ALNS + MILP1/2/3, each MILP warm-started from
`greedy_complete`'s baseline), and writes `benchmark_table.{csv,json,tex}`.

| Flag | Default | Meaning |
|---|---|---|
| `--outdir` | `results/` | Output directory |
| `--phase` | `all` | `generate`/`solve`/`build`/`all` |
| `--sizes` | `S M L` | Size buckets `ScenarioSize`: `small`/`medium`/`large` |
| `--types` | `R C M` | Request spatial types `ScenarioType`: `random`/`clustered`/`mixed` |
| `--timings` | `L T` | Request timing shifts `ScenarioTiming`: `loose`/`tight` |
| `--soc-test` | `both` | Initial-agent-SoC range(s) `ScenarioSocRange`: `normal`/`stress`/`custom`/`both` |
| `--soc-custom` | none | `LB UB` custom SoC bounds (`0 < LB <= UB <= 1.0`), with `--soc-test custom` |
| `--nr-repeats` | `2` | Repetitions per combination |
| `--seed` | `42` | Base seed |
| `--smoke` | off | Small, quick sanity run (<=2 min) |
| `--retry-failed` | off | Also retry `failed` runs/rows |
| `--solver-name` | `cbc` | Passed to pyomo's `SolverFactory` for MILP1/2/3 |
| `--solver-config-type` | `cbc` | `cbc`/`gurobi`/`highs` (option-key convention) |
| `--milp-timelimit` | `60.0` | Seconds, MILP solving budget |
| `--alns-max-iter` | `1000` | ALNS exit criteria, number of `NoImprovement` iterations|
| `--table-name` | `benchmark_table` | Output file basename |
| `--rows-per-block` | `40` | `.tex` rows per `table*` block |
| `--plots` | off | Per-result `instance.plot()` + ALNS convergence/operator charts |

Generated table columns: scenario, solver, baseline objective, status, solver objective,
lower/upper bound, gap%, improvement (fraction; `.tex` renders it as %), count of
accepted requests, solve time. Bounds (and gap%) are blank for ALNS and any trivial
instance, reported only by MILP solutions. `scenario` is tagged with its SoC range,
e.g. `DS_SRL0_soc80-100`, same as the ablation suite's `point_name` below.

## run_workday_suite.py

By default, generates and solves for a 3-agent fleet, with `ScenarioType.Mixed`
requests across all 8 combinations of `start_time` x `base_rate` x `timing` (2x2x2);
solves every pending workday (`compare_solvers_one_workday`: MILP3 vs. ALNS against
the identical request submission/arrival stream); writes `summary_table.{csv,json,tex}`
(one row per workday) plus a per-workday `requests_table.{csv,json,tex}`.

| Flag | Default | Meaning |
|---|---|---|
| `--outdir` | `results_workday/` | Output directory |
| `--start-times` | `N S` | `WorkdayStart` times: normal `N`, all agents start at t=0. staggered `S`, agent i (0-indexed) starts at `i * workday/5` |
| `--base-rates` | `5.0 8.0` | Baseline request arrivals per hour (before surges) |
| `--timings` | `L T` | Same as above, also drives surge parameters|
| `--workday` | `480.0` | Simulated workday length (minutes) |
| `--workday-buffer` | `90.0` | Extra minutes run past `--workday` so late-submitted requests still resolve |
| `--table-name` | `summary_table` | Output file basename for the workday summary |
| `--requests-table-name` | `requests_table` | Output file basename for the compiled workday requests data |
| `--clock-display-start` | `08:00` | Workday starting clock time for formatting clock-time columns |
| `--plots` | off | Dual-axis (cumulative accept/reject rate vs. agent SoC) plot per (workday, solver)|

Plus `--nr-repeats` (default **1**, to keep the default sweep at exactly 8 workdays)
/ `--seed`/`--smoke`/`--retry-failed`/`--solver-name`/`--solver-config-type`/
`--milp-timelimit`/`--alns-max-iter`/`--rows-per-block` as `run_suite.py`. The script
runs with fixed fleet and request spatial type, call `generate_workday_suite(...)`
with `scenario_sizes=[...]`/`scenario_types=[...]` to test arbitrary ranges.


## run_ablation_suite.py

By default, generates + solves + builds **both** corpora (normal-SoC and SoC-stress)
and their comparison, in one command. Full default run: ~25 minutes at normal hardware.

| Flag | Default | Meaning |
|---|---|---|
| `--outdir` | `results_ablation/` | Output directory; corpora go in `normal/`/`stress/` subdirectories, the comparison in `comparison/` |
| `--corpus` | `both` | `both`/`normal`/`stress`/`custom` -- which corpus/corpora to generate/solve/build. `normal`/`stress`/`custom` alone skip the comparison |
| `--soc-custom` | none | `LB UB` custom SoC bounds (`0 < LB <= UB <= 1.0`), with `--corpus custom` |
| `--agent-counts` | `1 2` | Agent counts to sweep |
| `--request-counts` | `10 15 ... 60` | Request counts to sweep, shared by both corpora |
| `--nr-probes` | `3` | Requests held out per point/combination as insertion probes |
| `--variants` | all four | Which `InsertionVariant`s to measure |
| `--nr-measure-repeats` | `5` | Repeated timed calls per (point, probe, variant), averaged for stability |
| `--max-requests-for-naive` | `200` | V0/V1 are omitted beyond this number of requests (quadratic-ish, extremely expensive) |
| `--table-name` | `ablation_table` | Per-corpus output file basename |
| `--comparison-table-name` | `corpus_comparison` | Comparison output file basename |
| `--bucket-size` | `10` | Comparison route-length bucket width (stops) |
| `--metric` | `speedup_V2_V3` | Comparison metric: `InsertionAblationMetric` |

Plus `--types`/`--timings`/`--nr-repeats`/`--seed`/`--smoke`/`--retry-failed`/
`--rows-per-block` as before. The built-in SoC ranges are `(0.8, 1.0)` normal,
`(0.5, 0.7)` stress; `--corpus custom --soc-custom LB UB` tests any other range,
written under its own `custom/` subdirectory (comparison against it is skipped,
same as `normal`/`stress` alone).

Generated table columns: one row per point (`nr_agents`,`nr_requests`, spatial type,
timing, repetition), each variant's mean per-probe wall-clock time, `V0/V3` and
`V2/V3` speedup ratios, achieved `mean_route_length`, and V3's own analysis metrics:
pruned-pairs and prefix/suffix-propagation. The comparison table buckets both corpora
by `mean_route_length` and reports the chosen `metric`'s mean per bucket side by side,
in addition to the B/A ratio.

## compare_ablation_corpora.py

Buckets two already-solved ablation corpora by achieved `mean_route_length`, reports
the mean of chosen `metric` per bucket side by side, plus the B/A ratio; useful to
compare arbitrary corpora. The default `run_ablation_suite.py` run calls this after
solving/generating the two corpora.

| Flag | Default | Meaning |
|---|---|---|
| `--corpus-a` / `--corpus-b` | required | Outdirs of the two corpora |
| `--outdir` | `results_ablation_comparison/` | Output directory |
| `--table-name` | `corpus_comparison` | Output file basename (`.csv`/`.json`/`.tex`) |

Plus `--bucket-size` and `--metric` as before.


## Manifest format

Each suite writes `manifest.json` to its `--outdir`: one JSON object, rewritten in
full on every update, keyed by `(scenario_name, solver_strategy)` (static suite),
`workday_name` (workday suite -- one `compare_solvers_one_workday` call covers both
solvers), or `point_name` (ablation suite, tagged with its `soc_range`). Row `status`
is one of `ManifestStatus`: `pending`/`done`/`failed` plus phase-transient values
(`created`, `existing` for an already-generated row, and `infeasible`).

## Output layout

```
results/                    # run_suite.py
├── manifest.json
├── scenarios/{scenario_name}.json
├── results/{scenario_name}_{solver}.json
├── results/{scenario_name}_alns_alns_stats.json   # ALNS runs only
└── benchmark_table.{csv,json,tex}

results_workday/            # run_workday_suite.py
├── manifest.json
├── results/{workday_name}.json
├── results/{workday_name}_workday_log.json
├── requests_tables/{workday_name}_requests_table.{csv,json,tex}
└── summary_table.{csv,json,tex}

results_ablation/           # run_ablation_suite.py
├── normal/
│   ├── manifest.json
│   ├── results/{point_name}.json
│   └── ablation_table.{csv,json,tex}
├── stress/                # same layout, soc_range=(0.5, 0.7)
└── comparison/
    └── corpus_comparison.{csv,json,tex}
```

## Extending the suites

Size-bucket agent/request ranges live in `core.py`'s `scenario_bucket()`; the solver
list is `generate_benchmark_suite`'s own `solver_strategies` parameter, defaults to
every `SolverStrategy`. `modems.workday_benchmark` reuses `scenario_bucket()`.

