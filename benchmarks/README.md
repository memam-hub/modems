# Benchmark suite

Four core scripts, backed by `modems.benchmark`, `modems.workday_benchmark`, and
`modems.insertion_ablation`, plus two comparison scripts. `run_*.py` each run 
generate + solve + build-table in one command by default (`--phase` isolates one part), 
printing live progress, and is safe to interrupt/resume.

```bash
python3 run_suite.py            # static: one scenario, all 4 solvers
python3 run_workday_suite.py    # dynamic: full simulated day, MILP3 vs ALNS
python3 run_ablation_suite.py   # V0-V3 insertion ablation, two SoC corpora + comparison
python3 compare_ablation_corpora.py --corpus-a A --corpus-b B  # compare two runs
python3 compare_workday_objectives.py --closed C --open O      # closed vs open workdays
```

`_cli_common.py` is utility to handle shared arguments and progress printing.

## run_suite.py

Generates scenarios (size buckets x spatial types x timings x SoC ranges), solves
every pending `(scenario, solver)` row (ALNS + MILP1/2/3, each MILP warm-started from
`greedy_complete`'s baseline), and writes `benchmark_table.{csv,json,tex}`.

| Flag | Default | Meaning |
|---|---|---|
| `--outdir` | `single_suite/` | Output directory |
| `--phase` | `all` | `generate`/`solve`/`build`/`all` |
| `--sizes` | `small medium large` | `ScenarioSize` buckets (agents / request range from `scenario_bucket()`) |
| `--types` | `random clustered mixed` | `ScenarioType`: request spatial distribution |
| `--timings` | `uniform peaks` | `ScenarioTiming`: shape of the earliest-pickup times, even or with two rush-hour plateaus |
| `--soc-test` | `both` | Initial-agent-SoC range(s) `ScenarioSocRange`: `normal`/`stress`/`custom`/`both` |
| `--soc-custom` | none | `LB UB` custom SoC bounds (`0 < LB <= UB <= 1.0`), with `--soc-test custom` |
| `--objective` | `closed` | `ObjectiveType`: `closed` (`C`) includes the return-to-hub leg in the mission time, `open` (`O`) does not; case-insensitive, the letter is accepted. Sets every solver's problem type; stored in the manifest at generation |
| `--nr-repeats` | `2` | Repetitions per combination |
| `--seed` | `42` | Base seed |
| `--smoke` | off | Small, quick sanity run (<=2 min) |
| `--retry-failed` | off | Also retry `failed` runs/rows |
| `--solver-name` | `cbc` | Passed to pyomo's `SolverFactory` for MILP1/2/3 |
| `--solver-config-type` | `cbc` | `cbc`/`gurobi`/`highs` (option-key convention) |
| `--milp-timelimit` | `60.0` | Seconds, MILP solving budget |
| `--alns-max-iter` | `1000` | ALNS exit criteria, number of `NoImprovement` iterations |
| `--table-name` | `benchmark_table` | Output file basename |
| `--rows-per-block` | `40` | `.tex` rows per `table*` block |
| `--plots` | off | Per-result `instance.plot()` + ALNS convergence/operator charts |

Generated table columns: scenario, objective type, solver, baseline objective, status,
solver objective, lower/upper bound, gap%, improvement% over the baseline (negative when
the solver ends worse), count of accepted requests, solve time. Bounds (and gap%) are blank for ALNS and
any trivial instance, reported only by MILP solutions.

The objective is not part of the scenario seed, so runs with `--objective closed` and
`--objective open` can share one `--outdir` and solve identical instances.

### Naming convention

Scenario names follow the pattern `S<obj><idx>_<size><type><timing><lb>_<ub>`:

| Component | Flag | Value |
|---|---|---|
| `obj` | `--objective` | `C` closed, `O` open |
| `idx` | `--nr-repeats` | Repetition index, starting at `0` |
| `size` | `--sizes` | `S` small, `M` medium, `L` large |
| `type` | `--types` | `R` random, `C` clustered, `M` mixed |
| `timing` | `--timings` | `U` uniform, `P` peaks |
| `lb` | `--soc-test` / `--soc-custom` | SoC lower bound x 100: `80` normal, `50` stress, or the custom `LB` |
| `ub` | `--soc-test` / `--soc-custom` | SoC upper bound x 100: `100` normal, `70` stress, or the custom `UB` |

Examples:

- `SC0_LMU50_70`: closed objective, repetition 0, large, mixed, uniform, SoC in
  `[0.5, 0.7]` (stress).
- `SO1_SCP80_100`: open objective, repetition 1, small, clustered, peaks, SoC in
  `[0.8, 1.0]` (normal).

## run_workday_suite.py

By default, generates and solves workdays for a 3-agent fleet (`large`) with `mixed`
requests, across all 8 combinations of start time x base rate x timing (2x2x2); solves
every pending workday (`compare_solvers_one_workday`: MILP3 vs. ALNS on the identical
request submission stream); writes `summary_table.{csv,json,tex}` (one row per
workday) plus a per-workday `requests_table.{csv,json,tex}`.

| Flag | Default | Meaning |
|---|---|---|
| `--outdir` | `workday_suite/` | Output directory |
| `--sizes` | `large` | `ScenarioSize`, sets the fleet size (agents from `scenario_bucket()`) |
| `--types` | `mixed` | `ScenarioType` of the requests |
| `--timings` | `uniform peaks` | `ScenarioTiming`: shape of the base demand over the workday; `peaks` puts half of it in two plateaus at 1/3 and 2/3 of the day |
| `--start-times` | `normal staggered` | `WorkdayStartTime`: `normal`, all agents start at t=0; `staggered`, agent i (0-indexed) starts at `i * workday/5` |
| `--base-rates` | `5 8` | Baseline request submissions per hour (integers) |
| `--nr-surges` | `3` | Surges per workday: 30-minute windows that each add one hour of base-rate demand, never overlapping each other or the peaks, and kept 30 (`uniform`) or 15 (`peaks`) minutes apart from both. Raises if they do not fit: at most 8 (`uniform`) or 5 (`peaks`) in a 480-minute workday |
| `--objective` | `closed` | Same as above; with `open`, an available agent stays at its last node, going to a hub only to recharge |
| `--workday` | `480.0` | Workday length (minutes) over which requests are submitted; each simulation then drains until every accepted request is delivered (capped at 240 minutes past the workday) |
| `--table-name` | `summary_table` | Output file basename for the workday summary |
| `--requests-table-name` | `requests_table` | Output file basename for the compiled workday requests data |
| `--clock-display-start` | `08:00` | Workday starting clock time for formatting clock-time columns |
| `--plots` | off | Dual-axis (cumulative accept/reject rate vs. agent SoC) plot per (workday, solver) |

Plus `--nr-repeats` (default **1**) / `--seed`/`--smoke`/`--retry-failed`/
`--solver-name`/`--solver-config-type`/`--milp-timelimit`/`--alns-max-iter`/
`--rows-per-block` as `run_suite.py`.

### Naming convention

Workday names follow the pattern `W<obj><idx>_<size><type><timing><start><br>_<srg>`,
where `obj`, `idx`, `size`, `type`, and `timing` read as in the scenario names, and:

| Component | Flag | Value |
|---|---|---|
| `start` | `--start-times` | `N` normal, `S` staggered |
| `br` | `--base-rates` | Base rate (requests per hour) |
| `srg` | `--nr-surges` | Number of surges per workday |

Examples:

- `WC0_LMUN5_3`: closed objective, repetition 0, large (3 agents), mixed, uniform,
  normal start, 5 requests per hour, 3 surges.
- `WO1_MRPS8_4`: open objective, repetition 1, medium (2 agents), random, peaks,
  staggered start, 8 requests per hour, 4 surges.


## run_ablation_suite.py

By default, generates + solves + builds **both** corpora (normal-SoC and SoC-stress)
and their comparison, in one command. Full default run: ~25 minutes at normal hardware.

| Flag | Default | Meaning |
|---|---|---|
| `--outdir` | `ablation_suite/` | Output directory; corpora go in `normal/`/`stress/` subdirectories, the comparison in `comparison/` |
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
`--rows-per-block` as `run_suite.py`. The built-in SoC ranges are `(0.8, 1.0)` normal,
`(0.5, 0.7)` stress; `--corpus custom --soc-custom LB UB` tests any other range,
written under its own `custom/` subdirectory (comparison against it is skipped,
same as `normal`/`stress` alone).

Generated table columns: one row per point (`nr_agents`,`nr_requests`, spatial type,
timing, repetition), each variant's mean per-probe wall-clock time, `V0/V3` and
`V2/V3` speedup ratios, achieved `mean_route_length`, and V3's own analysis metrics:
pruned-pairs and prefix/suffix-propagation. The comparison table buckets both corpora
by `mean_route_length` and reports the chosen `metric`'s mean per bucket side by side,
in addition to the B/A ratio.

### Naming convention

Ablation point names follow the pattern `I<idx>_a<agt>r<req><type><timing><lb>_<ub>`,
with no objective letter (insertion enumeration has no notion of an objective), where
`idx`, `type`, `timing`, `lb`, and `ub` read as in the scenario names, and:

| Component | Flag | Value |
|---|---|---|
| `agt` | `--agent-counts` | Number of agents |
| `req` | `--request-counts` | Number of requests |

Examples:

- `I0_a1r10CU50_70`: repetition 0, 1 agent, 10 requests, clustered, uniform, SoC in
  `[0.5, 0.7]` (stress).
- `I1_a2r60RP80_100`: repetition 1, 2 agents, 60 requests, random, peaks, SoC in
  `[0.8, 1.0]` (normal).


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

Plus `--bucket-size` and `--metric` as `run_ablation_suite.py`.

## compare_workday_objectives.py

Compares two solved workday suites, one run with `--objective closed` and one with
`--objective open`, which were generated with identical arguments and `--seed`.
Objective type does not impact request generation, so workdays have identical request 
streams across suites. Each workday is observed four times (MILP3/ALNS x closed/open).

Writes `combined_manifest.json` and `combined_summary.{csv,json,tex}` (each workday's
closed row directly followed by its open row, with demand load and per-served-request 
travel time/energy) and three figures:

| Figure | Content |
|---|---|
| `var_decision_impact.png` | Rows: acceptance, mean delay, mean excess ride, travel time and energy per served request. Columns: objective (open - closed, per solver), solver (ALNS - MILP3, per objective), start time (staggered - normal, per solver). Mean difference with its 95% CI |
| `demand_vs_accept_delay.png` | Acceptance and mean delay against the demand load (requests per agent-hour): binned means with 95% CI bands per solver x objective, one marker per workday (hollow: with surges) |
| `most_demand_<name>.png` | The representative workday (highest demand load among workdays with surges): rolling acceptance and agent SoC for its four runs, above the shared demand |

Delay and excess ride time are means over served requests only.

| Flag | Default | Meaning |
|---|---|---|
| `--closed` / `--open` | required | Outdirs of the closed and open suites |
| `--outdir` | `results_workday_comparison/` | Output directory |
| `--table-name` | `combined_summary` | Combined table basename (`.csv`/`.json`/`.tex`) |
| `--all-workdays` | off | Also plot every paired workday into `<outdir>/workdays/` |
| `--window` | `30` | Minutes of submissions behind each rolling acceptance point |


## Manifest format

Each suite writes `manifest.json` to its `--outdir`: one JSON object, rewritten in
full on every update, keyed by `(scenario_name, solver_strategy)` (static suite),
`workday_name` (workday suite -- one `compare_solvers_one_workday` call covers both
solvers), or `point_name` (ablation suite, tagged with its `soc_range`). Row `status`
is one of `ManifestStatus`: `pending`/`done`/`failed` plus phase-transient values
(`created`, `existing` for an already-generated row, and `infeasible`).

## Output layout

```
single_suite/               # run_suite.py
├── manifest.json
├── scenarios/{scenario_name}.json
├── results/{scenario_name}_{solver}.json
├── results/{scenario_name}_alns_alns_stats.json   # ALNS runs only
└── benchmark_table.{csv,json,tex}

workday_suite/              # run_workday_suite.py
├── manifest.json
├── results/{workday_name}.json
├── results/{workday_name}_workday_log.json
├── requests_tables/{workday_name}_requests_table.{csv,json,tex}
└── summary_table.{csv,json,tex}

ablation_suite/             # run_ablation_suite.py
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

