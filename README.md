# modems

MILP and ALNS solvers for the **MODEMS** on-campus passenger mobility routing service:
a dynamic variant of the electric Autonomous Dial-a-Ride Problem, solved either with
exact methods (three MILP formulations, MILP1/MILP2/MILP3) or heuristically (Adaptive
Large Neighborhood Search ALNS), with a rolling-horizon workflow for multi-epoch,
full-workday operation.


## Requirements

- Python >= 3.12, pure Python, no compiled components.
- **MILP Solver:** [CBC](https://github.com/coin-or/Cbc) is the default, installed
  automatically via `pulp[cbc]` -- no license needed. [HiGHS](https://highs.dev/)
  and [Gurobi](https://www.gurobi.com/) are also supported and validated.
  `ModemsMilp.solve()` takes `solver_name` (the string passed to pyomo's
  `SolverFactory`, e.g., `"cbc"`, `"appsi_highs"`, `"gurobi"`) and `solver_config_type`
  (a `SolverConfigType` -- for the solver option-key convention/configuration: `cbc`,
  `gurobi`, or `highs`).


## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -e .
```

Optional extras: `pip install -e ".[highs]"` (free, no license) or
`pip install -e ".[gurobi]"` (needs a license -- free for academics at
[gurobi.com/academia](https://www.gurobi.com/academia/academic-program-and-licenses/)).
Everything defaults to CBC and works without either.


## Quickstart

```python
from modems import ModemsScenarioGenerator, ModemsMilp, MilpType, \
    SolverConfigType, ModemsAlns, ProblemType

generator = ModemsScenarioGenerator(seed=42)
scenario = generator.generate_random_scenario(nr_agents=2, nr_requests=5)

milp = ModemsMilp(scenario=scenario, milp_type=MilpType.milp3,
                   problem_type=ProblemType.closed_selective,
                   milp_params={"eps": 0.01, "zeta": 1.0,
                                "eta": 100.0, "rho": 2.5,
                                "big_m": 150.0})
milp.solve(solver_name="cbc", solver_config_type=SolverConfigType.cbc,
           solver_options={"timelimit": 60.0})
```

`ModemsAlns` follows analogously: build with `(scenario, problem_type, model_params)`,
then `.solve(seed=..., max_iter=...)`.
To get the right `ProblemType` for any solver from an objective (`ObjectiveType`),
use `resolve_problem_type(strategy, objective)`: `closed` objective includes the
final return-to-hub leg in the mission time, while `open` excludes it. This does not
impact the generated routes, all routes end at a reachable hub for energy feasibility.
All solvers support both routing objectives. In terms of selectivity:
ALNS is selective only (by design), raising on a non-selective `problem_type`.
MILP1 is non-selective only, MILP2 is selective only, and MILP3 supports both
(selective by default). Try `benchmarks/run_suite.py --smoke` for a full
runnable example across every solver.


## Scenario model

- **Network**: default 3 hubs + 15 stations (18 nodes). All human-facing indices
  (`h_1`, `s_1..s_15`, `agent_1`, `request_1`) are 1-based. Travel times come from
  Euclidean coordinate distances at a calibrated shuttle speed.
- **`ModemsScenario`**: agents + requests + network, with spatial distribution `type`
  (`ScenarioType`: random/clustered/mixed), `timing` (`ScenarioTiming`: uniform/peaks,
  the shape of the earliest-pickup times: even, or with two rush-hour plateaus),
  and a canonical `size` (`ScenarioSize`: small/medium/large, inferred from
  agent/request counts). These enums also accept their first letter (`"R"`, `"T"`,
  `"S"`, case-insensitive), which is how they appear in benchmark names.
- **`ProblemContext(scenario, problem_type, strategy, model_params)`**:
  precomputed, read-only lookup tables shared by every solver. Default `model_params`:
  `eps=0.01, zeta=1.0, eta=100.0, rho=2.5` (enforced `eps < zeta < eta`, `rho >= 1`);
  fixed pickup time-window `omega=5.0` min; MILP `big_m=150`.
- **`preprocess(ctx, partial_plan=None)`** returns `(base_solution,
  unassigned_request_names)`; `None` signals solution infeasibility. `partial_plan` is
  a `ModemsSolution` for the current snapshot that carries over scheduled requests'
  visit order, service times, and slack decisions under a rolling-horizon replan.


## Rolling horizon

`RollingHorizonSimulator`/`WorkdaySimulation` (`rolling_horizon.py`) drive
epoch-by-epoch replanning. `WorkdaySimulation.run()` returns `list[EpochLog]`;
`build_workday_log(epoch_log, request_submissions)` aggregates a full day into a
`WorkdayLog`: every submitted request's outcome/timing as `RequestRecord`s, and
every agent's full-day node-by-node walk as `AgentNodeVisit`s using physical names
(`h_2`, `s_5`), both in absolute time offset from clock-start.
`summarize_workday(epoch_log)` exports the same log into scalar stats:
acceptance rate, waiting/excess-ride/delay time, energy, and solve time.

`solver_mode="single"` (with `single_solver="milp3"`/`"alns"`) solves and adopts the
same solver each epoch for a controlled comparison; `solver_mode="operational"` races
MILP3 and ALNS and adopts whichever is better each epoch; the default `"benchmarking"`
races all four solvers instead for a full comparison. `compare_solvers_one_workday()`
runs the identical request submission/arrival stream through two independent
single-solver simulators; `compare_solvers_over_workdays()` repeats this across
N seeded workdays. Every `ModemsRequest` carries a persistent `request_id`, and every
`ModemsAgent` a persistent `agent_id`, to identify them across epochs.

`objective` (`ObjectiveType`, default `closed`, accepted by the simulator and both
`compare_solvers_*` functions) sets every solver's problem type. With `open`, an
agent that is available after its last delivery (or while idle) remains stationary
instead of driving to a hub; it is sent to a hub only when it needs to recharge.

Workday demand comes from `ModemsScenarioGenerator.generate_workday_requests()`: a
Poisson process over earliest-pickup times in `[15, workday_length]` whose rate is
`base_rate_per_hour` shaped by the timing (`uniform`, or `peaks` with half the demand
in two plateaus at 1/3 and 2/3 of the day), plus `nr_surges` 30-minute surges that each
add one hour of base-rate demand. Surges never overlap each other or the peaks, and
keep a timing-dependent gap from both (30 minutes for `uniform`, 15 for `peaks`);
asking for more surges than fit raises a `ValueError` (`max_surges()`). Each request
is submitted 15–45 minutes before its earliest pickup (never before t=0). After the
workday, `WorkdaySimulation` keeps replanning (drains) until every accepted request is
delivered, with a safety cap of `DEFAULT_MAX_DRAIN` minutes.


## Insertion algorithms

`algorithms.py` implements Algorithm 1 (`preprocess`), Algorithm 2 (`greedy_complete`),
and Algorithm 3 (`alns_feasible_insertions` for ALNS, `milp_feasible_insertions` for
MILP warm-starts). Both return `InsertionCandidate`s: a fully propagated `journey`
plus `delta_obj`, ready for direct adoption with no re-validation needed.


## Plotting

`ModemsSolution` supports `plot_timing()`, `plot_soc()`, and `plot_load()`, alongside
`RoadNetwork`'s network/route plots. `ModemsInstance.plot(outdir, name=)` generates
all of them in one call and returns a dict of file paths, omitting `soc_plot` for
MILP1 to avoid confusion, since it has a worst-case (different) SoC consumption model.


## Loading results back in

Every solved `ModemsMilp`/`ModemsAlns` exposes `.instance` (`ModemsInstance`), whose
`to_json()`/`from_json()` round-trip losslessly into a live `ModemsInstance`.


## Package layout

```
src/modems/
├── core.py               # ModemsRequest, ModemsAgent, ObjectiveType, ProblemType,
│                         # SolverStrategy, resolve_problem_type,
│                         # ScenarioType/Timing/Size, ModemsScenario, ProblemContext
├── network.py            # RoadNetwork
├── solution.py           # NodeState, ModemsJourney, ModemsSolution,
│                         # ModemsSolutionInfo, ModemsInstance
├── algorithms.py         # preprocess, greedy_complete, alns_feasible_insertions,
│                         # milp_feasible_insertions (Algorithms 1-3)
├── milp.py               # MilpType, SolverConfigType, ModemsMilp
├── alns.py               # ModemsAlns
├── generator.py          # ModemsScenarioGenerator
├── rolling_horizon.py    # RollingHorizonSimulator, WorkdaySimulation, EpochLog,
│                         # RequestRecord, AgentNodeVisit, WorkdayLog
├── benchmark.py          # static (single-scenario, 4-solver) benchmark suite
├── workday_benchmark.py  # dynamic (full simulated workday) benchmark suite
└── insertion_ablation.py # Algorithm 3 (V0-V3) performance ablation suite
```


## Testing

```bash
pip install -e ".[dev]"
pytest                       # everything
pytest -m "not integration"  # fast unit tests only
pytest -m integration        # real CBC/HiGHS/alns solver tests only
```


## Benchmark suite

The three independent suites for evaluation purposes live under `benchmarks/`, check
[`benchmarks/README.md`](benchmarks/README.md) for the full CLI reference. Each script
generates + solves + builds (tables) in one command by default, prints live progress,
and is safe to interrupt/resume:

```bash
cd benchmarks
python3 run_suite.py            # static: one scenario, all 4 solvers
python3 run_workday_suite.py    # dynamic: full simulated day, MILP3 vs ALNS
python3 run_ablation_suite.py   # Algorithm 3 V0-V3 performance ablation
```

To run a quick sanity check, all scripts can be invoked with a `--smoke` argument:

```bash
cd benchmarks
python3 run_suite.py --smoke
python3 run_workday_suite.py --smoke
python3 run_ablation_suite.py --smoke
```

To generate the exact results reported in the paper, run the following commands
(note: this can span several hours depending on the hardware specs):

```bash
cd benchmarks

python3 run_ablation_suite.py \
  --outdir ablation_suite \
  --request-counts 10 30 60 \
  --agent-counts 1 2 \
  --types R C M \
  --timings U P \
  --nr-repeats 1    # insertion ablation

python3 run_suite.py \
  --outdir single_suite \
  --sizes S M L \
  --types R C M \
  --timings U P \
  --soc-test normal \
  --nr-repeats 2 \
  --solver-name gurobi \
  --solver-config-type gurobi   # single scenario suite

python3 run_workday_suite.py \
  --outdir workday_suite \
  --timings U P \
  --start-times N S \
  --base-rates 5 8 \
  --nr-surges 3 \
  --nr-repeats 3 \
  --solver-name gurobi \
  --solver-config-type gurobi \
  --plots     # multiple workdays with plots

python3 run_suite.py \
  --outdir single_plots \
  --sizes L \
  --types M \
  --timings P \
  --soc-test stress \
  --nr-repeats 1 \
  --solver-name gurobi \
  --solver-config-type gurobi \
  --plots     # single scenario with plots
```


## License

MIT -- see [`LICENSE`](LICENSE).


## Citation

Code archived at DOI:

[https://doi.org/10.5281/zenodo.22927939](https://doi.org/10.5281/zenodo.22927939)

