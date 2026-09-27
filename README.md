# modems

MILP and ALNS solvers for the **MODEMS** on-campus passenger mobility service: a
dynamic electric Autonomous Dial-a-Ride Problem, solved exactly (three MILP
formulations, MILP1/MILP2/MILP3) or heuristically (Adaptive Large Neighborhood Search,
ALNS), with a rolling-horizon simulator for multi-epoch, full-workday operation.


## Installation

Python >= 3.12, pure Python.

```bash
python3 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -e .                 # CBC included (via pulp), no license needed
pip install -e ".[highs]"        # optional: HiGHS, free
pip install -e ".[gurobi]"       # optional: Gurobi, needs a license (free for academics)
```


## Quickstart

```python
from modems import ModemsAlns, ModemsMilp, ModemsScenarioGenerator, resolve_problem_type

scenario = ModemsScenarioGenerator(seed=42).generate_random_scenario(
    nr_agents=2, nr_requests=5
)

milp = ModemsMilp(scenario, "milp3", resolve_problem_type("milp3", "closed"))
milp.solve(solver_name="cbc", solver_config_type="cbc", solver_options={"timelimit": 60})

alns = ModemsAlns(scenario, resolve_problem_type("alns", "closed"))
alns.solve(seed=1, max_iter=500)

print(milp.instance.solution_info.objective, alns.instance.solution_info.objective)
```

`solver_name` is the name passed to pyomo's `SolverFactory` (`"cbc"`, `"appsi_highs"`,
`"gurobi"`); `solver_config_type` (`cbc`, `highs`, `gurobi`) selects how generic options
such as `timelimit` are translated for that solver. Every solved model exposes
`.instance` (`ModemsInstance`), which round-trips losslessly through
`to_json()`/`from_json()` and plots everything with `.plot(outdir)`.


## Concepts

- **Scenario** (`ModemsScenario`): agents, requests, and a road network (default: 3
  hubs, 15 stations). Scenarios are classified by spatial `type` (random, clustered,
  mixed), `timing` of the pickups (uniform, or two rush-hour peaks), and `size`
  (small, medium, large). Enums accept their first letter, as in benchmark names.
- **Objective** (`ObjectiveType`): `closed` counts the final return to a hub in the
  mission time, `open` does not. Routes always end at a reachable hub either way.
- **Selectivity**: ALNS and MILP2 may reject requests, MILP1 must serve all of them,
  MILP3 supports both (selective by default). `resolve_problem_type(strategy,
  objective)` picks the matching `ProblemType`.
- **Model parameters** (`ProblemContext`): `eps=0.01`, `zeta=2.0`, `eta=100.0`,
  `rho=2.5`, `omega=5.0` by default (with `eps < zeta < eta`, `rho >= 1`), plus the
  MILP `big_m=150`. A scenario that can never be served (e.g., a request larger than
  every vehicle) is rejected when the `ProblemContext` is built.
- **Rolling horizon** (`WorkdaySimulation`): replans every epoch as requests arrive
  during a simulated workday, with either solver or both racing, then keeps replanning
  after the day ends until every accepted request is delivered. Workday demand is a
  Poisson process shaped by the timing profile, plus optional surges.


## Benchmarks

Three resumable suites live under [`benchmarks/`](benchmarks/README.md) (full CLI
reference there). Each script generates, solves, and builds its outputs in one
command, prints live progress, and is safe to interrupt and rerun:

| Script | Evaluates | Main figure |
|---|---|---|
| `run_suite.py` | Static scenarios, all four solvers | `solver_comparison.png` |
| `run_workday_suite.py` | Full simulated workdays, MILP3 vs ALNS | per-workday SoC/acceptance |
| `run_ablation_suite.py` | Algorithm 3 (V0-V3), normal vs low SoC | `insertion_variants.png`, `normal_vs_stress.png` |
| `compare_workday_objectives.py` | Two workday suites, closed vs open objective | `var_decision_impact.png`, `var_decision_boxes.png` |

Result tables are written as `csv,json` by default; add `--export-formats all` (or,
e.g., `csv,tex`) for LaTeX tables. Seeds depend only on the scenario or workday, never
on a decision (solver, objective, start time, insertion variant), so every decision is
compared on identical demand.

```bash
cd benchmarks
python3 run_suite.py --smoke           # quick sanity checks, a few minutes each
python3 run_workday_suite.py --smoke
python3 run_ablation_suite.py --smoke
bash run_full_benchmark.sh             # results in the paper: takes several hours
```


## Package layout

```
src/modems/
├── core.py                # requests, agents, scenarios, problem types, ProblemContext
├── network.py             # RoadNetwork
├── generator.py           # ModemsScenarioGenerator (static scenarios, workday demand)
├── solution.py            # journeys, solutions, ModemsInstance
├── algorithms.py          # preprocessing, greedy construction, feasible insertions
├── milp.py                # ModemsMilp (MILP1-3)
├── alns.py                # ModemsAlns
├── rolling_horizon.py     # epoch-by-epoch simulation of a workday
├── benchmark.py           # static benchmark suite
├── workday_benchmark.py   # workday benchmark suite
├── workday_comparison.py  # closed vs open workday comparison
├── insertion_ablation.py  # Algorithm 3 ablation suite
└── plotting.py            # shared figure helpers
```


## Testing

```bash
pip install -e ".[dev]"
pytest                       # everything (about a minute)
pytest -m "not integration"  # fast unit tests only (seconds)
```


## License

MIT -- see [`LICENSE`](LICENSE).


## Citation

Code archived at [https://doi.org/10.5281/zenodo.22927939](https://doi.org/10.5281/zenodo.22927939).
