"""
modems: MILP and ALNS solvers for the MODEMS on-campus passenger mobility
routing problem (electric autonomous Dial-a-Ride problem variant).

Quickstart:

    from modems import ModemsScenarioGenerator, ModemsMilp, MilpType, \\
        SolverConfigType, ModemsAlns, ProblemType

    generator = ModemsScenarioGenerator(seed=42)
    scenario = generator.generate_random_scenario(nr_agents=2, nr_requests=5)

    milp = ModemsMilp(scenario=scenario, milp_type=MilpType.milp3,
                       problem_type=ProblemType.closed_selective,
                       milp_params={"eps": 0.01, "zeta": 1.0,
                                    "eta": 100.0, "rho": 2.5,
                                    "omega": 5.0, "big_m": 150.0})
    milp.solve(solver_name="cbc", solver_config_type=SolverConfigType.cbc,
               solver_options={"timelimit": 60.0})

To resolve the problem type from solver + objective, use resolve_problem_type:
resolve_problem_type(SolverStrategy.alns, ObjectiveType.open) -> open_selective.

See benchmarks/run_suite.py --smoke for a full runnable example (generate,
solve, and build a summary table across every solver in one command).
"""

__version__ = "0.1.2"

# --- algorithms: solver-agnostic construction/completion/insertion (Algorithms 1-3)
from .algorithms import (
    InsertionCandidate,
    alns_feasible_insertions,
    greedy_complete,
    milp_feasible_insertions,
    preprocess,
)

# --- alns: Adaptive Large Neighborhood Search
from .alns import ModemsAlns, plot_alns_metrics_from_stats

# --- benchmark: suite generation, resumable solving, and table building
from .benchmark import (
    ModemsBenchmarkResult,
    build_benchmark_table,
    generate_benchmark_suite,
    read_manifest,
    solve_benchmark_suite,
)

# --- core: problem definition (requests, agents, scenario, problem context)
from .core import (
    ModemsAgent,
    ModemsRequest,
    ModemsScenario,
    ObjectiveType,
    ProblemContext,
    ProblemType,
    RequestStatus,
    ScenarioSize,
    ScenarioTiming,
    ScenarioType,
    SolverStrategy,
    resolve_problem_type,
    scenario_bucket,
    scenario_size_of,
)

# --- generator: reproducible random-scenario generation
from .generator import (
    ModemsScenarioGenerator,
)

# --- insertion_ablation: Algorithm 3 (V0-V3) insertion-candidate-search ablation
from .insertion_ablation import (
    build_ablation_table,
    compare_route_length_buckets,
    generate_ablation_suite,
    read_ablation_manifest,
    solve_ablation_suite,
    write_corpus_comparison,
)

# --- milp: MILP1/MILP2/MILP3 formulations
from .milp import MilpType, ModemsMilp, SolverConfigType

# --- network: node naming/types, the road network container
from .network import NetworkNodeName, NetworkNodeType, RoadNetwork

# --- rolling_horizon: agent state advancement between planning epochs
from .rolling_horizon import (
    AgentAdvanceResult,
    AgentNodeVisit,
    AgentOperationalState,
    EpochLog,
    RequestOutcome,
    RequestRecord,
    RollingHorizonSimulator,
    WorkdayLog,
    WorkdaySimulation,
    advance_agent_state,
    build_workday_log,
    charging_duration_min,
    compare_solvers_one_workday,
    compare_solvers_over_workdays,
    summarize_workday,
)

# --- solution: route/schedule/result (ModemsJourney, ModemsSolution, ModemsInstance)
from .solution import (
    ModemsInstance,
    ModemsJourney,
    ModemsSolution,
    ModemsSolutionInfo,
    NodeState,
    SolutionStatus,
)

# --- workday_benchmark: resumable full-day rolling-horizon (MILP3-vs-ALNS) suite
from .workday_benchmark import (
    WorkdayStartTime,
    build_workday_requests_table,
    build_workday_summary_table,
    generate_workday_suite,
    plot_workday_soc_acceptance,
    read_workday_manifest,
    solve_workday_suite,
)

__all__ = [  # noqa: RUF022 -- grouped by module (with comments) on purpose
    "__version__",
    # core
    "ModemsAgent",
    "ModemsRequest",
    "ModemsScenario",
    "ObjectiveType",
    "ProblemContext",
    "ProblemType",
    "RequestStatus",
    "ScenarioSize",
    "ScenarioTiming",
    "ScenarioType",
    "SolverStrategy",
    "resolve_problem_type",
    "scenario_bucket",
    "scenario_size_of",
    # network
    "NetworkNodeName",
    "NetworkNodeType",
    "RoadNetwork",
    # solution
    "ModemsInstance",
    "ModemsJourney",
    "ModemsSolution",
    "ModemsSolutionInfo",
    "NodeState",
    "SolutionStatus",
    # algorithms
    "InsertionCandidate",
    "alns_feasible_insertions",
    "greedy_complete",
    "milp_feasible_insertions",
    "preprocess",
    # generator
    "ModemsScenarioGenerator",
    # milp
    "MilpType",
    "SolverConfigType",
    "ModemsMilp",
    # alns
    "ModemsAlns",
    "plot_alns_metrics_from_stats",
    # benchmark
    "ModemsBenchmarkResult",
    "build_benchmark_table",
    "generate_benchmark_suite",
    "read_manifest",
    "solve_benchmark_suite",
    # rolling_horizon
    "AgentAdvanceResult",
    "AgentNodeVisit",
    "AgentOperationalState",
    "EpochLog",
    "RequestOutcome",
    "RequestRecord",
    "RollingHorizonSimulator",
    "WorkdayLog",
    "WorkdaySimulation",
    "advance_agent_state",
    "build_workday_log",
    "charging_duration_min",
    "compare_solvers_one_workday",
    "compare_solvers_over_workdays",
    "summarize_workday",
    # workday_benchmark
    "WorkdayStartTime",
    "build_workday_requests_table",
    "build_workday_summary_table",
    "generate_workday_suite",
    "plot_workday_soc_acceptance",
    "read_workday_manifest",
    "solve_workday_suite",
    # insertion_ablation
    "build_ablation_table",
    "compare_route_length_buckets",
    "generate_ablation_suite",
    "read_ablation_manifest",
    "solve_ablation_suite",
    "write_corpus_comparison",
]
