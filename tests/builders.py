"""
Deterministic builders shared across the test suite.

  - line_scenario(): a hand-built network where every node sits on a line and travel
    times are exact distances. Timing, load, and SoC can be derived by hand, so tests
    assert exact numbers rather than loose properties.
  - generated(): seeded ModemsScenarioGenerator instances for property/oracle tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from modems.algorithms import greedy_complete, preprocess
from modems.core import (
    ModemsAgent,
    ModemsRequest,
    ModemsScenario,
    ObjectiveType,
    ProblemContext,
    SolverStrategy,
    resolve_problem_type,
)
from modems.generator import ModemsScenarioGenerator
from modems.network import NetworkNodeType, RoadNetwork
from modems.solution import ModemsSolution


def line_network(positions: list[float], nr_hubs: int = 1) -> RoadNetwork:
    """
    A network with nodes on a line at the given positions (hubs first); travel time
    between two nodes is their absolute distance, satisfying triangle inequality
    """
    x = np.asarray(positions, dtype=float)
    locations = np.column_stack([x, np.zeros_like(x)])
    travel_times = np.abs(x[:, None] - x[None, :])
    return RoadNetwork(
        nr_hubs=nr_hubs,
        nr_stations=len(positions) - nr_hubs,
        locations=locations,
        travel_times=travel_times,
    )


def line_scenario(
    requests: list[ModemsRequest],
    agents: list[ModemsAgent] | None = None,
    positions: list[float] | None = None,
) -> ModemsScenario:
    """
    Scenario on line_network(); one hub at 0 and stations s_i at i,
    with a single agent starting at the hub at t=0 with a full charge
    """
    positions = positions or [float(i) for i in range(0, 7)]
    agents = agents or [ModemsAgent(NetworkNodeType.hub, 1, agent_id="k1")]
    return ModemsScenario(agents, requests, line_network(positions))


def make_ctx(
    scenario: ModemsScenario,
    strategy: SolverStrategy | str = SolverStrategy.alns,
    objective: ObjectiveType | str = ObjectiveType.closed,
    selective: bool | None = None,
    **model_params: Any,
) -> ProblemContext:
    """ProblemContext with the strategy-default (or given) selectivity"""
    problem_type = resolve_problem_type(strategy, objective, selective)
    return ProblemContext(scenario, problem_type, strategy, model_params or None)


def greedy_solution(ctx: ProblemContext) -> ModemsSolution:
    """Algorithm 1 + 2 on a context with only new requests; asserts success"""
    base, unassigned = preprocess(ctx)
    assert base is not None
    solution, _, ok = greedy_complete(base, unassigned)
    assert ok
    return solution


def generated(seed: int = 1, **kwargs: Any) -> ModemsScenario:
    """Wrapper for seeded generate_random_scenario()"""
    return ModemsScenarioGenerator(seed=seed).generate_random_scenario(**kwargs)


def duration_limited_scenario() -> ModemsScenario:
    """
    One agent with just enough SoC to travel ~6.6 min (MILP1 duration budget D^k):
    request_1 (s1 -> s2, round trip 4 min) fits, request_2 (s5 -> s6, 12 min once
    appended) does not, so a non-selective problem has no feasible solution
    """
    agent = ModemsAgent("h", 1, soc_initial=0.30, soc_min_operational=0.25)
    requests = [
        ModemsRequest(1, 2, earliest_pickup=10.0, request_id="fits"),
        ModemsRequest(5, 6, earliest_pickup=20.0, request_id="too_far"),
    ]
    return line_scenario(requests, agents=[agent])
