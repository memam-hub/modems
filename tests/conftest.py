"""
Shared pytest fixtures used across the test suite.

Scenario fixtures are intentionally tiny (1-2 agents, 2-5 requests): solver-facing
tests (test_milp.py, test_alns.py, test_benchmark.py, test_rolling_horizon.py)
run the real CBC/HiGHS/alns backends, so keeping instances small is what makes the
integration-marked tests fast enough to run routinely rather than only in CI.
"""

from __future__ import annotations

from typing import Any

import pytest

from modems.core import ModemsScenario
from modems.generator import ModemsScenarioGenerator
from modems.solution import DEFAULT_PARAMS_MILP


@pytest.fixture
def default_model_params() -> dict[str, Any]:
    """Return a fresh copy of the shared model-penalty parameter dict"""
    return dict(DEFAULT_PARAMS_MILP)


@pytest.fixture
def small_scenario() -> ModemsScenario:
    """Build a small (2 agents, 3 requests) seeded scenario for unit tests"""
    gen = ModemsScenarioGenerator(seed=1)
    return gen.generate_random_scenario(nr_agents=2, nr_requests=3)


@pytest.fixture
def tiny_scenario() -> ModemsScenario:
    """Build a minimal (1 agent, 2 requests) seeded scenario for solver tests"""
    gen = ModemsScenarioGenerator(seed=1)
    return gen.generate_random_scenario(nr_agents=1, nr_requests=2)
