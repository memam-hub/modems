"""
Shared fixtures. Solver-facing tests keep instances tiny (1-2 agents, 2-5 requests)
so the integration-marked tests (real CBC/HiGHS/alns backends) stay fast enough to
run on every change. Deterministic builders live in tests/builders.py
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")  # headless plotting, before pyplot loads

from typing import Any  # noqa: E402

import pytest  # noqa: E402

from modems.core import ModemsScenario  # noqa: E402
from modems.solution import DEFAULT_PARAMS_MILP  # noqa: E402

from .builders import generated  # noqa: E402


@pytest.fixture
def default_model_params() -> dict[str, Any]:
    """Fresh copy of the default MILP parameter dict"""
    return dict(DEFAULT_PARAMS_MILP)


@pytest.fixture
def small_scenario() -> ModemsScenario:
    """Seeded scenario with 2 agents and 3 requests"""
    return generated(seed=1, nr_agents=2, nr_requests=3)


@pytest.fixture
def tiny_scenario() -> ModemsScenario:
    """Seeded scenario with 1 agent and 2 requests, for real solver calls"""
    return generated(seed=1, nr_agents=1, nr_requests=2)


@pytest.fixture(autouse=True)
def fast_plots(monkeypatch: pytest.MonkeyPatch) -> None:
    """Figures are saved at 300 dpi in production; tests only need the file"""
    import matplotlib.figure
    import matplotlib.pyplot as plt

    save_plt, save_fig = plt.savefig, matplotlib.figure.Figure.savefig
    monkeypatch.setattr(
        plt, "savefig", lambda *a, **k: save_plt(*a, **{**k, "dpi": 10})
    )
    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        lambda self, *a, **k: save_fig(self, *a, **{**k, "dpi": 10}),
    )
