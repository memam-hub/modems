from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .benchmark import ManifestStatus
from .core import ObjectiveType, SolverStrategy
from .network import RoadNetwork
from .plotting import draw_hboxes, save_figure, setup_fig_grid
from .rolling_horizon import RequestOutcome, WorkdayLog
from .workday_benchmark import (
    BNCH_WORKDAY_PFX,
    SUMMARY_FIELDS,
    WorkdayStartTime,
    agent_soc_series,
    read_workday_manifest,
    workday_summary_row,
    write_workday_summary_files,
)

OBJECTIVES = (ObjectiveType.closed, ObjectiveType.open)
SOLVERS = (SolverStrategy.milp3, SolverStrategy.alns)
SOLVER_LABEL = {SolverStrategy.milp3: "MILP3", SolverStrategy.alns: "ALNS"}
SOLVER_COLOR = {SolverStrategy.milp3: "tab:blue", SolverStrategy.alns: "tab:red"}
OBJECTIVE_STYLE = {ObjectiveType.closed: "-", ObjectiveType.open: "--"}
OBJECTIVE_MARKER = {ObjectiveType.closed: "o", ObjectiveType.open: "^"}
DEFAULT_ROLLING_WINDOW = 30.0  # minutes of submissions behind each acceptance point
DEFAULT_DEMAND_BIN = 15.0  # minutes per bar of the demand strip
COMBINED_EXTRA_FIELDS = [
    "pair",
    "demand_load",
    *(
        f"{m}_per_served_{s}"
        for m in ("travel_time", "energy")
        for s in ("milp3", "alns")
    ),
]


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    value: Callable[[dict[str, Any]], float | None]  # of one solver's result metrics


def _per_served(total_key: str) -> Callable[[dict[str, Any]], float | None]:
    def value(result: dict[str, Any]) -> float | None:
        served = result["nr_requests_completed"]
        return result[total_key] / served if served else None

    return value


def _percentage(key: str) -> Callable[[dict[str, Any]], float | None]:
    """A rate in percent; None when undefined (no decided requests)"""
    return lambda result: None if result[key] is None else 100.0 * result[key]


def _if_served(key: str) -> Callable[[dict[str, Any]], float | None]:
    """Means over served requests only exist if something was served"""
    return lambda result: result[key] if result["nr_requests_completed"] else None


METRICS = [
    Metric("acceptance", "Acceptance (pp)", _percentage("acceptance_rate")),
    Metric("delay", "Mean delay (min)", _if_served("mean_delay_time")),
    Metric(
        "excess_ride", "Mean excess ride (min)", _if_served("mean_excess_ride_time")
    ),
    Metric("travel", "Travel / served (min)", _per_served("total_agent_travel_time")),
    Metric("energy", "Energy / served (SoC)", _per_served("total_energy_consumed")),
]


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------

# two-sided 95% Student-t critical values for df = 1..30
_T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
         2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
         2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]  # fmt: skip


def t_critical(df: int) -> float:
    """Two-sided 95% t critical value (the normal 1.96 beyond df = 30)"""
    if df < 1:
        return math.inf
    return _T975[df - 1] if df <= 30 else 1.96


def mean_ci(values: list[float]) -> tuple[float, float] | None:
    """(mean, 95% CI half-width) of a sample; None if empty, inf half-width if n=1"""
    if not values:
        return None
    x = np.asarray(values, dtype=float)
    if len(x) == 1:
        return float(x[0]), math.inf
    return float(x.mean()), t_critical(len(x) - 1) * float(x.std(ddof=1)) / math.sqrt(
        len(x)
    )


# --------------------------------------------------------------------------------------
# Loading and pairing
# --------------------------------------------------------------------------------------


def pair_name(workday_name: str) -> str:
    """Workday name without its objective letter: WC0_SRUN5_0 -> W0_SRUN5_0"""
    n = len(BNCH_WORKDAY_PFX)
    return workday_name[:n] + workday_name[n + 1 :]


@dataclass
class WorkdayRun:
    """One solved workday of one suite: its manifest row and per-solver results"""

    name: str
    objective: ObjectiveType
    row: dict[str, Any]
    results: dict[SolverStrategy, dict[str, Any]]
    nr_submissions: int

    @property
    def pair(self) -> str:
        return pair_name(self.name)

    @property
    def demand_load(self) -> float:
        """Submitted requests per agent-hour"""
        nr_hours = self.row["workday_length"] / 60.0
        return self.nr_submissions / (self.row["nr_agents"] * nr_hours)

    def logs(self) -> dict[SolverStrategy, WorkdayLog]:
        with open(self.row["workday_log_file"]) as f:
            raw_json = json.load(f)
        return {SolverStrategy(k): WorkdayLog.from_dict(v) for k, v in raw_json.items()}


_PATH_FIELDS = ("result_file", "workday_log_file")


def _read_manifest_resolved(outdir: str) -> dict[str, dict[str, Any]]:
    """
    read_workday_manifest() with file paths usable from any working directory: paths
    are stored as given at solve time, e.g., relative to the folder the suite was run
    from (the suite's parent), so a relative path that does not exist as-is is
    resolved against the suite's parent folder
    """
    parent = os.path.dirname(os.path.abspath(outdir))
    manifest = read_workday_manifest(outdir)
    for row in manifest.values():
        for field in _PATH_FIELDS:
            path = row.get(field)
            if path and not os.path.isabs(path) and not os.path.exists(path):
                row[field] = os.path.join(parent, path)
    return manifest


def _load_suite(outdir: str, objective: ObjectiveType) -> dict[str, WorkdayRun]:
    """Solved workdays of one suite keyed by pair name; raises on a wrong objective"""
    runs = {}
    for name, row in _read_manifest_resolved(outdir).items():
        if ObjectiveType(row["objective_type"]) != objective:
            raise ValueError(
                f"{outdir!r} must only contain {objective} workdays, found {name!r} "
                f"({row['objective_type']})"
            )
        if ManifestStatus(row["status"]) != ManifestStatus.done:
            continue
        with open(row["result_file"]) as f:
            data = json.load(f)
        runs[pair_name(name)] = WorkdayRun(
            name=name,
            objective=objective,
            row=row,
            results={s: data[s.value] for s in SOLVERS},
            nr_submissions=data["nr_submissions"],
        )
    return runs


def _check_pairs(closed: dict[str, WorkdayRun], opened: dict[str, WorkdayRun]) -> None:
    """Paired workdays must have the same seed (same request demand stream)"""
    for pair in closed.keys() & opened.keys():
        seeds = closed[pair].row["seed"], opened[pair].row["seed"]
        if seeds[0] != seeds[1]:
            raise ValueError(
                f"{pair}: closed and open workdays have different seeds {seeds}; "
                "generate both suites with the same --seed and parameters"
            )


# --------------------------------------------------------------------------------------
# Combined manifest and summary table
# --------------------------------------------------------------------------------------


def _combined_rows(closed_dir: str, open_dir: str) -> tuple[list[dict], list[dict]]:
    """Both manifests and both summaries, ordered by workday, closed before open"""
    manifest_rows, summary_rows = [], []
    for _, outdir in zip(OBJECTIVES, (closed_dir, open_dir)):
        for name, row in _read_manifest_resolved(outdir).items():
            manifest_rows.append({**row, "pair": pair_name(name)})
            summary = workday_summary_row(name, row)
            summary["pair"] = pair_name(name)
            for s in ("milp3", "alns"):
                served = summary[f"nr_served_{s}"]
                for metric, total in (
                    ("travel_time", "travel_time"),
                    ("energy", "energy_consumed"),
                ):
                    value = summary[f"{total}_{s}"]
                    summary[f"{metric}_per_served_{s}"] = (
                        value / served if served else None
                    )
            if summary["nr_submissions"] is not None:
                nr_hours = row["workday_length"] / 60.0
                summary["demand_load"] = summary["nr_submissions"] / (
                    row["nr_agents"] * nr_hours
                )
            else:
                summary["demand_load"] = None
            summary_rows.append(summary)

    def order(r: dict) -> tuple:
        return r["pair"], OBJECTIVES.index(ObjectiveType(r["objective_type"]))

    return sorted(manifest_rows, key=order), sorted(summary_rows, key=order)


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def demand_key(run: WorkdayRun) -> tuple:
    """The workday parameters that define its demand, i.e., everything but decisions"""
    row = run.row
    return tuple(
        row[k]
        for k in ("size", "type", "timing", "base_rate", "nr_surges", "repetition")
    )


def start_time_pairs(
    closed: dict[str, WorkdayRun], opened: dict[str, WorkdayRun]
) -> list[tuple[dict[ObjectiveType, WorkdayRun], dict[ObjectiveType, WorkdayRun]]]:
    """
    (normal, staggered) runs per objective of every workday observed with both start
    times on identical demand (same seed)
    """
    by_key: dict[tuple, dict[WorkdayStartTime, dict[ObjectiveType, WorkdayRun]]] = {}
    for runs in (closed, opened):
        for run in runs.values():
            start = WorkdayStartTime(run.row["start_time"])
            by_key.setdefault(demand_key(run), {}).setdefault(start, {})[
                run.objective
            ] = run
    pairs = []
    for key in sorted(by_key, key=str):
        normal = by_key[key].get(WorkdayStartTime.normal, {})
        staggered = by_key[key].get(WorkdayStartTime.staggered, {})
        same_demand = {
            o: normal[o]
            for o in normal.keys() & staggered.keys()
            if normal[o].row["seed"] == staggered[o].row["seed"]
        }
        if same_demand:
            pairs.append((same_demand, {o: staggered[o] for o in same_demand}))
    return pairs


def impact_estimates(
    closed: dict[str, WorkdayRun], opened: dict[str, WorkdayRun]
) -> list[dict[str, Any]]:
    """
    Figure 1 data: one entry per (metric, decision, group), the mean difference and
    95% CI half-width, and the number of workdays behind it
      - objective, per solver: paired open - closed over workdays in both suites
      - solver, per objective: paired ALNS - MILP3 over that suite's workdays
      - start time, per solver: paired staggered - normal over workdays with identical
        demand (check start_time_pairs), averaged over both objectives per workday
    """
    pairs = sorted(closed.keys() & opened.keys())
    start_pairs = start_time_pairs(closed, opened)
    estimates = []

    def add(metric, decision, group, estimate, n):
        mean, half = estimate if estimate is not None else (None, None)
        estimates.append(
            dict(
                metric=metric.key,
                decision=decision,
                group=group,
                mean=mean,
                ci=half,
                n=n,
            )
        )

    for metric in METRICS:
        for s in SOLVERS:
            diffs = [
                metric.value(opened[p].results[s]) - metric.value(closed[p].results[s])
                for p in pairs
                if metric.value(opened[p].results[s]) is not None
                and metric.value(closed[p].results[s]) is not None
            ]
            add(metric, "objective", SOLVER_LABEL[s], mean_ci(diffs), len(diffs))
        for objective, runs in zip(OBJECTIVES, (closed, opened)):
            diffs = [
                metric.value(r.results[SolverStrategy.alns])
                - metric.value(r.results[SolverStrategy.milp3])
                for r in runs.values()
                if metric.value(r.results[SolverStrategy.alns]) is not None
                and metric.value(r.results[SolverStrategy.milp3]) is not None
            ]
            add(metric, "solver", objective.value, mean_ci(diffs), len(diffs))
        for s in SOLVERS:
            diffs = []
            for normal, staggered in start_pairs:
                # staggered - normal of each objective present, averaged per workday
                per_objective = [
                    metric.value(staggered[o].results[s])
                    - metric.value(normal[o].results[s])
                    for o in normal.keys() & staggered.keys()
                    if metric.value(staggered[o].results[s]) is not None
                    and metric.value(normal[o].results[s]) is not None
                ]
                if per_objective:
                    diffs.append(float(np.mean(per_objective)))
            add(metric, "start_time", SOLVER_LABEL[s], mean_ci(diffs), len(diffs))
    return estimates


def _group_style(group: str) -> tuple[str, str]:
    """(colour, marker face) of a Figure 1 group: a solver or an objective"""
    for s, label in SOLVER_LABEL.items():
        if group == label:
            return SOLVER_COLOR[s], SOLVER_COLOR[s]
    return "k", "k" if group == ObjectiveType.closed else "white"


DECISION_TITLES = {
    "objective": "Objective: open - closed",
    "solver": "Solver: ALNS - MILP3",
    "start_time": "Start time: staggered - normal",
}


def plot_impacts(estimates: list[dict[str, Any]], path: str) -> str:
    """Figure 1: rows = outcomes, columns = decisions, dot + 95% CI per group"""
    decisions = list(DECISION_TITLES)
    fig, axes = setup_fig_grid(len(METRICS), len(decisions), 18.0, 3.0 * len(METRICS))
    for i, metric in enumerate(METRICS):
        row = [e for e in estimates if e["metric"] == metric.key]
        finite = [
            abs(e["mean"]) + (e["ci"] if math.isfinite(e["ci"]) else 0.0)
            for e in row
            if e["mean"] is not None
        ]
        limit = 1.15 * max(finite, default=1.0) or 1.0
        for j, decision in enumerate(decisions):
            ax = axes[i, j]
            cells = [e for e in row if e["decision"] == decision]
            ax.axvline(0.0, color="k", linewidth=1.5, zorder=1)
            for y, e in enumerate(cells):
                if e["mean"] is None:
                    ax.text(0.0, y, "n/a", ha="center", va="center", color="gray")
                    continue
                ci = e["ci"] if math.isfinite(e["ci"]) else 0.0
                color, face = _group_style(e["group"])
                ax.errorbar(
                    e["mean"],
                    y,
                    xerr=ci,
                    fmt="o",
                    markersize=10,
                    capsize=6,
                    color=color,
                    markerfacecolor=face,
                    markeredgewidth=2,
                    zorder=3,
                )
                ax.text(
                    limit,
                    y,
                    f" n={e['n']}",
                    va="center",
                    ha="left",
                    fontsize=11,
                    color="gray",
                )
            ax.set_yticks(range(len(cells)), [e["group"] for e in cells])
            ax.set_ylim(-0.6, len(cells) - 0.4)
            ax.set_xlim(-limit, limit)
            if i == 0:
                ax.set_title(DECISION_TITLES[decision])
            if j == 0:
                ax.set_ylabel(metric.label)
    fig.tight_layout()
    return save_figure(fig, path)


def plot_demand(runs: list[WorkdayRun], path: str) -> str:
    """
    Figure 2: acceptance and mean delay against demand load (requests per agent-hour),
    binned means with 95% CI band per solver x objective; workdays with surges hollow
    """
    panels = [METRICS[0], METRICS[1]]
    fig, axes = setup_fig_grid(1, len(panels), 18.0, 7.0)
    loads = np.array([r.demand_load for r in runs])
    nr_bins = int(min(6, max(1, len(set(loads.round(6))) // 3)))
    edges = np.unique(np.quantile(loads, np.linspace(0, 1, nr_bins + 1)))
    handles = []
    for ax, metric in zip(axes[0], panels):
        for objective in OBJECTIVES:
            for s in SOLVERS:
                color, style = SOLVER_COLOR[s], OBJECTIVE_STYLE[objective]
                points = [
                    (r.demand_load, metric.value(r.results[s]), r.row["nr_surges"] > 0)
                    for r in runs
                    if r.objective == objective
                    and metric.value(r.results[s]) is not None
                ]
                if not points:
                    continue
                x, y, surge = (np.array(v) for v in zip(*points))
                marker = OBJECTIVE_MARKER[objective]
                ax.scatter(
                    x[~surge],
                    y[~surge],
                    color=color,
                    marker=marker,
                    alpha=0.35,
                    s=40,
                    zorder=2,
                )
                ax.scatter(
                    x[surge],
                    y[surge],
                    facecolors="none",
                    edgecolors=color,
                    marker=marker,
                    alpha=0.6,
                    s=50,
                    zorder=2,
                )
                centers, means, halves = [], [], []
                for lo, hi in zip(edges[:-1], edges[1:]):
                    inside = (x >= lo) & ((x <= hi) if hi == edges[-1] else (x < hi))
                    if inside.any():
                        mean, half = mean_ci(list(y[inside]))
                        centers.append(float(x[inside].mean()))
                        means.append(mean)
                        halves.append(half if math.isfinite(half) else 0.0)
                means_a, halves_a = np.array(means), np.array(halves)
                (line,) = ax.plot(
                    centers,
                    means,
                    style,
                    color=color,
                    linewidth=3,
                    zorder=3,
                    label=f"{SOLVER_LABEL[s]}, {objective}",
                )
                low, high = means_a - halves_a, means_a + halves_a
                if metric.key == "acceptance":
                    low, high = np.clip(low, 0.0, 100.0), np.clip(high, 0.0, 100.0)
                else:
                    low = np.maximum(low, 0.0)  # delays are non-negative
                ax.fill_between(centers, low, high, color=color, alpha=0.12, zorder=1)
                if ax is axes[0, 0]:
                    handles.append(line)
        ax.set_xlabel("Demand load (requests per agent-hour)")
        ax.set_ylabel(metric.label.replace("(pp)", "(%)"))
        if metric.key == "acceptance":
            ax.set_ylim(top=102.0)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=4,
        title="circles: closed, triangles: open, hollow: workdays with surges",
    )
    fig.tight_layout()
    return save_figure(fig, path)


def rolling_acceptance(
    log: WorkdayLog, window: float
) -> tuple[list[float], list[float]]:
    """Acceptance % among the requests submitted in the last `window` minutes"""
    decided = sorted(
        (r.submission_time, r.outcome == RequestOutcome.accepted)
        for r in log.requests
        if r.outcome in (RequestOutcome.accepted, RequestOutcome.rejected)
    )
    times, rates = [], []
    for t, _ in decided:
        recent = [accepted for s, accepted in decided if t - window < s <= t]
        times.append(t)
        rates.append(100.0 * sum(recent) / len(recent))
    return times, rates


def plot_workday(
    closed: WorkdayRun,
    opened: WorkdayRun,
    path: str,
    window: float = DEFAULT_ROLLING_WINDOW,
    demand_bin: float = DEFAULT_DEMAND_BIN,
) -> str:
    """
    Figure 3: rows = solver, columns = objective; rolling acceptance (left axis) and
    agent SoC (right axis) per run, above one demand strip shared by all four runs
    """
    import matplotlib.pyplot as plt

    plt.rc("font", family="serif", size=16)
    fig = plt.figure(figsize=(18.0, 12.0))
    grid = fig.add_gridspec(3, 2, height_ratios=[3, 3, 1.2], hspace=0.35, wspace=0.25)
    logs = {o: run.logs() for o, run in zip(OBJECTIVES, (closed, opened))}
    # the day ends at clock_end, or at the last committed arrival if later
    t_end = max(
        t[-1]
        for by_solver in logs.values()
        for log in by_solver.values()
        for _, t, _ in agent_soc_series(log)
    )
    strip = fig.add_subplot(grid[2, :])
    handles = []
    for i, s in enumerate(SOLVERS):
        for j, objective in enumerate(OBJECTIVES):
            log = logs[objective][s]
            ax = fig.add_subplot(grid[i, j], sharex=strip)
            ax.grid(True, alpha=0.5, zorder=-2)
            ax.set_ylim(-2, 102)
            ax.set_title(f"{SOLVER_LABEL[s]}, {objective}")
            if j == 0:
                ax.set_ylabel(f"Acceptance, last {window:g} min (%)")
            ax.tick_params(labelbottom=False)
            times, rates = rolling_acceptance(log, window)
            if times:
                times, rates = [*times, t_end], [*rates, rates[-1]]
            (acc,) = ax.step(
                times,
                rates,
                where="post",
                color="tab:green",
                linewidth=3,
                label="Rolling acceptance",
            )
            ax_soc = ax.twinx()
            ax_soc.set_ylim(-0.02, 1.02)
            if j == 1:
                ax_soc.set_ylabel("Agent SoC")
            soc_lines = [
                ax_soc.plot(
                    t,
                    soc,
                    "--",
                    linewidth=2,
                    color=RoadNetwork._agent_color(k),
                    label=f"{name} SoC",
                )[0]
                for k, (name, t, soc) in enumerate(agent_soc_series(log))
            ]
            if not handles:
                handles = [acc, *soc_lines]
    submissions = [
        r.submission_time
        for r in logs[ObjectiveType.closed][SolverStrategy.milp3].requests
    ]
    bins = np.arange(0.0, t_end + demand_bin, demand_bin)
    strip.hist(submissions, bins=bins, color="gray", edgecolor="white")
    strip.grid(True, alpha=0.5, zorder=-2)
    strip.set_xlim(0.0, t_end)
    strip.set_xlabel("Time (min)")
    strip.set_ylabel(f"Requests / {demand_bin:g} min")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=4)
    fig.suptitle(
        f"{closed.pair}: demand load {closed.demand_load:.1f} requests per agent-hour"
    )
    return save_figure(fig, path)


def representative_pair(
    closed: dict[str, WorkdayRun], opened: dict[str, WorkdayRun]
) -> str | None:
    """Highest demand load among paired workdays with surges (else among all)"""
    pairs = sorted(closed.keys() & opened.keys())
    with_surges = [p for p in pairs if closed[p].row["nr_surges"] > 0]
    candidates = with_surges or pairs
    return max(candidates, key=lambda p: (closed[p].demand_load, p), default=None)


# --------------------------------------------------------------------------------------
# Box-plot figures (alternative views of Figures 1 and 2)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Factor:
    """A grouping axis of the box-plot grids: its levels and how to read one run"""

    title: str
    levels: tuple[str, ...]
    level: Callable[[WorkdayRun, SolverStrategy], str]


def _solver_factor() -> Factor:
    return Factor(
        "Solver", tuple(SOLVER_LABEL[s] for s in SOLVERS), lambda r, s: SOLVER_LABEL[s]
    )


def _row_factor(title: str, field: str, levels: list) -> Factor:
    return Factor(title, tuple(str(v) for v in levels), lambda r, s: str(r.row[field]))


def _box_records(runs: list[WorkdayRun]) -> list[tuple[WorkdayRun, SolverStrategy]]:
    """One record per run and solver: 2 objectives x 2 solvers per workday"""
    return [(r, s) for r in runs for s in SOLVERS]


def _box_color(level_index: int, shade_index: int) -> tuple:
    """Colour by the first factor's level, lighter for the second factor's 2nd level"""
    import matplotlib.colors as mcolors

    base = mcolors.to_rgb(
        ("tab:blue", "tab:orange", "tab:green", "tab:purple")[level_index % 4]
    )
    return tuple(c + (1.0 - c) * 0.55 * shade_index for c in base)


def plot_box_grid(
    runs: list[WorkdayRun],
    columns: list[tuple[Factor, Factor]],
    path: str,
    title: str = "",
) -> str:
    """
    Rows = outcomes, columns = pairs of factors; each cell draws one horizontal
    box-and-whisker per combination of its two factors' levels, pooling everything
    else. Whiskers extend to the furthest value within 1.5 x IQR (Tukey), with
    outliers as dots. Every cell of a row shares the x-axis [min, max] of that
    outcome across all runs
    """
    records = _box_records(runs)
    fig, axes = setup_fig_grid(len(METRICS), len(columns), 18.0, 3.2 * len(METRICS))
    for i, metric in enumerate(METRICS):
        values = [metric.value(r.results[s]) for r, s in records]
        values = [v for v in values if v is not None]
        low, high = (min(values), max(values)) if values else (0.0, 1.0)
        pad = 0.05 * (high - low) or 0.05 * max(abs(high), 1.0)
        for j, (first, second) in enumerate(columns):
            ax = axes[i, j]
            data, labels, colors = [], [], []
            for a, level_a in enumerate(first.levels):
                for b, level_b in enumerate(second.levels):
                    group = [
                        metric.value(r.results[s])
                        for r, s in records
                        if first.level(r, s) == level_a
                        and second.level(r, s) == level_b
                        and metric.value(r.results[s]) is not None
                    ]
                    data.append(group)
                    labels.append(f"{level_a} · {level_b}")
                    colors.append(_box_color(a, b))
            draw_hboxes(ax, data, labels, colors)
            ax.set_xlim(low - pad, high + pad)
            if i == 0:
                ax.set_title(f"{first.title} × {second.title}")
            if j == 0:
                ax.set_ylabel(metric.label.replace("(pp)", "(%)"))
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return save_figure(fig, path)


def plot_impact_boxes(runs: list[WorkdayRun], path: str) -> str:
    """Box-plot view of the decision impacts: the three decisions, pairwise"""
    objective = Factor(
        "Objective", tuple(o.value for o in OBJECTIVES), lambda r, s: r.objective.value
    )
    start = _row_factor("Start time", "start_time", [t.value for t in WorkdayStartTime])
    solver = _solver_factor()
    columns = [(objective, solver), (solver, start), (start, objective)]
    return plot_box_grid(runs, columns, path)


def plot_demand_boxes(runs: list[WorkdayRun], path: str) -> str:
    """
    Box-plot view of the demand figure: the solver against the demand parameters varied by the
    suite (base arrival rate and timing profile), pairwise
    """
    rates = sorted({r.row["base_rate"] for r in runs})
    timings = [
        t for t in ("uniform", "peaks") if any(r.row["timing"] == t for r in runs)
    ]
    rate = Factor(
        "Base rate (req/h)",
        tuple(str(v) for v in rates),
        lambda r, s: str(r.row["base_rate"]),
    )
    timing = _row_factor("Timing", "timing", timings)
    solver = _solver_factor()
    columns = [(rate, solver), (timing, solver), (rate, timing)]
    return plot_box_grid(runs, columns, path)


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def compare_workday_objectives(
    closed_dir: str,
    open_dir: str,
    outdir: str,
    table_name: str = "combined_summary",
    all_workdays: bool = False,
    window: float = DEFAULT_ROLLING_WINDOW,
    formats: str | Iterable[str] | None = None,
) -> dict[str, Any]:
    """
    Compare a closed-objective and an open-objective workday suite generated with the
    same arguments and seed, so every workday is observed on identical demand with
    each objective, solver, and start time. Writes combined_manifest.json, the
    combined summary table (the selected formats, check resolve_export_formats), and
    the figures. Returns the combined summary rows, the decision impact estimates, the
    written figure paths, and the number of workday pairs per decision
    """
    closed = _load_suite(closed_dir, ObjectiveType.closed)
    opened = _load_suite(open_dir, ObjectiveType.open)
    _check_pairs(closed, opened)
    os.makedirs(outdir, exist_ok=True)

    manifest_rows, summary_rows = _combined_rows(closed_dir, open_dir)
    with open(os.path.join(outdir, "combined_manifest.json"), "w") as f:
        json.dump(manifest_rows, f, indent=4, default=str)
    write_workday_summary_files(
        summary_rows,
        outdir,
        table_name,
        fieldnames=SUMMARY_FIELDS + COMBINED_EXTRA_FIELDS,
        formats=formats,
    )

    figures: dict[str, str] = {}
    estimates = impact_estimates(closed, opened)
    if closed or opened:
        figures["impacts"] = plot_impacts(
            estimates, os.path.join(outdir, "var_decision_impact.png")
        )
        runs = [*closed.values(), *opened.values()]
        figures["demand"] = plot_demand(
            runs, os.path.join(outdir, "demand_vs_accept_delay.png")
        )
        figures["impact_boxes"] = plot_impact_boxes(
            runs, os.path.join(outdir, "var_decision_boxes.png")
        )
        figures["demand_boxes"] = plot_demand_boxes(
            runs, os.path.join(outdir, "demand_param_boxes.png")
        )
    representative = representative_pair(closed, opened)
    if representative is not None:
        figures["workday"] = plot_workday(
            closed[representative],
            opened[representative],
            os.path.join(outdir, f"most_demand_{representative}.png"),
            window=window,
        )
    if all_workdays:
        workdays_dir = os.path.join(outdir, "workdays")
        os.makedirs(workdays_dir, exist_ok=True)
        for p in sorted(closed.keys() & opened.keys()):
            plot_workday(
                closed[p],
                opened[p],
                os.path.join(workdays_dir, f"{p}.png"),
                window=window,
            )
    return {
        "rows": summary_rows,
        "estimates": estimates,
        "representative": representative,
        "figures": figures,
        "nr_start_time_pairs": len(start_time_pairs(closed, opened)),
        "nr_paired": len(closed.keys() & opened.keys()),
        "unpaired": sorted(closed.keys() ^ opened.keys()),
    }
