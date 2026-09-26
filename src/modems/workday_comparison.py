from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .benchmark import ManifestStatus
from .core import ObjectiveType, SolverStrategy
from .network import RoadNetwork
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
DEFAULT_DEMAND_BIN = 10.0  # minutes per bar of the demand strip
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


def t_critical(df: float) -> float:
    """Two-sided 95% t critical value (Cornish-Fisher expansion beyond df = 30)"""
    if df < 1:
        return math.inf
    if df <= 30:
        return _T975[int(df) - 1]  # conservative for non-integer (Welch) df
    z = 1.959964
    return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)


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


def welch_ci(a: list[float], b: list[float]) -> tuple[float, float] | None:
    """(mean(b) - mean(a), Welch 95% CI half-width); None unless both have n >= 2"""
    if len(a) < 2 or len(b) < 2:
        return None
    xa, xb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    va, vb = xa.var(ddof=1) / len(xa), xb.var(ddof=1) / len(xb)
    diff = float(xb.mean() - xa.mean())
    if va + vb == 0.0:
        return diff, 0.0
    df = (va + vb) ** 2 / (va**2 / (len(xa) - 1) + vb**2 / (len(xb) - 1))
    return diff, t_critical(df) * math.sqrt(va + vb)


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


def _load_suite(outdir: str, objective: ObjectiveType) -> dict[str, WorkdayRun]:
    """Solved workdays of one suite keyed by pair name; raises on a wrong objective"""
    runs = {}
    for name, row in read_workday_manifest(outdir).items():
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
        for name, row in read_workday_manifest(outdir).items():
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


def _setup_fig_grid(nrows: int, ncols: int, width: float, height: float, **kwargs):
    import matplotlib.pyplot as plt

    plt.rc("font", family="serif", size=16)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, **kwargs)
    fig.set_size_inches(width, height, forward=True)
    for ax in axes.flat:
        ax.grid(True, alpha=0.5, zorder=-2)
    return fig, axes


def _save(fig, path: str) -> str:
    import matplotlib.pyplot as plt

    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return path


def impact_estimates(
    closed: dict[str, WorkdayRun], opened: dict[str, WorkdayRun]
) -> list[dict[str, Any]]:
    """
    Figure 1 data: one entry per (metric, decision, group), the mean difference and
    95% CI half-width, and the number of workdays behind it
      - objective, per solver: paired open - closed over workdays in both suites
      - solver, per objective: paired ALNS - MILP3 over that suite's workdays
      - start time, per solver: Welch difference of means, staggered - normal, of the
        per-workday average over both objectives (closed/open share their demand)
    """
    pairs = sorted(closed.keys() & opened.keys())
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
            by_start: dict[WorkdayStartTime, list[float]] = {}
            for p in closed.keys() | opened.keys():
                values = [
                    metric.value(runs[p].results[s])
                    for runs in (closed, opened)
                    if p in runs and metric.value(runs[p].results[s]) is not None
                ]
                if values:
                    run = closed.get(p) or opened[p]
                    start = WorkdayStartTime(run.row["start_time"])
                    by_start.setdefault(start, []).append(float(np.mean(values)))
            normal, staggered = (
                by_start.get(WorkdayStartTime.normal, []),
                by_start.get(WorkdayStartTime.staggered, []),
            )
            add(
                metric,
                "start_time",
                SOLVER_LABEL[s],
                welch_ci(normal, staggered),
                len(normal) + len(staggered),
            )
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
    fig, axes = _setup_fig_grid(len(METRICS), len(decisions), 18.0, 3.0 * len(METRICS))
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
    return _save(fig, path)


def plot_demand(runs: list[WorkdayRun], path: str) -> str:
    """
    Figure 2: acceptance and mean delay against demand load (requests per agent-hour),
    binned means with 95% CI band per solver x objective; workdays with surges hollow
    """
    panels = [METRICS[0], METRICS[1]]
    fig, axes = _setup_fig_grid(1, len(panels), 18.0, 7.0)
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
    return _save(fig, path)


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
    return _save(fig, path)


def representative_pair(
    closed: dict[str, WorkdayRun], opened: dict[str, WorkdayRun]
) -> str | None:
    """Highest demand load among paired workdays with surges (else among all)"""
    pairs = sorted(closed.keys() & opened.keys())
    with_surges = [p for p in pairs if closed[p].row["nr_surges"] > 0]
    candidates = with_surges or pairs
    return max(candidates, key=lambda p: (closed[p].demand_load, p), default=None)


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
) -> dict[str, Any]:
    """
    Compare a closed-objective and an open-objective workday suite (see the module
    docstring). Returns the combined summary rows, the Figure 1 estimates, and the
    paths of every written figure
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
        "nr_paired": len(closed.keys() & opened.keys()),
        "unpaired": sorted(closed.keys() ^ opened.keys()),
    }
