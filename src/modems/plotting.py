from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


def setup_fig_grid(
    nrows: int, ncols: int, width: float, height: float, **kwargs: Any
) -> tuple[Any, np.ndarray]:
    """Figure with an nrows x ncols grid of axes, styled consistently"""
    import matplotlib.pyplot as plt

    plt.rc("font", family="serif", size=18)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, **kwargs)
    fig.set_size_inches(width, height, forward=True)
    for ax in axes.flat:
        ax.grid(True, alpha=0.5, zorder=-2)
    return fig, axes


def save_figure(fig: Any, path: str) -> str:
    """Save at 300 dpi, close the figure, and return the path"""
    import matplotlib.pyplot as plt

    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return path


def _horizontal() -> dict[str, Any]:
    """Horizontal boxplot keyword: orientation= since matplotlib 3.10, else vert="""
    import matplotlib

    major, minor = (int(v) for v in matplotlib.__version__.split(".")[:2])
    return (
        {"orientation": "horizontal"} if (major, minor) >= (3, 10) else {"vert": False}
    )


def draw_hboxes(
    ax: Any,
    groups: Sequence[Sequence[float]],
    labels: Sequence[str],
    colors: Sequence[Any],
    show_counts: bool = True,
    hollow: Sequence[Sequence[float]] | None = None,
) -> None:
    """
    One horizontal box-and-whisker per group, the first group on top. Whiskers reach
    the furthest value within 1.5 x IQR (Tukey), with outliers as dots. Empty groups
    keep their row (and label) with nothing drawn. If given, hollow holds one more
    group per group (e.g., a baseline), drawn as an unfilled box edged in the group's
    color just above its filled box, in the same row; each box gets its own n=
    """
    if hollow is not None and len(hollow) != len(groups):
        raise ValueError(
            f"hollow needs one group per group: {len(hollow)} != {len(groups)}"
        )
    positions = list(range(len(groups), 0, -1))
    offset, width = (0.0, 0.6) if hollow is None else (0.18, 0.32)
    filled_positions = [y - offset for y in positions]
    boxes = ax.boxplot(
        [list(g) if len(g) else [math.nan] for g in groups],
        positions=filled_positions,
        widths=width,
        **_horizontal(),
        patch_artist=True,
        medianprops=dict(color="k", linewidth=2),
        flierprops=dict(marker="o", markersize=8),
    )
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
    counted = list(zip(filled_positions, groups))
    if hollow is not None:
        hollow_positions = [y + offset for y in positions]
        boxes = ax.boxplot(
            [list(g) if len(g) else [math.nan] for g in hollow],
            positions=hollow_positions,
            widths=width,
            **_horizontal(),
            patch_artist=True,
            medianprops=dict(color="k", linewidth=2),
            flierprops=dict(marker="o", markersize=8, markerfacecolor="none"),
        )
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor("none")
            patch.set_edgecolor(color)
            patch.set_linewidth(2)
        counted += list(zip(hollow_positions, hollow))
    ax.set_yticks(positions, labels)
    if show_counts:
        for y, group in counted:
            ax.text(
                1.0,
                y,
                f" n={len(group)}",
                transform=ax.get_yaxis_transform(),
                va="center",
                ha="left",
                fontsize=11,
                color="gray",
            )


def draw_grouped_vboxes(
    ax: Any,
    categories: Sequence[Any],
    series: Sequence[tuple[str, Any, Sequence[Sequence[float]]]],
) -> list[Any]:
    """
    Vertical boxes grouped per category (e.g., request count): series is a list of
    (label, color, one group of values per category), drawn side by side within each
    category. Returns one legend handle per series
    """
    from matplotlib.patches import Patch

    width = 0.8 / max(len(series), 1)
    handles = []
    for k, (label, color, groups) in enumerate(series):
        offset = (k - (len(series) - 1) / 2) * width
        positions = [i + offset for i in range(len(categories))]
        data = [list(g) if len(g) else [math.nan] for g in groups]
        boxes = ax.boxplot(
            data,
            positions=positions,
            widths=0.85 * width,
            patch_artist=True,
            medianprops=dict(color="k", linewidth=2),
            flierprops=dict(marker="o", markersize=8),
        )
        for patch in boxes["boxes"]:
            patch.set_facecolor(color)
        handles.append(Patch(facecolor=color, edgecolor="k", label=label))
    ax.set_xticks(range(len(categories)), [str(c) for c in categories])
    ax.set_xlim(-0.6, len(categories) - 0.4)
    return handles


def values_of(rows: Sequence[dict[str, Any]], key: str, **where: Any) -> list[float]:
    """Non-missing values of rows[key] among the rows matching every where filter"""
    return [
        float(r[key])
        for r in rows
        if r.get(key) is not None and all(r.get(k) == v for k, v in where.items())
    ]
