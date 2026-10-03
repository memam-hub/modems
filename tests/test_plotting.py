"""Tests for modems.plotting: the shared figure helpers"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from modems.plotting import draw_hboxes


def test_hboxes_without_hollow_keep_one_full_width_box_per_row() -> None:
    fig, ax = plt.subplots()
    draw_hboxes(ax, [[1.0, 2.0, 3.0], [2.0, 4.0]], ["a", "b"], ["red", "blue"])
    assert len(ax.patches) == 2
    assert [t.get_text() for t in ax.get_yticklabels()] == ["a", "b"]
    assert [t.get_text() for t in ax.texts] == [" n=3", " n=2"]
    plt.close(fig)


def test_hollow_boxes_sit_above_their_filled_box_unfilled_and_color_edged() -> None:
    fig, ax = plt.subplots()
    draw_hboxes(
        ax,
        [[1.0, 2.0, 3.0], [2.0, 4.0]],
        ["a", "b"],
        ["red", "blue"],
        hollow=[[5.0, 6.0], []],
    )
    filled, hollow = ax.patches[:2], ax.patches[2:]
    assert len(filled) == 2 and len(hollow) == 2
    for box in filled:
        assert box.get_facecolor()[3] == 1.0  # filled
    for box, color in zip(hollow, ("red", "blue")):
        assert box.get_facecolor()[3] == 0.0  # unfilled
        assert box.get_edgecolor()[:3] == pytest.approx(
            plt.matplotlib.colors.to_rgb(color)
        )
    # same row (tick), hollow box above the filled one
    assert [t.get_text() for t in ax.get_yticklabels()] == ["a", "b"]
    filled_y = [p.get_path().vertices[:, 1].mean() for p in filled[:1]]
    hollow_y = [p.get_path().vertices[:, 1].mean() for p in hollow[:1]]
    assert hollow_y[0] > filled_y[0]
    # one n= per box, the empty hollow group included
    assert sorted(t.get_text() for t in ax.texts) == sorted(
        [" n=3", " n=2", " n=2", " n=0"]
    )
    plt.close(fig)


def test_hollow_needs_one_group_per_group() -> None:
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="one group per group"):
        draw_hboxes(ax, [[1.0], [2.0]], ["a", "b"], ["red", "blue"], hollow=[[1.0]])
    plt.close(fig)
