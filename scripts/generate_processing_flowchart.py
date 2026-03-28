#!/usr/bin/env python3
"""Generate a manuscript flowchart for the EEG processing and evaluation pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath


BOXES = [
    {
        "title": "Exported EEG",
        "body": "21 participants\nAF7, AF8, TP9, TP10\n256 Hz, 10 s segments",
        "xy": (0.03, 0.66),
        "color": "#E3F2FD",
        "inside_loop": False,
    },
    {
        "title": "Preprocessing",
        "body": "Per-channel demeaning\nUpsample, 60 Hz low-pass\nDownsample to 256 Hz",
        "xy": (0.22, 0.66),
        "color": "#E8F5E9",
        "inside_loop": False,
    },
    {
        "title": "Windowing",
        "body": "Non-overlapping windows\n2, 4, 6, 8, 10 s",
        "xy": (0.41, 0.66),
        "color": "#E8F5E9",
        "inside_loop": False,
    },
    {
        "title": "Feature Extraction",
        "body": "247 EEG-derived measures\nSpectral, temporal,\nentropy, asymmetry, synchrony",
        "xy": (0.60, 0.60),
        "color": "#FFF3E0",
        "inside_loop": True,
    },
    {
        "title": "LOPO Validation Fold",
        "body": "Hold out 1 participant\nTrain on the remainder\nRepeat across all participants",
        "xy": (0.79, 0.60),
        "color": "#F3E5F5",
        "inside_loop": True,
    },
    {
        "title": "Training Rebalancing",
        "body": "Training participants only\nGroup equalization\nSMOTE",
        "xy": (0.79, 0.16),
        "color": "#FCE4EC",
        "inside_loop": True,
    },
    {
        "title": "Feature Selection",
        "body": "Training participants only\nSelectKBest (f-classif)\ninner-k sweep 1 to 70",
        "xy": (0.60, 0.16),
        "color": "#FCE4EC",
        "inside_loop": True,
    },
    {
        "title": "Model Fitting",
        "body": "Advanced hybrid CNN-LSTM\nor linear SVM",
        "xy": (0.41, 0.16),
        "color": "#F3E5F5",
        "inside_loop": True,
    },
    {
        "title": "Held-Out Scoring",
        "body": "Score held-out participant i\nAverage window probabilities\nStore fold output",
        "xy": (0.22, 0.16),
        "color": "#E3F2FD",
        "inside_loop": True,
    },
    {
        "title": "Outputs",
        "body": "Aggregate across 21 folds\nPatient-level metrics\nRecurrence, Jaccard,\nKuncheva, effect direction",
        "xy": (0.03, 0.16),
        "color": "#FFF8E1",
        "inside_loop": False,
    },
]

BOX_WIDTH = 0.16
BOX_HEIGHT = 0.20


def draw_box(
    ax: plt.Axes,
    title: str,
    body: str,
    xy: tuple[float, float],
    color: str,
    inside_loop: bool,
) -> None:
    x, y = xy
    facecolor = "#EEF7F5" if inside_loop else color
    edgecolor = "#2E6F73" if inside_loop else "#455A64"
    title_color = "#24575A" if inside_loop else "#263238"
    body_color = "#35595B" if inside_loop else "#37474F"
    linewidth = 2.0 if inside_loop else 1.5

    patch = FancyBboxPatch(
        (x, y),
        BOX_WIDTH,
        BOX_HEIGHT,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + BOX_WIDTH / 2,
        y + BOX_HEIGHT * 0.72,
        title,
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color=title_color,
        zorder=3,
    )
    ax.text(
        x + BOX_WIDTH / 2,
        y + BOX_HEIGHT * 0.34,
        body,
        ha="center",
        va="center",
        fontsize=10.2,
        color=body_color,
        linespacing=1.25,
        zorder=3,
    )


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#546E7A") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.7,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=4,
        )
    )


def draw_loop_region(ax: plt.Axes) -> None:
    vertices = [
        (0.575, 0.84),
        (0.965, 0.84),
        (0.965, 0.08),
        (0.205, 0.08),
        (0.205, 0.39),
        (0.575, 0.39),
        (0.575, 0.84),
        (0.575, 0.84),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.CLOSEPOLY,
    ]
    patch = PathPatch(
        MplPath(vertices, codes),
        facecolor="#E8F4F3",
        edgecolor="#4D8D8F",
        linewidth=2.3,
        hatch="xx",
        alpha=0.32,
        joinstyle="round",
        zorder=0,
    )
    ax.add_patch(patch)
    ax.text(
        0.60,
        0.805,
        "LOPO analysis block",
        ha="left",
        va="bottom",
        fontsize=12.1,
        fontweight="bold",
        color="#24575A",
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="#4D8D8F", linewidth=1.0),
        zorder=5,
    )
    ax.text(
        0.60,
        0.775,
        "Repeat once per held-out participant.",
        ha="left",
        va="bottom",
        fontsize=9.7,
        color="#24575A",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.95),
        zorder=5,
    )


def build_figure(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 8.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_loop_region(ax)

    for item in BOXES:
        draw_box(
            ax,
            item["title"],
            item["body"],
            item["xy"],
            item["color"],
            item["inside_loop"],
        )

    # Top row arrows
    for left, right in zip(BOXES[:4], BOXES[1:5]):
        arrow(
            ax,
            (left["xy"][0] + BOX_WIDTH, left["xy"][1] + BOX_HEIGHT / 2),
            (right["xy"][0], right["xy"][1] + BOX_HEIGHT / 2),
            "#2E6F73" if left["inside_loop"] and right["inside_loop"] else "#546E7A",
        )

    # Transition from top to bottom row
    arrow(
        ax,
        (BOXES[4]["xy"][0] + BOX_WIDTH / 2, BOXES[4]["xy"][1]),
        (BOXES[5]["xy"][0] + BOX_WIDTH / 2, BOXES[5]["xy"][1] + BOX_HEIGHT),
        "#2E6F73",
    )

    # Bottom row arrows, right to left
    bottom = BOXES[5:]
    for left, right in zip(bottom, bottom[1:]):
        arrow(
            ax,
            (left["xy"][0], left["xy"][1] + BOX_HEIGHT / 2),
            (right["xy"][0] + BOX_WIDTH, right["xy"][1] + BOX_HEIGHT / 2),
            "#2E6F73" if left["inside_loop"] and right["inside_loop"] else "#546E7A",
        )

    ax.text(0.03, 0.93, "Resting-State EEG Processing and Evaluation Pipeline", fontsize=18, fontweight="bold", color="#263238")
    ax.text(
        0.03,
        0.89,
        "In each fold, one participant is held out once; rebalancing and feature selection are performed only on the remaining training participants.",
        fontsize=11.5,
        color="#455A64",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="paper/figures/figure0_processing_pipeline.png",
        help="Output image path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_figure(Path(args.output))


if __name__ == "__main__":
    main()
