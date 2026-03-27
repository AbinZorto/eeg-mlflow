#!/usr/bin/env python3
"""Generate a manuscript flowchart for the EEG processing and evaluation pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BOXES = [
    {
        "title": "Exported EEG",
        "body": "21 participants\nAF7, AF8, TP9, TP10\n256 Hz, 10 s segments",
        "xy": (0.03, 0.66),
        "color": "#E3F2FD",
    },
    {
        "title": "Preprocessing",
        "body": "Per-channel demeaning\nUpsample, 60 Hz low-pass\nDownsample to 256 Hz",
        "xy": (0.22, 0.66),
        "color": "#E8F5E9",
    },
    {
        "title": "Windowing",
        "body": "Non-overlapping windows\n2, 4, 6, 8, 10 s",
        "xy": (0.41, 0.66),
        "color": "#E8F5E9",
    },
    {
        "title": "Feature Extraction",
        "body": "247 EEG-derived measures\nSpectral, temporal,\nentropy, asymmetry, synchrony",
        "xy": (0.60, 0.66),
        "color": "#FFF3E0",
    },
    {
        "title": "Outer Split",
        "body": "Leave one participant out\nHeld-out participant\nnever enters training",
        "xy": (0.79, 0.66),
        "color": "#F3E5F5",
    },
    {
        "title": "Training Rebalancing",
        "body": "Training folds only\nLOPO-group equalization\nSMOTE",
        "xy": (0.79, 0.16),
        "color": "#FCE4EC",
    },
    {
        "title": "Feature Selection",
        "body": "Training folds only\nSelectKBest (f-classif)\ninner-k sweep 1 to 70",
        "xy": (0.60, 0.16),
        "color": "#FCE4EC",
    },
    {
        "title": "Model Fitting",
        "body": "Advanced hybrid CNN-LSTM\nor linear SVM",
        "xy": (0.41, 0.16),
        "color": "#F3E5F5",
    },
    {
        "title": "Held-Out Scoring",
        "body": "Window probabilities\nMean probability\nper participant",
        "xy": (0.22, 0.16),
        "color": "#E3F2FD",
    },
    {
        "title": "Outputs",
        "body": "Patient-level metrics\nRecurrence, Jaccard,\nKuncheva, effect direction",
        "xy": (0.03, 0.16),
        "color": "#FFF8E1",
    },
]

BOX_WIDTH = 0.16
BOX_HEIGHT = 0.20


def draw_box(ax: plt.Axes, title: str, body: str, xy: tuple[float, float], color: str) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        BOX_WIDTH,
        BOX_HEIGHT,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.5,
        edgecolor="#455A64",
        facecolor=color,
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
        color="#263238",
    )
    ax.text(
        x + BOX_WIDTH / 2,
        y + BOX_HEIGHT * 0.34,
        body,
        ha="center",
        va="center",
        fontsize=10.2,
        color="#37474F",
        linespacing=1.25,
    )


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.7,
            color="#546E7A",
            shrinkA=0,
            shrinkB=0,
        )
    )


def build_figure(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 8.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for item in BOXES:
        draw_box(ax, item["title"], item["body"], item["xy"], item["color"])

    # Top row arrows
    for left, right in zip(BOXES[:4], BOXES[1:5]):
        arrow(
            ax,
            (left["xy"][0] + BOX_WIDTH, left["xy"][1] + BOX_HEIGHT / 2),
            (right["xy"][0], right["xy"][1] + BOX_HEIGHT / 2),
        )

    # Transition from top to bottom row
    arrow(
        ax,
        (BOXES[4]["xy"][0] + BOX_WIDTH / 2, BOXES[4]["xy"][1]),
        (BOXES[5]["xy"][0] + BOX_WIDTH / 2, BOXES[5]["xy"][1] + BOX_HEIGHT),
    )

    # Bottom row arrows, right to left
    bottom = BOXES[5:]
    for left, right in zip(bottom, bottom[1:]):
        arrow(
            ax,
            (left["xy"][0], left["xy"][1] + BOX_HEIGHT / 2),
            (right["xy"][0] + BOX_WIDTH, right["xy"][1] + BOX_HEIGHT / 2),
        )

    ax.text(0.03, 0.93, "Resting-State EEG Processing and Evaluation Pipeline", fontsize=18, fontweight="bold", color="#263238")
    ax.text(
        0.03,
        0.89,
        "All selection and class rebalancing steps are confined to the training folds before held-out participant scoring.",
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
