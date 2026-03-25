from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


PALETTE = {
    "remission": "#1d5f8b",
    "non_remission": "#a64d3d",
    "summary": "#153b63",
    "summary_band": "#88a9c7",
    "neutral": "#8ea0b2",
    "neutral_dark": "#627487",
    "grid": "#d9e3ec",
    "text": "#203243",
    "spine": "#8ea0b2",
    "figure_face": "#f5f8fb",
    "axes_face": "#fcfdff",
}


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["figure_face"],
            "axes.facecolor": PALETTE["axes_face"],
            "axes.edgecolor": PALETTE["spine"],
            "axes.labelcolor": PALETTE["text"],
            "axes.titlecolor": PALETTE["text"],
            "axes.titlesize": 15,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.85,
            "xtick.color": PALETTE["text"],
            "ytick.color": PALETTE["text"],
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.frameon": True,
            "legend.facecolor": "#ffffff",
            "legend.edgecolor": "#d4dce6",
            "legend.framealpha": 0.96,
            "font.family": "DejaVu Sans",
            "savefig.facecolor": PALETTE["figure_face"],
            "savefig.bbox": "tight",
        }
    )


def style_axes(ax, *, ygrid: bool = True, xgrid: bool = False) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.8 if ygrid else 0.0)
    ax.grid(axis="x", alpha=0.35 if xgrid else 0.0)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.0)
        ax.spines[spine].set_color(PALETTE["spine"])
    ax.tick_params(axis="both", which="major", length=0)


def add_subtitle(ax, text: str) -> None:
    ax.text(
        0.0,
        1.02,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        color="#5d6f81",
    )


def save_figure(fig, output_path: Path, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
