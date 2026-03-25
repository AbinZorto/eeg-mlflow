from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from src.utils.paper_plot_style import PALETTE, save_figure


@dataclass(frozen=True)
class CompositePanel:
    label: str
    figure_id: str
    title: str
    path: Path
    reason: Optional[str] = None


def _draw_panel_label(ax, label: str) -> None:
    ax.text(
        0.015,
        0.985,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
        color=PALETTE["text"],
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "#ffffff",
            "edgecolor": "#d4dce6",
            "linewidth": 0.8,
            "alpha": 0.98,
        },
    )


def _draw_placeholder(ax, panel: CompositePanel) -> None:
    ax.set_facecolor(PALETTE["axes_face"])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    _draw_panel_label(ax, panel.label)
    ax.text(
        0.5,
        0.58,
        panel.title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="semibold",
        color=PALETTE["text"],
    )
    ax.text(
        0.5,
        0.42,
        f"Unavailable\n{panel.reason or 'missing_source_panel'}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color=PALETTE["neutral_dark"],
    )


def _draw_image_panel(ax, panel: CompositePanel) -> None:
    image = mpimg.imread(panel.path)
    ax.imshow(image, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    _draw_panel_label(ax, panel.label)


def compose_composite_figure(
    *,
    title: str,
    subtitle: str,
    panels: Sequence[CompositePanel],
    output_path: Path,
    dpi: int,
    figsize: Optional[tuple[float, float]] = None,
) -> None:
    panel_count = max(1, len(panels))
    if panel_count <= 2:
        nrows = 1
        ncols = panel_count
        resolved_figsize = figsize or ((14.0, 6.2) if panel_count == 2 else (7.2, 6.2))
        title_y = 0.97
        subtitle_y = 0.935
        top = 0.84
        bottom = 0.08
    else:
        nrows = 2
        ncols = 2
        resolved_figsize = figsize or (14.0, 11.0)
        title_y = 0.985
        subtitle_y = 0.955
        top = 0.91
        bottom = 0.03

    fig, axes = plt.subplots(nrows, ncols, figsize=resolved_figsize, facecolor=PALETTE["figure_face"])
    if hasattr(axes, "flat"):
        axes_flat = list(axes.flat)
    else:
        axes_flat = [axes]

    for ax, panel in zip(axes_flat, panels):
        if panel.path.exists() and panel.reason is None:
            _draw_image_panel(ax, panel)
        else:
            _draw_placeholder(ax, panel)

    for ax in axes_flat[len(panels):]:
        ax.axis("off")

    fig.suptitle(
        title,
        x=0.02,
        y=title_y,
        ha="left",
        va="top",
        fontsize=19,
        fontweight="semibold",
        color=PALETTE["text"],
    )
    fig.text(
        0.02,
        subtitle_y,
        subtitle,
        ha="left",
        va="top",
        fontsize=10,
        color=PALETTE["neutral_dark"],
    )
    fig.subplots_adjust(left=0.03, right=0.99, top=top, bottom=bottom, wspace=0.08, hspace=0.08)
    save_figure(fig, output_path, dpi)
