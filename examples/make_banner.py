"""Render the project banner as a PNG.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>

``docs/assets/banner.svg`` is the crisp version the documentation site uses.
This renders the same design to PNG for the README, because PyPI rewrites
README images through its own proxy and sanitizes the HTML, and a raster is the
one format certain to survive that. The PNG renders on GitHub and on PyPI
alike.

    python examples/make_banner.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle

ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"

W, H = 12.80, 3.40          # inches at 100 dpi -> 1280 x 340
INK, SOFT = "#16202e", "#64748b"
INDIGO, TEAL, AMBER = "#4f46e5", "#0d9488", "#f59e0b"

PARULA = ["#352a87", "#0f72dd", "#07a0cd", "#4dbd91", "#d1c33b", "#fafa1f"]

PILLS = [
    ("ADL", "#eef2ff", "#c7d2fe", "#4338ca"),
    ("GECM", "#ecfeff", "#a5f3fc", "#0e7490"),
    ("IRF", "#f0fdfa", "#99f6e4", "#0f766e"),
    ("LRM", "#fffbeb", "#fde68a", "#b45309"),
    ("Interactions", "#fdf2f8", "#fbcfe8", "#be185d"),
]


def gradient(ax, x0, y0, x1, y1, colors, horizontal=True, zorder=0):
    """Paint a horizontal or vertical gradient band in axes coordinates."""
    cmap = LinearSegmentedColormap.from_list("_g", colors)
    grid = np.linspace(0, 1, 512).reshape(1, -1)
    if not horizontal:
        grid = grid.T
    ax.imshow(grid, extent=(x0, x1, y0, y1), aspect="auto", cmap=cmap,
              zorder=zorder, interpolation="bicubic")


def main() -> None:
    fig = plt.figure(figsize=(W, H), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1280)
    ax.set_ylim(0, 340)
    ax.axis("off")

    # Page background, with the faintest wash of colour.
    ax.add_patch(Rectangle((0, 0), 1280, 340, color="#fbfcfe", zorder=0))
    gradient(ax, 0, 0, 1280, 340, ["#eef2ff", "#f0f9ff", "#ecfdf5"], zorder=0)
    # The accent rule along the top.
    gradient(ax, 0, 335, 1280, 340, ["#6366f1", "#0ea5e9", "#14b8a6"], zorder=3)

    # ----------------------------------------------------------- wordmark
    ax.text(74, 230, "tseffects", fontsize=52, fontweight="bold",
            color=INK, va="center", ha="left", zorder=4)
    gradient(ax, 76, 196, 194, 201, ["#6366f1", "#0ea5e9", "#14b8a6"], zorder=4)

    # The left column has to clear the figure card, which starts at x = 726.
    ax.text(74, 172, "Dynamic effects from single-equation",
            fontsize=15, fontweight="semibold", color="#334155", zorder=4)
    ax.text(74, 148, "time series models, with interactions",
            fontsize=15, fontweight="semibold", color="#334155", zorder=4)
    ax.text(74, 120,
            "Impulse and step responses, long-run multipliers",
            fontsize=12.5, color=SOFT, zorder=4)
    ax.text(74, 99,
            "and conditional effects — with delta-method errors.",
            fontsize=12.5, color=SOFT, zorder=4)

    # --------------------------------------------------------------- pills
    x = 74
    for label, fill, edge, text_color in PILLS:
        width = 24 + 7.6 * len(label)
        ax.add_patch(FancyBboxPatch(
            (x, 46), width, 27, boxstyle="round,pad=0,rounding_size=13.5",
            facecolor=fill, edgecolor=edge, linewidth=1.1, zorder=4))
        ax.text(x + width / 2, 59.5, label, fontsize=10, fontweight="bold",
                color=text_color, ha="center", va="center", zorder=5)
        x += width + 9

    # ------------------------------------------------- the response figure
    card_x, card_y, card_w, card_h = 726, 44, 486, 252
    ax.add_patch(FancyBboxPatch(
        (card_x, card_y), card_w, card_h,
        boxstyle="round,pad=0,rounding_size=18",
        facecolor="white", edgecolor="#e3e9f2", linewidth=1.2, zorder=4))

    # The Parula strip, naming the default colour scale, kept on its own row
    # above the plot so it cannot collide with the legend.
    gradient(ax, card_x + 28, card_y + 228, card_x + 150, card_y + 234,
             PARULA, zorder=5)
    ax.text(card_x + 157, card_y + 231, "Parula", fontsize=8.5,
            color="#94a3b8", va="center", zorder=5)

    bx, by = card_x + 58, card_y + 46          # plot origin inside the card
    bw, bh = 330, 150
    ax.plot([bx, bx + bw], [by, by], color="#94a3b8", lw=1.2, zorder=5)
    ax.plot([bx, bx], [by, by + bh], color="#94a3b8", lw=1.2, zorder=5)
    ax.plot([bx, bx + bw], [by + 18, by + 18], color="#cbd5e1", lw=1.1,
            ls=(0, (4, 4)), zorder=5)

    s = np.linspace(0, 1, 160)
    step = 1 - np.exp(-3.1 * s)                       # a step response
    pulse = np.exp(-4.6 * s) * (1 - np.exp(-28 * s))  # an impulse response

    y_step = by + 18 + step * 98
    ax.fill_between(bx + s * bw * 0.86, y_step - 10 - 34 * s,
                    y_step + 10 + 46 * s, color=INDIGO, alpha=0.13, zorder=5)
    ax.plot(bx + s * bw * 0.86, y_step, color=INDIGO, lw=3.2,
            solid_capstyle="round", zorder=6)

    y_pulse = by + 18 + pulse * 74
    ax.fill_between(bx + s * bw * 0.86, by + 18 - 4 - 10 * s,
                    y_pulse + 6 + 10 * s, color=TEAL, alpha=0.11, zorder=5)
    ax.plot(bx + s * bw * 0.86, y_pulse, color=TEAL, lw=2.6,
            dashes=(4, 2.4), solid_capstyle="round", zorder=6)

    # The long run, set apart past the end of the path.
    lrm_x = bx + bw * 0.955
    ax.plot([lrm_x, lrm_x], [by + 84, by + 144], color=AMBER, lw=3.2,
            solid_capstyle="round", zorder=6)
    ax.plot([lrm_x], [by + 114], "o", color=AMBER, ms=7.5, zorder=7)
    ax.text(lrm_x, by - 13, "LRM", fontsize=9.5, fontweight="bold",
            color="#b45309", ha="center", va="center", zorder=6)

    ax.text(bx + bw * 0.43, by - 31, "periods since the shock (s)",
            fontsize=9.5, color=SOFT, ha="center", va="center", zorder=6)
    ax.text(bx - 26, by + bh / 2, "response of y", fontsize=9.5, color=SOFT,
            ha="center", va="center", rotation=90, zorder=6)

    legend_y = card_y + 231
    ax.plot([card_x + 268, card_x + 292], [legend_y, legend_y], color=INDIGO,
            lw=3.0, solid_capstyle="round", zorder=6)
    ax.text(card_x + 298, legend_y, "step", fontsize=9.5, color="#475569",
            va="center", zorder=6)
    ax.plot([card_x + 344, card_x + 368], [legend_y, legend_y], color=TEAL,
            lw=2.6, dashes=(4, 2.4), solid_capstyle="round", zorder=6)
    ax.text(card_x + 374, legend_y, "impulse", fontsize=9.5, color="#475569",
            va="center", zorder=6)

    out = ASSETS / "banner.png"
    fig.savefig(out, dpi=100, facecolor="#fbfcfe")
    plt.close(fig)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
