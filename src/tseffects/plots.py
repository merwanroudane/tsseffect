"""Plots, in matplotlib.

The R package draws with ggplot2 and a stripped-down ``theme_bw``: no panel
border, no grid, black axis lines.  :func:`apply_theme` reproduces that, so the
figures look like the ones in the R vignette and drop into a paper without
further fiddling.

Everything here takes an :class:`~tseffects.results.EffectResult` and an
optional ``ax``, and returns the axes, so plots compose with any other
matplotlib work.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "apply_theme",
    "plot_effects",
    "plot_gdrf",
    "plot_interact_lines",
    "plot_interact_heatmap",
    "OKABE_ITO",
]

#: Okabe-Ito, the colour-blind-safe qualitative palette R's ``palette.colors()``
#: returns.  Black is first and is skipped when picking series colours, exactly
#: as the R package does.
OKABE_ITO = (
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#999999",
)

#: Low/high endpoints for the diverging fills, white being the midpoint at zero.
HEATMAP_PALETTES = {
    "Blue-Red": ("#023FA5", "#8E063B"),
    "Blue-Red 2": ("#4A6FE3", "#D33F6A"),
    "Blue-Red 3": ("#261B94", "#8E063B"),
    "Red-Green": ("#8E063B", "#1B7837"),
    "Purple-Green": ("#762A83", "#1B7837"),
    "Purple-Brown": ("#762A83", "#8C510A"),
    "Green-Brown": ("#1B7837", "#8C510A"),
    "Blue-Yellow 2": ("#0072B2", "#E69F00"),
    "Blue-Yellow 3": ("#023FA5", "#F0E442"),
    "Green-Orange": ("#1B7837", "#D55E00"),
    "Cyan-Magenta": ("#56B4E9", "#CC79A7"),
    "Tropic": ("#00768F", "#A4451E"),
    "Broc": ("#1B4F72", "#6E5B20"),
    "Cork": ("#1B4F72", "#4F6E20"),
    "Vik": ("#00497A", "#7A3B00"),
    "Berlin": ("#9EB0FF", "#FFB09E"),
    "Lisbon": ("#E5E5E5", "#4A2000"),
    "Grays": ("#1A1A1A", "#D9D9D9"),
}


def apply_theme(ax) -> None:
    """Strip the axes back to two black lines and no grid."""
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.0)
    ax.set_facecolor("white")
    ax.tick_params(colors="black", which="both")


def _new_axes(ax, figsize=(7.0, 4.5)):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _fmt(value) -> str:
    """Format an axis tick the way R would: no trailing ``.0``."""
    value = float(value)
    return f"{value:g}"


def _baseline_at(s_limit: int) -> float:
    """Where the pre-shock baseline sits on the x axis.

    One period before the shock is what the quantity means, but on a long
    horizon a tick at -1 collides with the tick at 0.  Scaling the gap with the
    horizon keeps the two labels apart, and mirrors how the long run is pushed
    out past the end of the path.
    """
    return -float(max(1, round(s_limit * 0.08)))


def _lrm_at(s_limit: int) -> float:
    return float(s_limit + round(s_limit * 0.2))


def _period_ticks(s_limit: int, include_baseline: bool, lrm_at=None):
    """Five evenly spaced period ticks, plus the baseline and LRM markers."""
    base = list(np.linspace(0, s_limit, 5))
    ticks = [_fmt(v) for v in base]
    positions = list(base)
    if include_baseline:
        # "Baseline" is a wide label sitting just left of the tick at 0, and on
        # a narrow panel the two overlap.  Drop the 0 tick: the shock is still
        # visibly where the line starts, so the label earns nothing.
        positions, ticks = positions[1:], ticks[1:]
        positions.insert(0, _baseline_at(s_limit))
        ticks.insert(0, "Baseline")
    if lrm_at is not None:
        positions.append(float(lrm_at))
        ticks.append("LRM")
    return positions, ticks


def _numeric_periods(frame, s_limit: int):
    """Map the ``Period`` column onto x-axis positions.

    ``"baseline"`` sits before the shock and ``"LRM"`` is pushed out past the
    last period, so each reads as separate from the path.
    """
    lrm_at = _lrm_at(s_limit)
    baseline_at = _baseline_at(s_limit)
    out = []
    for value in frame["Period"]:
        if value == "baseline":
            out.append(baseline_at)
        elif value == "LRM":
            out.append(lrm_at)
        else:
            out.append(float(value))
    return np.asarray(out, dtype=float), lrm_at


def _line_colors(line_colors, n: int):
    """Resolve the colour specification for ``n`` series."""
    if isinstance(line_colors, str):
        if line_colors == "okabe-ito":
            return list(OKABE_ITO[1 : n + 1])
        if line_colors == "bw":
            return [f"{g / 100:.2f}" for g in np.linspace(0, 60, n)]
        raise ValueError(
            "line_colors must be 'okabe-ito', 'bw', or one colour per series"
        )
    colors = list(line_colors)
    if len(colors) != n:
        raise ValueError(
            f"Number of supplied line_colors ({len(colors)}) is not equal to the "
            f"number of series to plot ({n})"
        )
    return colors


def _diverging_cmap(name):
    """A white-centred diverging colormap from a palette name or colormap.

    The house families come first -- Parula, Blue-Red and the rest -- then this
    module's own named pairs, then any matplotlib colormap.
    """
    from matplotlib.colors import LinearSegmentedColormap

    if not isinstance(name, str):
        return name

    from ._colors import COLORSCALES, diverging_cmap as house_diverging

    if name.strip() in COLORSCALES or name.strip().title() in COLORSCALES:
        return house_diverging(name)

    if name not in HEATMAP_PALETTES:
        import matplotlib

        try:
            return matplotlib.colormaps[name]
        except KeyError:
            raise ValueError(
                f"Unknown heatmap palette {name!r}. Use one of: "
                + ", ".join(sorted(set(HEATMAP_PALETTES) | set(COLORSCALES)))
                + ", or any matplotlib colormap name"
            ) from None
    low, high = HEATMAP_PALETTES[name]
    return LinearSegmentedColormap.from_list(name, [low, "#FFFFFF", high])


def plot_gdrf(result, ax=None, color="black", fill_alpha=0.2, figsize=(7.0, 4.5)):
    """Draw a response function: the path, its confidence band, and the LRM.

    A horizontal line at zero is drawn for discrete effects, where zero means
    "no effect", and omitted for fitted values, where it means nothing.
    """
    ax = _new_axes(ax, figsize)
    frame = result.estimates
    value_col = result.value_column
    s_limit = int(result.meta["s_limit"])
    effect_type = result.meta.get("effect_type", "discrete")

    x, lrm_at = _numeric_periods(frame, s_limit)
    is_lrm = frame["Period"].to_numpy() == "LRM"
    path = ~is_lrm

    ax.fill_between(
        x[path],
        frame.loc[path, "Lower"],
        frame.loc[path, "Upper"],
        color=color,
        alpha=fill_alpha,
        linewidth=1.0,
        edgecolor=color,
    )
    ax.plot(x[path], frame.loc[path, value_col], color=color, lw=1.8)

    if effect_type == "discrete":
        ax.axhline(0, color="black", lw=1.0)

    if is_lrm.any():
        lrm = frame.loc[is_lrm].iloc[0]
        ax.plot(
            [lrm_at, lrm_at], [lrm["Lower"], lrm["Upper"]], color=color, lw=2.0
        )
        ax.plot([lrm_at], [lrm[value_col]], "o", color=color, ms=7)

    positions, labels = _period_ticks(
        s_limit,
        include_baseline=effect_type == "fitted",
        lrm_at=lrm_at if is_lrm.any() else None,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Number of Periods Since Initial Shock (s)")
    ax.set_ylabel(result.meta.get("y_label", value_col))
    apply_theme(ax)
    return ax


def plot_interact_lines(
    result, ax=None, line_colors="okabe-ito", fill_alpha=0.1, figsize=(7.5, 4.5)
):
    """Draw a family of conditional effect lines.

    ``line_options="z.lines"`` puts the period on the x axis and one line per
    value of the moderator; ``"s.lines"`` puts the moderator on the x axis and
    one line per period, which is the picture that shows where in the range of z
    the effect is actually distinguishable from zero.
    """
    ax = _new_axes(ax, figsize)
    frame = result.estimates
    s_limit = int(result.meta["s_limit"])
    effect_type = result.meta.get("effect_type", "discrete")
    line_options = result.meta.get("line_options", "z.lines")
    z_label = result.meta.get("z_label", "z")
    rounding = int(result.meta.get("z_label_rounding", 3))
    has_lrm = bool(result.meta.get("lrm"))

    if line_options == "z.lines":
        z_values = list(result.meta["z_vals"])
        colors = _line_colors(line_colors, len(z_values))
        x_all, lrm_at = _numeric_periods(frame, s_limit)
        frame = frame.assign(_x=x_all)
        lrm_offsets = np.linspace(
            lrm_at - (len(z_values) - 1) / 2,
            lrm_at + (len(z_values) - 1) / 2,
            len(z_values),
        )

        for i, z in enumerate(z_values):
            rows = frame[np.isclose(frame["Z"], z)]
            path = rows[rows["Period"] != "LRM"]
            label = (
                f"{z:.{rounding}f}" if max(abs(np.asarray(z_values))) < 10 else f"{z:g}"
            )
            ax.fill_between(
                path["_x"],
                path["Lower"],
                path["Upper"],
                color=colors[i],
                alpha=fill_alpha,
                linewidth=1.0,
                edgecolor=colors[i],
            )
            ax.plot(path["_x"], path["Effect"], color=colors[i], lw=1.8, label=label)
            if has_lrm:
                lrm = rows[rows["Period"] == "LRM"]
                if not lrm.empty:
                    row = lrm.iloc[0]
                    ax.plot(
                        [lrm_offsets[i]] * 2,
                        [row["Lower"], row["Upper"]],
                        color=colors[i],
                        lw=2.0,
                    )
                    ax.plot(
                        [lrm_offsets[i]], [row["Effect"]], "o", color=colors[i], ms=7
                    )

        if effect_type == "discrete":
            ax.axhline(0, color="black", lw=1.0)
        positions, labels = _period_ticks(
            s_limit,
            include_baseline=effect_type == "fitted",
            lrm_at=lrm_at if has_lrm else None,
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Number of Periods Since Initial Shock (s)")
        ax.set_ylabel(result.meta.get("y_label", "Effect"))
        ax.legend(title=f"Value of {z_label}", frameon=False)
    else:  # s.lines
        s_values = [
            s_limit + 1 if s == "LRM" else int(s) for s in result.meta.get("s_vals", [0])
        ]
        colors = _line_colors(line_colors, len(s_values))
        for i, s in enumerate(s_values):
            key = "LRM" if (has_lrm and s == s_limit + 1) else s
            rows = frame[frame["Period"] == key].sort_values("Z")
            if rows.empty:
                continue
            if key == "LRM":
                label = "s = ∞ (Long-run)"
            elif s == 0:
                label = "s = 0 (Short-run)"
            else:
                label = f"s = {s}"
            ax.fill_between(
                rows["Z"],
                rows["Lower"],
                rows["Upper"],
                color=colors[i],
                alpha=fill_alpha,
                linewidth=1.0,
                edgecolor=colors[i],
            )
            ax.plot(rows["Z"], rows["Effect"], color=colors[i], lw=1.8, label=label)
        if effect_type == "discrete":
            ax.axhline(0, color="black", lw=1.0)
        ax.set_xlabel(f"Value of {z_label}")
        ax.set_ylabel(result.meta.get("y_label", "Effect"))
        ax.legend(title="Value of s", frameon=False)

    apply_theme(ax)
    return ax


def plot_interact_heatmap(
    result, ax=None, heatmap_colors="Blue-Red", figsize=(7.5, 5.0), colorbar=True
):
    """Draw the conditional effect surface over period and moderator.

    With ``heatmap_options="significant"`` the cells whose confidence interval
    covers zero are set to zero, so the coloured region is exactly the region
    where the effect is distinguishable from zero.
    """
    from matplotlib.colors import TwoSlopeNorm

    ax = _new_axes(ax, figsize)
    frame = result.estimates.copy()
    s_limit = int(result.meta["s_limit"])
    effect_type = result.meta.get("effect_type", "discrete")
    has_lrm = bool(result.meta.get("lrm"))
    option = result.meta.get("heatmap_options", "all")

    x, lrm_at = _numeric_periods(frame, s_limit)
    frame["_x"] = x

    if option == "significant":
        insignificant = (frame["Lower"] < 0) & (frame["Upper"] > 0)
        frame["_fill"] = np.where(insignificant, 0.0, frame["Effect"])
    else:
        frame["_fill"] = frame["Effect"]

    grid = frame.pivot_table(index="Z", columns="_x", values="_fill")
    z_axis = grid.index.to_numpy(dtype=float)
    x_axis = grid.columns.to_numpy(dtype=float)
    values = grid.to_numpy(dtype=float)

    # Match ggplot2's scale_fill_gradient2: midpoint at 0, limits spanning zero.
    vmin = min(0.0, float(np.nanmin(values)))
    vmax = max(0.0, float(np.nanmax(values)))
    # TwoSlopeNorm insists on vmin < vcenter < vmax, so when every effect has
    # the same sign one side has to be nudged off zero.  That half of the scale
    # then simply goes unused, which is what gradient2 shows too.
    eps = max(abs(vmin), abs(vmax), 1.0) * 1e-9
    norm = TwoSlopeNorm(vmin=min(vmin, -eps), vcenter=0.0, vmax=max(vmax, eps))

    mesh = ax.pcolormesh(
        x_axis,
        z_axis,
        values,
        cmap=_diverging_cmap(heatmap_colors),
        norm=norm,
        shading="nearest",
    )
    positions, labels = _period_ticks(
        s_limit,
        include_baseline=effect_type == "fitted",
        lrm_at=lrm_at if has_lrm else None,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Number of Periods Since Initial Shock (s)")
    ax.set_ylabel(f"Value of {result.meta.get('z_label', 'z')}")
    if colorbar:
        label = result.meta.get("fill_label", "Effect")
        if option == "significant":
            label = "Statistically Significant\n" + label
        ax.figure.colorbar(mesh, ax=ax, label=label)
    apply_theme(ax)
    return ax


def plot_effects(result, ax=None, **kwargs):
    """Draw whichever plot suits the result.

    Response functions get :func:`plot_gdrf`; interaction results get
    :func:`plot_interact_lines` or :func:`plot_interact_heatmap` according to
    the ``plot_type`` they were computed with.
    """
    if result.meta.get("model") == "interact":
        if result.meta.get("plot_type") == "heatmap":
            return plot_interact_heatmap(result, ax=ax, **kwargs)
        return plot_interact_lines(result, ax=ax, **kwargs)
    return plot_gdrf(result, ax=ax, **kwargs)
