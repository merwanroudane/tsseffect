r"""Dynamic conditional effects from a multiplicative interaction.

When a dynamic model carries an interaction, :math:`x_t \times z_t`, the effect
of a shock to x depends on *both* how long ago the shock landed and the value of
the moderator z.  The quantity of interest is therefore a surface over
:math:`(s, z)` rather than a single response function, which is why the result
of :func:`interact_adl` has a ``Z`` column and why its default plot is either a
family of lines or a heatmap.

This follows Warner, Vande Kamp and Jordan, *Conditional Relationships in
Dynamic Models*, Political Science Research and Methods (2026),
`doi:10.1017/psrm.2026.10087 <https://doi.org/10.1017/psrm.2026.10087>`_.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import sympy as sp

from . import _checks as chk
from ._delta import BatchDelta, z_quantile
from ._util import as_model, check_vrbl, model_column, sanitize_vrbl
from .baselines import yhat_calculator
from .calculators import convolution_calculator, general_calculator, pulse_calculator
from .gdrf import _table_meta
from .results import EffectResult

__all__ = ["interact_adl", "combine_interaction_terms"]

PLOT_TYPES = ("lines", "heatmap")
LINE_OPTIONS = ("z.lines", "s.lines")
HEATMAP_OPTIONS = ("all", "significant")

#: Maximum number of lines a single plot will carry before it stops being read.
MAX_LINES = 5

#: Number of moderator values swept for a range (s.lines or a heatmap).
RANGE_POINTS = 50


def combine_interaction_terms(x_vrbl, x_z_vrbl) -> dict:
    r"""Fold the interaction terms into the marginal effect of x at each lag.

    At lag :math:`q` the marginal effect of x is
    :math:`\beta_q + \sum_k \gamma_{q,k} z`, so the "coefficient" the response
    function needs is that whole expression.  Several interaction terms can
    share a lag -- a contemporaneous and a cross-lagged one, say -- and all of
    them are added.

    Returns
    -------
    dict
        ``{expression: lag order}``, with the moderator left as the free symbol
        ``z_val``.

    Examples
    --------
    >>> from tseffects import combine_interaction_terms
    >>> combine_interaction_terms({"x": 0, "l_1_x": 1},
    ...                           {"x_z": 0, "x_l_1_z": 0, "z_l_1_x": 1})
    {'x+x_z*z_val+x_l_1_z*z_val': 0, 'l_1_x+z_l_1_x*z_val': 1}
    """
    limit = max(list(x_vrbl.values()) + list(x_z_vrbl.values()))
    combined: dict[str, int] = {}
    for lag in range(limit + 1):
        # Only one plain x coefficient is possible per lag.
        x_terms = [n for n, q in x_vrbl.items() if q == lag]
        xz_terms = [n for n, q in x_z_vrbl.items() if q == lag]
        left = x_terms[0] if x_terms else "0"
        right = "+".join(f"{n}*z_val" for n in xz_terms) if xz_terms else "0"
        combined[f"{left}+{right}"] = lag
    return combined


def _default_z_vals(model_info, z_vrbl, plot_type, line_options, z_vals):
    """Pick the moderator values to evaluate at, as the R function does.

    Discrete values for a family of lines, a swept range for ``s.lines`` and
    heatmaps.  Defaults are built from the moderator's own distribution: the
    mean plus or minus one standard deviation for lines, two for a range.
    """
    first_z = next(iter(z_vrbl))
    column = model_column(model_info, first_z)
    mean, sd = float(np.nanmean(column)), float(np.nanstd(column, ddof=1))

    wants_range = plot_type == "heatmap" or (
        plot_type == "lines" and line_options == "s.lines"
    )

    if z_vals is not None:
        z_vals = np.asarray(z_vals, dtype=float).ravel()
        if wants_range:
            if z_vals.size != 2:
                raise ValueError(
                    f"{'heatmap' if plot_type == 'heatmap' else 's.lines'} "
                    "requires two z_vals: the lower and upper limit of the range "
                    "of z values to sweep"
                )
            return np.linspace(z_vals[0], z_vals[1], RANGE_POINTS)
        if z_vals.size > MAX_LINES:
            raise ValueError(
                f"Do not supply more than {MAX_LINES} discrete values of z to "
                "plot. The plot is too busy"
            )
        return z_vals

    if wants_range:
        return np.linspace(mean - 2 * sd, mean + 2 * sd, RANGE_POINTS)
    distinct = np.unique(column[~np.isnan(column)])
    if distinct.size < 6:
        # A genuinely discrete moderator: use its own values.
        return distinct
    return np.linspace(mean - sd, mean + sd, 3)


def _lrm_interact(combined_x_vrbl, y_vrbl, shock_size, custom_history) -> str:
    numerator = " + ".join(f"({n})" for n in combined_x_vrbl)
    scale = f"{shock_size}"
    if custom_history is not None:
        scale = f"{custom_history[-1]} * {shock_size}"
    if not y_vrbl:
        return f"{scale} * ({numerator})"
    denominator = " + ".join(f"({n})" for n in y_vrbl)
    return f"({scale} * ({numerator}))/(1 - ({denominator}))"


def interact_adl(
    model,
    x_vrbl,
    z_vrbl,
    x_z_vrbl,
    y_vrbl=None,
    shock_history="impulse",
    cumulative: bool = False,
    effect_type: str = "discrete",
    prediction_values=None,
    baseline_y=None,
    baseline_y_se: float = 0.0,
    shock_size: float = 1.0,
    plot_type: str = "lines",
    line_options: str = "z.lines",
    heatmap_options: str = "significant",
    z_vals=None,
    s_vals=(0, "LRM"),
    level: float = 0.95,
    s_limit: int = 20,
    se_type: str = "const",
    return_lrm: bool = True,
    z_label: str | None = None,
    z_label_rounding: int = 3,
) -> EffectResult:
    r"""Dynamic conditional effects from an interactive ADL model.

    Parameters
    ----------
    model : statsmodels results, or (coefficients, vcov)
        A fitted OLS model containing the interactive ADL estimates.
    x_vrbl : dict
        ``{coefficient name: lag order}`` for the independent variable.
    z_vrbl : dict
        ``{coefficient name: lag order}`` for the moderating variable.  The
        first entry names the series whose distribution supplies default values
        of z.
    x_z_vrbl : dict
        ``{coefficient name: lag order}`` for the interaction terms.  The lag
        order is the one that pertains to the **x** side of the product: for
        contemporaneous x times lagged z (``x_l_1_z``) that is 0, and for lagged
        x times contemporaneous z (``z_l_1_x``) it is 1.
    y_vrbl : dict, optional
        ``{coefficient name: lag order}`` for the lagged dependent variable.
    shock_history : {"impulse", "pulse", "step"}, int or sequence
        What happens to x, as in :func:`~tseffects.gdrf_adl`.
    cumulative : bool
        Accumulate the effects over periods.
    effect_type : {"discrete", "fitted"}
        The conditional effect itself, or fitted values of y.
    prediction_values, baseline_y, baseline_y_se, shock_size
        Baseline handling, as in :func:`~tseffects.gdrf_adl`.  Do not put any
        ``z_vrbl`` entry in ``prediction_values``; those values come from
        ``z_vals``.
    plot_type : {"lines", "heatmap"}
        Which picture the result is for.  This also sets how finely the
        moderator is swept: a handful of discrete values for a family of lines,
        or 50 points across a range for ``s.lines`` and heatmaps.
    line_options : {"z.lines", "s.lines"}
        ``"z.lines"`` puts the period on the x axis and draws one line per value
        of z; ``"s.lines"`` puts z on the x axis and draws one line per period.
    heatmap_options : {"significant", "all"}
        Whether to blank out the cells whose confidence interval covers zero.
    z_vals : sequence, optional
        Values of the moderator.  At most five for ``z.lines``; exactly two
        (lower and upper limit) for ``s.lines`` and heatmaps.
    s_vals : sequence
        Periods to draw for ``line_options="s.lines"``.  ``"LRM"`` asks for the
        long run and falls back to the last period when no LRM exists.
    level, s_limit, se_type, return_lrm
        As in :func:`~tseffects.gdrf_adl`.
    z_label : str, optional
        Axis label for the moderator.  Defaults to the first ``z_vrbl`` name.
    z_label_rounding : int
        Decimal places for the moderator values in the legend.

    Returns
    -------
    EffectResult
        ``.estimates`` has one row per period and value of z, with ``Period``,
        ``Z``, ``Effect``, ``SE``, ``Lower`` and ``Upper``.

    Examples
    --------
    >>> import statsmodels.formula.api as smf
    >>> from tseffects import load_toy_data, interact_adl
    >>> fit = smf.ols("y ~ l_1_y + x + l_1_x + z + l_1_z + x_z + z_l_1_x"
    ...               " + x_l_1_z + l_1_x_l_1_z", data=load_toy_data()).fit()
    >>> out = interact_adl(fit, x_vrbl={"x": 0, "l_1_x": 1},
    ...                    z_vrbl={"z": 0, "l_1_z": 1},
    ...                    x_z_vrbl={"x_z": 0, "z_l_1_x": 1,
    ...                              "x_l_1_z": 0, "l_1_x_l_1_z": 1},
    ...                    y_vrbl={"l_1_y": 1}, z_vals=[-1, 0, 1], s_limit=5)
    >>> sorted(float(z) for z in out.estimates["Z"].unique())
    [-1.0, 0.0, 1.0]
    >>> out.estimates.columns.tolist()
    ['Period', 'Z', 'Effect', 'SE', 'Lower', 'Upper']
    """
    model_info = as_model(model, se_type=se_type)
    x_vrbl = check_vrbl(sanitize_vrbl(x_vrbl), "x_vrbl")
    z_vrbl = check_vrbl(sanitize_vrbl(z_vrbl), "z_vrbl")
    x_z_vrbl = check_vrbl(sanitize_vrbl(x_z_vrbl), "x_z_vrbl")
    y_vrbl = check_vrbl(sanitize_vrbl(y_vrbl), "y_vrbl", allow_none=True)

    chk.check_gdrf_arguments(
        effect_type,
        prediction_values,
        baseline_y,
        baseline_y_se,
        shock_size,
        0,
        "levels",
        kind="interact",
    )
    chk.check_adl_arguments(
        model_info,
        x_vrbl,
        y_vrbl,
        z_vrbl=z_vrbl,
        x_z_vrbl=x_z_vrbl,
        se_type=se_type,
        kind="interact",
    )

    if plot_type not in PLOT_TYPES:
        raise ValueError("plot_type must be one of 'lines' or 'heatmap'")
    if plot_type == "lines" and line_options not in LINE_OPTIONS:
        raise ValueError("line_options must be one of 's.lines' or 'z.lines'")
    if plot_type == "heatmap" and heatmap_options not in HEATMAP_OPTIONS:
        raise ValueError("heatmap_options must be one of 'all' or 'significant'")
    if effect_type == "fitted" and heatmap_options == "significant":
        warnings.warn(
            "'significant' is not applicable to fitted values; fitted values are "
            "predictions, not estimated effects. Using 'all' instead",
            stacklevel=2,
        )
        heatmap_options = "all"

    shock = chk.resolve_shock_history(shock_history, s_limit, cumulative)

    # This function does no levels/differences translation, so d_x = d_y = 0.
    has_lrm = chk.lrm_exists(
        shock["custom"],
        shock["h_order_plot"],
        cumulative,
        0,
        0,
        effect_type,
        return_lrm,
    )

    s_vals = list(s_vals)
    if plot_type == "lines" and line_options == "s.lines":
        if len(s_vals) > MAX_LINES:
            raise ValueError(
                f"Do not supply more than {MAX_LINES} discrete values of s to "
                "plot. The plot is too busy"
            )
        if not has_lrm and s_vals == [0, "LRM"]:
            s_vals = [0, s_limit]
            warnings.warn(
                f"Using the last period (s = {s_limit}) as the value of s to plot "
                "instead of infinity (LRM)",
                stacklevel=2,
            )

    z_values = _default_z_vals(model_info, z_vrbl, plot_type, line_options, z_vals)

    combined_x_vrbl = combine_interaction_terms(x_vrbl, x_z_vrbl)
    pulses = pulse_calculator(combined_x_vrbl, y_vrbl, limit=s_limit)
    if shock["custom"] is not None:
        final = convolution_calculator(0, 0, shock["custom"], pulses)
    else:
        final = general_calculator(0, 0, shock["h_order"], s_limit, pulses)

    coef = model_info.coef_vector
    vcov = model_info.vcov

    if effect_type == "discrete":
        scale = sp.Float(float(shock_size)) if float(shock_size) != 1 else sp.Integer(1)
        exprs = [scale * f if scale != 1 else f for f in final.formulae]
        baseline_expr = None
    else:
        yhats = yhat_calculator(
            final.formulae,
            d_y=0,
            model_info=model_info,
            y_vrbl=y_vrbl,
            inferences_y="levels",
            prediction_values=prediction_values,
            baseline_y=baseline_y,
            shock_size=shock_size,
            z_vrbl=z_vrbl,
            x_z_vrbl=x_z_vrbl,
            use_z=True,
        )
        baseline_expr, exprs = yhats[0], yhats[1:]

    # Every period compiled once, then evaluated at each value of z.  Keeping z
    # symbolic until evaluation is what lets one compilation serve the whole
    # grid instead of one per (period, z) cell.
    batch = BatchDelta(exprs, model_info.names, extra=["z_val"])

    records = []
    if baseline_expr is not None:
        baseline_batch = BatchDelta(
            [baseline_expr], model_info.names, extra=["z_val"]
        )
        for z in z_values:
            row = baseline_batch.evaluate(coef, vcov, level, {"z_val": z})[0]
            records.append(("baseline", float(z), *row))

    for z in z_values:
        rows = batch.evaluate(coef, vcov, level, {"z_val": z})
        for s in range(s_limit + 1):
            records.append((s, float(z), *rows[s]))

    lrm_formula = None
    if has_lrm:
        lrm_formula = _lrm_interact(
            combined_x_vrbl, y_vrbl, shock_size, shock["custom"]
        )
        lrm_batch = BatchDelta([lrm_formula], model_info.names, extra=["z_val"])
        for z in z_values:
            row = lrm_batch.evaluate(coef, vcov, level, {"z_val": z})[0]
            records.append(("LRM", float(z), *row))

    frame = pd.DataFrame(
        records, columns=["Period", "Z", "Effect", "SE", "Lower", "Upper"]
    )
    frame["Period"] = frame["Period"].astype(object)

    if effect_type == "fitted" and baseline_y_se:
        frame["SE"] = np.sqrt(frame["SE"] ** 2 + float(baseline_y_se) ** 2)
        half = z_quantile(level) * frame["SE"]
        frame["Lower"] = frame["Effect"] - half
        frame["Upper"] = frame["Effect"] + half

    keys = [f"s = {s}" for s in range(s_limit + 1)]
    formulae = dict(zip(keys, (str(e) for e in exprs)))
    formulae_expr = dict(zip(keys, exprs))
    shocks = dict(zip(keys, final.shocks))
    if baseline_expr is not None:
        formulae = {"baseline": str(baseline_expr), **formulae}
        formulae_expr = {"baseline": baseline_expr, **formulae_expr}
        shocks = {"baseline": [0], **shocks}
    if lrm_formula is not None:
        formulae["LRM"] = str(sp.sympify(lrm_formula))
        formulae_expr["LRM"] = sp.sympify(lrm_formula)

    kind = "Fitted Values" if effect_type == "fitted" else "Discrete Effect"
    prefix = "Cumulative " if cumulative else ""
    meta = {
        "effect_type": effect_type,
        "cumulative": cumulative,
        "s_limit": s_limit,
        "level": level,
        "lrm": has_lrm,
        "custom_history": shock["custom"],
        "h_order": shock["h_order_plot"],
        "model": "interact",
        "table": _table_meta(
            model_info,
            x_vrbl=x_vrbl,
            y_vrbl=y_vrbl,
            shock_history=shock_history,
            cumulative=cumulative,
            custom=shock["custom"],
            s_limit=s_limit,
            shock_size=shock_size,
            se_type=se_type,
            effect_type=effect_type,
            model_label="interactive ADL",
            has_lrm=has_lrm,
        ),
        "plot_type": plot_type,
        "line_options": line_options,
        "heatmap_options": heatmap_options,
        "z_vals": np.asarray(z_values, dtype=float),
        "s_vals": s_vals,
        "z_label": z_label or next(iter(z_vrbl)),
        "z_label_rounding": z_label_rounding,
        "y_label": f"{prefix}{kind} of Shock History",
        "fill_label": f"{prefix}{kind}\nof Shock History",
        "combined_x_vrbl": combined_x_vrbl,
    }
    return EffectResult(frame, formulae, formulae_expr, shocks, meta)
