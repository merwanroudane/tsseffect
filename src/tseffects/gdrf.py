r"""General Dynamic Response Functions for ADL and GECM models.

These are the functions most users want.  Each takes a model that has already
been estimated by ordinary least squares, a description of which coefficients
belong to the independent and dependent variables at which lags, and returns the
dynamic effect of a shock to x on y, period by period, with delta-method
standard errors.

The four arguments that do the real work are ``shock_history`` (what happens to
x), ``d_x`` / ``d_y`` (how many times each variable was differenced before
estimation) and ``inferences_x`` / ``inferences_y`` (whether you want to read
the shock and the response in levels or in differences).  Together they pick out
a member of the GDRF family: the Impulse Response Function, the Step Response
Function, their cumulations, and the cases where a shock applied in differences
has to be unwound into the levels of y.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import sympy as sp

from . import _checks as chk
from ._delta import BatchDelta, DeltaFn, z_quantile
from ._util import as_model, check_vrbl, sanitize_vrbl
from .baselines import yhat_calculator
from .calculators import convolution_calculator, gecm_to_adl, general_calculator, pulse_calculator
from .results import EffectResult

__all__ = ["gdrf_adl", "adl_effects", "gdrf_gecm", "gecm_effects"]


def _evaluate(formulae, model_info, level: float, subs=None) -> np.ndarray:
    """Delta-method every formula at once, returning an ``(n, 4)`` array."""
    names = list(subs) if subs else model_info.names
    batch = BatchDelta(
        formulae,
        names,
        substitutions=subs,
        target_names=model_info.names if subs else None,
    )
    return batch.evaluate(model_info.coef_vector, model_info.vcov, level)


def _add_baseline_se(frame: pd.DataFrame, value_col: str, baseline_y_se: float, level: float):
    """Fold a user-asserted baseline standard error in, in quadrature.

    The baseline is an assertion about the level of y, independent of the
    model's sampling error, so the two variances simply add.
    """
    if not baseline_y_se:
        return frame
    frame = frame.copy()
    frame["SE"] = np.sqrt(frame["SE"] ** 2 + float(baseline_y_se) ** 2)
    half = z_quantile(level) * frame["SE"]
    frame["Lower"] = frame[value_col] - half
    frame["Upper"] = frame[value_col] + half
    return frame


def _placeholder_coefficients(translated) -> tuple[dict, dict]:
    """Stand a bare symbol in for each translated ADL coefficient.

    Returns the two ``{symbol: lag}`` mappings the response recursion needs,
    and the ``{symbol: expression}`` map that takes them back to the GECM's own
    coefficients.
    """
    placeholders: dict[str, dict] = {"x_vrbl_adl": {}, "y_vrbl_adl": {}}
    subs: dict[str, str] = {}
    for key, prefix in (("x_vrbl_adl", "_adl_b"), ("y_vrbl_adl", "_adl_a")):
        for expression, lag in translated[key].items():
            name = f"{prefix}{lag}"
            placeholders[key][name] = lag
            subs[name] = expression
    return placeholders, subs


def _lrm_adl(x_vrbl, y_vrbl, shock_size, custom_history) -> str:
    """The ADL Long-Run Multiplier, :math:`\\sum_q \\beta_q / (1 - \\sum_i \\alpha_i)`."""
    numerator = "+".join(f"({n})" for n in x_vrbl)
    scale = f"{shock_size}"
    if custom_history is not None:
        scale = f"{custom_history[-1]} * {shock_size}"
    if not y_vrbl:
        return f"{scale} * ({numerator})"
    denominator = "+".join(f"({n})" for n in y_vrbl)
    return f"({scale} * ({numerator}))/(1-({denominator}))"


def _y_label(effect_type, cumulative, custom, s_limit, h_order, plot_d_y, plot_d_x) -> str:
    """The axis label, naming the member of the GDRF family being shown."""
    if custom is not None:
        kind = "Fitted Values" if effect_type == "fitted" else "Discrete Effect"
        prefix = "Cumulative " if cumulative else ""
        return f"{prefix}{kind} of Shock History"
    index = f"({s_limit}, {h_order}, {plot_d_y}, {plot_d_x})"
    if effect_type == "fitted":
        prefix = "Cumulative Fitted Values of " if cumulative else "Fitted Values of "
        return f"{prefix}GDRF$_{{{index}}}$"
    prefix = "Cumulative " if cumulative else ""
    return f"{prefix}GDRF$_{{{index}}}$"


def _assemble(
    final,
    model_info,
    *,
    effect_type,
    shock_size,
    d_y,
    y_vrbl,
    inferences_y,
    prediction_values,
    baseline_y,
    baseline_y_se,
    level,
    s_limit,
    lrm_formula,
    value_col="GDRF",
    subs=None,
):
    """Shared tail of the ADL and GECM paths: evaluate, frame, append the LRM.

    ``subs`` carries the GECM reparameterization: the formulae are written in
    placeholder ADL coefficients, and this maps each back to its expression in
    the GECM's own coefficients.
    """
    if effect_type == "discrete":
        scale = sp.Float(float(shock_size)) if float(shock_size) != 1 else sp.Integer(1)
        exprs = [scale * f if scale != 1 else f for f in final.formulae]
        periods: list = list(range(s_limit + 1))
        shock_keys = [f"s = {s}" for s in range(s_limit + 1)]
        shocks = list(final.shocks)
    else:
        exprs = yhat_calculator(
            final.formulae,
            d_y=d_y,
            model_info=model_info,
            y_vrbl=y_vrbl,
            inferences_y=inferences_y,
            prediction_values=prediction_values,
            baseline_y=baseline_y,
            shock_size=shock_size,
        )
        periods = ["baseline"] + list(range(s_limit + 1))
        shock_keys = ["baseline"] + [f"s = {s}" for s in range(s_limit + 1)]
        shocks = [[0]] + list(final.shocks)

    values = _evaluate(exprs, model_info, level, subs=subs)
    frame = pd.DataFrame(
        {
            "Period": pd.Series(periods, dtype=object),
            value_col: values[:, 0],
            "SE": values[:, 1],
            "Lower": values[:, 2],
            "Upper": values[:, 3],
        }
    )

    if effect_type == "fitted" and inferences_y == "levels":
        frame = _add_baseline_se(frame, value_col, baseline_y_se, level)

    # Report the formulae in the model's own coefficient names, substituting
    # the placeholders back but not multiplying out -- the nested form is both
    # cheap and a readable account of the GECM-to-ADL translation.
    shown = [e.subs(subs) for e in exprs] if subs else exprs
    formulae = dict(zip(shock_keys, (str(e) for e in shown)))
    formulae_expr = dict(zip(shock_keys, shown))
    shock_map = dict(zip(shock_keys, shocks))

    if lrm_formula is not None:
        # The LRM is written directly in the model's coefficients, so it needs
        # no reparameterization.
        expr = sp.sympify(lrm_formula)
        fn = DeltaFn(expr, model_info.names)
        est = fn(model_info.coef_vector, model_info.vcov, level)
        frame = pd.concat(
            [
                frame,
                pd.DataFrame(
                    {
                        "Period": pd.Series(["LRM"], dtype=object),
                        value_col: [est.estimate],
                        "SE": [est.se],
                        "Lower": [est.lower],
                        "Upper": [est.upper],
                    }
                ),
            ],
            ignore_index=True,
        )
        formulae["LRM"] = str(expr)
        formulae_expr["LRM"] = expr

    return frame, formulae, formulae_expr, shock_map


def gdrf_adl(
    model,
    x_vrbl,
    y_vrbl=None,
    d_x: int = 0,
    d_y: int = 0,
    shock_history="pulse",
    inferences_x: str = "levels",
    inferences_y: str = "levels",
    effect_type: str = "discrete",
    cumulative: bool = False,
    prediction_values=None,
    baseline_y=None,
    baseline_y_se: float = 0.0,
    shock_size: float = 1.0,
    level: float = 0.95,
    s_limit: int = 20,
    se_type: str = "const",
    return_lrm: bool = True,
) -> EffectResult:
    r"""Dynamic effects from an autoregressive distributed lag model.

    Parameters
    ----------
    model : statsmodels results, or (coefficients, vcov)
        A fitted OLS model containing the ADL estimates.
    x_vrbl : dict
        ``{coefficient name: lag order}`` for the independent variable and its
        lags, e.g. ``{"x": 0, "l_1_x": 1}``.
    y_vrbl : dict, optional
        ``{coefficient name: lag order}`` for the lagged dependent variable.
        ``None`` means finite dynamics, which is allowed but warned about.
    d_x, d_y : int
        How many times the independent and dependent variables were differenced
        before the model was estimated.
    shock_history : {"pulse", "impulse", "step"}, int or sequence
        What happens to x.  ``"pulse"`` (:math:`h = -1`) is a one-period,
        one-unit increase; ``"step"`` (:math:`h = 0`) is a permanent increase.
        Any integer :math:`h` is accepted, as is a sequence of length
        ``s_limit + 1`` giving an arbitrary history period by period.
    inferences_x, inferences_y : {"levels", "differences"}
        Whether the shock is applied, and the response read, in the levels of
        the variable or in its differences.  Only meaningful when the
        corresponding ``d_`` is non-zero.
    effect_type : {"discrete", "fitted"}
        ``"discrete"`` reports the effect itself; ``"fitted"`` adds it to a
        baseline level of y and reports the trajectory.
    cumulative : bool
        Accumulate the effects over periods.  With ``shock_history="pulse"``
        this gives the Cumulative Impulse Response Function.
    prediction_values : dict, optional
        Values at which to hold the other regressors when computing a
        steady-state baseline (``effect_type="fitted"``, ``d_y = 0``).  Set any
        differenced variables to 0, and give multiple lags of one variable the
        same value.
    baseline_y : float, optional
        A baseline level of y for ``effect_type="fitted"``.  Required when
        ``d_y > 0`` and ``inferences_y="levels"``, since a differenced model
        carries no information about the level of y.
    baseline_y_se : float
        An asserted standard error for that baseline, added in quadrature.
    shock_size : float
        Size of the shock to x, in the units of x.
    level : float
        Confidence level for the intervals.
    s_limit : int
        Number of periods after the shock to report.
    se_type : str
        ``"const"`` for the classical covariance matrix, or any of
        ``"HC0"``-``"HC5"`` for a heteroskedasticity-consistent one.
    return_lrm : bool
        Report the Long-Run Multiplier when it is well defined.

    Returns
    -------
    EffectResult
        With ``.estimates``, ``.formulae`` and a ``.plot()`` method.

    Notes
    -----
    A Long-Run Multiplier exists only when :math:`h - d_x + d_y = 0`.  Outside
    that case the effects either decay to zero or diverge, and a warning says
    which.

    Examples
    --------
    >>> import statsmodels.formula.api as smf
    >>> from tseffects import load_toy_data, gdrf_adl
    >>> data = load_toy_data()
    >>> fit = smf.ols("y ~ l_1_y + x + l_1_x", data=data).fit()
    >>> out = gdrf_adl(fit, x_vrbl={"x": 0, "l_1_x": 1}, y_vrbl={"l_1_y": 1},
    ...                shock_history="pulse", s_limit=5)
    >>> out.estimates.shape
    (6, 5)
    >>> out.formulae["s = 2"]
    'l_1_x*l_1_y + l_1_y**2*x'
    """
    model_info = as_model(model, se_type=se_type)
    x_vrbl = check_vrbl(sanitize_vrbl(x_vrbl), "x_vrbl")
    y_vrbl = check_vrbl(sanitize_vrbl(y_vrbl), "y_vrbl", allow_none=True)

    chk.check_gdrf_arguments(
        effect_type,
        prediction_values,
        baseline_y,
        baseline_y_se,
        shock_size,
        d_y,
        inferences_y,
        kind="GDRF",
    )
    chk.check_adl_arguments(
        model_info,
        x_vrbl,
        y_vrbl,
        d_x=d_x,
        d_y=d_y,
        inferences_x=inferences_x,
        inferences_y=inferences_y,
        se_type=se_type,
        kind="GDRF",
    )

    shock = chk.resolve_shock_history(shock_history, s_limit, cumulative)
    adjusted = chk.adjust_inferences(inferences_y, inferences_x, int(d_y), int(d_x))

    pulses = pulse_calculator(x_vrbl, y_vrbl, limit=s_limit)
    if shock["custom"] is not None:
        final = convolution_calculator(
            adjusted["calc_d_x"], adjusted["calc_d_y"], shock["custom"], pulses
        )
    else:
        final = general_calculator(
            adjusted["calc_d_x"],
            adjusted["calc_d_y"],
            shock["h_order"],
            s_limit,
            pulses,
        )

    has_lrm = chk.lrm_exists(
        shock["custom"],
        shock["h_order_plot"],
        cumulative,
        adjusted["calc_d_x"],
        adjusted["calc_d_y"],
        effect_type,
        return_lrm,
    )
    lrm_formula = (
        _lrm_adl(x_vrbl, y_vrbl, shock_size, shock["custom"]) if has_lrm else None
    )

    frame, formulae, formulae_expr, shocks = _assemble(
        final,
        model_info,
        effect_type=effect_type,
        shock_size=shock_size,
        d_y=int(d_y),
        y_vrbl=y_vrbl,
        inferences_y=inferences_y,
        prediction_values=prediction_values,
        baseline_y=baseline_y,
        baseline_y_se=baseline_y_se,
        level=level,
        s_limit=s_limit,
        lrm_formula=lrm_formula,
    )

    h_label = shock["h_order"] if effect_type == "fitted" else shock["h_order_plot"]
    meta = {
        "effect_type": effect_type,
        "cumulative": cumulative,
        "s_limit": s_limit,
        "level": level,
        "lrm": has_lrm,
        "custom_history": shock["custom"],
        "h_order": h_label,
        "plot_d_x": adjusted["plot_d_x"],
        "plot_d_y": adjusted["plot_d_y"],
        "y_label": _y_label(
            effect_type,
            cumulative,
            shock["custom"],
            s_limit,
            h_label,
            adjusted["plot_d_y"],
            adjusted["plot_d_x"],
        ),
        "model": "adl",
    }
    return EffectResult(frame, formulae, formulae_expr, shocks, meta)


def adl_effects(
    model,
    x_vrbl,
    y_vrbl=None,
    shock_history="pulse",
    level: float = 0.95,
    s_limit: int = 20,
    se_type: str = "const",
) -> EffectResult:
    """Impulse or step response from a stationary ADL model, in levels.

    A thin wrapper on :func:`gdrf_adl` for the common case where both variables
    are in levels and the discrete effect is what is wanted.

    Examples
    --------
    >>> import statsmodels.formula.api as smf
    >>> from tseffects import load_toy_data, adl_effects
    >>> fit = smf.ols("y ~ l_1_y + x + l_1_x", data=load_toy_data()).fit()
    >>> out = adl_effects(fit, {"x": 0, "l_1_x": 1}, {"l_1_y": 1},
    ...                   shock_history="step", s_limit=5)
    >>> round(float(out.lrm["GDRF"]), 8) == round(
    ...     float((fit.params["x"] + fit.params["l_1_x"])
    ...           / (1 - fit.params["l_1_y"])), 8)
    True
    """
    return gdrf_adl(
        model,
        x_vrbl=x_vrbl,
        y_vrbl=y_vrbl,
        d_x=0,
        d_y=0,
        shock_history=shock_history,
        level=level,
        s_limit=s_limit,
        se_type=se_type,
    )


def gdrf_gecm(
    model,
    x_vrbl,
    y_vrbl,
    x_d_vrbl,
    y_d_vrbl=None,
    x_vrbl_d_x: int = 0,
    y_vrbl_d_y: int = 0,
    x_d_vrbl_d_x: int = 1,
    y_d_vrbl_d_y: int = 1,
    shock_history="pulse",
    inferences_x: str = "levels",
    inferences_y: str = "levels",
    effect_type: str = "discrete",
    cumulative: bool = False,
    baseline_y=None,
    baseline_y_se: float = 0.0,
    shock_size: float = 1.0,
    level: float = 0.95,
    s_limit: int = 20,
    se_type: str = "const",
    return_lrm: bool = True,
) -> EffectResult:
    r"""Dynamic effects from a Generalized Error-Correction Model.

    The GECM and the ADL are the same model written two ways, so this
    translates the coefficients with :func:`~tseffects.gecm_to_adl` and then
    runs the ADL machinery.  No information is lost in the translation, and the
    standard errors are computed from the GECM's own covariance matrix.

    Parameters
    ----------
    model : statsmodels results, or (coefficients, vcov)
        A fitted OLS model containing the GECM estimates.
    x_vrbl, y_vrbl : dict
        The *first lag* of the independent and dependent variables in levels --
        the error-correction terms.  A GECM has exactly one of each, so these
        must be single-entry mappings with lag order 1.
    x_d_vrbl : dict
        The differenced independent variable and its lags, e.g.
        ``{"d_x": 0, "l_1_d_x": 1}``.
    y_d_vrbl : dict, optional
        The lagged differenced dependent variable.  ``None`` means the model has
        no short-run dynamics in y.
    x_vrbl_d_x, y_vrbl_d_y : int
        Order of differencing of the levels terms, normally 0.
    x_d_vrbl_d_x, y_d_vrbl_d_y : int
        Order of differencing of the differenced terms, normally 1.  Each must
        be exactly one more than its levels counterpart.
    shock_history, inferences_x, inferences_y, effect_type, cumulative
        As in :func:`gdrf_adl`.  A GECM recovers inferences in levels by
        construction, so both ``inferences_`` arguments must stay ``"levels"``.
    baseline_y, baseline_y_se, shock_size, level, s_limit, se_type, return_lrm
        As in :func:`gdrf_adl`.  Steady-state ``prediction_values`` are not
        available for a GECM; supply ``baseline_y`` instead.

    Returns
    -------
    EffectResult

    Examples
    --------
    >>> import statsmodels.formula.api as smf
    >>> from tseffects import load_toy_data, gdrf_gecm
    >>> fit = smf.ols("d_y ~ l_1_y + l_1_d_y + l_1_x + d_x + l_1_d_x",
    ...               data=load_toy_data()).fit()
    >>> out = gdrf_gecm(fit, x_vrbl={"l_1_x": 1}, y_vrbl={"l_1_y": 1},
    ...                 x_d_vrbl={"d_x": 0, "l_1_d_x": 1},
    ...                 y_d_vrbl={"l_1_d_y": 1}, shock_history="step", s_limit=5)
    >>> round(float(out.lrm["GDRF"]), 8) == round(
    ...     float(-fit.params["l_1_x"] / fit.params["l_1_y"]), 8)
    True
    """
    model_info = as_model(model, se_type=se_type)
    x_vrbl = check_vrbl(sanitize_vrbl(x_vrbl), "x_vrbl")
    y_vrbl = check_vrbl(sanitize_vrbl(y_vrbl), "y_vrbl")
    x_d_vrbl = check_vrbl(sanitize_vrbl(x_d_vrbl), "x_d_vrbl")
    y_d_vrbl = check_vrbl(sanitize_vrbl(y_d_vrbl), "y_d_vrbl", allow_none=True)

    chk.check_gdrf_arguments(
        effect_type,
        None,
        baseline_y,
        baseline_y_se,
        shock_size,
        y_vrbl_d_y,
        inferences_y,
        kind="GDRF",
    )
    chk.check_gecm_arguments(
        model_info,
        x_vrbl,
        y_vrbl,
        x_d_vrbl,
        y_d_vrbl,
        x_vrbl_d_x,
        y_vrbl_d_y,
        x_d_vrbl_d_x,
        y_d_vrbl_d_y,
        inferences_x,
        inferences_y,
        se_type=se_type,
        kind="GDRF",
    )

    shock = chk.resolve_shock_history(shock_history, s_limit, cumulative)
    adjusted = chk.adjust_inferences(
        inferences_y, inferences_x, int(y_vrbl_d_y), int(x_vrbl_d_x)
    )

    translated = gecm_to_adl(x_vrbl, y_vrbl, x_d_vrbl, y_d_vrbl)
    # Each translated ADL coefficient is a *sum* of GECM coefficients.  Writing
    # those sums straight into the response recursion would raise a sum to the
    # power of the horizon, so a placeholder symbol stands in for each one and
    # the delta method applies the chain rule through the (linear) translation.
    placeholders, subs = _placeholder_coefficients(translated)
    pulses = pulse_calculator(
        placeholders["x_vrbl_adl"], placeholders["y_vrbl_adl"], limit=s_limit
    )
    if shock["custom"] is not None:
        final = convolution_calculator(
            adjusted["calc_d_x"], adjusted["calc_d_y"], shock["custom"], pulses
        )
    else:
        final = general_calculator(
            adjusted["calc_d_x"],
            adjusted["calc_d_y"],
            shock["h_order"],
            s_limit,
            pulses,
        )

    has_lrm = chk.lrm_exists(
        shock["custom"],
        shock["h_order_plot"],
        cumulative,
        adjusted["calc_d_x"],
        adjusted["calc_d_y"],
        effect_type,
        return_lrm,
    )
    lrm_formula = None
    if has_lrm:
        # In a GECM the long-run multiplier is -theta_x / theta_y, read straight
        # off the two levels terms.
        x_name = next(iter(x_vrbl))
        y_name = next(iter(y_vrbl))
        lrm_formula = f"({shock_size}* (-1) * ({x_name}))/({y_name})"

    frame, formulae, formulae_expr, shocks = _assemble(
        final,
        model_info,
        effect_type=effect_type,
        shock_size=shock_size,
        d_y=int(y_vrbl_d_y),
        y_vrbl=placeholders["y_vrbl_adl"],
        inferences_y=inferences_y,
        prediction_values=None,
        baseline_y=baseline_y,
        baseline_y_se=baseline_y_se,
        level=level,
        s_limit=s_limit,
        lrm_formula=lrm_formula,
        subs=subs,
    )

    h_label = shock["h_order"] if effect_type == "fitted" else shock["h_order_plot"]
    meta = {
        "effect_type": effect_type,
        "cumulative": cumulative,
        "s_limit": s_limit,
        "level": level,
        "lrm": has_lrm,
        "custom_history": shock["custom"],
        "h_order": h_label,
        "plot_d_x": adjusted["plot_d_x"],
        "plot_d_y": adjusted["plot_d_y"],
        "y_label": _y_label(
            effect_type,
            cumulative,
            shock["custom"],
            s_limit,
            h_label,
            adjusted["plot_d_y"],
            adjusted["plot_d_x"],
        ),
        "model": "gecm",
        "adl_coefficients": translated,
    }
    return EffectResult(frame, formulae, formulae_expr, shocks, meta)


def gecm_effects(
    model,
    x_vrbl,
    y_vrbl,
    x_d_vrbl,
    y_d_vrbl=None,
    shock_history="pulse",
    level: float = 0.95,
    s_limit: int = 20,
    se_type: str = "const",
) -> EffectResult:
    """Impulse or step response from a standard GECM in levels and differences.

    A thin wrapper on :func:`gdrf_gecm` for the usual case: variables in levels
    at lag 1, their first differences, and the discrete effect.

    Examples
    --------
    >>> import statsmodels.formula.api as smf
    >>> from tseffects import load_toy_data, gecm_effects
    >>> fit = smf.ols("d_y ~ l_1_y + l_1_d_y + l_1_x + d_x + l_1_d_x",
    ...               data=load_toy_data()).fit()
    >>> out = gecm_effects(fit, {"l_1_x": 1}, {"l_1_y": 1},
    ...                    {"d_x": 0, "l_1_d_x": 1}, {"l_1_d_y": 1},
    ...                    shock_history="step", s_limit=5)
    >>> out.estimates["Period"].tolist()[-1]
    'LRM'
    """
    return gdrf_gecm(
        model,
        x_vrbl=x_vrbl,
        y_vrbl=y_vrbl,
        x_d_vrbl=x_d_vrbl,
        y_d_vrbl=y_d_vrbl,
        x_vrbl_d_x=0,
        y_vrbl_d_y=0,
        x_d_vrbl_d_x=1,
        y_d_vrbl_d_y=1,
        shock_history=shock_history,
        level=level,
        s_limit=s_limit,
        se_type=se_type,
    )
