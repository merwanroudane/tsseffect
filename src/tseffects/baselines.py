r"""Turning effects into fitted values.

By default a GDRF is an *effect*: how much the dependent variable moves
relative to wherever it was.  :func:`yhat_calculator` adds that effect to a
baseline level of the dependent variable, so the trajectory itself can be read
off the plot.

Two cases matter.  When the dependent variable is in levels
(:math:`d_y = 0`) the baseline can be the model-implied steady state

.. math:: \bar y = \frac{\text{Intercept} + \sum_v \bar v \,\beta_v}{1 - \sum_i \alpha_i},

which is built as an *expression* so the delta method carries model uncertainty
into the baseline as well.  When the dependent variable is differenced the model
knows nothing about the level of y, so the baseline has to be supplied by the
user, optionally with its own standard error.
"""

from __future__ import annotations

import warnings

import sympy as sp

from ._util import INTERCEPT

#: Symbol standing for the value of the moderating variable in an interaction.
Z_VAL = sp.Symbol("z_val")


def get_value(var: str, prediction_values, model_info) -> float:
    """The prediction value for one regressor.

    Uses the supplied value when there is one, and otherwise falls back to the
    mean of that series in the model frame -- warning, because a mean is rarely
    what someone means for a differenced or interacted term.
    """
    if prediction_values:
        value = prediction_values.get(var)
        if value is not None:
            return float(value)
    mean = model_info.mean_of(var)
    if mean is None:
        raise ValueError(
            f"{var!r} is in the model but has no prediction value and is not in "
            "the model frame. Supply it through prediction_values"
        )
    warnings.warn(
        f"{var} not in prediction_values; mean of series used", stacklevel=3
    )
    return mean


def steady_state_expr(
    model_info,
    prediction_values,
    y_vrbl,
    z_vrbl=None,
    x_z_vrbl=None,
    use_z: bool = False,
):
    r"""The model-implied steady state of y, as a SymPy expression.

    Every regressor except the lagged dependent variables and the constant is
    held at its prediction value; the lagged dependent variables become the
    :math:`1 - \sum_i \alpha_i` denominator.  In an interaction model the
    moderator enters as the free symbol ``z_val``, and an interaction term
    enters as its own prediction value times ``z_val``.
    """
    y_names = set(y_vrbl or {})
    z_names = set(z_vrbl or {})
    xz_names = set(x_z_vrbl or {})

    if use_z and z_names & set(prediction_values or {}):
        raise ValueError(
            "Do not supply a value for any z_vrbl in prediction_values. These "
            "come from z_vals"
        )

    numerator = sp.Symbol(INTERCEPT) if INTERCEPT in model_info.coef else sp.Integer(0)
    for name in model_info.names:
        if name in y_names or name == INTERCEPT:
            continue
        beta = sp.Symbol(name)
        if use_z and name in z_names:
            numerator += Z_VAL * beta
        elif use_z and name in xz_names:
            numerator += sp.Float(get_value(name, prediction_values, model_info)) * Z_VAL * beta
        else:
            numerator += sp.Float(get_value(name, prediction_values, model_info)) * beta

    if not y_names:
        # A finite distributed lag model has no autoregressive denominator.
        return numerator
    denominator = sp.Integer(1) - sum(
        (sp.sympify(name) for name in y_vrbl), start=sp.Integer(0)
    )
    return numerator / denominator


def yhat_calculator(
    formulae,
    d_y: int,
    model_info,
    y_vrbl=None,
    inferences_y: str = "levels",
    prediction_values=None,
    baseline_y=None,
    shock_size: float = 1.0,
    z_vrbl=None,
    x_z_vrbl=None,
    use_z: bool = False,
) -> list:
    """Prepend a baseline level of y to each effect formula.

    Parameters
    ----------
    formulae : list of sympy.Expr
        Effect formulae, from :func:`~tseffects.general_calculator` or
        :func:`~tseffects.convolution_calculator`.
    d_y : int
        Order of differencing of the dependent variable in the fitted model.
    model_info : ModelInfo
        The fitted model, for the steady state and any prediction-value means.
    y_vrbl : dict, optional
        The lagged dependent variables, which form the steady-state denominator.
    inferences_y : {"levels", "differences"}
        Whether the response is read in levels or differences.
    prediction_values : dict, optional
        Values at which to hold the other regressors for the steady state.
    baseline_y : float, optional
        A user-supplied baseline, which takes precedence when ``d_y == 0`` and
        is required when ``d_y > 0`` and ``inferences_y == "levels"``.
    shock_size : float
        Scales the whole effect, in the units of x.
    z_vrbl, x_z_vrbl, use_z
        Interaction bookkeeping; ``use_z`` makes the moderator enter the
        baseline as the free symbol ``z_val``.

    Returns
    -------
    list of sympy.Expr
        ``len(formulae) + 1`` expressions: the baseline, then the fitted value
        in each period.
    """
    if d_y == 0:
        if baseline_y is not None:
            baseline = sp.Float(float(baseline_y))
        else:
            if prediction_values is None and not use_z:
                raise ValueError(
                    "No baseline or prediction values for y in levels"
                )
            baseline = steady_state_expr(
                model_info,
                prediction_values,
                y_vrbl,
                z_vrbl=z_vrbl,
                x_z_vrbl=x_z_vrbl,
                use_z=use_z,
            )
    elif d_y == 1:
        if inferences_y == "differences":
            # Still in differences: there is no level to start from.
            baseline = sp.Integer(0)
        else:
            if baseline_y is None:
                raise ValueError(
                    "No baseline for y in levels. A model in differences carries "
                    "no information about the level of y, so supply baseline_y"
                )
            baseline = sp.Float(float(baseline_y))
    else:
        raise NotImplementedError(
            f"fitted values are defined for d_y of 0 or 1, not {d_y}. Use "
            "effect_type = 'discrete' for higher orders of differencing"
        )

    scale = sp.Float(float(shock_size)) if float(shock_size) != 1 else sp.Integer(1)
    # No expansion here: the effect formulae already carry whichever form
    # pulse_calculator chose, and multiplying a GECM formula out would blow up.
    return [baseline] + [baseline + scale * f for f in formulae]
