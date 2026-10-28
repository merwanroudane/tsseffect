"""Argument validation.

These mirror ``GDRF.dummy.checks``, ``adl.dummy.checks`` and
``gecm.dummy.checks`` in the R package, including which situations warn and
which stop, so a specification that R rejects is rejected here with the same
reasoning.
"""

from __future__ import annotations

import warnings

import numpy as np

from ._vcov import VCOV_TYPES

EFFECT_TYPES = ("discrete", "fitted")
INFERENCES = ("levels", "differences")


def _effect_label(kind: str) -> str:
    return "dynamic conditional effect" if kind == "interact" else "GDRF"


def check_effect_type(effect_type: str) -> None:
    if effect_type not in EFFECT_TYPES:
        raise ValueError(
            "Invalid effect_type. effect_type must be one of 'discrete' or 'fitted'"
        )


def check_se_type(se_type: str) -> None:
    if se_type not in VCOV_TYPES:
        raise ValueError(
            f"Invalid se_type {se_type!r}. se_type must be one of: "
            + ", ".join(VCOV_TYPES)
        )


def check_integer(value, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} must be specified")
    if isinstance(value, bool) or not isinstance(value, (int, np.integer, float)):
        raise ValueError(f"{name} must be an integer")
    if float(value) % 1 != 0:
        raise ValueError(f"{name} must be an integer")
    return int(value)


def check_gdrf_arguments(
    effect_type: str,
    prediction_values,
    baseline_y,
    baseline_y_se,
    shock_size,
    d_y: int,
    inferences_y: str,
    kind: str = "GDRF",
) -> None:
    """The shared baseline/shock-size checks for anything producing a GDRF."""
    label = _effect_label(kind)
    check_effect_type(effect_type)

    if isinstance(shock_size, bool) or not isinstance(
        shock_size, (int, float, np.integer, np.floating)
    ):
        raise ValueError("shock_size must be numeric")
    if np.ndim(shock_size) != 0:
        raise ValueError(
            "shock_size should be a single value by which to scale the entire "
            "shock_history"
        )

    if effect_type == "discrete":
        if prediction_values is not None:
            raise ValueError(
                f"Do not supply prediction_values when effect_type = 'discrete'. "
                f"The {label} is the discrete effect, which takes no predicted values"
            )
        if baseline_y is not None and baseline_y != 0:
            raise ValueError(
                f"Do not supply baseline_y when effect_type = 'discrete'. The "
                f"{label} is the discrete effect, which is always compared to 0"
            )
        if baseline_y_se is not None and baseline_y_se != 0:
            raise ValueError(
                f"Do not supply baseline_y_se when effect_type = 'discrete'. The "
                f"{label} is the discrete effect, which has its own standard errors"
            )
        return

    # Everything below is effect_type == "fitted".
    if prediction_values is None and baseline_y is None:
        raise ValueError(
            "You must provide either a baseline_y value or prediction_values (to "
            "calculate a steady state). If unsure, set baseline_y = 0 (no assumed "
            "starting value for y)"
        )
    if prediction_values is not None and not isinstance(prediction_values, dict):
        raise ValueError(
            "prediction_values must be a mapping of {variable: value}. Be sure to "
            "set any differenced variables to 0"
        )
    if prediction_values is not None and baseline_y_se not in (0, 0.0):
        raise ValueError(
            "If using prediction_values, leave baseline_y_se at 0 and let the model "
            "supply model-based uncertainty around the steady state"
        )
    if baseline_y is not None:
        if isinstance(baseline_y, bool) or not isinstance(
            baseline_y, (int, float, np.integer, np.floating)
        ):
            raise ValueError("If providing a baseline_y, it must be numeric")
        if np.ndim(baseline_y) != 0:
            raise ValueError("Only provide a single baseline_y value")
    if baseline_y_se is not None:
        if isinstance(baseline_y_se, bool) or not isinstance(
            baseline_y_se, (int, float, np.integer, np.floating)
        ):
            raise ValueError("If providing a baseline_y_se, it must be numeric")
        if np.ndim(baseline_y_se) != 0:
            raise ValueError("Only provide a single baseline_y_se value")

    if d_y == 0:
        if baseline_y is not None:
            if prediction_values is not None:
                warnings.warn(
                    "Both baseline_y and prediction_values supplied; baseline_y "
                    "takes precedence and prediction_values will be ignored",
                    stacklevel=3,
                )
        else:
            warnings.warn(
                "If any differenced variables are included in the model, ensure "
                "they are set to 0 in prediction_values for a meaningful "
                "steady-state prediction. Multiple lags of the same variable "
                "should also be set to the same value",
                stacklevel=3,
            )
    elif d_y > 0:
        if prediction_values is not None:
            warnings.warn(
                "prediction_values is ignored when d_y > 0: there is no "
                "steady-state prediction for y",
                stacklevel=3,
            )
        if inferences_y == "differences":
            if baseline_y is not None and baseline_y != 0:
                warnings.warn(
                    "Assuming anything other than baseline_y = 0 when "
                    "inferences_y = 'differences' suggests the model is unstable "
                    "(y is always changing). Using 0 as baseline_y",
                    stacklevel=3,
                )
            if shock_size == 1:
                warnings.warn(
                    "effect_type = 'fitted' with inferences_y = 'differences' and "
                    "shock_size = 1 is identical to effect_type = 'discrete'; "
                    "returning formulae unchanged",
                    stacklevel=3,
                )
        elif inferences_y == "levels" and baseline_y is None:
            raise ValueError(
                "There is no baseline for y if effect_type = 'fitted' and d_y > 0. "
                "Specify a baseline through baseline_y (with optional uncertainty "
                "through baseline_y_se). If you just want to observe changes with "
                "no meaningful baseline, use baseline_y = 0 and baseline_y_se = 0"
            )


def check_adl_arguments(
    model_info,
    x_vrbl,
    y_vrbl,
    z_vrbl=None,
    x_z_vrbl=None,
    d_x=None,
    d_y=None,
    inferences_x=None,
    inferences_y=None,
    se_type: str = "const",
    kind: str = "GDRF",
) -> None:
    """Checks specific to an ADL specification."""
    label = _effect_label(kind)
    check_se_type(se_type)

    if x_vrbl is None:
        raise ValueError(
            f"Variables in {label} terms (x and possibly y) must be specified "
            "through x_vrbl and possibly y_vrbl"
        )
    if y_vrbl is None:
        warnings.warn(
            f"No y_vrbl in {label} terms implies a static or finite dynamics "
            "model: are you sure you want this?",
            stacklevel=3,
        )

    model_info.require(x_vrbl, "x_vrbl")
    model_info.require(y_vrbl, "y_vrbl")

    if kind == "GDRF":
        d_x = check_integer(d_x, "d_x")
        d_y = check_integer(d_y, "d_y")
        if inferences_x not in INFERENCES:
            raise ValueError(
                "Invalid inferences_x. The shock_history for x must be either "
                "'levels' or 'differences'"
            )
        if inferences_y not in INFERENCES:
            raise ValueError(
                "Invalid inferences_y. The response function for y must be either "
                "'levels' or 'differences'"
            )
        if inferences_x == "differences" and d_x == 0:
            raise ValueError(
                "Invalid inferences_x. The shock_history for x cannot be in a "
                "higher order of differencing (d_x) than the original independent "
                "variable"
            )
        if inferences_y == "differences" and d_y == 0:
            raise ValueError(
                "Invalid inferences_y. The response function for y cannot be in a "
                "higher order of differencing (d_y) than the original dependent "
                "variable"
            )

    if kind == "interact":
        if z_vrbl is None or x_z_vrbl is None:
            raise ValueError(
                "Variables in the interaction term (x, z and x:z) must be "
                "specified through x_vrbl, z_vrbl and x_z_vrbl"
            )
        model_info.require(z_vrbl, "z_vrbl")
        model_info.require(x_z_vrbl, "x_z_vrbl")


def check_gecm_arguments(
    model_info,
    x_vrbl,
    y_vrbl,
    x_d_vrbl,
    y_d_vrbl,
    x_vrbl_d_x,
    y_vrbl_d_y,
    x_d_vrbl_d_x,
    y_d_vrbl_d_y,
    inferences_x: str,
    inferences_y: str,
    se_type: str = "const",
    kind: str = "GDRF",
) -> None:
    """Checks specific to a GECM specification."""
    label = _effect_label(kind)
    check_se_type(se_type)

    if x_vrbl is None or y_vrbl is None or x_d_vrbl is None:
        raise ValueError(
            f"Variables in {label} terms (x and y) and lagged differences must be "
            "specified through x_vrbl, y_vrbl, x_d_vrbl and (possibly) y_d_vrbl "
            "for a GECM"
        )

    x_vrbl_d_x = check_integer(x_vrbl_d_x, "x_vrbl_d_x")
    y_vrbl_d_y = check_integer(y_vrbl_d_y, "y_vrbl_d_y")
    x_d_vrbl_d_x = check_integer(x_d_vrbl_d_x, "x_d_vrbl_d_x")
    y_d_vrbl_d_y = check_integer(y_d_vrbl_d_y, "y_d_vrbl_d_y")

    if x_d_vrbl_d_x - x_vrbl_d_x != 1:
        raise ValueError(
            "In a GECM, the variable in differences should be one order of "
            "differencing from the variable in levels. Check the order of "
            "differencing of x"
        )
    if y_d_vrbl is not None and y_d_vrbl_d_y - y_vrbl_d_y != 1:
        raise ValueError(
            "In a GECM, the variable in differences should be one order of "
            "differencing from the variable in levels. Check the order of "
            "differencing of y"
        )

    if len(x_vrbl) != 1 or len(y_vrbl) != 1:
        raise ValueError(
            "In a GECM, include only the first lag of the variable in levels "
            "(x_vrbl and y_vrbl)"
        )
    if list(x_vrbl.values())[0] != 1 or list(y_vrbl.values())[0] != 1:
        raise ValueError(
            "In a GECM, include only the first lag of the variable in levels "
            "(x_vrbl and y_vrbl)"
        )

    if inferences_x != "levels" or inferences_y != "levels":
        raise ValueError(
            f"In a GECM, inferences regarding the {label} of x on y are "
            "automatically recovered in levels"
        )

    model_info.require(x_vrbl, "x_vrbl")
    model_info.require(y_vrbl, "y_vrbl")
    model_info.require(x_d_vrbl, "x_d_vrbl")
    model_info.require(y_d_vrbl, "y_d_vrbl")


def resolve_shock_history(shock_history, s_limit: int, cumulative: bool) -> dict:
    """Validate the shock history and convert a word to its integer order.

    ``"pulse"`` / ``"impulse"`` is :math:`h = -1` and ``"step"`` is
    :math:`h = 0`; an integer is taken as :math:`h` directly; a sequence of
    length ``s_limit + 1`` is a custom history.  When ``cumulative`` is set, a
    custom history is cumulated and an integer order is incremented by one.

    Returns
    -------
    dict
        ``h_order`` (used for the calculation), ``h_order_plot`` (the order the
        user asked for, used for axis labels) and ``shock_history``.
    """
    if shock_history is None:
        raise ValueError("shock_history must be specified")

    custom = None
    h_order = None

    if isinstance(shock_history, str):
        if shock_history in ("pulse", "impulse"):
            h_order = -1
        elif shock_history == "step":
            h_order = 0
        else:
            raise ValueError(
                "Invalid shock_history. shock_history must be one of 'pulse', "
                "'impulse' or 'step', a sequence for a custom shock history, or "
                "an integer h representing the order of the GDRF"
            )
    elif isinstance(shock_history, bool):
        raise ValueError("Invalid shock_history: booleans are not shock histories")
    elif isinstance(shock_history, (int, np.integer)):
        h_order = int(shock_history)
    elif isinstance(shock_history, (float, np.floating)):
        if float(shock_history) % 1 != 0:
            raise ValueError(
                "Invalid shock_history. A single-value shock_history is the "
                "integer order h"
            )
        h_order = int(shock_history)
    else:
        custom = np.asarray(shock_history, dtype=float).ravel()
        if custom.size == 1:
            h_order, custom = int(custom[0]), None
        elif custom.size != s_limit + 1:
            raise ValueError(
                f"Invalid shock_history. A custom shock history must be of length "
                f"s_limit + 1 ({s_limit + 1}). The supplied shock history has "
                f"length {custom.size}"
            )

    h_order_plot = h_order
    if cumulative:
        if custom is not None:
            custom = np.cumsum(custom)
        else:
            h_order = h_order + 1

    return {
        "h_order": h_order,
        "h_order_plot": h_order_plot,
        "shock_history": custom if custom is not None else h_order_plot,
        "custom": custom,
    }


def adjust_inferences(inferences_y: str, inferences_x: str, d_y: int, d_x: int) -> dict:
    """Split the orders of differencing into calculation and labelling parts.

    Asking for inferences in the *differences* of a variable means the
    calculation needs no unwinding for it, but the plot label still has to say
    the response is in differences -- so each order is carried in two places.
    """
    if inferences_y == "differences":
        calc_d_y, plot_d_y = 0, d_y
    else:
        calc_d_y, plot_d_y = d_y, 0
    if inferences_x == "differences":
        calc_d_x, plot_d_x = 0, d_x
    else:
        calc_d_x, plot_d_x = d_x, 0
    return {
        "calc_d_x": calc_d_x,
        "calc_d_y": calc_d_y,
        "plot_d_x": plot_d_x,
        "plot_d_y": plot_d_y,
    }


def lrm_exists(
    custom_history,
    h_order_plot,
    cumulative: bool,
    calc_d_x: int,
    calc_d_y: int,
    effect_type: str,
    return_lrm: bool,
) -> bool:
    r"""Whether a Long-Run Multiplier is well defined and should be reported.

    The LRM exists when :math:`h - d_x + d_y = 0`: the effects then settle on a
    finite non-zero value.  If that expression is negative the effects decay to
    zero, and if it is positive they diverge, so there is nothing to report
    either way.  Fitted values are excluded entirely, since they are
    predictions rather than effects.
    """
    if not return_lrm or effect_type == "fitted":
        return False

    if custom_history is not None:
        if not cumulative:
            # A custom history without cumulation has no limiting value to read.
            return False
        warnings.warn(
            "The LRM condition (h - d_x + d_y = 0) holds for integer h but is "
            "not guaranteed to hold for a custom shock history. Interpret the "
            "LRM with caution",
            stacklevel=3,
        )
        return True

    h = int(h_order_plot) + (1 if cumulative else 0)
    condition = h - calc_d_x + calc_d_y
    if condition == 0:
        return True
    if condition < 0:
        warnings.warn(
            "return_lrm = True but the effects decay to 0 as s -> infinity. No "
            "LRM is returned",
            stacklevel=3,
        )
    else:
        warnings.warn(
            "return_lrm = True but the effects diverge to +/-infinity as "
            "s -> infinity. No LRM is returned",
            stacklevel=3,
        )
    return False
