r"""The symbolic engine: impulse responses, generalized shock histories, and
the GECM-to-ADL translation.

These four functions are the mathematical core of the package, and they do no
numerical work at all.  Each returns SymPy expressions written in the *names*
of the model's coefficients, which :mod:`tseffects._delta` later evaluates.
That separation is what makes a single standard-error machinery cover every
quantity of interest.

The notation follows Vande Kamp, Jordan and Rajan.  For an autoregressive
distributed lag model

.. math::

    \Delta^{d_y} y_t = \sum_{i} \alpha_i \Delta^{d_y} y_{t-i}
                     + \sum_{q} \beta_q \Delta^{d_x} x_{t-q} + \varepsilon_t

the period-specific impulse response (the "pulse") satisfies the recursion

.. math:: \xi_s = \beta_s + \sum_i \alpha_i \xi_{s-i},

and the General Dynamic Response Function of order :math:`h` is a weighted sum
of those pulses, the weights being generalized binomial coefficients in
:math:`m = h - d_x + d_y`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import sympy as sp
from scipy.special import comb

from ._util import by_lag, check_vrbl

__all__ = [
    "Formulae",
    "pulse_calculator",
    "general_calculator",
    "convolution_calculator",
    "gecm_to_adl",
    "shock_weights",
]


@dataclass
class Formulae:
    """Effect formulae for periods ``s = 0, 1, ..., limit``.

    Attributes
    ----------
    formulae : list of sympy.Expr
        One expression per period.
    shocks : list of list of float
        The weights applied to the pulses in that period -- the generalized
        binomial coefficients for an integer shock history, or the (possibly
        differenced) custom shock history otherwise.  Returning them is how the
        R package lets users check what was actually applied.
    """

    formulae: list
    shocks: list
    flat: bool = True

    def __len__(self) -> int:
        return len(self.formulae)

    def as_strings(self) -> list[str]:
        """The formulae as readable strings, one per period.

        See :attr:`flat` for why GECM formulae are left in nested form.
        """
        return [str(f) for f in self.formulae]

    def expanded(self) -> list:
        """The formulae multiplied out.

        Only worth asking for when :attr:`flat` is ``False``: for a GECM the
        expanded polynomial grows exponentially in the horizon, which is
        exactly why it is not expanded by default.
        """
        return [sp.expand(f) for f in self.formulae]


def _term(lookup: dict[int, str], order: int):
    """The coefficient at a given lag order, or zero when that lag is absent.

    The names may themselves be expressions rather than bare symbols -- that is
    how :func:`gecm_to_adl` passes a translated ADL coefficient such as
    ``l_1_x + l_1_d_x - d_x`` through to :func:`pulse_calculator`.
    """
    name = lookup.get(order)
    if name is None or name == "0":
        return sp.Integer(0)
    return sp.sympify(name)


def _all_atomic(lookup: dict[int, str]) -> bool:
    """Whether every coefficient in a lookup is a bare symbol."""
    return all(_term(lookup, k).is_Symbol for k in lookup)


def _maybe_expand(expr, flat: bool):
    return sp.expand(expr) if flat else expr


def _is_expanded(expr) -> bool:
    """Whether an expression is already multiplied out.

    Lets :func:`general_calculator` and :func:`convolution_calculator` inherit
    the choice :func:`pulse_calculator` made, without being told it.
    """
    for node in sp.preorder_traversal(sp.sympify(expr)):
        if node.is_Mul and any(arg.is_Add for arg in node.args):
            return False
        if node.is_Pow and node.base.is_Add:
            return False
    return True


def pulse_calculator(x_vrbl, y_vrbl=None, limit: int = 20) -> list:
    r"""Impulse Response Function formulae for a dynamic linear model.

    Builds :math:`\xi_s = \beta_s + \sum_i \alpha_i \xi_{s-i}` for
    :math:`s = 0, \ldots, \texttt{limit}`: the response of the dependent
    variable to a one-unit, one-period increase in the independent variable.

    Parameters
    ----------
    x_vrbl : dict
        ``{coefficient name: lag order}`` for the independent variable and its
        lags, e.g. ``{"x": 0, "l_1_x": 1}``.
    y_vrbl : dict, optional
        ``{coefficient name: lag order}`` for the lagged dependent variable.
        ``None`` means a finite-dynamics (distributed lag) model, which is
        allowed.
    limit : int
        Number of periods after the shock to compute, so the result has
        ``limit + 1`` entries.

    Returns
    -------
    list of sympy.Expr
        The pulse formula for each period, starting at ``s = 0``.

    Examples
    --------
    >>> from tseffects import pulse_calculator
    >>> pulses = pulse_calculator({"x": 0, "l_1_x": 1}, {"l_1_y": 1}, limit=3)
    >>> pulses[1]
    l_1_x + l_1_y*x

    A model with no lagged dependent variable is handled too; the response is
    then just the distributed lag weights.

    >>> pulse_calculator({"x": 0, "l_1_x": 1}, None, limit=2)
    [x, l_1_x, 0]
    """
    check_vrbl(x_vrbl, "x_vrbl")
    check_vrbl(y_vrbl, "y_vrbl", allow_none=True)
    limit = int(limit)
    if limit < 0:
        raise ValueError("limit must be non-negative")

    x_by_lag = by_lag(x_vrbl)
    y_by_lag = by_lag(y_vrbl)
    max_y = max(y_by_lag) if y_by_lag else 0

    # Whether to multiply the recursion out as we go.  When the autoregressive
    # coefficients are plain symbols the expanded form is small -- and is the
    # flat form mpoly prints, which makes the formulae readable.  When they are
    # compound expressions, as they are after a GECM translation, expanding
    # raises a sum to the power of the horizon and the term count explodes, so
    # the recursion is left nested.  Either form evaluates and differentiates
    # to the same thing; only the printing differs.
    flat = _all_atomic(y_by_lag)

    pulses: list = [sp.Integer(0)] * (limit + 1)
    for s in range(limit + 1):
        if s == 0:
            # No dynamics have had time to operate: the contemporaneous
            # coefficient, if the model has one.
            pulses[0] = _maybe_expand(_term(x_by_lag, 0), flat)
            continue
        carried = sp.Integer(0)
        for i in range(1, min(s, max_y) + 1):
            if i in y_by_lag:
                carried += _term(y_by_lag, i) * pulses[s - i]
        pulses[s] = _maybe_expand(_term(x_by_lag, s) + carried, flat)
    return pulses


def shock_weights(h: int, d_x: int, d_y: int, s: int) -> list[float]:
    r"""The generalized binomial weights for one period.

    With :math:`m = h - d_x + d_y`, the weight on the pulse :math:`\xi_{s-j}` is

    .. math::

        w_j = \bigl(2 \cdot 0^{|m| - m} - 1\bigr)^{j}
              \binom{|m| - 1 + (1 + j)\,0^{|m| - m}}{j},

    reading :math:`0^0 = 1`.  The familiar cases fall out of this:
    :math:`m = -1` gives the Impulse Response Function (weight 1 on the current
    pulse, 0 elsewhere), :math:`m = 0` the Step Response Function (all weights
    1, i.e. a running sum), :math:`m = 1` its cumulation, and :math:`m = -2` a
    first difference of the pulses.
    """
    m = int(h) - int(d_x) + int(d_y)
    # R's 0^(|m| - m): one when m >= 0, zero otherwise.
    gate = 1 if abs(m) - m == 0 else 0
    sign_base = 2 * gate - 1
    top = abs(m) - 1 + gate
    weights = []
    for j in range(s + 1):
        n = top + j * gate
        # exact=True keeps the binomials as Python ints, which matters once the
        # shock history order and the period both get large.
        weights.append(sign_base**j * int(comb(n, j, exact=True)))
    return weights


def general_calculator(d_x: int, d_y: int, h: int, limit: int, pulses) -> Formulae:
    r"""Generalized (im)pulse formulae for an integer shock history.

    Weights the period-specific impulse responses by the generalized binomial
    coefficients of :func:`shock_weights` and sums them, giving the General
    Dynamic Response Function :math:`\mathrm{GDRF}_{(s,\,h,\,d_y,\,d_x)}`.

    Parameters
    ----------
    d_x, d_y : int
        How many times the independent and dependent variables were differenced
        before estimation.
    h : int
        Order of the shock history.  ``-1`` is the Impulse Response Function and
        ``0`` the Step Response Function.
    limit : int
        Number of periods after the shock.
    pulses : list of sympy.Expr
        Impulse responses, normally from :func:`pulse_calculator`.

    Returns
    -------
    Formulae

    Examples
    --------
    >>> from tseffects import pulse_calculator, general_calculator
    >>> pulses = pulse_calculator({"x": 0, "l_1_x": 1}, {"l_1_y": 1}, limit=3)
    >>> step = general_calculator(d_x=0, d_y=0, h=0, limit=3, pulses=pulses)
    >>> step.shocks[2]
    [1, 1, 1]
    """
    limit = int(limit)
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if len(pulses) < limit + 1:
        raise ValueError(
            f"pulses has {len(pulses)} elements but limit = {limit} needs "
            f"{limit + 1}"
        )

    flat = _is_expanded(pulses[-1])
    formulae: list = []
    shocks: list = []
    for s in range(limit + 1):
        weights = shock_weights(h, d_x, d_y, s)
        total = sp.Integer(0)
        for j, w in enumerate(weights):
            if w == 0:
                continue
            total += sp.Integer(w) * pulses[s - j]
        formulae.append(_maybe_expand(total, flat))
        shocks.append(weights)
    return Formulae(formulae=formulae, shocks=shocks, flat=flat)


def convolution_calculator(d_x: int, d_y: int, shocks, pulses) -> Formulae:
    r"""Effect formulae for an arbitrary, user-supplied shock history.

    Any shock history can be decomposed into a sum of dated impulses, so the
    response in period :math:`s` is the convolution

    .. math:: \sum_{k=0}^{s} \tilde h_k \, \xi_{s-k},

    where :math:`\tilde h` is the supplied history differenced :math:`d_x` times
    (because the model's regressor is already in differences).  The result is
    then cumulated :math:`d_y` times to express it in the levels of the
    dependent variable.

    Parameters
    ----------
    d_x, d_y : int
        Orders of differencing of the independent and dependent variables.
    shocks : array_like
        The shock history applied to the independent variable *in levels*, of
        length ``len(pulses)``.
    pulses : list of sympy.Expr
        Impulse responses, normally from :func:`pulse_calculator`.

    Returns
    -------
    Formulae

    Examples
    --------
    An interrupted trend, rather than a one-off shock:

    >>> from tseffects import pulse_calculator, convolution_calculator
    >>> pulses = pulse_calculator({"x": 0, "l_1_x": 1}, {"l_1_y": 1}, limit=3)
    >>> out = convolution_calculator(0, 0, [1, 2, 3, 0], pulses)
    >>> out.shocks[-1]
    [1.0, 2.0, 3.0, 0.0]
    """
    d_x, d_y = int(d_x), int(d_y)
    if d_x < 0 or d_y < 0:
        raise ValueError("d_x and d_y must be non-negative")

    translated = np.asarray(shocks, dtype=float).ravel()
    if translated.size != len(pulses):
        raise ValueError(
            f"shocks has length {translated.size} but there are {len(pulses)} "
            "pulse formulae; a custom shock history must cover every period"
        )

    # Differencing the shock history keeps the first element and differences the
    # rest, so the cumulated history matches the levels one period by period.
    for _ in range(d_x):
        translated = np.concatenate(([translated[0]], np.diff(translated)))

    flat = _is_expanded(pulses[-1])
    response: list = []
    for s in range(len(pulses)):
        total = sp.Integer(0)
        for k in range(s + 1):
            if translated[k] == 0:
                continue
            # Keep the user's numbers as given rather than rationalizing them.
            weight = float(translated[k])
            total += (
                sp.Integer(int(weight)) if weight.is_integer() else sp.Float(weight)
            ) * pulses[s - k]
        response.append(_maybe_expand(total, flat))

    for _ in range(d_y):
        running = sp.Integer(0)
        cumulated = []
        for term in response:
            running = _maybe_expand(running + term, flat)
            cumulated.append(running)
        response = cumulated

    shocks_by_period = [
        [float(v) for v in translated[: s + 1]] for s in range(len(pulses))
    ]
    return Formulae(formulae=response, shocks=shocks_by_period, flat=flat)


def _fill_lags(vrbl, start: int) -> dict[int, str]:
    """Fill in the lag orders a specification skips, with ``"0"`` as the name.

    The GECM translation indexes coefficients by lag order and needs a slot for
    every order up to the maximum, even the ones the model omits.
    """
    present = by_lag(vrbl)
    if not present:
        return {}
    return {i: present.get(i, "0") for i in range(start, max(present) + 1)}


def gecm_to_adl(x_vrbl, y_vrbl, x_d_vrbl, y_d_vrbl=None) -> dict:
    r"""Translate Generalized Error-Correction Model coefficients to ADL ones.

    The GECM and the ADL are isomorphic reparameterizations of the same model,
    so this loses no information and lets one set of ADL formulae serve both.
    For the levels/differences GECM

    .. math::

        \Delta y_t = \theta_y y_{t-1} + \theta_x x_{t-1}
                   + \sum_i \alpha_i \Delta y_{t-i}
                   + \sum_q \beta_q \Delta x_{t-q} + \varepsilon_t

    the ADL coefficients are :math:`\beta^{ADL}_0 = \beta_0`,
    :math:`\beta^{ADL}_1 = \theta_x + \beta_1 - \beta_0`,
    :math:`\beta^{ADL}_q = \beta_q - \beta_{q-1}` in between, and
    :math:`\beta^{ADL}_{Q+1} = -\beta_Q` at the end; likewise
    :math:`\alpha^{ADL}_1 = \theta_y + \alpha_1 + 1` and
    :math:`\alpha^{ADL}_p = \alpha_p - \alpha_{p-1}`.  Note the ADL
    representation is one order higher than the GECM.

    Parameters
    ----------
    x_vrbl, y_vrbl : dict
        The first lag of the independent and dependent variables *in levels*;
        a GECM has exactly one of each.
    x_d_vrbl : dict
        The differenced independent variable and its lags, which may start at
        lag 0.
    y_d_vrbl : dict, optional
        The lagged differenced dependent variable.  ``None`` is allowed: the
        model then has no short-run dynamics in y.

    Returns
    -------
    dict
        ``{"x_vrbl_adl": {expression: lag}, "y_vrbl_adl": {expression: lag}}``,
        where each key is a string expression in the GECM's coefficient names,
        ready for :func:`pulse_calculator`.

    Examples
    --------
    >>> from tseffects import gecm_to_adl
    >>> out = gecm_to_adl({"l_1_x": 1}, {"l_1_y": 1},
    ...                   {"d_x": 0, "l_1_d_x": 1}, {"l_1_d_y": 1})
    >>> out["y_vrbl_adl"]
    {'l_1_y+l_1_d_y+1': 1, '(-1)*l_1_d_y': 2}
    """
    check_vrbl(x_vrbl, "x_vrbl")
    check_vrbl(y_vrbl, "y_vrbl")
    check_vrbl(x_d_vrbl, "x_d_vrbl")
    check_vrbl(y_d_vrbl, "y_d_vrbl", allow_none=True)

    x_levels = _fill_lags(x_vrbl, start=1)
    y_levels = _fill_lags(y_vrbl, start=1)
    x_diffs = _fill_lags(x_d_vrbl, start=0)
    y_diffs = _fill_lags(y_d_vrbl, start=1) if y_d_vrbl else {}

    max_xd = max(x_diffs)
    x_adl: dict[str, int] = {}
    for q in range(max_xd + 2):
        if q == 0:
            expr = x_diffs[0]
        elif q == 1 and max_xd == 0:
            # Only a contemporaneous difference: there is no beta_1 to add.
            expr = f"{x_levels[1]}-{x_diffs[0]}"
        elif q == 1:
            expr = f"{x_levels[1]}+{x_diffs[1]}-{x_diffs[0]}"
        elif q == max_xd + 1:
            expr = f"(-1)*{x_diffs[q - 1]}"
        else:
            expr = f"(-1)*{x_diffs[q - 1]}+{x_diffs[q]}"
        x_adl[expr] = q

    if not y_diffs:
        # No short-run y dynamics: the ADL has a single lag of y.
        y_adl = {f"{y_levels[1]}+1": 1}
    else:
        max_yd = max(y_diffs)
        y_adl = {}
        for p in range(1, max_yd + 2):
            if p == 1:
                expr = f"{y_levels[1]}+{y_diffs[1]}+1"
            elif p == max_yd + 1:
                expr = f"(-1)*{y_diffs[p - 1]}"
            else:
                expr = f"(-1)*{y_diffs[p - 1]}+{y_diffs[p]}"
            y_adl[expr] = p

    if len(x_adl) != max_xd + 2:  # pragma: no cover - duplicate expressions
        warnings.warn(
            "the GECM translation produced duplicate ADL expressions; check "
            "for repeated coefficient names",
            stacklevel=2,
        )
    return {"x_vrbl_adl": x_adl, "y_vrbl_adl": y_adl}
