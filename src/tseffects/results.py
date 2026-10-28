"""The object the effect functions return.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ._tables import Summary, build_frame


@dataclass
class EffectResult:
    """Estimated dynamic effects, their formulae, and a plot on request.

    The R package decides what to hand back from four ``return.*`` flags; this
    carries everything and lets you take what you need.

    Attributes
    ----------
    estimates : pandas.DataFrame
        One row per period (and per value of the moderator, for an interaction
        model) with the estimate, its standard error and confidence bounds.
        ``Period`` holds the integer period, or ``"baseline"`` / ``"LRM"``.
    formulae : dict of str
        The algebraic formula behind each row, keyed ``"s = 0"``, ``"s = 1"``,
        ... and ``"LRM"``.  These are what makes the delta-method standard
        errors checkable by hand.
    formulae_expr : dict of sympy.Expr
        The same formulae as SymPy objects, for further symbolic work.
    shocks : dict
        The weights applied to the impulse responses in each period -- the
        generalized binomial coefficients, or the (differenced) custom history.
    meta : dict
        Labelling and bookkeeping used by :meth:`plot`: ``effect_type``,
        ``s_limit``, ``h_order``, ``cumulative``, ``level``, ``lrm`` and the
        plot-side orders of differencing.
    """

    estimates: pd.DataFrame
    formulae: dict = field(default_factory=dict)
    formulae_expr: dict = field(default_factory=dict)
    shocks: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def value_column(self) -> str:
        """``"GDRF"`` for a response function, ``"Effect"`` for an interaction."""
        return "Effect" if "Effect" in self.estimates.columns else "GDRF"

    def to_frame(self) -> pd.DataFrame:
        """The estimates with z statistics, p-values and significance markers.

        The p-values are two-sided normal, to match the delta-method intervals.

        Examples
        --------
        >>> import statsmodels.formula.api as smf
        >>> from tseffects import load_toy_data, gdrf_adl
        >>> fit = smf.ols("y ~ l_1_y + x + l_1_x", data=load_toy_data()).fit()
        >>> out = gdrf_adl(fit, x_vrbl={"x": 0, "l_1_x": 1},
        ...                y_vrbl={"l_1_y": 1}, shock_history="step", s_limit=3)
        >>> out.to_frame().columns.tolist()
        ['Period', 'GDRF', 'SE', 'Lower', 'Upper', 'z', 'p', 'stars']
        """
        return build_frame(self.estimates, self.value_column)

    def summary(self) -> Summary:
        """An aligned table of the effects, ready for a manuscript.

        The returned object prints as text, and carries ``as_latex()``,
        ``as_markdown()`` and ``as_html()`` renderings of the same numbers.

        Examples
        --------
        >>> import statsmodels.formula.api as smf
        >>> from tseffects import load_toy_data, gdrf_adl
        >>> fit = smf.ols("y ~ l_1_y + x + l_1_x", data=load_toy_data()).fit()
        >>> out = gdrf_adl(fit, x_vrbl={"x": 0, "l_1_x": 1},
        ...                y_vrbl={"l_1_y": 1}, shock_history="step", s_limit=3)
        >>> out.summary().as_markdown().splitlines()[0].startswith('Period')
        True
        """
        return Summary(
            self.to_frame(),
            self.value_column,
            self.meta.get("table", {}),
            self.meta.get("table", {}).get("title", "Dynamic effects"),
        )

    def expanded_formulae(self) -> dict:
        """The formulae multiplied out into a flat sum of terms.

        ADL formulae are already flat, so this changes nothing there.  GECM
        formulae are kept nested, because each ADL coefficient is a sum of GECM
        coefficients and multiplying them out raises a sum to the power of the
        horizon -- the term count grows exponentially in ``s_limit``.  Ask for
        this only at short horizons, and only when the flat algebra is what you
        want to read.
        """
        import sympy as sp

        return {k: str(sp.expand(v)) for k, v in self.formulae_expr.items()}

    @property
    def lrm(self) -> pd.Series | None:
        """The Long-Run Multiplier row, if one was computed."""
        rows = self.estimates[self.estimates["Period"] == "LRM"]
        if rows.empty:
            return None
        return rows.iloc[0] if len(rows) == 1 else rows

    def plot(self, ax=None, **kwargs):
        """Draw the effects.  See :func:`tseffects.plots.plot_effects`."""
        from .plots import plot_effects

        return plot_effects(self, ax=ax, **kwargs)

    def __repr__(self) -> str:
        kind = self.meta.get("effect_type", "discrete")
        n = len(self.estimates)
        head = (
            f"<EffectResult: {n} rows, effect_type={kind!r}, "
            f"s_limit={self.meta.get('s_limit')}>"
        )
        return head + "\n" + self.estimates.to_string(index=False, max_rows=12)

    def _repr_html_(self) -> str:  # pragma: no cover - notebook convenience
        return self.estimates.to_html(index=False, max_rows=12)
