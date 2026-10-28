r"""Dynamic effects from single-equation time series models, with interactions.

Autoregressive distributed lag (ADL) models and their reparameterized
equivalent, the Generalized Error-Correction Model (GECM), are easy to
estimate.  What is harder is getting a rich set of dynamic inferences back out
of them: the coefficients alone do not tell you how a shock to x moves y three
periods later, nor what happens when the shock is permanent rather than
transitory, nor what happens when the model was estimated in differences but
you want the answer in levels.

This package recovers those inferences.  Everything is built from the Impulse
Response Function and the Step Response Function, generalized into one family
-- the General Dynamic Response Function -- indexed by the order of the shock
history :math:`h` and the orders of differencing :math:`d_x` and :math:`d_y`:

.. math::

    \mathrm{GDRF}_{(s,\,h,\,d_y,\,d_x)}
      = \sum_{j=0}^{s} w_j(h, d_x, d_y)\, \xi_{s-j},

where :math:`\xi_s` is the period-:math:`s` impulse response and the weights
are generalized binomial coefficients.  Arbitrary shock histories are handled
too, since any history decomposes into a sum of dated impulses.

Every quantity is produced as an algebraic *formula* in the model's coefficient
names first, and only then evaluated by the delta method.  That is what lets one
standard-error machinery cover impulse responses, cumulated responses, long-run
multipliers, fitted trajectories and conditional effects alike -- and it means
you can read the formula behind any number the package reports.

Getting started
---------------
>>> import statsmodels.formula.api as smf
>>> from tseffects import load_toy_data, gdrf_adl
>>> data = load_toy_data()
>>> fit = smf.ols("y ~ l_1_y + x + l_1_x", data=data).fit()
>>> out = gdrf_adl(fit,
...                x_vrbl={"x": 0, "l_1_x": 1},
...                y_vrbl={"l_1_y": 1},
...                shock_history="pulse", s_limit=10)
>>> out.estimates.head(3).round(4)
  Period    GDRF      SE   Lower   Upper
0      0  0.6384  0.2608  0.1273  1.1496
1      1  0.3918  0.3184 -0.2322  1.0158
2      2  0.2987  0.2475 -0.1864  0.7839

``out.plot()`` draws it, and ``out.formulae`` gives the algebra behind each row.

This is a Python port of the R package **tseffects** by Soren Jordan, Garrett N.
Vande Kamp and Reshi Rajan.  The method is documented in Warner, Vande Kamp and
Jordan, "Conditional Relationships in Dynamic Models", *Political Science
Research and Methods* (2026),
`doi:10.1017/psrm.2026.10087 <https://doi.org/10.1017/psrm.2026.10087>`_, and in
Vande Kamp and Jordan, "The Necessity of Moving Averages in Dynamic Linear
Regression Models", *American Journal of Political Science* 69(1): 176-193
(2025).
"""

from ._delta import BatchDelta, DeltaFn, DeltaResult, delta_method
from ._vcov import VCOV_TYPES, vcov_hc
from .baselines import yhat_calculator
from .calculators import (
    Formulae,
    convolution_calculator,
    gecm_to_adl,
    general_calculator,
    pulse_calculator,
    shock_weights,
)
from .datasets import load_approval, load_toy_data
from .gdrf import adl_effects, gdrf_adl, gdrf_gecm, gecm_effects
from .interact import combine_interaction_terms, interact_adl
from .plots import (
    OKABE_ITO,
    apply_theme,
    plot_effects,
    plot_gdrf,
    plot_interact_heatmap,
    plot_interact_lines,
)
from .results import EffectResult

__version__ = "0.4.1"

__all__ = [
    # Main entry points
    "gdrf_adl",
    "adl_effects",
    "gdrf_gecm",
    "gecm_effects",
    "interact_adl",
    # Symbolic building blocks
    "pulse_calculator",
    "general_calculator",
    "convolution_calculator",
    "gecm_to_adl",
    "yhat_calculator",
    "combine_interaction_terms",
    "shock_weights",
    "Formulae",
    # Inference
    "delta_method",
    "DeltaFn",
    "BatchDelta",
    "DeltaResult",
    "vcov_hc",
    "VCOV_TYPES",
    # Results and plots
    "EffectResult",
    "plot_effects",
    "plot_gdrf",
    "plot_interact_lines",
    "plot_interact_heatmap",
    "apply_theme",
    "OKABE_ITO",
    # Data
    "load_toy_data",
    "load_approval",
    "__version__",
]
