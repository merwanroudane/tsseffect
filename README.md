# tseffects (Python)

**Dynamic effects from single-equation time series models, with interactions.**

Autoregressive distributed lag (ADL) models and their reparameterized
equivalent, the Generalized Error-Correction Model (GECM), are the workhorse
dynamic linear models — and they are easy to estimate. What is hard is getting
a rich set of dynamic inferences back out of them. The coefficients alone do
not tell you how a shock to `x` moves `y` three periods later; nor what happens
when the shock is permanent rather than transitory; nor what happens when the
model was estimated in differences but you want the answer in levels; nor how
any of that changes when the model carries an interaction.

This package recovers those inferences, with correct standard errors.

It is a faithful Python port of the R package
[**tseffects**](https://github.com/sorenjordan/tseffects) by Soren Jordan,
Garrett N. Vande Kamp and Reshi Rajan. Every number it produces has been checked
against the R package: see [Validation](#validation).

---

## Install

```bash
pip install tseffects
```

From a checkout:

```bash
pip install -e .
```

Requires Python 3.9+, and `numpy`, `scipy`, `pandas`, `sympy`, `statsmodels`,
`matplotlib`.

---

## The idea in one paragraph

Everything is built from two quantities: the **Impulse Response Function** (a
one-period, one-unit increase in `x`) and the **Step Response Function** (a
permanent increase). These are two members of one family, the **General Dynamic
Response Function**, indexed by the order of the shock history `h` and the
orders of differencing `d_x` and `d_y`:

$$\mathrm{GDRF}_{(s,\,h,\,d_y,\,d_x)} = \sum_{j=0}^{s} w_j(h, d_x, d_y)\,\xi_{s-j}$$

where $\xi_s$ is the period-$s$ impulse response — itself the recursion
$\xi_s = \beta_s + \sum_i \alpha_i \xi_{s-i}$ — and the weights $w_j$ are
generalized binomial coefficients in $m = h - d_x + d_y$. Setting $m = -1$ gives
the impulse response, $m = 0$ the step response, $m = 1$ its cumulation, and
$m = -2$ a first difference. Arbitrary shock histories are handled too, since any
history decomposes into a sum of dated impulses.

The key design choice, inherited from the R package, is that **every quantity is
built as an algebraic formula in the model's coefficient names first**, and only
then evaluated by the delta method. That is what lets a single standard-error
machinery cover impulse responses, cumulated responses, long-run multipliers,
fitted trajectories and conditional effects alike — and it means you can always
read the formula behind any number the package reports.

---

## Quick start

```python
import statsmodels.formula.api as smf
from tseffects import load_toy_data, gdrf_adl

data = load_toy_data()
fit = smf.ols("y ~ l_1_y + x + l_1_x", data=data).fit()

out = gdrf_adl(
    fit,
    x_vrbl={"x": 0, "l_1_x": 1},   # x and its lags: name -> lag order
    y_vrbl={"l_1_y": 1},           # the lagged dependent variable
    shock_history="step",          # a permanent one-unit increase in x
    s_limit=20,
)

print(out.estimates)   # Period, GDRF, SE, Lower, Upper -- plus the LRM row
out.plot()             # matplotlib axes, themed like the R vignette
```

You describe the model by naming **which coefficient sits at which lag**. That
is the whole interface: `{"x": 0, "l_1_x": 1}` says the coefficient called `x` is
the contemporaneous one and `l_1_x` is at lag 1. Nothing is guessed from the
formula string, so any naming convention works.

Three things come back on the result object:

| | |
|---|---|
| `out.estimates` | a `DataFrame`: one row per period, with the estimate, standard error and confidence bounds |
| `out.formulae` | the algebra behind each row, e.g. `s = 2` → `l_1_x*l_1_y + l_1_y**2*x` |
| `out.shocks` | the weights applied to the impulse responses in each period |

`out.plot()` draws it. `out.lrm` is the Long-Run Multiplier row when one exists.

---

## What you can ask for

### Shock histories

```python
gdrf_adl(fit, ..., shock_history="pulse")            # impulse response, h = -1
gdrf_adl(fit, ..., shock_history="step")             # step response, h = 0
gdrf_adl(fit, ..., shock_history="pulse", cumulative=True)   # cumulative IRF
gdrf_adl(fit, ..., shock_history=1)                  # any integer order h
gdrf_adl(fit, ..., shock_history=[1, 2, 3, 0, 0, 4]) # an interrupted trend
```

A custom history must have `s_limit + 1` entries, one per period.

### Levels and differences

If the model was estimated in differences but the substantive question is about
levels, say so and the package unwinds it:

```python
fit = smf.ols("y ~ l_1_y + d_x + l_1_d_x", data=data).fit()

gdrf_adl(fit, x_vrbl={"d_x": 0, "l_1_d_x": 1}, y_vrbl={"l_1_y": 1},
         d_x=1, d_y=0,              # x was differenced once before estimation
         inferences_x="levels")     # but shock it in levels
```

`inferences_x` / `inferences_y` choose whether the shock is *applied*, and the
response *read*, in levels or in differences.

### Fitted values instead of effects

An effect is a change from wherever `y` was. A fitted value is the trajectory
itself, which needs a baseline:

```python
# y in levels: the baseline can be the model-implied steady state, and the
# delta method carries the uncertainty in that steady state through.
gdrf_adl(fit, ..., effect_type="fitted",
         prediction_values={"x": 1, "l_1_x": 1})

# y in differences: the model knows nothing about the level of y, so assert one
gdrf_adl(fit, ..., effect_type="fitted", baseline_y=3, baseline_y_se=2.5)
```

### GECM

The GECM and the ADL are the same model written two ways, so the coefficients
are translated and the ADL machinery does the rest. No information is lost, and
the standard errors come from the GECM's own covariance matrix:

```python
from tseffects import gdrf_gecm

fit = smf.ols("d_y ~ l_1_y + l_1_d_y + l_1_x + d_x + l_1_d_x", data=data).fit()

gdrf_gecm(fit,
          x_vrbl={"l_1_x": 1},                   # the error-correction terms
          y_vrbl={"l_1_y": 1},
          x_d_vrbl={"d_x": 0, "l_1_d_x": 1},     # the short-run dynamics
          y_d_vrbl={"l_1_d_y": 1},
          shock_history="step")
```

In a GECM the long-run multiplier is read straight off the two levels terms as
$-\theta_x / \theta_y$.

### Interactions

When a dynamic model carries `x × z`, the effect of a shock to `x` depends on
both how long ago it landed *and* the value of `z` — a surface, not a curve:

```python
from tseffects import interact_adl

fit = smf.ols("y ~ l_1_y + x + l_1_x + z + l_1_z"
              " + x_z + z_l_1_x + x_l_1_z + l_1_x_l_1_z", data=data).fit()

out = interact_adl(
    fit,
    x_vrbl={"x": 0, "l_1_x": 1},
    z_vrbl={"z": 0, "l_1_z": 1},
    x_z_vrbl={"x_z": 0, "z_l_1_x": 1, "x_l_1_z": 0, "l_1_x_l_1_z": 1},
    y_vrbl={"l_1_y": 1},
    z_vals=[-1, 0, 1],
    plot_type="lines", line_options="z.lines",
)
out.plot()
```

For `x_z_vrbl` the lag order is the one on the **x** side of the product:
`x_l_1_z` (contemporaneous x times lagged z) is `0`, and `z_l_1_x` (lagged x
times contemporaneous z) is `1`.

Three pictures are available:

- `plot_type="lines", line_options="z.lines"` — period on the x axis, one line
  per value of `z`
- `plot_type="lines", line_options="s.lines"` — `z` on the x axis, one line per
  period; this is the picture that shows *where in the range of z* the effect is
  distinguishable from zero
- `plot_type="heatmap"` — the whole surface, optionally with insignificant cells
  blanked out (`heatmap_options="significant"`)

### Robust standard errors

`se_type` accepts `"const"` and the full `sandwich::vcovHC` family:
`"HC0"`, `"HC1"`, `"HC2"`, `"HC3"`, `"HC4"`, `"HC4m"`, `"HC5"`. (`statsmodels`
ships only HC0–HC3, so the rest are implemented here and checked against R.)

```python
gdrf_adl(fit, ..., se_type="HC3", level=0.90)
```

### When there is no long-run multiplier

A long-run multiplier exists only when $h - d_x + d_y = 0$. Outside that case
the effects either decay to zero or diverge, and the package says which rather
than reporting a meaningless number:

```
UserWarning: return_lrm = True but the effects decay to 0 as s -> infinity.
No LRM is returned
```

That is a feature, not a nuisance: it is the condition under which a "long-run
effect" is a well-defined object at all.

---

## Using the symbolic layer directly

The formula builders are public, and useful on their own — for deriving a
quantity by hand, or for checking what the package is actually computing:

```python
from tseffects import pulse_calculator, general_calculator, gecm_to_adl

pulses = pulse_calculator({"x": 0, "l_1_x": 1}, {"l_1_y": 1}, limit=5)
pulses[2]
# l_1_x*l_1_y + l_1_y**2*x

step = general_calculator(d_x=0, d_y=0, h=0, limit=5, pulses=pulses)
step.shocks[3]
# [1, 1, 1, 1]        -- a running sum, i.e. the step response

gecm_to_adl({"l_1_x": 1}, {"l_1_y": 1}, {"d_x": 0, "l_1_d_x": 1}, {"l_1_d_y": 1})
# {'x_vrbl_adl': {'d_x': 0, 'l_1_x+l_1_d_x-d_x': 1, '(-1)*l_1_d_x': 2},
#  'y_vrbl_adl': {'l_1_y+l_1_d_y+1': 1, '(-1)*l_1_d_y': 2}}
```

These return SymPy expressions, so you can differentiate, substitute or print
them as LaTeX. `delta_method` evaluates any expression you build yourself
against a fitted model.

---

## Working from coefficients alone

If the model was fitted somewhere else — Stata, EViews, a bootstrap, a
constrained estimator — pass the coefficients and covariance matrix directly:

```python
out = gdrf_adl((coef_dict, vcov_matrix), x_vrbl=..., y_vrbl=...)
```

`se_type` is then ignored, since there is no design matrix to recompute the
covariance from. Steady-state `prediction_values` need the model frame, so with
coefficients alone supply every value explicitly or use `baseline_y`.

---

## Validation

`validate/r_reference.R` runs 29 specifications through the R package and writes
its output; `validate/compare.py` reproduces each one in Python and compares
every number.

```bash
Rscript validate/r_reference.R
python validate/compare.py
```

As of the last run, **193/193 checks pass at a relative tolerance of 1e-9**,
covering:

- all eight covariance types, on seven different models (so `HC4m` and `HC5`,
  reimplemented here, match `sandwich` exactly)
- impulse, step, integer-`h` and cumulative shock histories
- custom shock histories, cumulated and not
- `x` in differences read in levels and in differences; likewise `y`
- fitted values from a steady state and from an asserted baseline with its own
  standard error
- GECM with one and with two lags of the differenced terms
- finite distributed lag models with no lagged dependent variable
- interactions: discrete, step, cumulative and fitted
- non-default confidence levels
- the generalized binomial weights for seven `(d_x, d_y, h)` combinations
- the GECM-to-ADL coefficient translation, string for string

---

## Performance, and why the formulae look the way they do

The symbolic layer is what makes the standard errors trustworthy, but it costs
something. Roughly, on the bundled toy data:

| | `s_limit=20` | `s_limit=60` | `s_limit=100` |
|---|---|---|---|
| `gdrf_adl`, step response | 0.8 s | 4.8 s | 12.0 s |
| `gdrf_gecm`, step response | 7.7 s | — | — |

Two implementation choices keep this tractable:

- **Every period is compiled together.** SymPy's `lambdify` generates and
  executes Python source, so compiling one function per period dominated the
  cost. The values and the whole Jacobian now go through two `lambdify` calls
  instead of two per period.
- **GECM formulae are carried in a placeholder parameterization.** Each
  translated ADL coefficient is a *sum* of GECM coefficients, so writing those
  sums into the response recursion raises a sum to the power of the horizon —
  the term count explodes. Instead a bare symbol stands in for each ADL
  coefficient, and the delta method applies the chain rule through the
  translation. Since the translation is linear, that Jacobian is a constant
  matrix and the answer is exact. Without this, a GECM at the default
  `s_limit=20` takes minutes.

A consequence: `out.formulae` for a GECM is reported in nested form, such as
`_adl_b0` substituted back to `d_x`, with the translation visible rather than
multiplied out. `out.expanded_formulae()` multiplies them out if you want the
flat algebra — at short horizons only.

Beyond roughly `s_limit=100` for an ADL, or `s_limit=30` for a GECM, you are
past the point where another period tells you anything: the response has long
since converged to the long-run multiplier, which is reported exactly and for
free.

## Differences from the R package

The mathematics is identical. The interface is not, where Python conventions
differ:

| R | Python |
|---|---|
| `GDRF.adl.plot` / `adl.plot` | `gdrf_adl` / `adl_effects` |
| `GDRF.gecm.plot` / `gecm.plot` | `gdrf_gecm` / `gecm_effects` |
| `interact.adl.plot` | `interact_adl` |
| `x.vrbl = c("x" = 0, "l_1_x" = 1)` | `x_vrbl={"x": 0, "l_1_x": 1}` |
| `dM.level` | `level` |
| `return.data` / `return.plot` / `return.formulae` | always returned on the result object; `.plot()` draws |
| `mpoly` for the algebra | `sympy` |
| `car::deltaMethod` | `tseffects.delta_method` |
| `sandwich::vcovHC` | `tseffects.vcov_hc` |
| `ggplot2` | `matplotlib` |

Three substantive notes:

- **Formula strings are printed in SymPy's ordering**, not `mpoly`'s. The
  expressions are the same; the term order differs. `out.formulae_expr` gives
  the SymPy objects if you want to compare them symbolically.
- **GECM formulae are not multiplied out**, for the reason given under
  [Performance](#performance-and-why-the-formulae-look-the-way-they-do).
  `out.expanded_formulae()` does it on request.
- **Fitted values are defined for `d_y` of 0 or 1**, as in R. Higher orders of
  differencing raise `NotImplementedError` rather than failing obscurely; use
  `effect_type="discrete"`.

---

## Method references

- Warner, Zach, Garrett N. Vande Kamp and Soren Jordan. 2026. "Conditional
  Relationships in Dynamic Models." *Political Science Research and Methods*.
  [doi:10.1017/psrm.2026.10087](https://doi.org/10.1017/psrm.2026.10087)
- Vande Kamp, Garrett N. and Soren Jordan. 2025. "The Necessity of Moving
  Averages in Dynamic Linear Regression Models." *American Journal of Political
  Science* 69(1): 176–193.
- De Boef, Suzanna and Luke Keele. 2008. "Taking Time Seriously." *American
  Journal of Political Science* 52(1): 184–200.
  [doi:10.1111/j.1540-5907.2007.00307.x](https://doi.org/10.1111/j.1540-5907.2007.00307.x)
- Philips, Andrew Q. 2018. "Have Your Cake and Eat It Too? Cointegration and
  Dynamic Inference from Autoregressive Distributed Lag Models." *American
  Journal of Political Science* 62(1): 230–244.
  [doi:10.1111/ajps.12318](https://doi.org/10.1111/ajps.12318)
- Pesaran, M. Hashem, Yongcheol Shin and Richard J. Smith. 2001. "Bounds Testing
  Approaches to the Analysis of Level Relationships." *Journal of Applied
  Econometrics* 16(3): 289–326.
  [doi:10.1002/jae.616](https://doi.org/10.1002/jae.616)
- Grant, Taylor and Matthew J. Lebo. 2016. "Error Correction Methods with
  Political Time Series." *Political Analysis* 24(1): 3–30.
  [doi:10.1093/pan/mpv027](https://doi.org/10.1093/pan/mpv027)

## Data

- `load_toy_data()` — 50 simulated periods with every lag, difference and
  interaction pre-computed. Convenient because the coefficient names *are* the
  column names.
- `load_approval()` — US presidential approval, 1981–2016, from Cavari, Amnon
  (2019), *Presidential Studies Quarterly* 49(4): 798–826,
  [doi:10.1111/psq.12594](https://doi.org/10.1111/psq.12594).

Both are the R package's datasets, unchanged.

## Credit and licence

The method and the original implementation are the work of **Soren Jordan,
Garrett N. Vande Kamp and Reshi Rajan** ([R package
`tseffects`](https://cran.r-project.org/package=tseffects), CRAN). This is a
port of that package to Python; please cite their papers for the method and the
R package alongside this one.

Released under the GPL (≥ 2), the licence of the original package.
