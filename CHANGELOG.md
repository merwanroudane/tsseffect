# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] — 2026-10-28

### Added

- `EffectResult.summary()` — an aligned effect table with a model and sample
  block, significance stars at 1/5/10 per cent, a dedicated row for the long-run
  multiplier, and a footer naming the variance estimator. The returned object
  also renders as `as_latex()` (booktabs, stars, a notes block), `as_markdown()`
  and `as_html()`, all built from one frame so they cannot disagree.
- `EffectResult.to_frame()` — the estimates with z statistics, two-sided normal
  p-values and significance markers.
- The colour families: `parula_colors`, `parula_cmap`, `matlab_jet_colors`,
  `turbo_colors`, `bluered_colors`, `sinha_colors`, `resolve_colorscale`,
  `sequential_cmap` and `diverging_cmap`. Heatmaps accept any of them by name.
- `examples/approval_empirical.py` — the complete worked example on the real
  Cavari (2019) approval data, which generates every figure in the README and on
  the documentation site.
- `CITATION.cff`, this changelog, and a GitHub Actions workflow running the test
  suite on Python 3.9 through 3.13.
- `__author__` and `__email__` on the package.

### Fixed

- The `"Baseline"` and `"0"` tick labels overlapped on fitted-value plots. The
  baseline now sits further from the shock, scaled to the horizon, and the
  redundant tick at zero is dropped.

## [0.4.2] — 2026-10-28

### Fixed

- The source distribution shipped `tests/test_*.py` but not `tests/conftest.py`,
  which holds every fixture, so the tests in it could not be run. A `MANIFEST.in`
  now includes the whole test directory, the R reference script and its Python
  comparison, and the example gallery — and prunes the rendered figures, which
  belong in the repository rather than in every download.

## [0.4.1] — 2026-10-28

First release: a port of the R package
[tseffects 0.4.1](https://cran.r-project.org/package=tseffects) by Soren Jordan,
Garrett N. Vande Kamp and Reshi Rajan.

### Added

- `gdrf_adl` and `adl_effects` — dynamic effects from an autoregressive
  distributed lag model.
- `gdrf_gecm` and `gecm_effects` — the same from a Generalized Error-Correction
  Model, via the exact ADL translation.
- `interact_adl` — conditional effects from a multiplicative interaction, as
  lines over periods, lines over the moderator, or a surface.
- The symbolic layer as public API: `pulse_calculator`, `general_calculator`,
  `convolution_calculator`, `gecm_to_adl`, `yhat_calculator`, `shock_weights`
  and `combine_interaction_terms`.
- `delta_method`, `DeltaFn` and `BatchDelta` for the inference, and `vcov_hc` for
  the full `sandwich::vcovHC` family — `const` and `HC0` through `HC5`, where
  `statsmodels` ships only `HC0`–`HC3`.
- Plots in matplotlib: response functions with confidence bands and a separated
  long-run multiplier, families of conditional effect lines, and effect surfaces
  that can blank out the cells whose interval covers zero.
- The two datasets from the R package: `load_toy_data()` and `load_approval()`.

### Validated

- 193 of 193 numerical checks against the R package at a relative tolerance of
  1e-9, across 29 specifications.
- The GECM path reproduces the isomorphic ADL fit to 1.3e-15, independently of R.

[0.5.0]: https://github.com/merwanroudane/tsseffect/releases/tag/v0.5.0
[0.4.2]: https://github.com/merwanroudane/tsseffect/releases/tag/v0.4.2
[0.4.1]: https://github.com/merwanroudane/tsseffect/releases/tag/v0.4.1
