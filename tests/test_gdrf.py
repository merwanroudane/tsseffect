"""The ADL and GECM entry points.

The numerical agreement with R is established by ``validate/compare.py``; what
is tested here are the invariants and identities that should hold whatever the
data are, plus the argument handling.
"""

import numpy as np
import pytest

from tseffects import adl_effects, gdrf_adl, gdrf_gecm, gecm_effects

from .conftest import ADL, GECM


class TestAdlShape:
    def test_pulse_has_one_row_per_period_and_no_lrm(self, fit_levels):
        out = gdrf_adl(fit_levels, shock_history="pulse", s_limit=10, **ADL)
        assert len(out.estimates) == 11
        assert out.lrm is None
        assert out.meta["lrm"] is False

    def test_step_appends_the_lrm(self, fit_levels):
        out = gdrf_adl(fit_levels, shock_history="step", s_limit=10, **ADL)
        assert len(out.estimates) == 12
        assert out.estimates["Period"].iloc[-1] == "LRM"
        assert "LRM" in out.formulae

    def test_fitted_prepends_a_baseline(self, fit_levels):
        out = gdrf_adl(
            fit_levels,
            shock_history="pulse",
            effect_type="fitted",
            prediction_values={"x": 1, "l_1_x": 1},
            s_limit=5,
            **ADL,
        )
        assert out.estimates["Period"].iloc[0] == "baseline"
        assert len(out.estimates) == 7
        # Fitted values are predictions, so no LRM is reported.
        assert out.lrm is None

    def test_columns(self, fit_levels):
        out = gdrf_adl(fit_levels, shock_history="pulse", s_limit=3, **ADL)
        assert out.estimates.columns.tolist() == [
            "Period",
            "GDRF",
            "SE",
            "Lower",
            "Upper",
        ]
        assert out.value_column == "GDRF"


class TestAdlIdentities:
    def test_s0_is_the_contemporaneous_coefficient(self, fit_levels):
        """At s = 0 nothing dynamic has happened yet."""
        out = gdrf_adl(fit_levels, shock_history="pulse", s_limit=5, **ADL)
        assert out.estimates["GDRF"].iloc[0] == pytest.approx(
            fit_levels.params["x"]
        )
        assert out.estimates["SE"].iloc[0] == pytest.approx(
            fit_levels.bse["x"]
        )

    def test_lrm_equals_sum_beta_over_one_minus_alpha(self, fit_levels):
        out = gdrf_adl(fit_levels, shock_history="step", s_limit=20, **ADL)
        p = fit_levels.params
        expected = (p["x"] + p["l_1_x"]) / (1 - p["l_1_y"])
        assert float(out.lrm["GDRF"]) == pytest.approx(expected)

    def test_step_is_the_cumulated_pulse(self, fit_levels):
        """The step response is the running sum of the impulse response."""
        pulse = gdrf_adl(fit_levels, shock_history="pulse", s_limit=10, **ADL)
        step = gdrf_adl(fit_levels, shock_history="step", s_limit=10, **ADL)
        assert np.allclose(
            np.cumsum(pulse.estimates["GDRF"]),
            step.estimates["GDRF"].iloc[:11],
        )

    def test_cumulative_pulse_equals_step(self, fit_levels):
        """`cumulative=True` on a pulse increments h, giving the step."""
        cum = gdrf_adl(
            fit_levels, shock_history="pulse", cumulative=True, s_limit=8, **ADL
        )
        step = gdrf_adl(fit_levels, shock_history="step", s_limit=8, **ADL)
        assert np.allclose(cum.estimates["GDRF"], step.estimates["GDRF"])
        assert np.allclose(cum.estimates["SE"], step.estimates["SE"])

    def test_step_response_converges_to_the_lrm(self, fit_levels):
        out = gdrf_adl(fit_levels, shock_history="step", s_limit=60, **ADL)
        path = out.estimates[out.estimates["Period"] != "LRM"]["GDRF"]
        assert float(path.iloc[-1]) == pytest.approx(
            float(out.lrm["GDRF"]), rel=1e-6
        )

    def test_shock_size_scales_linearly(self, fit_levels):
        one = gdrf_adl(fit_levels, shock_history="pulse", s_limit=5, **ADL)
        three = gdrf_adl(
            fit_levels, shock_history="pulse", shock_size=3, s_limit=5, **ADL
        )
        assert np.allclose(three.estimates["GDRF"], 3 * one.estimates["GDRF"])
        assert np.allclose(three.estimates["SE"], 3 * one.estimates["SE"])

    def test_custom_unit_impulse_matches_the_pulse(self, fit_levels):
        history = [1] + [0] * 10
        custom = gdrf_adl(fit_levels, shock_history=history, s_limit=10, **ADL)
        pulse = gdrf_adl(fit_levels, shock_history="pulse", s_limit=10, **ADL)
        assert np.allclose(custom.estimates["GDRF"], pulse.estimates["GDRF"])
        assert np.allclose(custom.estimates["SE"], pulse.estimates["SE"])

    def test_finite_dynamics_decays_to_exactly_zero(self, fit_fdl):
        out = gdrf_adl(
            fit_fdl,
            x_vrbl={"x": 0, "l_1_x": 1, "l_2_x": 2},
            y_vrbl=None,
            shock_history="pulse",
            s_limit=6,
        )
        assert np.allclose(out.estimates["GDRF"].iloc[3:], 0)
        assert np.allclose(out.estimates["SE"].iloc[3:], 0)


class TestInferences:
    def test_levels_and_differences_differ_for_a_differenced_x(self, fit_dx):
        args = dict(
            x_vrbl={"d_x": 0, "l_1_d_x": 1},
            y_vrbl={"l_1_y": 1},
            d_x=1,
            d_y=0,
            s_limit=6,
        )
        levels = gdrf_adl(fit_dx, shock_history="pulse",
                          inferences_x="levels", **args)
        diffs = gdrf_adl(fit_dx, shock_history="pulse",
                         inferences_x="differences", **args)
        assert not np.allclose(levels.estimates["GDRF"], diffs.estimates["GDRF"])

    def test_a_shock_in_differences_is_the_difference_of_the_levels_shock(
        self, fit_dx
    ):
        """inferences_x='differences' sets calc_d_x to 0, i.e. no unwinding."""
        args = dict(
            x_vrbl={"d_x": 0, "l_1_d_x": 1},
            y_vrbl={"l_1_y": 1},
            d_x=1,
            d_y=0,
            s_limit=6,
        )
        diffs = gdrf_adl(fit_dx, shock_history="pulse",
                         inferences_x="differences", **args)
        plain = gdrf_adl(fit_dx, x_vrbl={"d_x": 0, "l_1_d_x": 1},
                         y_vrbl={"l_1_y": 1}, d_x=0, d_y=0,
                         shock_history="pulse", s_limit=6)
        assert np.allclose(diffs.estimates["GDRF"], plain.estimates["GDRF"])

    def test_differences_inference_rejected_when_d_is_zero(self, fit_levels):
        with pytest.raises(ValueError, match="higher order of differencing"):
            gdrf_adl(fit_levels, d_x=0, d_y=0,
                     inferences_x="differences", **ADL)

    def test_invalid_inference_string(self, fit_levels):
        with pytest.raises(ValueError, match="Invalid inferences_y"):
            gdrf_adl(fit_levels, inferences_y="sideways", **ADL)


class TestFittedValues:
    def test_baseline_se_adds_in_quadrature(self, fit_dy):
        args = dict(
            x_vrbl={"x": 0, "l_1_x": 1},
            y_vrbl={"l_1_d_y": 1},
            d_x=0,
            d_y=1,
            effect_type="fitted",
            baseline_y=3,
            s_limit=5,
        )
        plain = gdrf_adl(fit_dy, shock_history="pulse", **args)
        padded = gdrf_adl(
            fit_dy, shock_history="pulse", baseline_y_se=2.0, **args
        )
        assert np.allclose(
            padded.estimates["SE"],
            np.sqrt(plain.estimates["SE"] ** 2 + 4.0),
        )

    def test_fitted_equals_baseline_plus_effect(self, fit_dy):
        """The fitted path is the discrete effect shifted by the baseline."""
        args = dict(
            x_vrbl={"x": 0, "l_1_x": 1},
            y_vrbl={"l_1_d_y": 1},
            d_x=0,
            d_y=1,
            s_limit=5,
        )
        effect = gdrf_adl(fit_dy, shock_history="pulse", **args)
        fitted = gdrf_adl(
            fit_dy,
            shock_history="pulse",
            effect_type="fitted",
            baseline_y=7.0,
            **args,
        )
        # Fitted values carry no LRM, so compare the period path only.
        effect_path = effect.estimates[effect.estimates["Period"] != "LRM"]
        path = fitted.estimates[fitted.estimates["Period"] != "baseline"]
        assert np.allclose(path["GDRF"].to_numpy(),
                           effect_path["GDRF"].to_numpy() + 7.0)
        # Shifting by a constant leaves the sampling variance unchanged.
        assert np.allclose(path["SE"].to_numpy(),
                           effect_path["SE"].to_numpy())

    def test_steady_state_baseline_matches_the_closed_form(self, fit_levels):
        out = gdrf_adl(
            fit_levels,
            shock_history="pulse",
            effect_type="fitted",
            prediction_values={"x": 2.0, "l_1_x": 2.0},
            s_limit=3,
            **ADL,
        )
        p = fit_levels.params
        expected = (p["Intercept"] + 2.0 * p["x"] + 2.0 * p["l_1_x"]) / (
            1 - p["l_1_y"]
        )
        assert out.estimates["GDRF"].iloc[0] == pytest.approx(expected)

    def test_prediction_values_rejected_for_discrete(self, fit_levels):
        with pytest.raises(ValueError, match="Do not supply prediction_values"):
            gdrf_adl(fit_levels, effect_type="discrete",
                     prediction_values={"x": 1}, **ADL)

    def test_fitted_needs_a_baseline(self, fit_levels):
        with pytest.raises(ValueError, match="baseline_y value or prediction_values"):
            gdrf_adl(fit_levels, effect_type="fitted", **ADL)

    def test_differenced_y_in_levels_needs_an_explicit_baseline(self, fit_dy):
        with pytest.raises(ValueError, match="no baseline for y"):
            gdrf_adl(
                fit_dy,
                x_vrbl={"x": 0, "l_1_x": 1},
                y_vrbl={"l_1_d_y": 1},
                d_x=0,
                d_y=1,
                effect_type="fitted",
                prediction_values={"x": 1, "l_1_x": 1},
                inferences_y="levels",
            )

    def test_higher_order_differencing_is_refused_clearly(self, fit_levels):
        with pytest.raises(NotImplementedError, match="d_y of 0 or 1"):
            gdrf_adl(fit_levels, d_x=0, d_y=2, effect_type="fitted",
                     baseline_y=0, **ADL)


class TestGecm:
    def test_lrm_is_minus_theta_x_over_theta_y(self, fit_gecm):
        out = gdrf_gecm(fit_gecm, shock_history="step", s_limit=10, **GECM)
        p = fit_gecm.params
        assert float(out.lrm["GDRF"]) == pytest.approx(
            -p["l_1_x"] / p["l_1_y"]
        )

    def test_s0_is_the_contemporaneous_difference_coefficient(self, fit_gecm):
        out = gdrf_gecm(fit_gecm, shock_history="pulse", s_limit=5, **GECM)
        assert out.estimates["GDRF"].iloc[0] == pytest.approx(
            fit_gecm.params["d_x"]
        )
        assert out.estimates["SE"].iloc[0] == pytest.approx(
            fit_gecm.bse["d_x"]
        )

    def test_matches_the_isomorphic_adl(self, data, fit_gecm):
        """The sharpest check there is on the translation.

        ``d_y ~ l_1_y + l_1_d_y + l_1_x + d_x + l_1_d_x`` and
        ``y ~ l_1_y + l_2_y + x + l_1_x + l_2_x`` are the same regression
        written two ways -- they span the same column space -- so every dynamic
        effect, and every standard error, must come out identical.

        The two fits have to be put on the same sample first: in this dataset
        the differenced columns carry leading missing values that the levels
        columns do not.
        """
        import statsmodels.formula.api as smf

        aligned = data.dropna(
            subset=["d_y", "l_1_d_y", "d_x", "l_1_d_x", "l_2_y", "l_2_x"]
        )
        fit_gecm_aligned = smf.ols(
            "d_y ~ l_1_y + l_1_d_y + l_1_x + d_x + l_1_d_x", data=aligned
        ).fit()
        fit_adl = smf.ols(
            "y ~ l_1_y + l_2_y + x + l_1_x + l_2_x", data=aligned
        ).fit()
        from_gecm = gdrf_gecm(
            fit_gecm_aligned, shock_history="step", s_limit=12, **GECM
        )
        from_adl = gdrf_adl(
            fit_adl,
            x_vrbl={"x": 0, "l_1_x": 1, "l_2_x": 2},
            y_vrbl={"l_1_y": 1, "l_2_y": 2},
            shock_history="step",
            s_limit=12,
        )
        assert np.allclose(from_gecm.estimates["GDRF"],
                           from_adl.estimates["GDRF"])
        assert np.allclose(from_gecm.estimates["SE"], from_adl.estimates["SE"])

    def test_step_converges_towards_the_lrm(self, fit_gecm):
        out = gdrf_gecm(fit_gecm, shock_history="step", s_limit=25, **GECM)
        path = out.estimates[out.estimates["Period"] != "LRM"]["GDRF"]
        lrm = float(out.lrm["GDRF"])
        # Monotone approach, and close by the end of the horizon.
        gaps = np.abs(path.to_numpy() - lrm)
        assert gaps[-1] < gaps[len(gaps) // 2] < gaps[0]
        assert gaps[-1] < 0.01 * abs(lrm)

    def test_formulae_are_reported_in_the_models_own_coefficients(self, fit_gecm):
        """The placeholder parameterization never reaches the user."""
        out = gdrf_gecm(fit_gecm, shock_history="pulse", s_limit=4, **GECM)
        for text in out.formulae.values():
            assert "_adl_" not in text
        s2 = out.formulae["s = 2"]
        assert "l_1_d_y" in s2 and "l_1_y" in s2 and "d_x" in s2

    def test_formulae_stay_nested_but_expand_on_request(self, fit_gecm):
        """GECM formulae are not multiplied out; see expanded_formulae()."""
        out = gdrf_gecm(fit_gecm, shock_history="pulse", s_limit=4, **GECM)
        # Nested: the translation is still visible as a bracketed sum.
        assert "(" in out.formulae["s = 2"]
        expanded = out.expanded_formulae()
        assert "**2" in expanded["s = 2"]
        assert len(expanded["s = 2"]) > len(out.formulae["s = 2"])

    def test_translation_is_exposed(self, fit_gecm):
        out = gdrf_gecm(fit_gecm, shock_history="pulse", s_limit=3, **GECM)
        adl = out.meta["adl_coefficients"]
        assert adl["y_vrbl_adl"] == {"l_1_y+l_1_d_y+1": 1, "(-1)*l_1_d_y": 2}

    def test_requires_the_first_lag_in_levels(self, fit_gecm):
        with pytest.raises(ValueError, match="only the first lag"):
            gdrf_gecm(
                fit_gecm,
                x_vrbl={"l_1_x": 2},
                y_vrbl={"l_1_y": 1},
                x_d_vrbl={"d_x": 0, "l_1_d_x": 1},
                y_d_vrbl={"l_1_d_y": 1},
            )

    def test_rejects_inferences_in_differences(self, fit_gecm):
        with pytest.raises(ValueError, match="automatically recovered in levels"):
            gdrf_gecm(fit_gecm, inferences_y="differences", **GECM)

    def test_rejects_inconsistent_differencing(self, fit_gecm):
        with pytest.raises(ValueError, match="one order of differencing"):
            gdrf_gecm(fit_gecm, x_d_vrbl_d_x=2, **GECM)


class TestWrappers:
    def test_adl_effects_matches_the_full_call(self, fit_levels):
        simple = adl_effects(fit_levels, shock_history="step", s_limit=7, **ADL)
        full = gdrf_adl(
            fit_levels, d_x=0, d_y=0, shock_history="step", s_limit=7, **ADL
        )
        assert np.allclose(simple.estimates["GDRF"], full.estimates["GDRF"])

    def test_gecm_effects_matches_the_full_call(self, fit_gecm):
        simple = gecm_effects(fit_gecm, shock_history="step", s_limit=7, **GECM)
        full = gdrf_gecm(fit_gecm, shock_history="step", s_limit=7, **GECM)
        assert np.allclose(simple.estimates["GDRF"], full.estimates["GDRF"])


class TestArgumentHandling:
    def test_unknown_coefficient_name(self, fit_levels):
        with pytest.raises(ValueError, match="x_vrbl not present"):
            gdrf_adl(fit_levels, x_vrbl={"nope": 0}, y_vrbl={"l_1_y": 1})

    def test_bad_shock_history_word(self, fit_levels):
        with pytest.raises(ValueError, match="Invalid shock_history"):
            gdrf_adl(fit_levels, shock_history="wiggle", **ADL)

    def test_custom_history_wrong_length(self, fit_levels):
        with pytest.raises(ValueError, match="length s_limit \\+ 1"):
            gdrf_adl(fit_levels, shock_history=[1, 2, 3], s_limit=10, **ADL)

    def test_bad_se_type(self, fit_levels):
        with pytest.raises(ValueError, match="Invalid se_type"):
            gdrf_adl(fit_levels, se_type="HC9", **ADL)

    def test_bad_effect_type(self, fit_levels):
        with pytest.raises(ValueError, match="Invalid effect_type"):
            gdrf_adl(fit_levels, effect_type="mysterious", **ADL)

    def test_level_narrows_the_interval(self, fit_levels):
        wide = gdrf_adl(fit_levels, shock_history="step", level=0.99,
                        s_limit=5, **ADL)
        narrow = gdrf_adl(fit_levels, shock_history="step", level=0.80,
                          s_limit=5, **ADL)
        assert (
            (wide.estimates["Upper"] - wide.estimates["Lower"]).to_numpy()
            > (narrow.estimates["Upper"] - narrow.estimates["Lower"]).to_numpy()
        ).all()

    def test_missing_lagged_dependent_variable_warns(self, fit_fdl):
        with pytest.warns(UserWarning, match="static or finite dynamics"):
            gdrf_adl(fit_fdl, x_vrbl={"x": 0, "l_1_x": 1}, y_vrbl=None,
                     s_limit=3)

    def test_decaying_effects_warn_rather_than_inventing_an_lrm(self, fit_levels):
        with pytest.warns(UserWarning, match="decay to 0"):
            out = gdrf_adl(fit_levels, shock_history="pulse", s_limit=5, **ADL)
        assert out.lrm is None

    def test_diverging_effects_warn(self, fit_levels):
        with pytest.warns(UserWarning, match="diverge"):
            gdrf_adl(fit_levels, shock_history=1, s_limit=5, **ADL)
