"""Dynamic conditional effects."""

import numpy as np
import pytest

from tseffects import combine_interaction_terms, gdrf_adl, interact_adl

from .conftest import INTERACT


class TestCombineTerms:
    def test_folds_interactions_into_the_marginal_effect(self):
        out = combine_interaction_terms(
            {"x": 0, "l_1_x": 1},
            {"x_z": 0, "x_l_1_z": 0, "z_l_1_x": 1, "l_1_x_l_1_z": 1},
        )
        assert out == {
            "x+x_z*z_val+x_l_1_z*z_val": 0,
            "l_1_x+z_l_1_x*z_val+l_1_x_l_1_z*z_val": 1,
        }

    def test_a_lag_with_no_interaction_term(self):
        out = combine_interaction_terms({"x": 0, "l_1_x": 1}, {"x_z": 0})
        assert out["l_1_x+0"] == 1

    def test_a_lag_with_no_plain_x_term(self):
        out = combine_interaction_terms({"x": 0}, {"x_z": 0, "z_l_1_x": 1})
        assert out["0+z_l_1_x*z_val"] == 1


class TestShape:
    def test_one_row_per_period_and_z(self, fit_interact):
        out = interact_adl(fit_interact, z_vals=[-1, 0, 1], s_limit=10,
                           **INTERACT)
        assert len(out.estimates) == 11 * 3
        assert out.estimates.columns.tolist() == [
            "Period", "Z", "Effect", "SE", "Lower", "Upper",
        ]
        assert out.value_column == "Effect"

    def test_step_adds_an_lrm_per_z(self, fit_interact):
        out = interact_adl(fit_interact, shock_history="step",
                           z_vals=[-1, 0, 1], s_limit=10, **INTERACT)
        lrm = out.estimates[out.estimates["Period"] == "LRM"]
        assert len(lrm) == 3
        assert sorted(float(z) for z in lrm["Z"]) == [-1.0, 0.0, 1.0]

    def test_fitted_adds_a_baseline_per_z(self, fit_interact):
        out = interact_adl(
            fit_interact,
            effect_type="fitted",
            prediction_values={"x": 1, "l_1_x": 1, "x_z": 1, "z_l_1_x": 1,
                               "x_l_1_z": 1, "l_1_x_l_1_z": 1},
            z_vals=[-1, 0, 1],
            s_limit=5,
            **INTERACT,
        )
        baseline = out.estimates[out.estimates["Period"] == "baseline"]
        assert len(baseline) == 3

    def test_discrete_z_values_are_found_automatically(self, data, fit_interact):
        """A moderator with few distinct values is used as-is."""
        out = interact_adl(fit_interact, s_limit=3, **INTERACT)
        assert len(out.meta["z_vals"]) == 3  # z is continuous -> mean +/- 1 sd

    def test_a_range_is_swept_for_heatmaps(self, fit_interact):
        out = interact_adl(fit_interact, plot_type="heatmap", s_limit=3,
                           **INTERACT)
        assert len(out.meta["z_vals"]) == 50


class TestIdentities:
    def test_at_z_zero_the_effect_is_the_plain_adl_effect(self, fit_interact):
        """With z = 0 the interaction terms drop out of the marginal effect."""
        conditional = interact_adl(fit_interact, z_vals=[0.0], s_limit=8,
                                   **INTERACT)
        plain = gdrf_adl(
            fit_interact,
            x_vrbl={"x": 0, "l_1_x": 1},
            y_vrbl={"l_1_y": 1},
            shock_history="impulse",
            s_limit=8,
        )
        assert np.allclose(
            conditional.estimates["Effect"].to_numpy(),
            plain.estimates["GDRF"].to_numpy(),
        )
        assert np.allclose(
            conditional.estimates["SE"].to_numpy(),
            plain.estimates["SE"].to_numpy(),
        )

    def test_s0_is_the_marginal_effect_of_x_at_z(self, fit_interact):
        out = interact_adl(fit_interact, z_vals=[2.0], s_limit=3, **INTERACT)
        p = fit_interact.params
        expected = p["x"] + 2.0 * p["x_z"] + 2.0 * p["x_l_1_z"]
        assert out.estimates["Effect"].iloc[0] == pytest.approx(expected)

    def test_the_effect_is_linear_in_z(self, fit_interact):
        """A multiplicative interaction makes the effect linear in z."""
        out = interact_adl(fit_interact, z_vals=[-2, 0, 2], s_limit=4,
                           **INTERACT)
        by_z = {
            float(z): out.estimates[np.isclose(out.estimates["Z"], z)][
                "Effect"
            ].to_numpy()
            for z in (-2, 0, 2)
        }
        assert np.allclose(by_z[0.0], (by_z[-2.0] + by_z[2.0]) / 2)

    def test_cumulative_impulse_equals_step(self, fit_interact):
        cum = interact_adl(fit_interact, shock_history="impulse",
                           cumulative=True, z_vals=[-1, 1], s_limit=6,
                           **INTERACT)
        step = interact_adl(fit_interact, shock_history="step",
                            z_vals=[-1, 1], s_limit=6, **INTERACT)
        assert np.allclose(cum.estimates["Effect"], step.estimates["Effect"])
        assert np.allclose(cum.estimates["SE"], step.estimates["SE"])

    def test_lrm_at_z(self, fit_interact):
        out = interact_adl(fit_interact, shock_history="step", z_vals=[1.5],
                           s_limit=10, **INTERACT)
        p = fit_interact.params
        numerator = (
            p["x"] + 1.5 * p["x_z"] + 1.5 * p["x_l_1_z"]
            + p["l_1_x"] + 1.5 * p["z_l_1_x"] + 1.5 * p["l_1_x_l_1_z"]
        )
        expected = numerator / (1 - p["l_1_y"])
        assert float(out.lrm["Effect"]) == pytest.approx(expected)

    def test_shock_size_scales_linearly(self, fit_interact):
        one = interact_adl(fit_interact, z_vals=[1], s_limit=4, **INTERACT)
        two = interact_adl(fit_interact, z_vals=[1], shock_size=2, s_limit=4,
                           **INTERACT)
        assert np.allclose(two.estimates["Effect"],
                           2 * one.estimates["Effect"])


class TestArgumentHandling:
    def test_too_many_lines(self, fit_interact):
        with pytest.raises(ValueError, match="more than 5 discrete values of z"):
            interact_adl(fit_interact, z_vals=[1, 2, 3, 4, 5, 6], **INTERACT)

    def test_heatmap_needs_exactly_two_limits(self, fit_interact):
        with pytest.raises(ValueError, match="requires two z_vals"):
            interact_adl(fit_interact, plot_type="heatmap", z_vals=[1, 2, 3],
                         **INTERACT)

    def test_bad_plot_type(self, fit_interact):
        with pytest.raises(ValueError, match="plot_type must be"):
            interact_adl(fit_interact, plot_type="pie", **INTERACT)

    def test_bad_line_options(self, fit_interact):
        with pytest.raises(ValueError, match="line_options must be"):
            interact_adl(fit_interact, line_options="t.lines", **INTERACT)

    def test_interaction_terms_are_required(self, fit_interact):
        with pytest.raises(ValueError, match="must be specified"):
            interact_adl(fit_interact, x_vrbl={"x": 0}, z_vrbl={"z": 0},
                         x_z_vrbl=None)

    def test_significant_is_dropped_for_fitted_values(self, fit_interact):
        with pytest.warns(UserWarning, match="not applicable to fitted values"):
            out = interact_adl(
                fit_interact,
                effect_type="fitted",
                plot_type="heatmap",
                heatmap_options="significant",
                baseline_y=0,
                z_vals=[-1, 1],
                s_limit=3,
                **INTERACT,
            )
        assert out.meta["heatmap_options"] == "all"

    def test_z_in_prediction_values_is_refused(self, fit_interact):
        with pytest.raises(ValueError, match="Do not supply a value for any z_vrbl"):
            interact_adl(
                fit_interact,
                effect_type="fitted",
                prediction_values={"x": 1, "l_1_x": 1, "z": 0, "x_z": 1,
                                   "z_l_1_x": 1, "x_l_1_z": 1,
                                   "l_1_x_l_1_z": 1},
                z_vals=[0],
                s_limit=2,
                **INTERACT,
            )

    def test_s_lines_falls_back_when_there_is_no_lrm(self, fit_interact):
        with pytest.warns(UserWarning, match="instead of infinity"):
            out = interact_adl(fit_interact, plot_type="lines",
                               line_options="s.lines", z_vals=[-1, 1],
                               s_limit=6, **INTERACT)
        assert out.meta["s_vals"] == [0, 6]
