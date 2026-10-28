"""The delta method, the covariance matrices, and name handling."""

import numpy as np
import pytest
import sympy as sp

from tseffects import VCOV_TYPES, DeltaFn, delta_method, vcov_hc
from tseffects._util import as_model, sanitize_name, sanitize_names, by_lag


class TestVcovHc:
    @pytest.mark.parametrize("ty", VCOV_TYPES)
    def test_symmetric_and_positive_diagonal(self, ty, fit_levels):
        V = vcov_hc(fit_levels.model.exog, fit_levels.resid,
                    fit_levels.df_resid, ty)
        assert np.allclose(V, V.T)
        assert (np.diag(V) > 0).all()

    def test_const_matches_statsmodels(self, fit_levels):
        V = vcov_hc(fit_levels.model.exog, fit_levels.resid,
                    fit_levels.df_resid, "const")
        assert np.allclose(V, fit_levels.cov_params().to_numpy())

    @pytest.mark.parametrize("ty", ["HC0", "HC1", "HC2", "HC3"])
    def test_hc0_to_hc3_match_statsmodels(self, ty, data):
        import statsmodels.formula.api as smf

        robust = smf.ols("y ~ l_1_y + x + l_1_x", data=data).fit(cov_type=ty)
        plain = smf.ols("y ~ l_1_y + x + l_1_x", data=data).fit()
        V = vcov_hc(plain.model.exog, plain.resid, plain.df_resid, ty)
        assert np.allclose(V, robust.cov_params().to_numpy())

    def test_hc_is_an_alias_for_hc0(self, fit_levels):
        args = (fit_levels.model.exog, fit_levels.resid, fit_levels.df_resid)
        assert np.allclose(vcov_hc(*args, "HC"), vcov_hc(*args, "HC0"))

    def test_hc1_inflates_hc0_by_n_over_df(self, fit_levels):
        args = (fit_levels.model.exog, fit_levels.resid, fit_levels.df_resid)
        n = len(fit_levels.resid)
        factor = n / fit_levels.df_resid
        assert np.allclose(vcov_hc(*args, "HC1"),
                           factor * vcov_hc(*args, "HC0"))

    def test_hc3_is_more_conservative_than_hc0(self, fit_levels):
        """HC3 divides by (1-h)^2, so its diagonal dominates HC0's."""
        args = (fit_levels.model.exog, fit_levels.resid, fit_levels.df_resid)
        assert (np.diag(vcov_hc(*args, "HC3"))
                > np.diag(vcov_hc(*args, "HC0"))).all()

    def test_bad_type(self, fit_levels):
        with pytest.raises(ValueError, match="Invalid se_type"):
            vcov_hc(fit_levels.model.exog, fit_levels.resid,
                    fit_levels.df_resid, "HC99")

    def test_mismatched_lengths(self, fit_levels):
        with pytest.raises(ValueError, match="different sample sizes"):
            vcov_hc(fit_levels.model.exog, fit_levels.resid[:-2],
                    fit_levels.df_resid, "HC0")


class TestDeltaMethod:
    def test_a_single_coefficient_recovers_its_own_standard_error(self, fit_levels):
        info = as_model(fit_levels)
        out = delta_method("x", info)
        assert out.estimate == pytest.approx(fit_levels.params["x"])
        assert out.se == pytest.approx(fit_levels.bse["x"])

    def test_a_sum_uses_the_covariance(self, fit_levels):
        info = as_model(fit_levels)
        out = delta_method("x + l_1_x", info)
        V = fit_levels.cov_params()
        expected = np.sqrt(
            V.loc["x", "x"] + V.loc["l_1_x", "l_1_x"] + 2 * V.loc["x", "l_1_x"]
        )
        assert out.se == pytest.approx(expected)

    def test_a_ratio_matches_the_analytic_gradient(self, fit_levels):
        """The long-run multiplier, differentiated by hand."""
        info = as_model(fit_levels)
        out = delta_method("(x + l_1_x)/(1 - l_1_y)", info)
        p, V = fit_levels.params, fit_levels.cov_params()
        denom = 1 - p["l_1_y"]
        grad = {
            "Intercept": 0.0,
            "x": 1 / denom,
            "l_1_x": 1 / denom,
            "l_1_y": (p["x"] + p["l_1_x"]) / denom**2,
        }
        g = np.array([grad[n] for n in V.columns])
        assert out.se == pytest.approx(np.sqrt(g @ V.to_numpy() @ g))

    def test_intervals_are_normal(self, fit_levels):
        from scipy.stats import norm

        info = as_model(fit_levels)
        out = delta_method("x", info, level=0.9)
        z = norm.ppf(0.95)
        assert out.lower == pytest.approx(out.estimate - z * out.se)
        assert out.upper == pytest.approx(out.estimate + z * out.se)

    def test_a_constant_has_no_standard_error(self, fit_levels):
        info = as_model(fit_levels)
        out = delta_method("3.5", info)
        assert out.estimate == pytest.approx(3.5)
        assert out.se == 0.0
        assert out.lower == out.upper == pytest.approx(3.5)

    def test_extra_symbols_are_treated_as_known(self, fit_levels):
        """z_val is substituted, not differentiated: it carries no variance."""
        info = as_model(fit_levels)
        fn = DeltaFn("x * z_val", info.names, extra=["z_val"])
        at_two = fn(info.coef_vector, info.vcov, 0.95, {"z_val": 2.0})
        at_one = fn(info.coef_vector, info.vcov, 0.95, {"z_val": 1.0})
        assert at_two.estimate == pytest.approx(2 * at_one.estimate)
        assert at_two.se == pytest.approx(2 * at_one.se)

    def test_unknown_symbol_is_refused(self, fit_levels):
        info = as_model(fit_levels)
        with pytest.raises(ValueError, match="neither coefficients nor"):
            DeltaFn("x + mystery", info.names)

    def test_missing_extra_value(self, fit_levels):
        info = as_model(fit_levels)
        fn = DeltaFn("x * z_val", info.names, extra=["z_val"])
        with pytest.raises(ValueError, match="missing values for: z_val"):
            fn(info.coef_vector, info.vcov, 0.95)

    def test_bad_level(self, fit_levels):
        info = as_model(fit_levels)
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            delta_method("x", info, level=1.0)

    def test_compiled_once_reused_many_times(self, fit_levels):
        info = as_model(fit_levels)
        fn = DeltaFn(sp.sympify("x + l_1_y * z_val"), info.names,
                     extra=["z_val"])
        values = [
            fn(info.coef_vector, info.vcov, 0.95, {"z_val": z}).estimate
            for z in (0.0, 1.0, 2.0)
        ]
        # Linear in z, so equally spaced inputs give equally spaced outputs.
        assert values[1] - values[0] == pytest.approx(values[2] - values[1])


class TestNameHandling:
    @pytest.mark.parametrize(
        "raw, clean",
        [
            ("(Intercept)", "Intercept"),
            ("Intercept", "Intercept"),
            ("const", "Intercept"),
            ("l.1.x", "l_1_x"),
            ("log(x)", "log_x_"),
            ("I(x^2)", "I_x^2_"),
            ("lag(x, 1)", "lag_x__1_"),
            ("2x", "_2x"),
        ],
    )
    def test_sanitize(self, raw, clean):
        assert sanitize_name(raw) == clean

    def test_warns_about_each_bad_character_once(self):
        with pytest.warns(UserWarning, match="containing \\."):
            sanitize_names(["a.b", "c.d"])

    def test_intercept_parentheses_are_left_alone(self):
        with pytest.warns(UserWarning):
            out = sanitize_names(["(Intercept)", "lag(x)"])
        assert out == ["Intercept", "lag_x_"]

    def test_by_lag_inverts_the_mapping(self):
        assert by_lag({"x": 0, "l_1_x": 1}) == {0: "x", 1: "l_1_x"}

    def test_by_lag_warns_on_a_duplicated_lag(self):
        with pytest.warns(UserWarning, match="share lag order"):
            out = by_lag({"x": 0, "x_alt": 0})
        assert out == {0: "x"}


class TestModelInput:
    def test_coefficients_and_vcov_directly(self, fit_levels):
        from tseffects import gdrf_adl

        coef = fit_levels.params.to_dict()
        vcov = fit_levels.cov_params().to_numpy()
        from_fit = gdrf_adl(fit_levels, x_vrbl={"x": 0, "l_1_x": 1},
                            y_vrbl={"l_1_y": 1}, shock_history="step",
                            s_limit=5)
        from_raw = gdrf_adl((coef, vcov), x_vrbl={"x": 0, "l_1_x": 1},
                            y_vrbl={"l_1_y": 1}, shock_history="step",
                            s_limit=5)
        assert np.allclose(from_fit.estimates["GDRF"],
                           from_raw.estimates["GDRF"])
        assert np.allclose(from_fit.estimates["SE"], from_raw.estimates["SE"])

    def test_a_pandas_series_of_coefficients(self, fit_levels):
        info = as_model((fit_levels.params, fit_levels.cov_params()))
        assert info.coef["x"] == pytest.approx(fit_levels.params["x"])

    def test_rejects_something_that_is_not_a_model(self):
        with pytest.raises(TypeError, match="fitted statsmodels"):
            as_model("not a model")

    def test_vcov_shape_is_checked(self, fit_levels):
        with pytest.raises(ValueError, match="coefficients"):
            as_model((fit_levels.params.to_dict(), np.eye(2)))
