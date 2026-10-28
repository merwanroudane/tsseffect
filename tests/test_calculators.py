"""The symbolic layer, against the R package's own testthat expectations.

The expected formulae in this file are copied verbatim from
``tests/testthat/test-general.calculator.R`` in the R package, and compared
symbolically so that a difference in term ordering does not count as a failure.
"""

import pytest
import sympy as sp

from tseffects import (
    convolution_calculator,
    gecm_to_adl,
    general_calculator,
    pulse_calculator,
    shock_weights,
)

X = {"x": 0, "l_1_x": 1}
Y = {"l_1_y": 1}
LIMIT = 5


def same(expr, text: str) -> bool:
    """Whether a SymPy expression equals a formula written out by hand."""
    return sp.simplify(sp.expand(expr) - sp.sympify(text)) == 0


class TestPulseCalculator:
    def test_recursion(self):
        pulses = pulse_calculator(X, Y, LIMIT)
        assert len(pulses) == LIMIT + 1
        assert same(pulses[0], "x")
        assert same(pulses[1], "l_1_x + l_1_y*x")
        assert same(pulses[2], "l_1_y*l_1_x + l_1_y**2*x")
        assert same(pulses[3], "l_1_y**2*l_1_x + l_1_y**3*x")

    def test_finite_dynamics(self):
        """With no lagged dependent variable the pulses are the lag weights."""
        pulses = pulse_calculator({"x": 0, "l_1_x": 1, "l_2_x": 2}, None, 4)
        assert same(pulses[0], "x")
        assert same(pulses[1], "l_1_x")
        assert same(pulses[2], "l_2_x")
        assert pulses[3] == 0
        assert pulses[4] == 0

    def test_skipped_lags_contribute_nothing(self):
        """A model with x at lags 0 and 2 but not 1 has a zero at lag 1."""
        pulses = pulse_calculator({"x": 0, "l_2_x": 2}, Y, 3)
        assert same(pulses[1], "l_1_y*x")
        assert same(pulses[2], "l_2_x + l_1_y**2*x")

    def test_second_order_autoregression(self):
        pulses = pulse_calculator(X, {"l_1_y": 1, "l_2_y": 2}, 3)
        assert same(pulses[2], "l_1_y*(l_1_x + l_1_y*x) + l_2_y*x")

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            pulse_calculator(None, Y, 3)
        with pytest.raises(ValueError, match="integer lag orders"):
            pulse_calculator({"x": "zero"}, Y, 3)
        with pytest.raises(ValueError, match="non-negative"):
            pulse_calculator(X, Y, -1)


class TestShockWeights:
    @pytest.mark.parametrize(
        "d_x, d_y, h, expected",
        [
            # m = -1: the impulse response -- all weight on the current pulse.
            (0, 0, -1, [1, 0, 0, 0]),
            # m = -2: a first difference of the pulses.
            (1, 0, -1, [1, -1, 0, 0]),
            # m = 0: the step response -- a running sum.
            (0, 1, -1, [1, 1, 1, 1]),
            (0, 0, 0, [1, 1, 1, 1]),
            # m = 1: the cumulated step response.
            (0, 0, 1, [1, 2, 3, 4]),
            # m = -3: a second difference.
            (2, 0, -1, [1, -2, 1, 0]),
        ],
    )
    def test_against_r(self, d_x, d_y, h, expected):
        assert shock_weights(h, d_x, d_y, 3) == expected

    def test_cumulating_a_step_gives_the_next_order(self):
        """Incrementing h is what `cumulative=True` does."""
        assert shock_weights(1, 0, 0, 4) == [1, 2, 3, 4, 5]


class TestGeneralCalculator:
    @pytest.mark.parametrize(
        "d_x, d_y, h, expected",
        [
            (0, 0, -1, "l_1_y**2 * l_1_x + l_1_y**3 * x"),
            (
                1,
                0,
                -1,
                "l_1_y**2*l_1_x + l_1_y**3*x - l_1_y*l_1_x - l_1_y**2*x",
            ),
            (
                0,
                1,
                -1,
                "l_1_y**2*l_1_x + l_1_y**3*x + l_1_y*l_1_x + l_1_y**2*x"
                " + l_1_x + l_1_y*x + x",
            ),
        ],
    )
    def test_formula_at_s3_matches_r(self, d_x, d_y, h, expected):
        pulses = pulse_calculator(X, Y, LIMIT)
        out = general_calculator(d_x, d_y, h, LIMIT, pulses)
        assert same(out.formulae[3], expected)

    def test_shape(self):
        pulses = pulse_calculator(X, Y, LIMIT)
        out = general_calculator(0, 0, -1, LIMIT, pulses)
        assert len(out.formulae) == LIMIT + 1
        assert len(out.shocks) == LIMIT + 1
        assert len(out.as_strings()) == LIMIT + 1

    def test_step_is_the_running_sum_of_the_pulses(self):
        pulses = pulse_calculator(X, Y, LIMIT)
        step = general_calculator(0, 0, 0, LIMIT, pulses)
        for s in range(LIMIT + 1):
            assert same(step.formulae[s], str(sp.expand(sum(pulses[: s + 1]))))

    def test_rejects_too_few_pulses(self):
        pulses = pulse_calculator(X, Y, 3)
        with pytest.raises(ValueError, match="limit = 5"):
            general_calculator(0, 0, -1, 5, pulses)


class TestConvolutionCalculator:
    def test_unit_impulse_reproduces_the_pulses(self):
        """A history of 1 then zeros is exactly the impulse response."""
        pulses = pulse_calculator(X, Y, LIMIT)
        out = convolution_calculator(0, 0, [1, 0, 0, 0, 0, 0], pulses)
        for s in range(LIMIT + 1):
            assert same(out.formulae[s], str(pulses[s]))

    def test_all_ones_reproduces_the_step(self):
        pulses = pulse_calculator(X, Y, LIMIT)
        custom = convolution_calculator(0, 0, [1] * (LIMIT + 1), pulses)
        step = general_calculator(0, 0, 0, LIMIT, pulses)
        for s in range(LIMIT + 1):
            assert same(custom.formulae[s] - step.formulae[s], "0")

    def test_interrupted_trend(self):
        pulses = pulse_calculator(X, Y, 3)
        out = convolution_calculator(0, 0, [1, 2, 3, 0], pulses)
        assert same(out.formulae[0], "x")
        assert same(out.formulae[1], "(l_1_x + l_1_y*x) + 2*x")
        assert out.shocks[-1] == [1.0, 2.0, 3.0, 0.0]

    def test_d_x_differences_the_history(self):
        pulses = pulse_calculator(X, Y, 3)
        out = convolution_calculator(1, 0, [1, 1, 1, 1], pulses)
        # Differencing a constant history leaves only the first impulse.
        assert out.shocks[-1] == [1.0, 0.0, 0.0, 0.0]

    def test_d_y_cumulates_the_response(self):
        pulses = pulse_calculator(X, Y, 3)
        plain = convolution_calculator(0, 0, [1, 0, 0, 0], pulses)
        cumulated = convolution_calculator(0, 1, [1, 0, 0, 0], pulses)
        assert same(cumulated.formulae[2] - sum(plain.formulae[:3]), "0")

    def test_rejects_wrong_length(self):
        pulses = pulse_calculator(X, Y, 3)
        with pytest.raises(ValueError, match="4 pulse formulae"):
            convolution_calculator(0, 0, [1, 2], pulses)


class TestGecmToAdl:
    def test_standard_case_matches_r(self):
        out = gecm_to_adl({"l_1_x": 1}, {"l_1_y": 1},
                          {"d_x": 0, "l_1_d_x": 1}, {"l_1_d_y": 1})
        assert out["x_vrbl_adl"] == {
            "d_x": 0,
            "l_1_x+l_1_d_x-d_x": 1,
            "(-1)*l_1_d_x": 2,
        }
        assert out["y_vrbl_adl"] == {"l_1_y+l_1_d_y+1": 1, "(-1)*l_1_d_y": 2}

    def test_longer_lags_matches_r(self):
        out = gecm_to_adl({"l_1_x": 1}, {"l_1_y": 1},
                          {"d_x": 0, "l_1_d_x": 1, "l_2_d_x": 2},
                          {"l_1_d_y": 1, "l_2_d_y": 2})
        assert list(out["x_vrbl_adl"].values()) == [0, 1, 2, 3]
        assert list(out["y_vrbl_adl"].values()) == [1, 2, 3]
        assert "(-1)*l_1_d_x+l_2_d_x" in out["x_vrbl_adl"]

    def test_no_short_run_y_dynamics(self):
        out = gecm_to_adl({"l_1_x": 1}, {"l_1_y": 1}, {"d_x": 0}, None)
        assert out["y_vrbl_adl"] == {"l_1_y+1": 1}
        assert out["x_vrbl_adl"] == {"d_x": 0, "l_1_x-d_x": 1}

    def test_translation_is_algebraically_an_adl(self):
        """The translated ADL(1,1) coefficients reproduce the GECM identity.

        For a GECM(1,1) the ADL has alpha_1 = theta_y + alpha_1 + 1, so the
        long-run multiplier -theta_x / theta_y must equal sum(beta) / (1 - alpha).
        """
        out = gecm_to_adl({"l_1_x": 1}, {"l_1_y": 1},
                          {"d_x": 0, "l_1_d_x": 1}, {"l_1_d_y": 1})
        betas = sum(sp.sympify(k) for k in out["x_vrbl_adl"])
        alphas = sum(sp.sympify(k) for k in out["y_vrbl_adl"])
        lrm_adl = sp.simplify(betas / (1 - alphas))
        lrm_gecm = sp.sympify("-l_1_x / l_1_y")
        assert sp.simplify(lrm_adl - lrm_gecm) == 0

    def test_rejects_missing_pieces(self):
        with pytest.raises(ValueError):
            gecm_to_adl({"l_1_x": 1}, {"l_1_y": 1}, None, None)
