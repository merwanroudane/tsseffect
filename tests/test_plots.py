"""The plotting layer.

These check that each plot draws, puts the data where it should, and labels its
axes -- not that it looks a particular way.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from tseffects import (
    OKABE_ITO,
    gdrf_adl,
    gdrf_gecm,
    interact_adl,
    plot_effects,
)
from tseffects.plots import _diverging_cmap, _line_colors

from .conftest import ADL, GECM, INTERACT


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


class TestGdrfPlot:
    def test_draws_the_path_and_the_band(self, fit_levels):
        out = gdrf_adl(fit_levels, shock_history="pulse", s_limit=10, **ADL)
        ax = out.plot()
        assert len(ax.lines) >= 1
        line = ax.lines[0]
        assert np.allclose(line.get_ydata(), out.estimates["GDRF"])
        assert len(ax.collections) >= 1  # the fill_between band

    def test_labels_the_axes(self, fit_levels):
        out = gdrf_adl(fit_levels, shock_history="step", s_limit=10, **ADL)
        ax = out.plot()
        assert "Periods Since Initial Shock" in ax.get_xlabel()
        assert "GDRF" in ax.get_ylabel()

    def test_the_lrm_gets_its_own_tick_past_the_path(self, fit_levels):
        out = gdrf_adl(fit_levels, shock_history="step", s_limit=20, **ADL)
        ax = out.plot()
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels[-1] == "LRM"
        positions = list(ax.get_xticks())
        assert positions[-1] > 20

    def test_no_lrm_tick_when_there_is_no_lrm(self, fit_levels):
        out = gdrf_adl(fit_levels, shock_history="pulse", s_limit=10, **ADL)
        ax = out.plot()
        assert "LRM" not in [t.get_text() for t in ax.get_xticklabels()]

    def test_discrete_effects_get_a_zero_line(self, fit_levels):
        out = gdrf_adl(fit_levels, shock_history="pulse", s_limit=5, **ADL)
        ax = out.plot()
        assert any(
            np.allclose(line.get_ydata(), 0) for line in ax.lines
        )

    def test_fitted_values_get_a_baseline_tick_and_no_zero_line(self, fit_levels):
        out = gdrf_adl(
            fit_levels,
            shock_history="pulse",
            effect_type="fitted",
            prediction_values={"x": 1, "l_1_x": 1},
            s_limit=10,
            **ADL,
        )
        ax = out.plot()
        assert "Baseline" in [t.get_text() for t in ax.get_xticklabels()]
        assert not any(np.allclose(line.get_ydata(), 0) for line in ax.lines)

    def test_gecm_plots(self, fit_gecm):
        out = gdrf_gecm(fit_gecm, shock_history="step", s_limit=10, **GECM)
        ax = out.plot()
        assert len(ax.lines) >= 1

    def test_accepts_an_existing_axes(self, fit_levels):
        out = gdrf_adl(fit_levels, shock_history="pulse", s_limit=5, **ADL)
        fig, ax = plt.subplots()
        assert out.plot(ax=ax) is ax

    def test_theme_removes_the_top_and_right_spines(self, fit_levels):
        out = gdrf_adl(fit_levels, shock_history="pulse", s_limit=5, **ADL)
        ax = out.plot()
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        assert ax.spines["left"].get_visible()


class TestInteractPlots:
    def test_z_lines_draws_one_line_per_z(self, fit_interact):
        out = interact_adl(fit_interact, z_vals=[-1, 0, 1], s_limit=10,
                           **INTERACT)
        ax = out.plot()
        labelled = [ln for ln in ax.lines if ln.get_label().lstrip("_")]
        assert len(labelled) >= 3
        assert ax.get_legend() is not None

    def test_s_lines_puts_the_moderator_on_the_x_axis(self, fit_interact):
        out = interact_adl(
            fit_interact,
            shock_history="step",
            plot_type="lines",
            line_options="s.lines",
            z_vals=[-2, 2],
            s_vals=[0, "LRM"],
            s_limit=10,
            **INTERACT,
        )
        ax = out.plot()
        assert "Value of z" in ax.get_xlabel()

    def test_heatmap_draws_a_mesh_with_a_colorbar(self, fit_interact):
        out = interact_adl(fit_interact, plot_type="heatmap",
                           heatmap_options="all", z_vals=[-2, 2], s_limit=10,
                           **INTERACT)
        ax = out.plot()
        assert len(ax.collections) == 1
        assert ax.figure.axes[-1] is not ax  # the colorbar axes

    def test_significant_heatmap_blanks_the_insignificant_cells(self, fit_interact):
        out = interact_adl(fit_interact, plot_type="heatmap",
                           heatmap_options="significant", z_vals=[-2, 2],
                           s_limit=10, **INTERACT)
        ax = out.plot()
        values = ax.collections[0].get_array().compressed()
        covers_zero = (out.estimates["Lower"] < 0) & (out.estimates["Upper"] > 0)
        assert covers_zero.any()  # otherwise the test proves nothing
        assert (values == 0).sum() == int(covers_zero.sum())

    def test_custom_colors(self, fit_interact):
        out = interact_adl(fit_interact, z_vals=[-1, 1], s_limit=5, **INTERACT)
        ax = out.plot(line_colors=["#111111", "#222222"])
        assert ax is not None


class TestColorHelpers:
    def test_okabe_ito_skips_black(self):
        assert _line_colors("okabe-ito", 3) == list(OKABE_ITO[1:4])

    def test_bw_is_a_grey_ramp(self):
        colors = _line_colors("bw", 3)
        assert len(colors) == 3
        assert all(0.0 <= float(c) <= 1.0 for c in colors)

    def test_wrong_number_of_colors(self):
        with pytest.raises(ValueError, match="not equal to the number"):
            _line_colors(["#000000"], 3)

    def test_unknown_palette_name(self):
        with pytest.raises(ValueError, match="must be 'okabe-ito'"):
            _line_colors("rainbow-sparkle", 3)

    def test_named_diverging_palette(self):
        cmap = _diverging_cmap("Blue-Red")
        assert np.allclose(cmap(0.5)[:3], (1.0, 1.0, 1.0), atol=0.02)

    def test_matplotlib_colormaps_are_accepted(self):
        assert _diverging_cmap("viridis") is not None

    def test_unknown_colormap(self):
        with pytest.raises(ValueError, match="Unknown heatmap palette"):
            _diverging_cmap("not-a-colormap")


def test_plot_effects_dispatches(fit_levels, fit_interact):
    gdrf = gdrf_adl(fit_levels, shock_history="pulse", s_limit=5, **ADL)
    assert plot_effects(gdrf) is not None
    heat = interact_adl(fit_interact, plot_type="heatmap", z_vals=[-1, 1],
                        s_limit=5, **INTERACT)
    assert len(plot_effects(heat).collections) == 1
