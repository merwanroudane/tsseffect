"""Every plot the package can draw, in one figure per panel.

    python examples/gallery.py

Writes PNGs into ``examples/figures``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

import tseffects as ts

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

data = ts.load_toy_data()

ADL = dict(x_vrbl={"x": 0, "l_1_x": 1}, y_vrbl={"l_1_y": 1})
GECM = dict(
    x_vrbl={"l_1_x": 1},
    y_vrbl={"l_1_y": 1},
    x_d_vrbl={"d_x": 0, "l_1_d_x": 1},
    y_d_vrbl={"l_1_d_y": 1},
)
INTERACT = dict(
    x_vrbl={"x": 0, "l_1_x": 1},
    z_vrbl={"z": 0, "l_1_z": 1},
    x_z_vrbl={"x_z": 0, "z_l_1_x": 1, "x_l_1_z": 0, "l_1_x_l_1_z": 1},
    y_vrbl={"l_1_y": 1},
)


def save(ax, name: str, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=11)
    ax.figure.tight_layout()
    ax.figure.savefig(OUT / f"{name}.png", dpi=150, facecolor="white")
    plt.close(ax.figure)
    print(f"  {name}.png")


def main() -> None:
    fit = smf.ols("y ~ l_1_y + x + l_1_x", data=data).fit()
    fit_dy = smf.ols("d_y ~ l_1_d_y + x + l_1_x", data=data).fit()
    fit_gecm = smf.ols(
        "d_y ~ l_1_y + l_1_d_y + l_1_x + d_x + l_1_d_x", data=data
    ).fit()
    fit_inter = smf.ols(
        "y ~ l_1_y + x + l_1_x + z + l_1_z + x_z + z_l_1_x + x_l_1_z"
        " + l_1_x_l_1_z",
        data=data,
    ).fit()

    print("ADL response functions")
    save(
        ts.gdrf_adl(fit, shock_history="pulse", s_limit=20, **ADL).plot(),
        "01_adl_impulse",
        "Impulse Response Function (h = -1)",
    )
    save(
        ts.gdrf_adl(fit, shock_history="step", s_limit=20, **ADL).plot(),
        "02_adl_step",
        "Step Response Function (h = 0), with the Long-Run Multiplier",
    )
    save(
        ts.gdrf_adl(
            fit, shock_history="pulse", cumulative=True, s_limit=20, **ADL
        ).plot(),
        "03_adl_cumulative_impulse",
        "Cumulative Impulse Response Function",
    )
    save(
        ts.gdrf_adl(
            fit, shock_history=[1, 2, 3, 3, 3, 0, 0, 0, 0, 0, 0], s_limit=10,
            **ADL,
        ).plot(),
        "04_adl_custom_history",
        "A bespoke shock history: a ramp that is then withdrawn",
    )

    print("Fitted values")
    save(
        ts.gdrf_adl(
            fit,
            shock_history="step",
            effect_type="fitted",
            prediction_values={"x": 1, "l_1_x": 1},
            s_limit=20,
            **ADL,
        ).plot(),
        "05_adl_fitted_steadystate",
        "Fitted values of y, from the model-implied steady state",
    )
    save(
        ts.gdrf_adl(
            fit_dy,
            x_vrbl={"x": 0, "l_1_x": 1},
            y_vrbl={"l_1_d_y": 1},
            d_x=0,
            d_y=1,
            shock_history="pulse",
            effect_type="fitted",
            baseline_y=3.0,
            baseline_y_se=2.0,
            s_limit=20,
        ).plot(),
        "06_adl_fitted_asserted_baseline",
        "Fitted values from an asserted baseline, with its own uncertainty",
    )

    print("GECM")
    save(
        ts.gdrf_gecm(fit_gecm, shock_history="pulse", s_limit=20, **GECM).plot(),
        "07_gecm_impulse",
        "GECM impulse response, via the ADL translation",
    )
    save(
        ts.gdrf_gecm(fit_gecm, shock_history="step", s_limit=20, **GECM).plot(),
        "08_gecm_step",
        "GECM step response; the LRM is -theta_x / theta_y",
    )

    print("Interactions")
    save(
        ts.interact_adl(
            fit_inter, z_vals=[-1, 0, 1], s_limit=20, **INTERACT
        ).plot(),
        "09_interact_zlines",
        "Conditional impulse responses, one line per value of z",
    )
    save(
        ts.interact_adl(
            fit_inter,
            shock_history="step",
            z_vals=[-1, 0, 1],
            s_limit=20,
            **INTERACT,
        ).plot(),
        "10_interact_zlines_step",
        "Conditional step responses, with the LRM at each value of z",
    )
    save(
        ts.interact_adl(
            fit_inter,
            shock_history="step",
            plot_type="lines",
            line_options="s.lines",
            z_vals=[-2.5, 2.5],
            s_vals=[0, "LRM"],
            s_limit=20,
            **INTERACT,
        ).plot(),
        "11_interact_slines",
        "Short-run and long-run effect across the range of z",
    )
    save(
        ts.interact_adl(
            fit_inter,
            shock_history="step",
            plot_type="heatmap",
            heatmap_options="all",
            z_vals=[-2.5, 2.5],
            s_limit=20,
            **INTERACT,
        ).plot(),
        "12_interact_heatmap_all",
        "The whole conditional effect surface",
    )
    save(
        ts.interact_adl(
            fit_inter,
            shock_history="step",
            plot_type="heatmap",
            heatmap_options="significant",
            z_vals=[-2.5, 2.5],
            s_limit=20,
            **INTERACT,
        ).plot(),
        "13_interact_heatmap_significant",
        "The same surface, with insignificant cells blanked out",
    )

    print("Robust standard errors, side by side")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, se_type in zip(axes, ("const", "HC1", "HC3")):
        out = ts.gdrf_adl(
            fit, shock_history="step", se_type=se_type, s_limit=20, **ADL
        )
        out.plot(ax=ax)
        ax.set_title(f"se_type = {se_type!r}", loc="left", fontsize=11)
        if ax is not axes[0]:
            ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(OUT / "14_se_types.png", dpi=150, facecolor="white")
    plt.close(fig)
    print("  14_se_types.png")

    print(f"\nFigures written to {OUT}")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()
