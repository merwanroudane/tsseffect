"""Consumer sentiment and US presidential approval, 1981-2016.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>

The complete worked example on real data.  It is the single source for the
README walkthrough, the documentation site's application section, and every
figure under ``docs/assets`` -- so the three can never drift apart.

    python examples/approval_empirical.py

Data: Cavari, Amnon (2019), "Evaluating the President on Your Priorities: Issue
Priorities, Policy Performance, and Presidential Approval, 1981-2016",
*Presidential Studies Quarterly* 49(4): 798-826, doi:10.1111/psq.12594.
140 quarters, shipped with the package as ``load_approval()``.

The question: a permanent one-point rise in consumer sentiment -- how far does
presidential approval move, how long does it take, and for how long can we tell
the effect from zero?

The answer is the reason this package exists.  The two sentiment coefficients
almost cancel (+0.31 and -0.26, a net +0.05), so a reader of the coefficient
table would conclude that sentiment barely matters.  The response function says
otherwise: the effect is about +0.33 points -- six times the naive sum -- and it
is significant at the 1 per cent level for the first year.  It also says
something the coefficient table cannot: beyond roughly two years the long-run
effect is no longer distinguishable from zero, because the uncertainty
compounds along with the autoregression.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import tseffects as ts

ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

#: The journal figure style, applied per figure so it never leaks elsewhere.
JOURNAL_RC = {
    "figure.figsize": (7.0, 4.3),
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.5,
    "legend.frameon": False,
    "lines.linewidth": 1.8,
}

#: The ADL(1,1) and the GECM that is its exact reparameterization.  Dropping
#: ``l_1_d_UMCSENT`` from the GECM is what makes the two span the same columns,
#: and therefore give identical effects and identical standard errors.
ADL_FORMULA = "APPROVE ~ l_1_APPROVE + UMCSENT + l_1_UMCSENT + DIVIDEDGOV"
GECM_FORMULA = "d_APPROVE ~ l_1_APPROVE + l_1_UMCSENT + d_UMCSENT + DIVIDEDGOV"

X_VRBL = {"UMCSENT": 0, "l_1_UMCSENT": 1}
Y_VRBL = {"l_1_APPROVE": 1}

#: Columns the models actually use.  Dropping missing values on *these* and no
#: others keeps the sample as large as the specification allows -- dropping on
#: the whole frame would silently throw away rows over unrelated columns and
#: change the answer.
NEEDED = [
    "APPROVE", "l_1_APPROVE", "d_APPROVE",
    "UMCSENT", "l_1_UMCSENT", "d_UMCSENT",
    "DIVIDEDGOV", "MIP_MACROECONOMICS",
]


def build_data() -> pd.DataFrame:
    """Load the approval series and construct the lags and differences.

    Nothing here is specific to this package: it is the ordinary business of
    getting a time series into the shape a dynamic model needs, and the README
    walks through it line by line.
    """
    data = ts.load_approval().copy()
    data["QUARTER"] = pd.period_range("1981Q1", periods=len(data), freq="Q").astype(str)
    for name in ("APPROVE", "UMCSENT"):
        data[f"l_1_{name}"] = data[name].shift(1)
        data[f"d_{name}"] = data[name].diff()
    return data.dropna(subset=NEEDED).reset_index(drop=True)


def save(fig, stem: str) -> None:
    """Write the vector file for a manuscript and the PNG for the web."""
    fig.savefig(ASSETS / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(ASSETS / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    wrote docs/assets/{stem}.png and .pdf")


def main() -> dict:
    data = build_data()
    out: dict[str, str] = {}

    print(f"\n[1] Data: {len(data)} quarters, "
          f"{data['QUARTER'].iloc[0]} to {data['QUARTER'].iloc[-1]}")
    out["nobs"] = str(len(data))
    out["span"] = f"{data['QUARTER'].iloc[0]}-{data['QUARTER'].iloc[-1]}"

    # ------------------------------------------------------------------ ADL
    print("\n[2] The ADL(1,1): what the coefficient table shows")
    fit = smf.ols(ADL_FORMULA, data=data).fit()
    print(fit.summary().tables[1])
    out["ols_table"] = str(fit.summary().tables[1])
    out["ols_r2"] = f"{fit.rsquared:.4f}"
    for name in ("l_1_APPROVE", "UMCSENT", "l_1_UMCSENT"):
        out[f"coef_{name}"] = f"{fit.params[name]:.4f}"
        out[f"se_{name}"] = f"{fit.bse[name]:.4f}"
        out[f"p_{name}"] = f"{fit.pvalues[name]:.4f}"
    net = fit.params["UMCSENT"] + fit.params["l_1_UMCSENT"]
    out["naive_sum"] = f"{net:.4f}"

    # ------------------------------------------- the step response and the LRM
    print("\n[3] The step response: what the coefficient table hides")
    step = ts.gdrf_adl(fit, x_vrbl=X_VRBL, y_vrbl=Y_VRBL,
                       shock_history="step", s_limit=20, se_type="HC3")
    print(step.summary())
    out["step_table"] = step.summary().as_text()
    out["step_markdown"] = step.summary().as_markdown()
    out["step_latex"] = step.summary().as_latex()

    frame = step.to_frame()
    lrm = step.lrm
    out["lrm"] = f"{float(lrm['GDRF']):.4f}"
    out["lrm_se"] = f"{float(lrm['SE']):.4f}"
    out["lrm_p"] = f"{float(frame['p'].iloc[-1]):.4f}"
    out["lrm_lo"] = f"{float(lrm['Lower']):.4f}"
    out["lrm_hi"] = f"{float(lrm['Upper']):.4f}"
    out["ratio"] = f"{float(lrm['GDRF']) / net:.1f}"

    # The period at which the effect stops being significant at 5 per cent.
    path = frame[frame["Period"] != "LRM"]
    significant = path[path["p"] < 0.05]["Period"].tolist()
    last_sig = max(int(p) for p in significant) if significant else None
    out["last_significant_s"] = str(last_sig)
    out["effect_s0"] = f"{float(path['GDRF'].iloc[0]):.4f}"
    out["effect_s0_p"] = f"{float(path['p'].iloc[0]):.4f}"
    print(f"\n    significant at 5% through s = {last_sig} "
          f"({last_sig} quarters), then no longer")

    with plt.rc_context(JOURNAL_RC):
        ax = step.plot(color="#4f46e5")
        ax.set_title("Step response of approval to a permanent rise in sentiment",
                     loc="left")
        ax.set_xlabel("Quarters since the shock (s)")
        ax.set_ylabel("Approval (percentage points)")
        save(ax.figure, "approval_step")

    # ------------------------------------------------------ impulse response
    print("\n[4] The impulse response: a one-quarter blip, which decays away")
    pulse = ts.gdrf_adl(fit, x_vrbl=X_VRBL, y_vrbl=Y_VRBL,
                        shock_history="pulse", s_limit=20, se_type="HC3",
                        return_lrm=False)
    out["pulse_markdown"] = pulse.summary().as_markdown()
    with plt.rc_context(JOURNAL_RC):
        ax = pulse.plot(color="#0d9488")
        ax.set_title("Impulse response: a one-quarter rise in sentiment", loc="left")
        ax.set_xlabel("Quarters since the shock (s)")
        ax.set_ylabel("Approval (percentage points)")
        save(ax.figure, "approval_impulse")

    # --------------------------------------------------------- fitted values
    print("\n[5] Fitted values: the trajectory of approval itself")
    means = {"UMCSENT": float(data["UMCSENT"].mean()),
             "l_1_UMCSENT": float(data["UMCSENT"].mean()),
             "DIVIDEDGOV": float(data["DIVIDEDGOV"].mean())}
    fitted = ts.gdrf_adl(fit, x_vrbl=X_VRBL, y_vrbl=Y_VRBL,
                         shock_history="step", effect_type="fitted",
                         prediction_values=means, s_limit=20, se_type="HC3")
    out["fitted_markdown"] = fitted.summary().as_markdown()
    out["baseline"] = f"{float(fitted.estimates['GDRF'].iloc[0]):.4f}"
    with plt.rc_context(JOURNAL_RC):
        ax = fitted.plot(color="#f59e0b")
        ax.set_title("Predicted approval, from its steady state", loc="left")
        ax.set_xlabel("Quarters since the shock (s)")
        ax.set_ylabel("Approval (%)")
        save(ax.figure, "approval_fitted")

    # ------------------------------------------------------------------ GECM
    print("\n[6] The same question through a GECM -- identical by construction")
    gecm_fit = smf.ols(GECM_FORMULA, data=data).fit()
    gecm = ts.gdrf_gecm(gecm_fit,
                        x_vrbl={"l_1_UMCSENT": 1}, y_vrbl={"l_1_APPROVE": 1},
                        x_d_vrbl={"d_UMCSENT": 0}, y_d_vrbl=None,
                        shock_history="step", s_limit=20, se_type="HC3")
    identical = bool(
        np.allclose(step.estimates["GDRF"], gecm.estimates["GDRF"])
        and np.allclose(step.estimates["SE"], gecm.estimates["SE"])
    )
    print(f"    ADL and GECM agree to machine precision: {identical}")
    out["gecm_identical"] = str(identical)
    out["gecm_markdown"] = gecm.summary().as_markdown()
    out["gecm_lrm"] = f"{float(gecm.lrm['GDRF']):.4f}"
    out["gecm_lrm_se"] = f"{float(gecm.lrm['SE']):.4f}"
    out["gecm_max_diff"] = f"{float(np.max(np.abs(step.estimates['GDRF'] - gecm.estimates['GDRF']))):.2e}"
    with plt.rc_context(JOURNAL_RC):
        ax = gecm.plot(color="#0284c7")
        ax.set_title("GECM step response -- the same curve, from the other form",
                     loc="left")
        ax.set_xlabel("Quarters since the shock (s)")
        ax.set_ylabel("Approval (percentage points)")
        save(ax.figure, "approval_gecm")

    # ------------------------------------------- conditional on issue salience
    print("\n[7] Does it depend on how salient the economy is? (Cavari's thesis)")
    data["UMC_MIP"] = data["UMCSENT"] * data["MIP_MACROECONOMICS"]
    data["l1UMC_MIP"] = data["l_1_UMCSENT"] * data["MIP_MACROECONOMICS"]
    inter_fit = smf.ols(
        "APPROVE ~ l_1_APPROVE + UMCSENT + l_1_UMCSENT + MIP_MACROECONOMICS"
        " + UMC_MIP + l1UMC_MIP + DIVIDEDGOV",
        data=data,
    ).fit()
    print(inter_fit.summary().tables[1])
    out["interact_ols_table"] = str(inter_fit.summary().tables[1])
    for name in ("UMC_MIP", "l1UMC_MIP"):
        out[f"coef_{name}"] = f"{inter_fit.params[name]:.4f}"
        out[f"p_{name}"] = f"{inter_fit.pvalues[name]:.4f}"

    salience = data["MIP_MACROECONOMICS"].quantile([0.1, 0.5, 0.9]).round(0).tolist()
    out["salience_levels"] = ", ".join(f"{v:.0f}" for v in salience)
    inter_kwargs = dict(
        x_vrbl=X_VRBL,
        z_vrbl={"MIP_MACROECONOMICS": 0},
        x_z_vrbl={"UMC_MIP": 0, "l1UMC_MIP": 1},
        y_vrbl=Y_VRBL,
        shock_history="step", s_limit=20, se_type="HC3",
        z_label="economic salience (% naming the economy)",
    )
    inter = ts.interact_adl(inter_fit, z_vals=salience, **inter_kwargs)
    out["interact_markdown"] = inter.summary().as_markdown()
    with plt.rc_context(JOURNAL_RC):
        ax = inter.plot()
        ax.set_title("Conditional step response, by how salient the economy is",
                     loc="left")
        ax.set_xlabel("Quarters since the shock (s)")
        ax.set_ylabel("Approval (percentage points)")
        save(ax.figure, "approval_interact_lines")

    heat = ts.interact_adl(
        inter_fit, plot_type="heatmap", heatmap_options="significant",
        z_vals=[float(data["MIP_MACROECONOMICS"].min()),
                float(data["MIP_MACROECONOMICS"].max())],
        **inter_kwargs,
    )
    with plt.rc_context(JOURNAL_RC):
        ax = heat.plot(heatmap_colors="Blue-Red")
        ax.set_title("Where the effect is distinguishable from zero", loc="left")
        ax.set_xlabel("Quarters since the shock (s)")
        save(ax.figure, "approval_interact_heatmap")

    # --------------------------------------------------- the headline figure
    print("\n[8] The headline figure")
    with plt.rc_context({**JOURNAL_RC, "figure.figsize": (11.5, 4.0)}):
        fig, axes = plt.subplots(1, 3)
        panels = [
            (pulse, "Impulse: one quarter", "#0d9488", "Approval (pp)"),
            (step, "Step: permanent", "#4f46e5", "Approval (pp)"),
            (fitted, "Predicted approval", "#f59e0b", "Approval (%)"),
        ]
        for ax, (res, title, color, ylab) in zip(axes, panels):
            res.plot(ax=ax, color=color)
            ax.set_title(title, loc="left", fontsize=11)
            ax.set_xlabel("Quarters since the shock (s)")
            ax.set_ylabel(ylab)
        save(fig, "approval_overview")

    (ASSETS / "approval_outputs.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(f"\n    wrote docs/assets/approval_outputs.json ({len(out)} values)")

    print("\n" + "=" * 78)
    print("WHAT THE COEFFICIENT TABLE SAYS")
    print(f"  sentiment now  : {out['coef_UMCSENT']} (p = {out['p_UMCSENT']})")
    print(f"  sentiment lag 1: {out['coef_l_1_UMCSENT']} (p = {out['p_l_1_UMCSENT']})")
    print(f"  the two nearly cancel, net {out['naive_sum']} -- 'sentiment hardly matters'")
    print()
    print("WHAT THE RESPONSE FUNCTION SAYS")
    print(f"  immediate effect : {out['effect_s0']} (p = {out['effect_s0_p']})")
    print(f"  long-run effect  : {out['lrm']} "
          f"(SE {out['lrm_se']}, p = {out['lrm_p']})")
    print(f"  that is {out['ratio']}x the naive sum of the coefficients,")
    print(f"  significant at 5% through s = {out['last_significant_s']} quarters,")
    print(f"  and no longer distinguishable from zero in the long run.")
    print("=" * 78)
    return out


if __name__ == "__main__":
    main()
