"""Compare the Python port against the R package's own output, case by case.

Run ``validate/r_reference.R`` in R first; it writes ``validate/r_out``.  Then::

    python validate/compare.py

Every estimate, standard error and confidence bound is compared, as are the
covariance matrices for all eight ``se_type`` values and the generalized
binomial weights.  The script exits non-zero if anything disagrees.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import tseffects as ts  # noqa: E402
from tseffects._vcov import vcov_hc  # noqa: E402

R_OUT = Path(__file__).resolve().parent / "r_out"
RTOL = 1e-9
ATOL = 1e-11

HC_TYPES = ("const", "HC0", "HC1", "HC2", "HC3", "HC4", "HC4m", "HC5")

FORMULAS = {
    "levels": "y ~ l_1_y + x + l_1_x",
    "dx": "y ~ l_1_y + d_x + l_1_d_x",
    "dy": "d_y ~ l_1_d_y + x + l_1_x",
    "gecm": "d_y ~ l_1_y + l_1_d_y + l_1_x + d_x + l_1_d_x",
    "gecm2": "d_y ~ l_1_y + l_1_d_y + l_2_d_y + l_1_x + d_x + l_1_d_x + l_2_d_x",
    "inter": (
        "y ~ l_1_y + x + l_1_x + z + l_1_z + x_z + z_l_1_x + x_l_1_z + l_1_x_l_1_z"
    ),
    "fdl": "y ~ x + l_1_x + l_2_x",
}

failures: list[str] = []
checks = 0


def report(name: str, ok: bool, detail: str = "") -> None:
    global checks
    checks += 1
    if ok:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(f"{name}: {detail}")


def close(a, b, label: str, name: str) -> None:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        report(f"{name} [{label}]", False, f"shape {a.shape} vs {b.shape}")
        return
    bad = ~np.isclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True)
    if bad.any():
        i = int(np.argmax(bad))
        report(
            f"{name} [{label}]",
            False,
            f"{bad.sum()}/{bad.size} differ, worst at {i}: "
            f"py={a.ravel()[i]!r} r={b.ravel()[i]!r}",
        )
    else:
        report(f"{name} [{label}]", True)


def compare_estimates(tag: str, result, value_col: str = "GDRF") -> None:
    path = R_OUT / f"{tag}.csv"
    if not path.exists():
        report(tag, False, "no R output")
        return
    r = pd.read_csv(path)
    p = result.estimates

    if len(r) != len(p):
        report(tag, False, f"{len(p)} python rows vs {len(r)} R rows")
        return

    r_period = [str(v).strip() for v in r["Period"]]
    p_period = [str(v).strip() for v in p["Period"]]
    # R writes integer periods as "0", "1", ...; normalise both sides.
    norm = lambda s: str(int(float(s))) if s.replace(".", "", 1).lstrip("-").isdigit() else s
    if [norm(v) for v in r_period] != [norm(v) for v in p_period]:
        report(tag, False, f"period labels differ: {p_period[:4]} vs {r_period[:4]}")
        return

    if "Z" in r.columns:
        close(p["Z"], r["Z"], "Z", tag)
    for col, r_col in ((value_col, value_col), ("SE", "SE"), ("Lower", "Lower"), ("Upper", "Upper")):
        close(p[col], r[r_col], r_col, tag)


def main() -> int:
    if not R_OUT.exists():
        print(f"R reference output not found at {R_OUT}")
        print("Run validate/r_reference.R in R first.")
        return 2

    data = pd.read_csv(R_OUT / "toy_data.csv")
    fits = {name: smf.ols(f, data=data).fit() for name, f in FORMULAS.items()}

    print("\n== coefficients match R's lm ==")
    for name, fit in fits.items():
        r = pd.read_csv(R_OUT / f"coef_{name}.csv")
        r_map = dict(zip(r["term"], r["estimate"]))
        py = {k.replace("(Intercept)", "Intercept"): v for k, v in fit.params.items()}
        r_map = {k.replace("(Intercept)", "Intercept"): v for k, v in r_map.items()}
        if set(py) != set(r_map):
            report(f"coef_{name}", False, f"{sorted(py)} vs {sorted(r_map)}")
            continue
        close([py[k] for k in sorted(py)], [r_map[k] for k in sorted(r_map)],
              "estimate", f"coef_{name}")

    print("\n== vcovHC, every type ==")
    for name, fit in fits.items():
        for ty in HC_TYPES:
            r = pd.read_csv(R_OUT / f"vcov_{name}_{ty}.csv", index_col=0)
            py = vcov_hc(fit.model.exog, fit.resid, fit.df_resid, ty)
            close(py, r.to_numpy(dtype=float), ty, f"vcov_{name}")

    print("\n== the symbolic pieces ==")
    pulses = ts.pulse_calculator({"x": 0, "l_1_x": 1}, {"l_1_y": 1}, limit=5)
    for dx, dy, h in [(0, 0, -1), (0, 0, 0), (1, 0, -1), (0, 1, -1),
                      (1, 0, 1), (2, 0, -1), (0, 2, -1)]:
        tag = f"V_general_dx{dx}_dy{dy}_h{h}"
        r = pd.read_csv(R_OUT / f"{tag}_shocks.csv")
        g = ts.general_calculator(dx, dy, h, 5, pulses)
        py = [w for s in range(6) for w in g.shocks[s]]
        close(py, r["weight"], "weights", tag)

    for tag, args in [
        ("V_gecm_to_adl", ({"l_1_x": 1}, {"l_1_y": 1},
                           {"d_x": 0, "l_1_d_x": 1}, {"l_1_d_y": 1})),
        ("V_gecm_to_adl_longer", ({"l_1_x": 1}, {"l_1_y": 1},
                                  {"d_x": 0, "l_1_d_x": 1, "l_2_d_x": 2},
                                  {"l_1_d_y": 1, "l_2_d_y": 2})),
        ("V_gecm_to_adl_minimal", ({"l_1_x": 1}, {"l_1_y": 1},
                                   {"d_x": 0}, None)),
    ]:
        r_lines = (R_OUT / f"{tag}.txt").read_text().strip().splitlines()
        out = ts.gecm_to_adl(*args)
        py_lines = []
        for key, side in (("x_vrbl_adl", "x"), ("y_vrbl_adl", "y")):
            terms = " | ".join(f"{k} = {v}" for k, v in out[key].items())
            py_lines.append(f"{side}: {terms}")
        report(tag, py_lines == r_lines,
               "" if py_lines == r_lines else f"\n    py: {py_lines}\n    R : {r_lines}")

    print("\n== ADL in levels ==")
    adl = dict(x_vrbl={"x": 0, "l_1_x": 1}, y_vrbl={"l_1_y": 1}, s_limit=10)
    compare_estimates("A_adl_pulse", ts.gdrf_adl(fits["levels"], shock_history="pulse", **adl))
    compare_estimates("B_adl_step", ts.gdrf_adl(fits["levels"], shock_history="step", **adl))
    compare_estimates("C_adl_pulse_cumulative",
                      ts.gdrf_adl(fits["levels"], shock_history="pulse", cumulative=True, **adl))
    compare_estimates("D_adl_h1", ts.gdrf_adl(fits["levels"], shock_history=1, **adl))

    print("\n== x in differences ==")
    dx_args = dict(x_vrbl={"d_x": 0, "l_1_d_x": 1}, y_vrbl={"l_1_y": 1},
                   d_x=1, d_y=0, s_limit=10)
    compare_estimates("E_adl_dx_levels",
                      ts.gdrf_adl(fits["dx"], shock_history="pulse",
                                  inferences_x="levels", **dx_args))
    compare_estimates("F_adl_dx_differences",
                      ts.gdrf_adl(fits["dx"], shock_history="pulse",
                                  inferences_x="differences", **dx_args))

    print("\n== y in differences ==")
    dy_args = dict(x_vrbl={"x": 0, "l_1_x": 1}, y_vrbl={"l_1_d_y": 1},
                   d_x=0, d_y=1, s_limit=10)
    compare_estimates("G_adl_dy_levels",
                      ts.gdrf_adl(fits["dy"], shock_history="pulse",
                                  inferences_y="levels", **dy_args))
    compare_estimates("H_adl_dy_differences",
                      ts.gdrf_adl(fits["dy"], shock_history="pulse",
                                  inferences_y="differences", **dy_args))

    print("\n== custom shock histories ==")
    history = [1, 2, 3, 0, 0, 4, 0, 0, 0, 1, 1]
    compare_estimates("I_adl_custom",
                      ts.gdrf_adl(fits["levels"], shock_history=history, **adl))
    compare_estimates("I2_adl_custom_cumulative",
                      ts.gdrf_adl(fits["levels"], shock_history=history,
                                  cumulative=True, **adl))

    print("\n== fitted values ==")
    compare_estimates("J_adl_fitted_steadystate",
                      ts.gdrf_adl(fits["levels"], shock_history="pulse",
                                  effect_type="fitted",
                                  prediction_values={"x": 1, "l_1_x": 1}, **adl))
    compare_estimates("K_adl_fitted_baseline",
                      ts.gdrf_adl(fits["dy"], shock_history="pulse",
                                  effect_type="fitted", baseline_y=3,
                                  baseline_y_se=2.5, shock_size=2, **dy_args))

    print("\n== GECM ==")
    gecm = dict(x_vrbl={"l_1_x": 1}, y_vrbl={"l_1_y": 1},
                x_d_vrbl={"d_x": 0, "l_1_d_x": 1}, y_d_vrbl={"l_1_d_y": 1},
                s_limit=10)
    compare_estimates("L_gecm_pulse", ts.gdrf_gecm(fits["gecm"], shock_history="pulse", **gecm))
    compare_estimates("M_gecm_step", ts.gdrf_gecm(fits["gecm"], shock_history="step", **gecm))
    compare_estimates(
        "N_gecm_longer_lags",
        ts.gdrf_gecm(fits["gecm2"], x_vrbl={"l_1_x": 1}, y_vrbl={"l_1_y": 1},
                     x_d_vrbl={"d_x": 0, "l_1_d_x": 1, "l_2_d_x": 2},
                     y_d_vrbl={"l_1_d_y": 1, "l_2_d_y": 2},
                     shock_history="step", s_limit=10),
    )

    print("\n== finite distributed lag ==")
    compare_estimates(
        "O_fdl_pulse",
        ts.gdrf_adl(fits["fdl"], x_vrbl={"x": 0, "l_1_x": 1, "l_2_x": 2},
                    y_vrbl=None, shock_history="pulse", s_limit=10),
    )

    print("\n== interactions ==")
    inter = dict(
        x_vrbl={"x": 0, "l_1_x": 1},
        z_vrbl={"z": 0, "l_1_z": 1},
        x_z_vrbl={"x_z": 0, "z_l_1_x": 1, "x_l_1_z": 0, "l_1_x_l_1_z": 1},
        y_vrbl={"l_1_y": 1},
        z_vals=[-1, 0, 1],
        s_limit=10,
    )
    compare_estimates("P_interact_zlines",
                      ts.interact_adl(fits["inter"], **inter), value_col="Effect")
    compare_estimates("Q_interact_step",
                      ts.interact_adl(fits["inter"], shock_history="step", **inter),
                      value_col="Effect")
    compare_estimates("R_interact_cumulative",
                      ts.interact_adl(fits["inter"], shock_history="impulse",
                                      cumulative=True, **inter),
                      value_col="Effect")
    compare_estimates(
        "S_interact_fitted",
        ts.interact_adl(fits["inter"], effect_type="fitted",
                        prediction_values={"x": 1, "l_1_x": 1, "x_z": 1,
                                           "z_l_1_x": 1, "x_l_1_z": 1,
                                           "l_1_x_l_1_z": 1}, **inter),
        value_col="Effect",
    )

    print("\n== robust standard errors end to end ==")
    for ty in HC_TYPES:
        compare_estimates(
            f"T_adl_pulse_{ty}",
            ts.gdrf_adl(fits["levels"], x_vrbl={"x": 0, "l_1_x": 1},
                        y_vrbl={"l_1_y": 1}, shock_history="step",
                        se_type=ty, s_limit=6),
        )

    print("\n== non-default confidence level ==")
    compare_estimates(
        "U_adl_step_level80",
        ts.gdrf_adl(fits["levels"], x_vrbl={"x": 0, "l_1_x": 1},
                    y_vrbl={"l_1_y": 1}, shock_history="step",
                    level=0.80, s_limit=6),
    )

    print(f"\n{checks - len(failures)}/{checks} checks passed")
    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("Every value matches the R package.")
    return 0


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    raise SystemExit(main())
