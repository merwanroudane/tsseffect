"""Name handling, model extraction and small shared helpers.

The R package relies on ``mpoly`` for its symbolic algebra, and ``mpoly``
refuses variable names containing ``.``, ``(``, ``)``, ``,`` or whitespace.
``tseffects:::mpoly.subber`` therefore rewrites coefficient names before any
formula is built and warns the user.  We keep exactly that behaviour so that
formulae printed by this package are comparable to the R ones, with one
addition: SymPy also rejects names that start with a digit, so those get an
underscore prefix.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ._vcov import VCOV_TYPES, vcov_hc

#: Names R / statsmodels / patsy use for the constant term.
INTERCEPT_ALIASES = ("(Intercept)", "Intercept", "const", "Const")

#: Internal name for the constant.  ``car::deltaMethod`` does the same rewrite.
INTERCEPT = "Intercept"

# The characters mpoly chokes on, paired with the label R puts in its warning.
_BAD_CHARS = ((".", "."), ("(", "("), (")", ")"), (",", ","), (" ", "whitespace"))


def sanitize_name(name: str) -> str:
    """Rewrite one coefficient name so it is a usable symbol.

    Mirrors ``mpoly.subber`` plus a guard for names starting with a digit.
    """
    if name in INTERCEPT_ALIASES:
        return INTERCEPT
    out = re.sub(r"[.(),\s]", "_", str(name))
    if out and out[0].isdigit():
        out = "_" + out
    return out


def sanitize_names(names, warn: bool = True) -> list[str]:
    """Rewrite a sequence of coefficient names, warning once per bad character."""
    names = [str(n) for n in names]
    non_intercept = [n for n in names if n not in INTERCEPT_ALIASES]
    if warn:
        for char, label in _BAD_CHARS:
            if any(char in n for n in non_intercept):
                warnings.warn(
                    f"Variable names containing {label} replaced with _",
                    stacklevel=3,
                )
    return [sanitize_name(n) for n in names]


def sanitize_vrbl(vrbl, warn: bool = True):
    """Rewrite the keys of a ``{name: lag}`` mapping, keeping order.

    ``None`` passes through, which is how the R functions signal "no lagged
    dependent variable" / "no interaction".
    """
    if vrbl is None:
        return None
    if isinstance(vrbl, (list, tuple)):
        # Allow [("x", 0), ("l_1_x", 1)] as well as a dict.
        vrbl = dict(vrbl)
    if not isinstance(vrbl, dict):
        raise TypeError(
            "variable specifications must be a mapping of {name: lag order}, "
            f"got {type(vrbl).__name__}"
        )
    keys = sanitize_names(vrbl.keys(), warn=warn)
    return {k: v for k, v in zip(keys, vrbl.values())}


def check_vrbl(vrbl, arg_name: str, allow_none: bool = False):
    """Validate a ``{name: lag}`` mapping the way the R dummy checks do."""
    if vrbl is None:
        if allow_none:
            return None
        raise ValueError(f"{arg_name} must be specified")
    if not vrbl:
        raise ValueError(f"{arg_name} must not be empty")
    for name, lag in vrbl.items():
        if not isinstance(name, str):
            raise ValueError(f"{arg_name} keys must be variable names (strings)")
        if isinstance(lag, bool) or not isinstance(lag, (int, np.integer)):
            raise ValueError(
                f"{arg_name} should map variable names to integer lag orders; "
                f"got {lag!r} for {name!r}"
            )
        if lag < 0:
            raise ValueError(f"{arg_name} lag orders must be non-negative")
    return vrbl


def by_lag(vrbl) -> dict[int, str]:
    """Invert ``{name: lag}`` to ``{lag: name}``.

    ``pulse.calculator`` looks variables up by lag order with
    ``which(x.vrbl == s)``, which silently takes the first match when two names
    share a lag order.  We keep that but say so out loud.
    """
    if vrbl is None:
        return {}
    out: dict[int, str] = {}
    for name, lag in vrbl.items():
        lag = int(lag)
        if lag in out:
            warnings.warn(
                f"Two variables share lag order {lag} ({out[lag]!r} and {name!r}); "
                f"using {out[lag]!r}. Combine them into one term if both belong "
                "in the effect.",
                stacklevel=3,
            )
            continue
        out[lag] = name
    return out


@dataclass
class ModelInfo:
    """Everything the effect calculations need out of a fitted linear model."""

    coef: dict[str, float]
    vcov: np.ndarray
    names: list[str] = field(default_factory=list)
    frame: pd.DataFrame | None = None

    def __post_init__(self):
        if not self.names:
            self.names = list(self.coef)
        self.vcov = np.asarray(self.vcov, dtype=float)
        if self.vcov.shape != (len(self.names), len(self.names)):
            raise ValueError(
                f"vcov is {self.vcov.shape} but there are {len(self.names)} "
                "coefficients"
            )

    @property
    def coef_vector(self) -> np.ndarray:
        return np.array([self.coef[n] for n in self.names], dtype=float)

    def require(self, vrbl, arg_name: str) -> None:
        """Raise if any name in ``vrbl`` is absent from the fitted model."""
        if vrbl is None:
            return
        missing = [n for n in vrbl if n not in self.coef]
        if missing:
            raise ValueError(
                f"{arg_name} not present in estimated model: {', '.join(missing)}"
            )

    def mean_of(self, var: str) -> float | None:
        """Mean of a model-frame column, used for unsupplied prediction values."""
        if self.frame is None or var not in self.frame.columns:
            return None
        return float(np.nanmean(np.asarray(self.frame[var], dtype=float)))


def as_model(model, se_type: str = "const", warn_names: bool = True) -> ModelInfo:
    """Coerce a fitted model into a :class:`ModelInfo`.

    Accepts a fitted ``statsmodels`` OLS result (the usual case), or a mapping /
    2-tuple of ``(coefficients, covariance matrix)`` for models fitted
    elsewhere.  With explicit coefficients the covariance is taken as given and
    ``se_type`` is ignored, since there is no design matrix to recompute it
    from.
    """
    if se_type not in VCOV_TYPES:
        raise ValueError(
            f"Invalid se_type {se_type!r}. se_type must be one of: "
            + ", ".join(VCOV_TYPES)
        )

    # (coef, vcov) supplied directly.
    if isinstance(model, tuple) and len(model) == 2:
        coef, vcov = model
        if isinstance(coef, pd.Series):
            names, values = list(coef.index), coef.to_numpy(dtype=float)
        elif isinstance(coef, dict):
            names, values = list(coef), np.array(list(coef.values()), dtype=float)
        else:
            raise TypeError("coefficients must be a dict or pandas Series")
        clean = sanitize_names(names, warn=warn_names)
        if isinstance(vcov, pd.DataFrame):
            vcov = vcov.to_numpy(dtype=float)
        return ModelInfo(
            coef=dict(zip(clean, values)), vcov=np.asarray(vcov, float), names=clean
        )

    params = getattr(model, "params", None)
    if params is None:
        raise TypeError(
            "model must be a fitted statsmodels regression result, or a "
            "(coefficients, vcov) tuple"
        )

    names = list(getattr(params, "index", []))
    values = np.asarray(params, dtype=float)
    if not names:
        names = list(getattr(getattr(model, "model", None), "exog_names", []))
    if len(names) != len(values):
        raise ValueError("could not recover coefficient names from the model")

    clean = sanitize_names(names, warn=warn_names)

    inner = getattr(model, "model", None)
    exog = getattr(inner, "exog", None)
    resid = getattr(model, "resid", None)
    if exog is not None and resid is not None:
        df_resid = float(getattr(model, "df_resid", len(resid) - np.shape(exog)[1]))
        vcov = vcov_hc(exog, resid, df_resid, se_type)
        frame = pd.DataFrame(np.asarray(exog, dtype=float), columns=clean)
    else:  # pragma: no cover - models without a design matrix
        if se_type != "const":
            raise ValueError(
                "se_type other than 'const' needs the model's design matrix and "
                "residuals; refit with statsmodels or pass the covariance matrix "
                "directly"
            )
        vcov = np.asarray(model.cov_params(), dtype=float)
        frame = None

    return ModelInfo(coef=dict(zip(clean, values)), vcov=vcov, names=clean, frame=frame)


def model_column(model_info: ModelInfo, name: str) -> np.ndarray:
    """Pull one column out of the model frame, by (sanitized) name."""
    if model_info.frame is None or name not in model_info.frame.columns:
        raise ValueError(
            f"{name!r} is not in the model frame, so its distribution cannot be "
            "used for defaults. Supply the values explicitly."
        )
    return np.asarray(model_info.frame[name], dtype=float)
