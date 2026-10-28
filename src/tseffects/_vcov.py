"""Heteroskedasticity-consistent covariance matrices.

The R package reads its covariance matrix off ``sandwich::vcovHC``, and accepts
every type that function does.  ``statsmodels`` only ships HC0-HC3, so the whole
family is implemented here against the same formulae ``sandwich`` uses:

.. math:: V = (X'X)^{-1} X' \\Omega X (X'X)^{-1}

with :math:`\\Omega = \\operatorname{diag}(\\omega_i)` and, writing
:math:`h_i` for the leverages and :math:`p = \\sum_i h_i = k`,

====== ==========================================================
type   :math:`\\omega_i`
====== ==========================================================
const  :math:`\\hat\\sigma^2 = \\sum_i e_i^2 / \\mathrm{df}`
HC0    :math:`e_i^2`
HC1    :math:`e_i^2 \\, n / \\mathrm{df}`
HC2    :math:`e_i^2 / (1 - h_i)`
HC3    :math:`e_i^2 / (1 - h_i)^2`
HC4    :math:`e_i^2 / (1 - h_i)^{\\delta_i}`, :math:`\\delta_i = \\min(4, n h_i / p)`
HC4m   :math:`e_i^2 / (1 - h_i)^{\\delta_i}`, :math:`\\delta_i = \\min(1, n h_i / p) + \\min(1.5, n h_i / p)`
HC5    :math:`e_i^2 / \\sqrt{(1 - h_i)^{\\alpha_i}}`, :math:`\\alpha_i = \\min(n h_i / p, \\max(4, 0.7 n h_{\\max} / p))`
====== ==========================================================

``HC`` is an alias for ``HC0``, as in ``sandwich``.
"""

from __future__ import annotations

import numpy as np

#: Accepted ``se_type`` values, matching the R package's own check.
VCOV_TYPES = (
    "const",
    "HC",
    "HC0",
    "HC1",
    "HC2",
    "HC3",
    "HC4",
    "HC4m",
    "HC5",
)


def _leverages(X: np.ndarray, xtx_inv: np.ndarray) -> np.ndarray:
    """Diagonal of the hat matrix, without forming the full n x n matrix."""
    return np.einsum("ij,jk,ik->i", X, xtx_inv, X)


def vcov_hc(exog, resid, df_resid: float, type_: str = "const") -> np.ndarray:
    """Covariance matrix of the OLS coefficients.

    Parameters
    ----------
    exog : array_like, shape (n, k)
        Design matrix, including the constant column.
    resid : array_like, shape (n,)
        OLS residuals.
    df_resid : float
        Residual degrees of freedom, ``n - k`` for an unweighted fit.
    type_ : str
        One of :data:`VCOV_TYPES`.
    """
    if type_ not in VCOV_TYPES:
        raise ValueError(
            f"Invalid se_type {type_!r}. se_type must be one of: "
            + ", ".join(VCOV_TYPES)
        )
    if type_ == "HC":
        type_ = "HC0"

    X = np.asarray(exog, dtype=float)
    e = np.asarray(resid, dtype=float).ravel()
    n, k = X.shape
    if e.shape[0] != n:
        raise ValueError("exog and resid imply different sample sizes")

    xtx_inv = np.linalg.pinv(X.T @ X)

    if type_ == "const":
        sigma2 = float(e @ e) / float(df_resid)
        return sigma2 * xtx_inv

    h = _leverages(X, xtx_inv)
    p = h.sum()  # equals k for a full-rank design
    e2 = e**2

    if type_ == "HC0":
        omega = e2
    elif type_ == "HC1":
        omega = e2 * n / float(df_resid)
    elif type_ == "HC2":
        omega = e2 / (1.0 - h)
    elif type_ == "HC3":
        omega = e2 / (1.0 - h) ** 2
    elif type_ == "HC4":
        delta = np.minimum(4.0, n * h / p)
        omega = e2 / (1.0 - h) ** delta
    elif type_ == "HC4m":
        ratio = n * h / p
        delta = np.minimum(1.0, ratio) + np.minimum(1.5, ratio)
        omega = e2 / (1.0 - h) ** delta
    elif type_ == "HC5":
        ratio = n * h / p
        alpha = np.minimum(ratio, np.maximum(4.0, 0.7 * n * h.max() / p))
        omega = e2 / np.sqrt((1.0 - h) ** alpha)
    else:  # pragma: no cover - guarded above
        raise ValueError(type_)

    meat = X.T @ (X * omega[:, None])
    return xtx_inv @ meat @ xtx_inv
