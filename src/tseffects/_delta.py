"""Delta-method evaluation of the effect formulae.

The R package hands each formula to ``car::deltaMethod``, which differentiates
it symbolically and reports

.. math::

    \\hat g = g(\\hat\\beta), \\qquad
    \\mathrm{SE}(\\hat g) = \\sqrt{\\nabla g(\\hat\\beta)' \\, V \\, \\nabla g(\\hat\\beta)}

with a normal confidence interval, since the input is a plain coefficient
vector rather than a model object.  :class:`DeltaFn` does the same with SymPy.

A formula may contain *extra* free symbols that are not coefficients -- in
practice ``z_val``, the value of the moderating variable in an interaction
model.  Those are held fixed (and treated as known, exactly as the R code does
by pasting the number into the formula string before calling ``deltaMethod``),
which lets one compiled :class:`DeltaFn` be reused across many values of z
instead of re-differentiating per value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp
from scipy.stats import norm


@dataclass(frozen=True)
class DeltaResult:
    """One evaluated quantity of interest."""

    estimate: float
    se: float
    lower: float
    upper: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.estimate, self.se, self.lower, self.upper)


def z_quantile(level: float) -> float:
    """The two-sided normal critical value ``car::deltaMethod`` uses."""
    if not 0.0 < level < 1.0:
        raise ValueError("level must be strictly between 0 and 1")
    return float(norm.ppf(1.0 - (1.0 - level) / 2.0))


class BatchDelta:
    """Many formulae, compiled together for delta-method evaluation.

    Compiling is the expensive step -- SymPy's ``lambdify`` generates and
    executes Python source -- and a response function needs one formula per
    period.  Doing them in a single pair of ``lambdify`` calls (one for the
    values, one for the Jacobian) instead of a pair per period is what keeps a
    twenty-period horizon fast.

    Parameters
    ----------
    exprs : sequence of sympy.Expr
        The quantities of interest, written in the model's coefficient names.
    names : sequence of str
        Coefficient names, in the same order as the rows and columns of the
        covariance matrix.
    extra : sequence of str, optional
        Free symbols that are *not* coefficients and are supplied at call time
        (in practice ``("z_val",)``, the value of a moderating variable).
        These are treated as known constants, so they carry no variance --
        which is what the R package does by substituting the number into the
        formula before calling ``deltaMethod``.
    substitutions : dict, optional
        Maps each name in ``names`` to an expression in the model's *actual*
        coefficients, which are then given in ``target_names``.  The chain rule
        is applied, so the standard errors are with respect to the real
        coefficients and their covariance matrix.

        This is how the GECM path stays cheap.  A GECM coefficient translates
        into an ADL coefficient that is a *sum* of GECM coefficients, and
        writing that sum into the response recursion raises it to the power of
        the horizon -- the expression grows exponentially.  Carrying a
        placeholder symbol instead keeps every formula the size it would be for
        a plain ADL, and because the translation is linear the chain-rule
        Jacobian is a constant matrix.
    target_names : sequence of str, optional
        The real coefficient names, required with ``substitutions``.
    """

    def __init__(self, exprs, names, extra=(), substitutions=None, target_names=None):
        self.exprs = [sp.sympify(e) for e in exprs]
        self.names = list(names)
        self.extra = list(extra)
        self.substitutions = (
            {k: sp.sympify(v) for k, v in substitutions.items()}
            if substitutions
            else None
        )
        if self.substitutions is not None:
            if target_names is None:
                raise ValueError("target_names is required with substitutions")
            missing = [n for n in self.names if n not in self.substitutions]
            if missing:
                raise ValueError(
                    "substitutions must cover every name: missing "
                    + ", ".join(missing)
                )
        self.target_names = list(target_names) if target_names else self.names

        coef_syms = [sp.Symbol(n) for n in self.names]
        extra_syms = [sp.Symbol(n) for n in self.extra]
        allowed = set(self.names) | set(self.extra)

        unknown = {
            s.name
            for e in self.exprs
            for s in e.free_symbols
            if s.name not in allowed
        }
        if unknown:
            raise ValueError(
                "formula references symbols that are neither coefficients nor "
                "supplied values: " + ", ".join(sorted(unknown))
            )

        # Only differentiate with respect to coefficients that appear in at
        # least one formula; the rest are a structural zero in every gradient.
        present = set().union(*(e.free_symbols for e in self.exprs)) if self.exprs else set()
        self._active = [i for i, s in enumerate(coef_syms) if s in present]
        active_syms = [coef_syms[i] for i in self._active]

        args = coef_syms + extra_syms
        jac = [[sp.diff(e, s) for s in active_syms] for e in self.exprs]

        # Common subexpression elimination matters for the nested formulae a
        # GECM produces, where the same sub-tree appears at every lag.
        self._f = sp.lambdify(args, self.exprs, modules="numpy", cse=True)
        self._g = (
            sp.lambdify(args, jac, modules="numpy", cse=True)
            if self._active
            else None
        )

        # The reparameterization and its Jacobian, for the chain rule.
        self._reparam = None
        if self.substitutions is not None:
            target_syms = [sp.Symbol(n) for n in self.target_names]
            sub_exprs = [self.substitutions[n] for n in self.names]
            unknown = {
                s.name
                for e in sub_exprs
                for s in e.free_symbols
                if s.name not in set(self.target_names)
            }
            if unknown:
                raise ValueError(
                    "substitutions reference unknown coefficients: "
                    + ", ".join(sorted(unknown))
                )
            sub_jac = [[sp.diff(e, s) for s in target_syms] for e in sub_exprs]
            self._reparam = (
                sp.lambdify(target_syms, sub_exprs, modules="numpy", cse=True),
                sp.lambdify(target_syms, sub_jac, modules="numpy", cse=True),
            )

    def __len__(self) -> int:
        return len(self.exprs)

    def _args(self, coef, extra_values):
        """Assemble the arguments, reparameterizing the coefficients if needed."""
        extra_values = extra_values or {}
        missing = [n for n in self.extra if n not in extra_values]
        if missing:
            raise ValueError("missing values for: " + ", ".join(missing))
        coef = np.asarray(coef, dtype=float)
        if self._reparam is not None:
            coef = np.asarray(self._reparam[0](*coef), dtype=float).ravel()
        return list(coef) + [float(extra_values[n]) for n in self.extra]

    def evaluate(
        self,
        coef: np.ndarray,
        vcov: np.ndarray,
        level: float = 0.95,
        extra_values=None,
    ) -> np.ndarray:
        """Estimates, standard errors and bounds, as an ``(n, 4)`` array."""
        args = self._args(coef, extra_values)
        n = len(self.exprs)

        estimates = np.asarray(self._f(*args), dtype=float).reshape(n)

        if self._g is None:
            ses = np.zeros(n, dtype=float)
        else:
            active = np.asarray(self._g(*args), dtype=float).reshape(n, -1)
            grad = np.zeros((n, len(self.names)), dtype=float)
            grad[:, self._active] = active
            if self._reparam is not None:
                # Chain rule: d(g)/d(theta) = d(a)/d(theta)' . d(g)/d(a).
                J = np.asarray(
                    self._reparam[1](*np.asarray(coef, dtype=float)), dtype=float
                ).reshape(len(self.names), len(self.target_names))
                grad = grad @ J
            V = np.asarray(vcov, dtype=float)
            # Row-wise g' V g, without building the full n x n product.
            var = np.einsum("ij,jk,ik->i", grad, V, grad)
            # Tiny negative values can appear from floating-point cancellation.
            ses = np.sqrt(np.maximum(var, 0.0))

        half = z_quantile(level) * ses
        return np.column_stack([estimates, ses, estimates - half, estimates + half])


class DeltaFn:
    """One formula compiled for repeated delta-method evaluation.

    A single-formula front end to :class:`BatchDelta`; see that class for the
    parameters.

    Examples
    --------
    >>> import numpy as np
    >>> fn = DeltaFn("a + 2*b", ["a", "b"])
    >>> out = fn(np.array([1.0, 3.0]), np.diag([0.25, 0.25]))
    >>> out.estimate, round(out.se, 6)
    (7.0, 1.118034)
    """

    def __init__(self, expr, names, extra=()):
        self.expr = sp.sympify(expr)
        self.names = list(names)
        self.extra = list(extra)
        self._batch = BatchDelta([self.expr], names, extra=extra)

    def __call__(
        self,
        coef: np.ndarray,
        vcov: np.ndarray,
        level: float = 0.95,
        extra_values=None,
    ) -> DeltaResult:
        """Evaluate the estimate, standard error and confidence interval."""
        row = self._batch.evaluate(coef, vcov, level, extra_values)[0]
        return DeltaResult(*(float(v) for v in row))


def delta_method(expr, model_info, level: float = 0.95, extra_values=None) -> DeltaResult:
    """One-shot delta method, for a single formula.

    ``DeltaFn`` is the right tool when the same formula is evaluated many times;
    this is the convenience wrapper for everything else.
    """
    extra = sorted(
        s.name for s in sp.sympify(expr).free_symbols if s.name not in model_info.coef
    )
    fn = DeltaFn(expr, model_info.names, extra=extra)
    return fn(model_info.coef_vector, model_info.vcov, level, extra_values)
