"""Publication-ready renderings of an effect table.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>

One table is computed once into a :class:`pandas.DataFrame` and rendered three
ways -- aligned text for the console, booktabs LaTeX for a manuscript, and
Markdown for the README and the documentation site -- so the three can never
disagree with each other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

__all__ = ["Summary", "stars_for", "build_frame"]

#: Significance thresholds and their markers, most demanding first.
STAR_LEVELS = ((0.01, "***"), (0.05, "**"), (0.10, "*"))


def stars_for(p_value: float) -> str:
    """The significance markers for a p-value, at 1, 5 and 10 per cent."""
    if not np.isfinite(p_value):
        return ""
    for threshold, marker in STAR_LEVELS:
        if p_value < threshold:
            return marker
    return ""


def build_frame(estimates: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Add z statistics, p-values and stars to an estimates table.

    The p-values are two-sided normal, matching the delta-method confidence
    intervals, which also use a normal quantile.
    """
    frame = estimates.copy()
    se = frame["SE"].to_numpy(dtype=float)
    est = frame[value_col].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, est / se, np.nan)
    p = 2.0 * (1.0 - norm.cdf(np.abs(z)))
    frame["z"] = z
    frame["p"] = np.where(np.isfinite(z), p, np.nan)
    frame["stars"] = [stars_for(v) for v in frame["p"]]
    return frame


def _fmt(value, width: int, decimals: int = 4) -> str:
    """A fixed-width number, or a centred dash when it is not available."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return f"{'--':>{width}}"
    return f"{float(value):>{width}.{decimals}f}"


class Summary:
    """An aligned effect table that prints itself.

    ``print(res.summary())`` and a bare ``res.summary()`` in a notebook both
    show the text table; :meth:`as_latex` and :meth:`as_markdown` render the
    same numbers for a manuscript or a web page.
    """

    WIDTH = 86

    def __init__(self, frame: pd.DataFrame, value_col: str, meta: dict, title: str):
        self.frame = frame
        self.value_col = value_col
        self.meta = meta
        self.title = title

    # ------------------------------------------------------------------ text
    def _header_block(self) -> list[str]:
        m = self.meta
        rows = [
            ("Dependent variable", m.get("y_name", "y"), "Observations", m.get("nobs", "--")),
            ("Shock variable", m.get("x_name", "x"), "Periods (s)", m.get("s_limit", "--")),
            ("Shock history", m.get("shock_label", "--"), "Shock size", m.get("shock_size", 1)),
            ("Model", m.get("model_label", "--"), "Std. errors", m.get("se_label", "const")),
        ]
        out = []
        for left_k, left_v, right_k, right_v in rows:
            out.append(
                f"{left_k:<21}: {str(left_v):<22} {right_k:<16}: {str(right_v):>10}"
            )
        return out

    def as_text(self) -> str:
        """The console table."""
        rule_eq = "=" * self.WIDTH
        rule_dash = "-" * self.WIDTH
        lines = [rule_eq, self.title.center(self.WIDTH), rule_eq]
        lines += self._header_block()
        lines.append(rule_dash)

        has_z = "Z" in self.frame.columns
        if has_z:
            head = (
                f"{'Period':>8}{'z value':>10}{'Effect':>12}{'Std.Err.':>11}"
                f"{'z':>9}{'P>|z|':>9}{'[95% Conf. Int.]':>22}"
            )
        else:
            head = (
                f"{'Period':>8}{'Effect':>14}{'Std.Err.':>12}"
                f"{'z':>10}{'P>|z|':>10}{'[95% Conf. Int.]':>24}"
            )
        lines += [head, rule_dash]

        special = []
        for _, row in self.frame.iterrows():
            period = str(row["Period"])
            body = (
                _fmt(row[self.value_col], 14 if not has_z else 12)
                + _fmt(row["SE"], 12 if not has_z else 11)
                + _fmt(row["z"], 10 if not has_z else 9, 3)
                + _fmt(row["p"], 10 if not has_z else 9, 3)
                + _fmt(row["Lower"], 12)
                + _fmt(row["Upper"], 12 if not has_z else 10)
                + f" {row['stars']}"
            )
            prefix = f"{period:>8}" + (_fmt(row["Z"], 10, 3) if has_z else "")
            # The long run and the baseline are set apart below their own rule.
            if period in ("LRM", "baseline"):
                special.append(prefix + body)
            else:
                lines.append(prefix + body)

        if special:
            lines.append(rule_dash)
            lines += special

        lines.append(rule_eq)
        lines.append("Significance: *** p<0.01, ** p<0.05, * p<0.10")
        lines.append(self.meta.get("se_note", ""))
        if self.meta.get("lrm_note"):
            lines.append(self.meta["lrm_note"])
        return "\n".join(line for line in lines if line is not None)

    def __str__(self) -> str:
        return self.as_text()

    def __repr__(self) -> str:
        return self.as_text()

    # ----------------------------------------------------------------- latex
    def as_latex(self, float_format: str = "%.4f", notes: bool = True) -> str:
        r"""A booktabs table, with standard errors under the estimates.

        Paste ``\newcommand{\sym}[1]{\ensuremath{^{#1}}}`` into the preamble
        for the significance markers.
        """
        has_z = "Z" in self.frame.columns
        ncol = 3 if has_z else 2
        spec = "l" * (ncol - 1) + "c"

        out = [
            "% Preamble: \\newcommand{\\sym}[1]{\\ensuremath{^{#1}}}",
            "% The notes block needs \\usepackage{threeparttable}; wrap the tabular",
            "% in \\begin{threeparttable} ... \\end{threeparttable}, or pass",
            "% notes=False and write the note yourself.",
            "\\begin{table}[htbp]\\centering",
            f"\\caption{{{_tex_escape(self.title)}}}\\label{{tab:tseffects}}",
            f"\\begin{{tabular}}{{{spec}}}",
            "\\toprule",
        ]
        header = ["Period"] + (["$z$ value"] if has_z else []) + ["Effect"]
        out.append(" & ".join(header) + " \\\\")
        out.append("\\midrule")

        for _, row in self.frame.iterrows():
            coef = float_format % float(row[self.value_col])
            se = float_format % float(row["SE"])
            cells = [_tex_escape(str(row["Period"]))]
            if has_z:
                cells.append(f"{float(row['Z']):.3f}")
            marker = f"\\sym{{{row['stars']}}}" if row["stars"] else ""
            out.append(" & ".join(cells + [f"{coef}{marker}"]) + " \\\\")
            blanks = [""] * (len(cells))
            out.append(" & ".join(blanks + [f"({se})"]) + " \\\\")

        out.append("\\midrule")
        out.append(
            f"Observations & {'& ' * (ncol - 2)}{self.meta.get('nobs', '--')} \\\\"
        )
        out.append("\\bottomrule")
        out.append("\\end{tabular}")
        if notes:
            out += [
                "\\begin{tablenotes}\\footnotesize",
                "\\item Standard errors in parentheses. \\sym{***} $p<0.01$, "
                "\\sym{**} $p<0.05$, \\sym{*} $p<0.10$.",
                f"\\item {_tex_escape(self.meta.get('se_note', ''))}",
                "\\end{tablenotes}",
            ]
        out.append("\\end{table}")
        return "\n".join(out)

    # -------------------------------------------------------------- markdown
    def as_markdown(self, float_format: str = "{:.4f}") -> str:
        """A Markdown table, for the README and the documentation site."""
        has_z = "Z" in self.frame.columns
        head = ["Period"] + (["z value"] if has_z else []) + [
            "Effect", "Std. Err.", "z", "P>\\|z\\|", "95% CI", ""
        ]
        align = ["---"] * len(head)
        rows = [" | ".join(head), " | ".join(align)]
        for _, row in self.frame.iterrows():
            cells = [str(row["Period"])]
            if has_z:
                cells.append(f"{float(row['Z']):.3f}")
            cells += [
                float_format.format(float(row[self.value_col])),
                float_format.format(float(row["SE"])),
                "--" if not np.isfinite(row["z"]) else f"{float(row['z']):.3f}",
                "--" if not np.isfinite(row["p"]) else f"{float(row['p']):.3f}",
                f"[{float(row['Lower']):.4f}, {float(row['Upper']):.4f}]",
                row["stars"],
            ]
            rows.append(" | ".join(cells))
        rows.append("")
        rows.append("*** p<0.01, ** p<0.05, * p<0.10. " + self.meta.get("se_note", ""))
        return "\n".join(rows)

    def as_html(self) -> str:
        """An HTML table, for embedding in a page."""
        return self.frame.to_html(index=False, float_format=lambda v: f"{v:.4f}")


def _tex_escape(text: str) -> str:
    """Escape the characters that break a LaTeX table.

    An underscore in a variable name is the first bug a real user hits.
    """
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text
