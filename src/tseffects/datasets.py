"""The two datasets shipped with the package.

Both come from the R package, unchanged.
"""

from __future__ import annotations

from importlib import resources

import pandas as pd

__all__ = ["load_toy_data", "load_approval"]


def _read(name: str) -> pd.DataFrame:
    with resources.as_file(resources.files("tseffects") / "data" / name) as path:
        return pd.read_csv(path)


def load_toy_data() -> pd.DataFrame:
    """Simulated, well-behaved interactive time series data.

    Fifty periods and 32 columns: ``x``, ``z`` and ``y`` in levels, their first
    five lags, first and second differences, and the four interactions of x and
    z with their lags (``x_z``, ``x_l_1_z``, ``z_l_1_x``, ``l_1_x_l_1_z``).
    Every transformation is pre-computed, which is what makes it convenient for
    worked examples: the coefficient names are the column names.

    Examples
    --------
    >>> from tseffects import load_toy_data
    >>> load_toy_data().shape
    (50, 32)
    """
    return _read("toy_ts_interaction_data.csv")


def load_approval() -> pd.DataFrame:
    """US presidential approval, 1981-2016, 140 quarters.

    From Cavari, Amnon (2019), "Evaluating the President on Your Priorities:
    Issue Priorities, Policy Performance, and Presidential Approval,
    1981-2016", *Presidential Studies Quarterly* 49(4): 798-826,
    `doi:10.1111/psq.12594 <https://doi.org/10.1111/psq.12594>`_.

    Columns include overall approval (``APPROVE``) and approval on the economy
    and foreign affairs, issue salience (``MIP_MACROECONOMICS``,
    ``MIP_FOREIGN``), macropartisanship, consumer sentiment (``UMCSENT``), the
    unemployment rate (``UNRATE``), and dummies for divided government,
    election years and the honeymoon period.

    Examples
    --------
    >>> from tseffects import load_approval
    >>> load_approval().shape
    (140, 14)
    """
    return _read("approval.csv")
