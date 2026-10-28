import pytest
import statsmodels.formula.api as smf

from tseffects import load_toy_data


@pytest.fixture(scope="session")
def data():
    return load_toy_data()


@pytest.fixture(scope="session")
def fit_levels(data):
    """ADL(1,1) with everything in levels."""
    return smf.ols("y ~ l_1_y + x + l_1_x", data=data).fit()


@pytest.fixture(scope="session")
def fit_dx(data):
    """ADL with the independent variable in first differences."""
    return smf.ols("y ~ l_1_y + d_x + l_1_d_x", data=data).fit()


@pytest.fixture(scope="session")
def fit_dy(data):
    """ADL with the dependent variable in first differences."""
    return smf.ols("d_y ~ l_1_d_y + x + l_1_x", data=data).fit()


@pytest.fixture(scope="session")
def fit_gecm(data):
    return smf.ols("d_y ~ l_1_y + l_1_d_y + l_1_x + d_x + l_1_d_x", data=data).fit()


@pytest.fixture(scope="session")
def fit_interact(data):
    return smf.ols(
        "y ~ l_1_y + x + l_1_x + z + l_1_z + x_z + z_l_1_x + x_l_1_z + l_1_x_l_1_z",
        data=data,
    ).fit()


@pytest.fixture(scope="session")
def fit_fdl(data):
    """A finite distributed lag model: no lagged dependent variable."""
    return smf.ols("y ~ x + l_1_x + l_2_x", data=data).fit()


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
