"""Colour scales, with MATLAB's Parula as the house default.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>

The same families as the R packages, with the same names and the same stops, so
a figure drawn here and a figure drawn there look identical.  ``"Parula"`` is
the default scale on every heatmap, surface and contour plot.

The one exception is a *diverging* effect surface centred on zero, where white
has to sit exactly at zero so the eye reads sign correctly.  There the default
is :func:`bluered_colors`, and :func:`diverging_cmap` builds the map.
"""

from __future__ import annotations

__all__ = [
    "PARULA_STOPS",
    "parula_colors",
    "parula_cmap",
    "matlab_jet_colors",
    "turbo_colors",
    "bluered_colors",
    "sinha_colors",
    "resolve_colorscale",
    "sequential_cmap",
    "diverging_cmap",
    "COLORSCALES",
]

#: The 64 RGB stops of MATLAB's R2014b Parula colormap.
PARULA_STOPS = (
    "#352a87", "#363093", "#3637a0", "#353dad", "#3243ba", "#2c4ac7",
    "#2053d4", "#0f5cdd", "#0363e1", "#0268e1", "#046de0", "#0871de",
    "#0d75dc", "#1079da", "#127dd8", "#1481d6", "#1485d4", "#1389d3",
    "#108ed2", "#0c93d2", "#0998d1", "#079ccf", "#06a0cd", "#06a4ca",
    "#06a7c6", "#07a9c2", "#0aacbe", "#0faeb9", "#15b1b4", "#1db3af",
    "#25b5a9", "#2eb7a4", "#38b99e", "#42bb98", "#4dbd91", "#59be8b",
    "#65c084", "#71c17d", "#7cc175", "#89c26d", "#95c265", "#a2c25c",
    "#aec254", "#bac24b", "#c5c343", "#d1c33b", "#dcc335", "#e6c32f",
    "#efc32b", "#f7c428", "#fcc726", "#feca26", "#fece26", "#fdd225",
    "#fcd626", "#fada26", "#f8de25", "#f6e225", "#f5e625", "#f5ea24",
    "#f6ee22", "#f7f221", "#f9f620", "#fafa1f",
)

#: MATLAB's classic ``jet``, kept for users migrating older figures.
JET_STOPS = (
    "#00007f", "#0000ff", "#007fff", "#00ffff", "#7fff7f", "#ffff00",
    "#ff7f00", "#ff0000", "#7f0000",
)

#: Google's ``turbo``: jet's dynamic range without its luminance artefacts.
TURBO_STOPS = (
    "#30123b", "#4145ab", "#4675ed", "#39a2fc", "#1bcfd4", "#24eca6",
    "#61fc6c", "#a4fc3b", "#d1e834", "#f3c63a", "#fe9b2d", "#f36315",
    "#d93806", "#b11901", "#7a0403",
)

#: A blue-white-red diverging ramp, for effects that change sign.
BLUERED_STOPS = (
    "#053061", "#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#ffffff",
    "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f",
)

#: The warm sequential ramp used in Sinha et al. (2023).
SINHA_STOPS = (
    "#2b1b47", "#5c2a6b", "#8e3680", "#bf4486", "#e45f80", "#f5856f",
    "#fcae63", "#fbd96e", "#f7f7a0",
)

COLORSCALES = {
    "Parula": PARULA_STOPS,
    "Jet": JET_STOPS,
    "Turbo": TURBO_STOPS,
    "BlueRed": BLUERED_STOPS,
    "Blue-Red": BLUERED_STOPS,
    "Sinha": SINHA_STOPS,
}


def _interpolate(stops, n: int) -> list[str]:
    """``n`` hex colours interpolated evenly along a list of stops."""
    from matplotlib.colors import LinearSegmentedColormap, to_hex

    if n < 1:
        raise ValueError("n must be at least 1")
    cmap = LinearSegmentedColormap.from_list("_tmp", list(stops), N=max(n, 2))
    if n == 1:
        return [to_hex(cmap(0.5))]
    return [to_hex(cmap(i / (n - 1))) for i in range(n)]


def parula_colors(n: int = 256) -> list[str]:
    """``n`` hex colours from MATLAB's R2014b Parula colormap.

    Examples
    --------
    >>> from tseffects import parula_colors
    >>> parula_colors(3)
    ['#352a87', '#33b8a1', '#fafa1f']
    """
    return _interpolate(PARULA_STOPS, n)


def parula_cmap(n: int = 256):
    """A matplotlib colormap from the Parula stops.

    Examples
    --------
    >>> from tseffects import parula_cmap
    >>> parula_cmap().name
    'parula'
    """
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("parula", list(PARULA_STOPS), N=n)


def matlab_jet_colors(n: int = 256) -> list[str]:
    """``n`` hex colours from MATLAB's classic ``jet`` colormap."""
    return _interpolate(JET_STOPS, n)


def turbo_colors(n: int = 256) -> list[str]:
    """``n`` hex colours from Google's ``turbo`` colormap."""
    return _interpolate(TURBO_STOPS, n)


def bluered_colors(n: int = 256) -> list[str]:
    """``n`` hex colours on a blue-white-red diverging ramp."""
    return _interpolate(BLUERED_STOPS, n)


def sinha_colors(n: int = 256) -> list[str]:
    """``n`` hex colours on the warm ramp of Sinha et al. (2023)."""
    return _interpolate(SINHA_STOPS, n)


def _stops(name) -> tuple:
    if not isinstance(name, str):
        raise TypeError("colour scale must be named by a string")
    key = name.strip()
    for candidate in (key, key.title(), key.capitalize()):
        if candidate in COLORSCALES:
            return COLORSCALES[candidate]
    raise ValueError(
        f"Unknown colour scale {name!r}. Choose one of: "
        + ", ".join(sorted(COLORSCALES))
    )


def resolve_colorscale(name: str = "Parula", n: int = 32) -> list:
    """A plotly-style colorscale, ``[[0.0, '#352a87'], ...]``.

    Examples
    --------
    >>> from tseffects import resolve_colorscale
    >>> resolve_colorscale("Parula", n=2)
    [[0.0, '#352a87'], [1.0, '#fafa1f']]
    """
    colors = _interpolate(_stops(name), n)
    last = max(len(colors) - 1, 1)
    return [[i / last, c] for i, c in enumerate(colors)]


def sequential_cmap(name: str = "Parula", n: int = 256):
    """A matplotlib colormap for a one-signed quantity."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, list(_stops(name)), N=n)


def diverging_cmap(name: str = "Blue-Red", n: int = 256):
    """A matplotlib colormap with white at the midpoint.

    For a named diverging family the stops are used as they are.  For a
    sequential family -- Parula, say -- the map is rebuilt as
    low-end / white / high-end, so zero still reads as white.
    """
    from matplotlib.colors import LinearSegmentedColormap

    stops = _stops(name)
    if name.strip().lower() in ("bluered", "blue-red"):
        return LinearSegmentedColormap.from_list(name, list(stops), N=n)
    return LinearSegmentedColormap.from_list(
        name, [stops[0], "#ffffff", stops[-1]], N=n
    )
