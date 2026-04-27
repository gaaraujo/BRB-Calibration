"""
Stable specimen colors for BRB-Calibration figures.

Colors are assigned in catalog ``ID`` sort order so each specimen keeps the same hue across
all plots. The palette uses matplotlib qualitative maps (``tab20``, ``tab20b``, ``tab20c``)
for the first 60 series, then extra ``turbo`` samples if needed.

Also exposes ``distinct_colors_rgba`` for non-specimen series (e.g. per-cycle overlays)
without repeating hues within a single figure.
"""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

_QUAL_POOL: list[tuple[float, float, float, float]] | None = None


def ordered_specimen_names(catalog: pd.DataFrame) -> list[str]:
    """Specimen names sorted by catalog ``ID`` (fallback: table order if ``ID`` missing)."""
    if catalog.empty or "Name" not in catalog.columns:
        return []
    if "ID" in catalog.columns:
        return catalog.sort_values("ID")["Name"].astype(str).tolist()
    return catalog["Name"].astype(str).tolist()


def _qualitative_rgba_pool() -> list[tuple[float, float, float, float]]:
    pool: list[tuple[float, float, float, float]] = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        cmap = plt.colormaps[cmap_name]
        colors_attr = getattr(cmap, "colors", None)
        if colors_attr is not None:
            for c in colors_attr:
                pool.append(mcolors.to_rgba(c))
            continue
        n = int(getattr(cmap, "N", 20))
        for i in range(n):
            pool.append(mcolors.to_rgba(cmap(i / max(n - 1, 1))))
    return pool


def _pool() -> list[tuple[float, float, float, float]]:
    global _QUAL_POOL
    if _QUAL_POOL is None:
        _QUAL_POOL = _qualitative_rgba_pool()
    return _QUAL_POOL


def distinct_colors_rgba(n: int) -> list[tuple[float, float, float, float]]:
    """Return ``n`` separated RGBA colors (e.g. one per cycle in a single-specimen debug plot)."""
    if n <= 0:
        return []
    pool = _pool()
    if n <= len(pool):
        return list(pool[:n])
    turbo = plt.colormaps["turbo"]
    n_extra = n - len(pool)
    tail = [
        mcolors.to_rgba(turbo(0.06 + 0.88 * (j + 1) / (n_extra + 1)))
        for j in range(n_extra)
    ]
    return list(pool) + tail


def specimen_color_by_name_map(catalog: pd.DataFrame) -> dict[str, tuple[float, float, float, float]]:
    """
    Map each catalog specimen name to RGBA. Order follows ``ordered_specimen_names``; colors do
    not repeat until more than 60 specimens (then ``turbo`` continues the list).
    """
    names = ordered_specimen_names(catalog)
    cols = distinct_colors_rgba(len(names))
    return {n: cols[i] for i, n in enumerate(names)}


def specimen_names_in_plot_order(catalog: pd.DataFrame, present: Iterable[str]) -> list[str]:
    """Catalog ID order restricted to names that appear in ``present``."""
    pset = {str(x) for x in present}
    return [n for n in ordered_specimen_names(catalog) if n in pset]
