"""
Per-specimen flags and weights from ``BRB-Specimens.csv``.

- ``individual_optimize`` -- eligibility for ``optimize_brb_mse`` (with resampled data).
- ``averaged_weight`` / ``generalized_weight`` -- non-negative. **Unordered** digitized rows (``digitized`` +
  ``path_ordered=false``) always have effective averaged/generalized weight **0**. Path-ordered rows use the
  CSV values; missing cells default to **1.0**.
"""
from __future__ import annotations

import math
from typing import Callable

import pandas as pd

from specimen_catalog import (  # noqa: E402
    AVERAGED_WEIGHT_COL,
    GENERALIZED_WEIGHT_COL,
    INDIVIDUAL_OPTIMIZE_COL,
    _parse_bool_cell,
    get_specimen_record,
    read_catalog,
    uses_unordered_inputs,
)


def _effective_averaged_weight_from_row(row: pd.Series, cat: pd.DataFrame) -> float:
    """Catalog averaged weight; zero for unordered digitized specimens."""
    name = str(row["Name"]).strip()
    rec = get_specimen_record(name, cat)
    if uses_unordered_inputs(rec):
        return 0.0
    if AVERAGED_WEIGHT_COL in row.index and pd.notna(row.get(AVERAGED_WEIGHT_COL)):
        w = float(row[AVERAGED_WEIGHT_COL])
    else:
        w = 1.0
    if not math.isfinite(w) or w < 0.0:
        raise ValueError(f"Invalid averaged_weight for {name!r}: {w}")
    return w


def _effective_generalized_weight_from_row(row: pd.Series, cat: pd.DataFrame) -> float:
    """Catalog generalized weight; zero for unordered digitized specimens."""
    name = str(row["Name"]).strip()
    rec = get_specimen_record(name, cat)
    if uses_unordered_inputs(rec):
        return 0.0
    if GENERALIZED_WEIGHT_COL in row.index and pd.notna(row.get(GENERALIZED_WEIGHT_COL)):
        w = float(row[GENERALIZED_WEIGHT_COL])
    else:
        w = 1.0
    if not math.isfinite(w) or w < 0.0:
        raise ValueError(f"Invalid generalized_weight for {name!r}: {w}")
    return w


def _weight_map_from_catalog(cat: pd.DataFrame, *, generalized: bool) -> dict[str, float]:
    """``Name`` -> weight for averaged or generalized use."""
    fn = _effective_generalized_weight_from_row if generalized else _effective_averaged_weight_from_row
    return {str(r["Name"]).strip(): fn(r, cat) for _, r in cat.iterrows()}


def make_averaged_weight_fn(catalog: pd.DataFrame | None = None) -> Callable[[str], float]:
    """``Name`` -> non-negative weight for **averaged** mean of parameters."""
    cat = catalog if catalog is not None else read_catalog()
    m = _weight_map_from_catalog(cat, generalized=False)

    def fn(name: str) -> float:
        """Return averaged weight for ``name``."""
        return float(m.get(str(name).strip(), 0.0))

    return fn


def make_generalized_weight_fn(catalog: pd.DataFrame | None = None) -> Callable[[str], float]:
    """``Name`` -> non-negative weight for **generalized** optimization objective."""
    cat = catalog if catalog is not None else read_catalog()
    m = _weight_map_from_catalog(cat, generalized=True)

    def fn(name: str) -> float:
        """Return generalized weight for ``name``."""
        return float(m.get(str(name).strip(), 0.0))

    return fn


def make_weight_fn(catalog: pd.DataFrame | None = None) -> Callable[[str], float]:
    """Deprecated alias: same as ``make_averaged_weight_fn``."""
    return make_averaged_weight_fn(catalog)


def weight_for_name(name: str) -> float:
    """Resolved averaged weight for ``Name`` (uses current catalog)."""
    return make_averaged_weight_fn()(name)


def names_for_individual_optimize(catalog: pd.DataFrame | None = None) -> frozenset[str]:
    """Specimens with ``individual_optimize=true``."""
    cat = catalog if catalog is not None else read_catalog()
    out: set[str] = set()
    for name in cat["Name"].astype(str).unique():
        if get_specimen_record(str(name).strip(), cat).individual_optimize:
            out.add(str(name).strip())
    return frozenset(out)


def weight_config_tag(catalog: pd.DataFrame | None = None) -> str:
    """Short provenance string for metrics CSVs."""
    return "catalog_weights"


def catalog_metrics_fields(name: str, catalog_by_name: pd.DataFrame) -> dict[str, object]:
    """``individual_optimize`` flag for metrics rows."""
    n = str(name).strip()
    if n not in catalog_by_name.index:
        return {"individual_optimize": False}
    row = catalog_by_name.loc[n]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    if INDIVIDUAL_OPTIMIZE_COL not in row.index:
        return {"individual_optimize": False}
    try:
        io = _parse_bool_cell(row[INDIVIDUAL_OPTIMIZE_COL])
    except ValueError:
        io = False
    return {"individual_optimize": io}
