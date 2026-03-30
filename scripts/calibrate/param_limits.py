"""
Load per-parameter box limits for ``optimize_brb_mse`` / ``optimize_generalized_brb_mse`` from
``config/calibration/params_limits.csv`` (see ``calibration_paths.PARAM_LIMITS_CSV``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from calibrate.calibration_paths import PARAM_LIMITS_CSV


def _parse_bound_cell(x: object, *, col: str, parameter: str, path: Path) -> float:
    """Parse a single lower/upper cell to ``float``; ``inf`` tokens allowed."""
    if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x):
        raise ValueError(f"{path}: empty {col!r} for parameter {parameter!r}")
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("inf", "+inf", "+infinity"):
            return float("inf")
        if s in ("-inf", "-infinity"):
            return float("-inf")
    return float(x)


def _resolve_columns(df: pd.DataFrame, path: Path) -> tuple[str, str, str]:
    """Map flexible column names to ``(parameter, lower, upper)`` source columns."""
    cols = {str(c).strip().lower(): c for c in df.columns}
    param_src = None
    for key in ("parameter", "param", "name"):
        if key in cols:
            param_src = cols[key]
            break
    if param_src is None:
        raise ValueError(
            f"{path}: expected a column named 'parameter', 'param', or 'name'; got {list(df.columns)}"
        )
    lower_src = None
    for key in ("lower", "min", "lo", "l"):
        if key in cols:
            lower_src = cols[key]
            break
    upper_src = None
    for key in ("upper", "max", "hi", "h", "u"):
        if key in cols:
            upper_src = cols[key]
            break
    if lower_src is None or upper_src is None:
        raise ValueError(
            f"{path}: need 'lower' (or min/lo) and 'upper' (or max/hi); got {list(df.columns)}"
        )
    return param_src, lower_src, upper_src


def load_param_limits(path: Path | None = None) -> dict[str, tuple[float, float]]:
    """
    Read ``parameter, lower, upper`` rows. Raises if the file is missing, empty, or invalid.
    """
    csv_path = Path(path).expanduser().resolve() if path is not None else PARAM_LIMITS_CSV
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Parameter limits CSV not found: {csv_path}. "
            "Create config/calibration/params_limits.csv or pass --param-limits."
        )
    df = pd.read_csv(csv_path, comment="#")
    if df.empty:
        raise ValueError(f"No data rows in {csv_path} (after comment lines).")
    pcol, lcol, ucol = _resolve_columns(df, csv_path)
    out: dict[str, tuple[float, float]] = {}
    seen: set[str] = set()
    for idx, row in df.iterrows():
        name = str(row[pcol]).strip()
        if not name or name.lower() == "nan":
            raise ValueError(f"{csv_path}: row {idx}: missing parameter name")
        if name in seen:
            raise ValueError(f"{csv_path}: duplicate parameter {name!r}")
        seen.add(name)
        lo = _parse_bound_cell(row[lcol], col="lower", parameter=name, path=csv_path)
        hi = _parse_bound_cell(row[ucol], col="upper", parameter=name, path=csv_path)
        if not np.isfinite(lo) or not np.isfinite(hi):
            pass  # allow one-sided inf if desired; lbfgsb_reparam handles (-inf, inf)
        if hi <= lo:
            raise ValueError(
                f"{csv_path}: parameter {name!r}: need upper > lower (got {lo}, {hi})"
            )
        out[name] = (lo, hi)
    return out


def bounds_dict_for(
    params_to_optimize: list[str],
    *,
    limits_path: Path | None = None,
) -> dict[str, tuple[float, float]]:
    """Map each name in ``params_to_optimize`` to limits from CSV, or ``(-inf, inf)`` if omitted."""
    limits = load_param_limits(limits_path)
    return {name: limits.get(name, (-np.inf, np.inf)) for name in params_to_optimize}
