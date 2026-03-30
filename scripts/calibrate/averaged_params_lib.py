"""Pure helpers for averaging SteelMPF columns over a parameters DataFrame (no OpenSees)."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd


def _weights_for_group(names: pd.Series, weight_fn: Callable[[str], float]) -> np.ndarray:
    """Per-row weights for ``names`` using ``weight_fn``."""
    return np.array([weight_fn(str(x)) for x in names], dtype=float)


def _weighted_mean_series(grp: pd.DataFrame, param_cols: list[str], w: np.ndarray) -> pd.Series:
    """Column-wise weighted mean of ``param_cols`` over ``grp``."""
    w = np.asarray(w, dtype=float)
    total_w = float(np.sum(w))
    if total_w <= 0.0:
        raise ValueError("Sum of positive weights is zero; cannot form weighted averaged parameters.")
    out: dict[str, float] = {}
    for c in param_cols:
        x = grp[c].to_numpy(dtype=float)
        out[c] = float(np.sum(w * x) / total_w)
    return pd.Series(out)


def compute_weighted_averaged_param_dict(
    params_df: pd.DataFrame,
    param_cols: list[str],
    *,
    by_set_id: bool,
    weight_fn: Callable[[str], float],
) -> dict[Any, pd.Series]:
    """
    Weighted mean of ``param_cols`` per group. Rows with weight 0 are omitted from aggregates.

    If ``by_set_id``, returns ``set_id -> Series``; else ``{"_global": Series}``.
    """
    if "Name" not in params_df.columns:
        raise ValueError("params_df must have a Name column.")
    missing = [c for c in param_cols if c not in params_df.columns]
    if missing:
        raise ValueError(f"Parameters CSV missing columns required for averaging: {missing}")

    name_col = params_df["Name"].astype(str)
    w_all = _weights_for_group(name_col, weight_fn)
    train_mask = w_all > 0.0
    train = params_df.loc[train_mask].copy()
    if train.empty:
        raise ValueError("No rows with positive weight; cannot form averaged parameters.")
    # Rows with NaN optimized parameters (e.g. catalog placeholders) must not enter the weighted mean.
    finite_mask = train[param_cols].apply(
        lambda r: bool(np.all(np.isfinite(r.to_numpy(dtype=float)))),
        axis=1,
    )
    train = train.loc[finite_mask].copy()
    if train.empty:
        raise ValueError(
            "No rows with positive weight and finite averaged columns "
            f"{param_cols}; cannot form averaged parameters."
        )

    out: dict[Any, pd.Series] = {}

    if by_set_id:
        for sid, grp in train.groupby(train["set_id"]):
            w = _weights_for_group(grp["Name"].astype(str), weight_fn)
            out[sid] = _weighted_mean_series(grp, param_cols, w)
        if not out:
            raise ValueError("by_set_id averaging produced no groups.")
        return out

    w = _weights_for_group(train["Name"].astype(str), weight_fn)
    out["_global"] = _weighted_mean_series(train, param_cols, w)
    return out


def merge_averaged_into_row(
    row: pd.Series,
    averaged: pd.Series,
    param_cols: list[str],
) -> pd.Series:
    """Copy ``row`` with ``param_cols`` taken from ``averaged``."""
    z = row.copy()
    for c in param_cols:
        z[c] = float(averaged[c])
    return z


def get_averaged_for_set_id(
    pool: dict[Any, pd.Series],
    set_id: Any,
    *,
    by_set_id: bool,
) -> pd.Series:
    """Return averaged ``Series`` for ``set_id``, or the global pool entry when ``by_set_id`` is false."""
    if not by_set_id:
        return pool["_global"]
    if set_id not in pool:
        raise KeyError(
            f"No averaged parameters for set_id={set_id!r}. "
            f"Available set_ids from training pool: {sorted(pool.keys(), key=str)}"
        )
    return pool[set_id]
