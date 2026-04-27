"""Pure helpers for averaging SteelMPF columns over a parameters DataFrame (no OpenSees)."""
from __future__ import annotations

import math
from collections.abc import Callable, MutableMapping
from typing import Any

import numpy as np
import pandas as pd

# Segment slopes stay per specimen (individual optimization row) when merging shared steel so
# averaged/generalized stages stay comparable to runs with per-specimen b_p / b_n.
BP_BN_SPECIMEN_LOCAL: tuple[str, ...] = ("b_p", "b_n")


def _finite_float_from_cell(v: Any) -> float | None:
    try:
        if pd.isna(v):
            return None
    except TypeError:
        if v is None:
            return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def restore_individual_bp_bn(
    dst: MutableMapping[str, Any],
    src: pd.Series,
    *,
    skip: frozenset[str] | None = None,
) -> None:
    """
    Write ``b_p`` / ``b_n`` from ``src`` (individual parameters row) into ``dst`` when finite.

    ``skip``: parameter names to leave unchanged (e.g. jointly optimized ``b_p`` / ``b_n``).
    """
    sk = skip or frozenset()
    for k in BP_BN_SPECIMEN_LOCAL:
        if k in sk:
            continue
        if k not in src.index:
            continue
        fv = _finite_float_from_cell(src.get(k))
        if fv is None:
            continue
        dst[k] = fv


def shared_steel_optimize_param_names(active: list[str]) -> list[str]:
    """Names optimized jointly across specimens (excludes per-specimen ``b_p`` / ``b_n``)."""
    return [p for p in active if p not in BP_BN_SPECIMEN_LOCAL]


def generalized_joint_optimize_param_names(active: list[str]) -> list[str]:
    """
    Parameters optimized jointly in ``optimize_generalized_brb_mse``.

    Unlike ``shared_steel_optimize_param_names``, this includes ``b_p`` / ``b_n`` when they appear
    in the generalized-stage ``optimize_params`` list (one shared value for the training cohort).
    """
    return list(active)


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
    param_cols_by_set_id: dict[Any, list[str]] | None = None,
) -> dict[Any, pd.Series]:
    """
    Weighted mean of parameter columns per group. Rows with weight 0 are omitted from aggregates.

    If ``by_set_id``, returns ``set_id -> Series``; else ``{"_global": Series}``.

    When ``param_cols_by_set_id`` is set and ``by_set_id`` is true, each ``set_id`` group uses
    ``param_cols_by_set_id.get(sid, param_cols)`` for which columns are averaged (must exist in
    ``params_df``). Global mode (``by_set_id`` false) always uses ``param_cols``; the by-set map
    is ignored.
    """
    if "Name" not in params_df.columns:
        raise ValueError("params_df must have a Name column.")

    def _require_cols(cols: list[str], *, label: str) -> None:
        missing = [c for c in cols if c not in params_df.columns]
        if missing:
            raise ValueError(
                f"Parameters CSV missing columns required for averaging ({label}): {missing}"
            )

    _require_cols(param_cols, label="default param_cols")

    name_col = params_df["Name"].astype(str)
    w_all = _weights_for_group(name_col, weight_fn)
    train_mask = w_all > 0.0
    train = params_df.loc[train_mask].copy()
    if train.empty:
        raise ValueError("No rows with positive weight; cannot form averaged parameters.")

    if param_cols_by_set_id is not None and by_set_id:
        for cols in param_cols_by_set_id.values():
            _require_cols(cols, label=f"set_id override {cols!r}")

    out: dict[Any, pd.Series] = {}

    if by_set_id:
        for sid, grp in train.groupby(train["set_id"]):
            sid_key = int(pd.to_numeric(sid, errors="raise"))
            cols = (
                list(param_cols_by_set_id.get(sid_key, param_cols))
                if param_cols_by_set_id is not None
                else param_cols
            )
            finite_mask = grp[cols].apply(
                lambda r: bool(np.all(np.isfinite(r.to_numpy(dtype=float)))),
                axis=1,
            )
            grp_ok = grp.loc[finite_mask].copy()
            if grp_ok.empty:
                raise ValueError(
                    "No rows with positive weight and finite values for columns "
                    f"{cols} within set_id={sid!r}; cannot form averaged parameters."
                )
            w = _weights_for_group(grp_ok["Name"].astype(str), weight_fn)
            out[sid_key] = _weighted_mean_series(grp_ok, cols, w)
        if not out:
            raise ValueError("by_set_id averaging produced no groups.")
        return out

    finite_mask = train[param_cols].apply(
        lambda r: bool(np.all(np.isfinite(r.to_numpy(dtype=float)))),
        axis=1,
    )
    train_g = train.loc[finite_mask].copy()
    if train_g.empty:
        raise ValueError(
            "No rows with positive weight and finite averaged columns "
            f"{param_cols}; cannot form averaged parameters."
        )
    w = _weights_for_group(train_g["Name"].astype(str), weight_fn)
    out["_global"] = _weighted_mean_series(train_g, param_cols, w)
    return out


def merge_averaged_into_row(
    row: pd.Series,
    averaged: pd.Series,
    param_cols: list[str],
    *,
    skip_bp_bn_restore: frozenset[str] | None = None,
) -> pd.Series:
    """
    Copy ``row`` with entries in ``param_cols`` taken from ``averaged`` when present there.

    Skips keys missing from ``averaged`` (e.g. generalized vector without ``b_p`` / ``b_n``).
    By default restores ``b_p`` / ``b_n`` from ``row`` after merging. Pass ``skip_bp_bn_restore``
    with names that should keep merged values (jointly optimized ``b_p`` / ``b_n``).
    """
    z = row.copy()
    for c in param_cols:
        if c not in averaged.index:
            continue
        z[c] = float(averaged[c])
    sk = skip_bp_bn_restore or frozenset()
    restore_individual_bp_bn(z, row, skip=sk)
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
    try:
        sid_key = int(pd.to_numeric(set_id, errors="raise"))
    except (ValueError, TypeError):
        sid_key = set_id
    if sid_key in pool:
        return pool[sid_key]
    if set_id in pool:
        return pool[set_id]
    raise KeyError(
        f"No averaged parameters for set_id={set_id!r}. "
        f"Available set_ids from training pool: {sorted(pool.keys(), key=str)}"
    )
