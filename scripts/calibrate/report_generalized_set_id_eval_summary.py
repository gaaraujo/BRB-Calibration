"""
Per-``set_id`` summary of generalized eval metrics (contributing path-ordered rows only).

Writes a CSV with, for each ``set_id``: specimen count, then for ``final_J_total``,
``final_J_feat_raw``, and ``final_J_E_raw``: **specimen_weight**-weighted mean (same idea as
aggregated metrics reports); largest/smallest specimen and raw values; weighted mean with
that largest (or smallest) specimen removed from the pool.

Also writes **train** vs **validation** CSVs for unordered cloud metrics ``J_binenv`` and
``J_binenv_L1`` (``final_unordered_J_binenv``, ``final_unordered_J_binenv_l1``): training rows are
generalized **contributors** (same mask as above); validation rows are all other specimens
in the metrics file for that ``set_id``. Train uses ``specimen_weight``; validation uses
uniform weights.

Used by ``optimize_generalized_brb_mse`` after ``generalized_params_eval_metrics.csv`` is
written (outputs default under ``summary_statistics/``); can also be run standalone on an existing metrics file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_SCRIPTS = _PROJECT_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from calibrate.calibration_paths import (  # noqa: E402
    GENERALIZED_PARAMS_EVAL_METRICS_PATH,
    GENERALIZED_SET_ID_EVAL_SUMMARY_CSV,
    GENERALIZED_UNORDERED_J_SUMMARY_TRAIN_CSV,
    GENERALIZED_UNORDERED_J_SUMMARY_VALIDATION_CSV,
)

_SUMMARY_FILENAME = "generalized_set_id_eval_summary.csv"

METRIC_SPECS: tuple[tuple[str, str], ...] = (
    ("final_J_total", "j_total"),
    ("final_J_feat_raw", "j_feat"),
    ("final_J_E_raw", "j_e"),
)

# Unordered cloud metrics: same rollup pattern as METRIC_SPECS (mean / largest / smallest / leave-one-out means).
UNORDERED_J_SPECS: tuple[tuple[str, str], ...] = (
    ("final_unordered_J_binenv", "J_binenv"),
    ("final_unordered_J_binenv_l1", "J_binenv_L1"),
)


def _human_unordered_j_columns(label: str) -> list[str]:
    """Spreadsheet-style headers for one unordered cloud metric label."""
    return [
        f"mean {label}",
        f"largest {label} specimen",
        f"largest indiv {label}",
        f"mean {label} without largest",
        f"smallest {label} specimen",
        f"smallest indiv {label}",
        f"mean {label} without smallest",
    ]


def _unordered_j_column_order() -> list[str]:
    cols = ["set_id", "num_specimens"]
    for _, lbl in UNORDERED_J_SPECS:
        cols.extend(_human_unordered_j_columns(lbl))
    return cols


def _as_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        return s.map(lambda v: str(v).lower() in ("true", "1", "yes")).astype(bool)
    return s.astype(bool)


def _contributing_mask(df: pd.DataFrame) -> pd.Series:
    jt = pd.to_numeric(df["final_J_total"], errors="coerce")
    return (
        _as_bool_series(df["contributes_to_aggregate"])
        & _as_bool_series(df["success"])
        & jt.notna()
        & np.isfinite(jt)
    )


def _weighted_mean(v: np.ndarray, w: np.ndarray) -> float:
    """``sum(w*v)/sum(w)`` over entries with finite v, finite w, w > 0."""
    m = np.isfinite(v) & np.isfinite(w) & (w > 0.0)
    if not np.any(m):
        return float("nan")
    sw = float(np.sum(w[m]))
    if sw <= 0.0:
        return float("nan")
    return float(np.sum(w[m] * v[m]) / sw)


def _block_stats(names: np.ndarray, values: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    """One metric block: weighted mean, argmax/argmin specimens, weighted means minus one outlier."""
    m = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(m):
        return {
            "mean": float("nan"),
            "largest_specimen": "",
            "largest_indiv": float("nan"),
            "mean_without_largest": float("nan"),
            "smallest_specimen": "",
            "smallest_indiv": float("nan"),
            "mean_without_smallest": float("nan"),
        }
    n = int(np.count_nonzero(m))
    nm = names[m]
    vm = values[m].astype(float)
    wm = weights[m].astype(float)
    mean_all = _weighted_mean(vm, wm)
    imax = int(np.argmax(vm))
    imin = int(np.argmin(vm))
    vmax = float(vm[imax])
    vmin = float(vm[imin])
    if n > 1:
        keep_max = np.ones(n, dtype=bool)
        keep_max[imax] = False
        mean_wo_max = _weighted_mean(vm[keep_max], wm[keep_max])
        keep_min = np.ones(n, dtype=bool)
        keep_min[imin] = False
        mean_wo_min = _weighted_mean(vm[keep_min], wm[keep_min])
    else:
        mean_wo_max = float("nan")
        mean_wo_min = float("nan")
    return {
        "mean": mean_all,
        "largest_specimen": str(nm[imax]),
        "largest_indiv": vmax,
        "mean_without_largest": mean_wo_max,
        "smallest_specimen": str(nm[imin]),
        "smallest_indiv": vmin,
        "mean_without_smallest": mean_wo_min,
    }


def build_set_id_eval_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Filter contributors and aggregate one row per ``set_id``."""
    required = {
        "Name",
        "set_id",
        "specimen_weight",
        "contributes_to_aggregate",
        "success",
        "final_J_total",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metrics frame missing columns: {sorted(missing)}")
    for col, _ in METRIC_SPECS:
        if col not in df.columns:
            raise ValueError(f"metrics frame missing column {col!r}")

    sub = df.loc[_contributing_mask(df)].copy()
    if sub.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for sid, g in sub.groupby("set_id", sort=True):
        rec: dict[str, Any] = {"set_id": sid, "num_specimens": int(len(g))}
        names = g["Name"].astype(str).to_numpy()
        w = pd.to_numeric(g["specimen_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        for col, key in METRIC_SPECS:
            v = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            stats = _block_stats(names, v, w)
            prefix = key
            rec[f"mean_{prefix}"] = stats["mean"]
            rec[f"largest_{prefix}_specimen"] = stats["largest_specimen"]
            rec[f"largest_indiv_{prefix}"] = stats["largest_indiv"]
            rec[f"mean_{prefix}_without_largest"] = stats["mean_without_largest"]
            rec[f"smallest_{prefix}_specimen"] = stats["smallest_specimen"]
            rec[f"smallest_indiv_{prefix}"] = stats["smallest_indiv"]
            rec[f"mean_{prefix}_without_smallest"] = stats["mean_without_smallest"]
        rows.append(rec)

    out = pd.DataFrame(rows)
    col_order = ["set_id", "num_specimens"]
    for _, key in METRIC_SPECS:
        col_order.extend(
            [
                f"mean_{key}",
                f"largest_{key}_specimen",
                f"largest_indiv_{key}",
                f"mean_{key}_without_largest",
                f"smallest_{key}_specimen",
                f"smallest_indiv_{key}",
                f"mean_{key}_without_smallest",
            ]
        )
    return out[col_order]


def _rows_unordered_j_split(df: pd.DataFrame, *, train: bool) -> list[dict[str, Any]]:
    """
    One row per ``set_id`` for either training (generalized contributors) or validation (all other rows).

    Training uses ``specimen_weight`` like the main generalized summary; validation uses uniform weights.
    """
    contrib = _contributing_mask(df)
    required = {
        "Name",
        "set_id",
        "specimen_weight",
        "contributes_to_aggregate",
        "success",
        "final_J_total",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metrics frame missing columns: {sorted(missing)}")
    for col, _ in UNORDERED_J_SPECS:
        if col not in df.columns:
            raise ValueError(f"metrics frame missing column {col!r}")

    rows: list[dict[str, Any]] = []
    for sid, g in df.groupby("set_id", sort=True):
        m = contrib.reindex(g.index).fillna(False).astype(bool)
        sub = g.loc[m] if train else g.loc[~m]
        n_eff = int(len(sub))
        rec: dict[str, Any] = {"set_id": sid, "num_specimens": n_eff}
        if n_eff == 0:
            for _, lbl in UNORDERED_J_SPECS:
                for h in _human_unordered_j_columns(lbl):
                    rec[h] = "" if "specimen" in h else float("nan")
            rows.append(rec)
            continue

        names = sub["Name"].astype(str).to_numpy()
        if train:
            w = pd.to_numeric(sub["specimen_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        else:
            w = np.ones(len(sub), dtype=float)

        for col_src, label in UNORDERED_J_SPECS:
            v = pd.to_numeric(sub[col_src], errors="coerce").to_numpy(dtype=float)
            stats = _block_stats(names, v, w)
            keys = _human_unordered_j_columns(label)
            rec[keys[0]] = stats["mean"]
            rec[keys[1]] = stats["largest_specimen"]
            rec[keys[2]] = stats["largest_indiv"]
            rec[keys[3]] = stats["mean_without_largest"]
            rec[keys[4]] = stats["smallest_specimen"]
            rec[keys[5]] = stats["smallest_indiv"]
            rec[keys[6]] = stats["mean_without_smallest"]
        rows.append(rec)
    return rows


def build_unordered_j_train_validation_summaries(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(train_summary, validation_summary)`` with spreadsheet-style column names."""
    cols = _unordered_j_column_order()
    if df.empty:
        empty = pd.DataFrame(columns=cols)
        return empty, empty
    train_rows = _rows_unordered_j_split(df, train=True)
    val_rows = _rows_unordered_j_split(df, train=False)
    tdf = pd.DataFrame(train_rows)
    vdf = pd.DataFrame(val_rows)
    for d in (tdf, vdf):
        for c in cols:
            if c not in d.columns:
                d[c] = np.nan
    return tdf[cols], vdf[cols]


def write_generalized_unordered_j_split_summaries(
    metrics: Path | pd.DataFrame,
    train_csv: Path | None = None,
    validation_csv: Path | None = None,
) -> tuple[Path, Path]:
    """
    Write train vs validation CSVs for ``J_binenv`` and ``J_binenv_L1`` (unordered cloud metrics).

    **Train** rows: same pool as ``build_set_id_eval_summary`` (``_contributing_mask``).
    **Validation** rows: all other generalized-eval rows for that ``set_id``.
    """
    train_csv = train_csv or GENERALIZED_UNORDERED_J_SUMMARY_TRAIN_CSV
    validation_csv = validation_csv or GENERALIZED_UNORDERED_J_SUMMARY_VALIDATION_CSV
    if isinstance(metrics, Path):
        df = pd.read_csv(metrics)
    else:
        df = metrics
    tdf, vdf = build_unordered_j_train_validation_summaries(df)
    train_csv = Path(train_csv).expanduser().resolve()
    validation_csv = Path(validation_csv).expanduser().resolve()
    train_csv.parent.mkdir(parents=True, exist_ok=True)
    validation_csv.parent.mkdir(parents=True, exist_ok=True)
    tdf.to_csv(train_csv, index=False)
    vdf.to_csv(validation_csv, index=False)
    return train_csv, validation_csv


def write_generalized_set_id_eval_summary(
    metrics: Path | pd.DataFrame,
    output_csv: Path | None = None,
) -> Path:
    """
    Build summary and write CSV. ``metrics`` may be a path to the generalized metrics CSV
    or an in-memory DataFrame (e.g. from ``optimize_generalized_brb_mse``).
    """
    if isinstance(metrics, Path):
        df = pd.read_csv(metrics)
        out = output_csv or metrics.with_name(_SUMMARY_FILENAME)
    else:
        df = metrics
        out = output_csv
    if out is None:
        raise ValueError("output_csv is required when metrics is a DataFrame")

    summary = build_set_id_eval_summary(df)
    out = Path(out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Write per-set_id J_total / J_feat / J_E summary CSV from generalized metrics.",
    )
    p.add_argument(
        "--metrics",
        type=Path,
        default=GENERALIZED_PARAMS_EVAL_METRICS_PATH,
        help="Input generalized_params_eval_metrics.csv",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=GENERALIZED_SET_ID_EVAL_SUMMARY_CSV,
        help=(
            f"Output CSV (default: {GENERALIZED_SET_ID_EVAL_SUMMARY_CSV} under summary_statistics/). "
            f"Pass a path to write next to --metrics instead."
        ),
    )
    args = p.parse_args()
    metrics_path = args.metrics.expanduser().resolve()
    out = write_generalized_set_id_eval_summary(metrics_path, args.output)
    print(f"Wrote {out} ({len(pd.read_csv(out))} set_id rows)")
    try:
        tp, vp = write_generalized_unordered_j_split_summaries(metrics_path)
        print(f"Wrote {tp} ({len(pd.read_csv(tp))} set_id rows)")
        print(f"Wrote {vp} ({len(pd.read_csv(vp))} set_id rows)")
    except Exception as exc:
        print(f"unordered J train/validation summaries skipped: {exc}")


if __name__ == "__main__":
    main()
