"""
Combined normalized overlays like ``plot_compare_calibration_overlays`` for **generalized** steel,
but with a **single** ``(b_p, b_n)`` per ``set_id``: the **specimen-weighted mean** over **training**
rows only (``contributes_to_aggregate``, ``success``, finite ``final_J_total`` in
``generalized_params_eval_metrics.csv`` — same notion of contributor as the parameter summary reports).

Writes ``{Name}_set{k}_simulated.csv`` under ``generalized_train_mean_bn_bp_simulated_force/`` and
``set{k}_combined_force_def_norm.png`` under ``overlays_train_mean_bn_bp/`` (defaults from
``calibration_paths``).

Typical (after ``optimize_generalized_brb_mse.py``)::

    python scripts/calibrate/plot_generalized_train_mean_bn_bp_overlays.py

By default, every ``set_id`` present in ``generalized_brb_parameters.csv`` is processed; pass
``--sets`` to restrict (e.g. ``--sets 1-10``).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_SCRIPTS = _PROJECT_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from calibrate.calibration_paths import (  # noqa: E402
    BRB_SPECIMENS_CSV,
    GENERALIZED_BRB_PARAMETERS_PATH,
    GENERALIZED_PARAMS_EVAL_METRICS_PATH,
    GENERALIZED_TRAIN_MEAN_BN_BP_SIMULATED_FORCE_DIR,
    PLOTS_GENERALIZED_TRAIN_MEAN_BN_BP_OVERLAYS,
)
from calibrate.plot_compare_calibration_overlays import (  # noqa: E402
    GRID_COLS_GENERALIZED_AVERAGED,
    _discover_simulated_index,
    _order_specimens,
    plot_combined_for_set,
)
from calibrate.plot_params_vs_filtered import (  # noqa: E402
    write_one_specimen_simulated_csvs,
    write_unordered_simulated_csv_for_set,
)
from calibrate.specimen_weights import make_generalized_weight_fn  # noqa: E402
from specimen_catalog import (  # noqa: E402
    list_names_digitized_unordered,
    path_ordered_resampled_force_csv_stems,
    read_catalog,
)


def _parse_set_ids(spec: str) -> list[int]:
    spec = spec.strip().lower().replace(" ", "")
    if "-" in spec:
        a, b = spec.split("-", 1)
        lo, hi = int(a), int(b)
        if hi < lo:
            lo, hi = hi, lo
        return list(range(lo, hi + 1))
    return [int(x) for x in spec.split(",") if x]


def _unique_set_ids_from_params(params_df: pd.DataFrame) -> list[int]:
    """Sorted unique finite ``set_id`` values from generalized parameters CSV."""
    if "set_id" not in params_df.columns:
        raise SystemExit("Parameters CSV missing 'set_id' column")
    sid = pd.to_numeric(params_df["set_id"], errors="coerce")
    vals = sorted({int(x) for x in sid.dropna().unique().tolist() if np.isfinite(x)})
    return vals


def _as_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        return s.map(lambda v: str(v).lower() in ("true", "1", "yes")).astype(bool)
    return s.astype(bool)


def _generalized_contributing_mask(metrics_df: pd.DataFrame) -> pd.Series:
    jt = pd.to_numeric(metrics_df["final_J_total"], errors="coerce")
    return (
        _as_bool_series(metrics_df["contributes_to_aggregate"])
        & _as_bool_series(metrics_df["success"])
        & jt.notna()
        & np.isfinite(jt)
    )


def _weighted_mean_bn_bp_for_set(
    metrics_df: pd.DataFrame,
    params_df: pd.DataFrame,
    set_id: int,
) -> tuple[float, float] | None:
    sid = pd.to_numeric(metrics_df["set_id"], errors="coerce")
    m = _generalized_contributing_mask(metrics_df) & (sid == int(set_id))
    sub_m = metrics_df.loc[m, ["Name", "set_id", "specimen_weight"]].copy()
    if sub_m.empty:
        return None
    pcols = ["Name", "set_id", "b_p", "b_n"]
    miss = [c for c in pcols if c not in params_df.columns]
    if miss:
        raise SystemExit(f"generalized parameters CSV missing columns {miss}")
    merged = sub_m.merge(
        params_df[pcols].drop_duplicates(subset=["Name", "set_id"]),
        on=["Name", "set_id"],
        how="inner",
    )
    if merged.empty:
        return None
    w = pd.to_numeric(merged["specimen_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    bp = pd.to_numeric(merged["b_p"], errors="coerce").to_numpy(dtype=float)
    bn = pd.to_numeric(merged["b_n"], errors="coerce").to_numpy(dtype=float)
    ok = (w > 0.0) & np.isfinite(bp) & np.isfinite(bn)
    if not np.any(ok):
        return None
    sw = float(np.sum(w[ok]))
    if sw <= 0.0 or not math.isfinite(sw):
        return None
    bp_bar = float(np.sum(w[ok] * bp[ok]) / sw)
    bn_bar = float(np.sum(w[ok] * bn[ok]) / sw)
    return bp_bar, bn_bar


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--params",
        type=Path,
        default=GENERALIZED_BRB_PARAMETERS_PATH,
        help="generalized_brb_parameters.csv",
    )
    p.add_argument(
        "--metrics",
        type=Path,
        default=GENERALIZED_PARAMS_EVAL_METRICS_PATH,
        help="generalized_params_eval_metrics.csv",
    )
    p.add_argument(
        "--sim-dir",
        type=Path,
        default=GENERALIZED_TRAIN_MEAN_BN_BP_SIMULATED_FORCE_DIR,
        help="Output directory for *_set*_simulated.csv",
    )
    p.add_argument(
        "--overlays-dir",
        type=Path,
        default=PLOTS_GENERALIZED_TRAIN_MEAN_BN_BP_OVERLAYS,
        help="Output directory for set*_combined_force_def_norm.png",
    )
    p.add_argument(
        "--sets",
        type=str,
        default=None,
        help='set_id list, e.g. "1-10" or "7" or "1,3,5"; default: all set_id in --params CSV',
    )
    args = p.parse_args()

    params_path = Path(args.params).expanduser().resolve()
    metrics_path = Path(args.metrics).expanduser().resolve()
    sim_dir = Path(args.sim_dir).expanduser().resolve()
    overlays_dir = Path(args.overlays_dir).expanduser().resolve()

    if not params_path.is_file():
        raise SystemExit(f"Missing parameters CSV: {params_path}")
    if not metrics_path.is_file():
        raise SystemExit(f"Missing metrics CSV: {metrics_path}")

    for col in ("Name", "set_id", "specimen_weight", "contributes_to_aggregate", "success", "final_J_total"):
        if col not in pd.read_csv(metrics_path, nrows=0).columns:
            raise SystemExit(f"Metrics CSV missing column {col!r}: {metrics_path}")

    params_df = pd.read_csv(params_path)
    metrics_df = pd.read_csv(metrics_path)
    if args.sets is None:
        set_ids = _unique_set_ids_from_params(params_df)
        if not set_ids:
            raise SystemExit(f"No finite set_id values in parameters CSV: {params_path}")
    else:
        set_ids = _parse_set_ids(args.sets)
    catalog = read_catalog(BRB_SPECIMENS_CSV)
    catalog_names = catalog["Name"].astype(str).tolist()
    catalog_by_name = catalog.set_index("Name")
    resampled_stems = set(path_ordered_resampled_force_csv_stems(catalog, project_root=_PROJECT_ROOT))
    unordered_eligible = set(list_names_digitized_unordered(catalog))
    sim_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    stage_weight_fn = make_generalized_weight_fn(catalog)

    any_png = False
    for set_id in set_ids:
        bn_bp = _weighted_mean_bn_bp_for_set(metrics_df, params_df, set_id)
        if bn_bp is None:
            print(f"set {set_id}: no contributing train rows with b_p/b_n; skip")
            continue
        bp_bar, bn_bar = bn_bp
        sid_num = pd.to_numeric(params_df["set_id"], errors="coerce")
        one_set = params_df.loc[sid_num == int(set_id)].copy()
        if one_set.empty:
            print(f"set {set_id}: no parameter rows; skip")
            continue
        one_set["b_p"] = bp_bar
        one_set["b_n"] = bn_bar
        print(
            f"set {set_id}: train-weighted mean b_p={bp_bar:.6g}, b_n={bn_bar:.6g} "
            f"({len(one_set)} specimen rows) -> sim + overlay"
        )

        for name in sorted(one_set["Name"].astype(str).unique()):
            prow_block = one_set[one_set["Name"].astype(str) == name]
            if name not in catalog_by_name.index:
                continue
            cat_row = catalog_by_name.loc[name]
            if isinstance(cat_row, pd.DataFrame):
                cat_row = cat_row.iloc[0]
            if name in resampled_stems:
                write_one_specimen_simulated_csvs(
                    name,
                    prow_block,
                    cat_row,
                    sim_dir,
                )
            elif name in unordered_eligible:
                prow = prow_block.iloc[0]
                write_unordered_simulated_csv_for_set(
                    specimen_id=name,
                    set_id=set_id,
                    prow=prow,
                    catalog_by_name=catalog_by_name,
                    sim_dir=sim_dir,
                    project_root=_PROJECT_ROOT,
                )

        idx = _discover_simulated_index(sim_dir)
        if set_id not in idx or not idx[set_id]:
            print(f"set {set_id}: no simulated CSVs found under {sim_dir}; skip overlay")
            continue
        specimen_names = _order_specimens(idx[set_id], catalog_names)
        if plot_combined_for_set(
            set_id,
            specimen_names,
            overlays_dir,
            params_path,
            sim_dir,
            catalog,
            grid_cols=GRID_COLS_GENERALIZED_AVERAGED,
            stage_weight_fn=stage_weight_fn,
        ):
            any_png = True
            print(
                f"Wrote {overlays_dir / f'set{set_id}_combined_force_def_norm.png'} "
                f"({len(specimen_names)} specimens)"
            )

    if not any_png:
        print("No combined figures written.")


if __name__ == "__main__":
    main()
