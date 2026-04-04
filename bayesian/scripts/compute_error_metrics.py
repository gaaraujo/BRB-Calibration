"""
Compute path-ordered calibration metrics matching the main repo's ``optimize_brb_mse`` breakdown:

- ``J_feat`` L2 / L1 (cycle-weighted landmark means)
- ``J_E`` L2 / L1 (mean cycle dissipated-energy error, normalized by ``S_E = S_delta S_P``; JSON uses ``S_D``, ``S_F``, ``S_E``)
- ``J_binenv`` L2 / L1 (32-bin force upper/lower envelope mismatch)

Reads experimental ``force_deformation.csv``, simulated force ``predicted_force.csv`` (one row, same length),
and ``cycle_meta.json``. Prints a per-cycle table: ``delta [in]``, ``strain [%]`` (100 × max |δ| / ``L_y``), ``w_c``,
and per-cycle ``J_feat`` / ``J_E`` means (same definitions as the aggregate scalars). Optional: compare ``J_feat`` L2 to the flat weighted row from ``calibration_data.csv``
and ``results.out`` (``||r-c||^2 / sum w_c``).

Text output prints ``S_delta`` [in], ``S_P`` [kip], ``S_E`` [in·kip] first (same numbers as JSON ``S_D``, ``S_F``, ``S_E``).
``--json`` adds ``scale_units`` for ``S_D`` / ``S_F`` / ``S_E``.

  python scripts/compute_error_metrics.py
  python scripts/compute_error_metrics.py --json
  python scripts/compute_error_metrics.py --verify-rows
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from io import StringIO
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.error_metrics import (
    FAILURE_PENALTY,
    compute_per_cycle_metric_rows,
    compute_repo_style_metrics,
)
from lib.landmark_vector import (
    jfeat_l2_squared,
    load_cycle_meta_json,
    load_force_deformation_csv,
    sum_w_c_contributing_cycles,
)
from lib.specimen_config import dy_from_config, load_specimen_config, resolve_path


def _load_one_row_floats(path: Path) -> np.ndarray:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"empty file: {path}")
    line = raw.splitlines()[0]
    delim = "," if "," in line else None
    return np.loadtxt(StringIO(line), delimiter=delim, dtype=np.float64).ravel()


def _load_predicted_force_row(path: Path) -> np.ndarray:
    return _load_one_row_floats(path)


def _fmt_fixed(x: float, decimals: int) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x:.{decimals}f}"


def _fmt_scientific(x: float, decimals: int) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x:.{decimals}e}"


def _print_cycle_table(
    rows: list[dict[str, float | int]],
    *,
    l_y_in: float,
    s_e: float,
) -> None:
    print()
    print("Per-cycle metrics (same landmark / energy definitions as aggregate J_feat / J_E)")
    print(
        f"strain [%] = 100 × delta_max / L_y   (L_y = {l_y_in:g} in from specimen_config.yaml);  "
        f"J_E columns use global S_E = {s_e:.6g} in·kip"
    )
    hdr = (
        "cycle_id",
        "delta [in]",
        "strain [%]",
        "w_c",
        "J_feat L2",
        "J_feat L1",
        "J_E L2",
        "J_E L1",
    )
    # Fixed decimals per column; J_E L2 uses scientific notation (wide dynamic range).
    w = (10, 12, 10, 12, 14, 14, 16, 14)
    print(
        f"{hdr[0]:>{w[0]}}  {hdr[1]:>{w[1]}}  {hdr[2]:>{w[2]}}  {hdr[3]:>{w[3]}}  "
        f"{hdr[4]:>{w[4]}}  {hdr[5]:>{w[5]}}  {hdr[6]:>{w[6]}}  {hdr[7]:>{w[7]}}"
    )
    sep = "  ".join("-" * n for n in w)
    print(sep)
    for r in rows:
        cid = int(r["cycle_id"])
        dlt = float(r["delta_in"])
        st = float(r["strain_pct"])
        wc = float(r["w_c"])
        j2 = float(r["j_feat_l2_mean"])
        j1 = float(r["j_feat_l1_mean"])
        e2 = float(r["j_e_l2"])
        e1 = float(r["j_e_l1"])
        print(
            f"{cid:>{w[0]}}  {_fmt_fixed(dlt, 6):>{w[1]}}  {_fmt_fixed(st, 4):>{w[2]}}  {_fmt_fixed(wc, 6):>{w[3]}}  "
            f"{_fmt_fixed(j2, 6):>{w[4]}}  {_fmt_fixed(j1, 6):>{w[5]}}  {_fmt_scientific(e2, 6):>{w[6]}}  {_fmt_fixed(e1, 6):>{w[7]}}"
        )


# Human-readable labels (values are dimensionless unless noted).
_TEXT_SCALAR_LABELS: dict[str, str] = {
    "S_D": "S_delta [in]",
    "S_F": "S_P [kip]",
    "S_E": "S_E [in·kip]",
}

_SCALE_UNITS_JSON: dict[str, str] = {
    "S_D": "in",
    "S_F": "kip",
    "S_E": "in·kip",
}


def _json_sanitize_value(v: object) -> float | None:
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        fv = float(v)
        return None if not math.isfinite(fv) else fv
    if isinstance(v, (int, np.integer)):
        return float(int(v))  # type: ignore[return-value]
    return None


def main() -> None:
    default_cfg = _ROOT / "config" / "specimen_config.yaml"
    p = argparse.ArgumentParser(description="Compute repo-style calibration error metrics.")
    p.add_argument("--config", type=Path, default=default_cfg)
    p.add_argument(
        "--predicted-force",
        type=Path,
        default=_ROOT / "predicted_force.csv",
        help="One-row simulated force [kip], same length as force_deformation.csv",
    )
    p.add_argument("--n-binenv-bins", type=int, default=32)
    p.add_argument("--json", action="store_true", help="Print one JSON object on stdout")
    p.add_argument(
        "--no-cycle-table",
        action="store_true",
        help="Do not print the per-cycle metrics table (text) or include ``cycles`` (JSON)",
    )
    p.add_argument(
        "--verify-rows",
        action="store_true",
        help="Also load calibration_data.csv + results.out and print J_feat L2 from flat rows",
    )
    p.add_argument(
        "--calibration-row",
        type=Path,
        default=_ROOT / "calibration_data.csv",
    )
    p.add_argument(
        "--prediction-row",
        type=Path,
        default=_ROOT / "results.out",
    )
    args = p.parse_args()

    cfg = load_specimen_config(args.config)
    base = args.config.parent
    fd_path = resolve_path(cfg, "force_deformation", base)
    meta_path = resolve_path(cfg, "cycle_meta", base)

    D, F_exp = load_force_deformation_csv(fd_path)
    F_sim = _load_predicted_force_row(args.predicted_force.expanduser().resolve())
    if F_sim.size != F_exp.size:
        raise ValueError(
            f"predicted force length {F_sim.size} != experimental length {F_exp.size}"
        )

    meta, s_f_m, s_d_m, n_meta = load_cycle_meta_json(meta_path)
    if n_meta is not None and len(D) != n_meta:
        raise ValueError(f"n_meta={n_meta} len(D)={len(D)}")
    if s_f_m is None or s_d_m is None:
        raise ValueError(f"cycle_meta missing s_f/s_d: {meta_path}")

    dy = dy_from_config(cfg)
    fy_v = float(cfg["fy_ksi"])
    a_v = float(cfg["A_sc_in2"])
    s_f_v = float(s_f_m)
    s_d_v = float(s_d_m)
    sum_w_c = sum_w_c_contributing_cycles(
        D, F_exp, meta, fy=fy_v, a_sc=a_v, dy=dy, s_f=s_f_v, s_d=s_d_v
    )

    m = compute_repo_style_metrics(
        D,
        F_exp,
        F_sim,
        meta,
        fy=fy_v,
        a_sc=a_v,
        dy=dy,
        s_d=s_d_v,
        s_f=s_f_v,
        n_binenv_bins=args.n_binenv_bins,
    )

    out: dict[str, float | None] = {
        "S_D": m.s_d,
        "S_F": m.s_f,
        "S_E": m.s_e,
        "J_feat_L2": m.j_feat_l2,
        "J_feat_L1": m.j_feat_l1,
        "J_E_L2": m.j_e_l2,
        "J_E_L1": m.j_e_l1,
        "J_binenv_L2": m.binenv_l2,
        "J_binenv_L1": m.binenv_l1,
        "sum_w_c_contributing": sum_w_c,
    }

    if args.verify_rows:
        c_path = args.calibration_row.expanduser().resolve()
        r_path = args.prediction_row.expanduser().resolve()
        if c_path.is_file() and r_path.is_file():
            c = _load_one_row_floats(c_path)
            r = _load_one_row_floats(r_path)
            j_row = jfeat_l2_squared(c, r, sum_w_c)
            out["J_feat_L2_from_weighted_rows"] = j_row
        else:
            out["J_feat_L2_from_weighted_rows"] = None

    l_y_in = float(cfg["L_y_in"])
    cycle_rows = compute_per_cycle_metric_rows(
        D,
        F_exp,
        F_sim,
        meta,
        fy=fy_v,
        a_sc=a_v,
        dy=dy,
        s_d=s_d_v,
        s_f=s_f_v,
        l_y_in=l_y_in,
    )

    if args.json:
        serializable: dict[str, float | None | list] = {}
        for k, v in out.items():
            if v is None:
                serializable[k] = None
            elif isinstance(v, (float, np.floating)):
                fv = float(v)
                serializable[k] = None if not math.isfinite(fv) else fv
            else:
                serializable[k] = v  # type: ignore[assignment]
        if not args.no_cycle_table:
            serializable["cycles"] = [
                {
                    "cycle_id": int(r["cycle_id"]),
                    "start": int(r["start"]),
                    "end": int(r["end"]),
                    "delta_in": _json_sanitize_value(r["delta_in"]),
                    "strain_pct": _json_sanitize_value(r["strain_pct"]),
                    "w_c": _json_sanitize_value(r["w_c"]),
                    "j_feat_l2_mean": _json_sanitize_value(r["j_feat_l2_mean"]),
                    "j_feat_l1_mean": _json_sanitize_value(r["j_feat_l1_mean"]),
                    "j_e_l2": _json_sanitize_value(r["j_e_l2"]),
                    "j_e_l1": _json_sanitize_value(r["j_e_l1"]),
                }
                for r in cycle_rows
            ]
        serializable["scale_units"] = dict(_SCALE_UNITS_JSON)
        print(json.dumps(serializable, indent=2, allow_nan=False))
        return

    items = list(out.items())
    label_w = max(len(_TEXT_SCALAR_LABELS.get(k, k)) for k, _ in items)
    for k, v in items:
        label = _TEXT_SCALAR_LABELS.get(k, k)
        if v is None:
            print(f"{label:<{label_w}} : (skipped — missing files)")
        elif isinstance(v, float) and v == FAILURE_PENALTY:
            print(f"{label:<{label_w}} : {v} (failure penalty / shape error)")
        else:
            print(f"{label:<{label_w}} : {v}")

    if not args.no_cycle_table:
        _print_cycle_table(cycle_rows, l_y_in=l_y_in, s_e=float(m.s_e))


if __name__ == "__main__":
    main()
