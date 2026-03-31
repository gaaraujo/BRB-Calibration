"""
Evaluate a SteelMPF parameter CSV (same columns as ``initial_brb_parameters.csv``): simulate,
write the same metrics as the calibration pipeline, per-specimen overlays, and combined
normalized montages per ``set_id``.

Standalone: lives under ``config/test/``. Not invoked by ``run.ps1``.

Usage (from repository root)::

    python config/test/run_eval_params_metrics.py --params config/test/params.csv

Defaults write under ``config/test/``:
  - ``params_eval_metrics.csv`` (override with ``--metrics-csv``)
  - ``simulated_force/{Name}_set{set_id}_simulated.csv``
  - ``overlays/{Name}_set{set_id}_force_def*.png``
  - ``overlays/set{set_id}_combined_force_def_norm.png``
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Repository root (parent of ``config/``)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_SCRIPTS = _PROJECT_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS / "postprocess"))

from calibrate.amplitude_mse_partition import (  # noqa: E402
    build_amplitude_weights,
    energy_scale_s_e,
)
from calibrate.calibration_io import metrics_dataframe  # noqa: E402
from calibrate.calibration_loss_settings import load_calibration_loss_settings  # noqa: E402
from calibrate.calibration_paths import CALIBRATION_LOSS_SETTINGS_CSV  # noqa: E402
from calibrate.cycle_feature_loss import (  # noqa: E402
    deformation_scale_s_d,
    load_p_y_kip_catalog,
)
from calibrate.digitized_unordered_eval_lib import compute_unordered_cloud_metrics  # noqa: E402
from calibrate.optimize_brb_mse import (  # noqa: E402
    AMPLITUDE_WEIGHTS_ARG_HELP,
    DEBUG_PARTITION,
    FAILURE_PENALTY,
    _loss_weight_snapshot,
    _metrics_dict_for_breakdown,
    force_scale_s_f,
    save_simulated_force_history_csv,
    simulate_and_loss_breakdown,
    _row_to_sim_params,
)
from calibrate.plot_compare_calibration_overlays import (  # noqa: E402
    _discover_simulated_index,
    _order_specimens,
    plot_combined_for_set,
)
from calibrate.plot_params_vs_filtered import (  # noqa: E402
    COLOR_NUMERICAL_COHORT,
    plot_force_def_overlays,
)
from calibrate.specimen_weights import catalog_metrics_fields  # noqa: E402
from model.corotruss import run_simulation  # noqa: E402
from postprocess.cycle_points import find_cycle_points, load_cycle_points_resampled  # noqa: E402
from specimen_catalog import (  # noqa: E402
    path_ordered_resampled_force_csv_stems,
    read_catalog,
    resolve_resampled_force_deformation_csv,
)

DEFAULT_PARAMS = _SCRIPT_DIR / "params.csv"
DEFAULT_OUT_DIR = _SCRIPT_DIR


def main() -> None:
    p = argparse.ArgumentParser(
        description="Simulate + metrics + overlays for a parameter CSV (config/test helper).",
    )
    p.add_argument(
        "--params",
        type=Path,
        default=DEFAULT_PARAMS,
        help=f"Parameter CSV (Name, set_id, SteelMPF columns). Default: {DEFAULT_PARAMS}",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (simulated_force/, overlays/; metrics CSV default path). Default: {DEFAULT_OUT_DIR}",
    )
    p.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help=(
            "Path for the metrics CSV (default: <out-dir>/params_eval_metrics.csv). "
            "Parent directories are created if needed."
        ),
    )
    p.add_argument(
        "--loss-settings",
        type=Path,
        default=None,
        help=f"Loss weights CSV. Default: {CALIBRATION_LOSS_SETTINGS_CSV}",
    )
    p.add_argument(
        "--amplitude-weights",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=AMPLITUDE_WEIGHTS_ARG_HELP,
    )
    p.add_argument(
        "--grid-cols",
        type=int,
        default=3,
        help="Columns in combined normalized montage figures (default: 3).",
    )
    args = p.parse_args()

    params_path = Path(args.params).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    loss_csv = (
        Path(args.loss_settings).expanduser().resolve()
        if args.loss_settings
        else CALIBRATION_LOSS_SETTINGS_CSV
    )
    loss = load_calibration_loss_settings(loss_csv)
    use_amp_w = (
        bool(args.amplitude_weights)
        if args.amplitude_weights is not None
        else loss.use_amplitude_weights
    )

    sim_dir = out_dir / "simulated_force"
    overlays_dir = out_dir / "overlays"
    metrics_path = (
        Path(args.metrics_csv).expanduser().resolve()
        if args.metrics_csv is not None
        else out_dir / "params_eval_metrics.csv"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    sim_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    if not params_path.is_file():
        raise SystemExit(f"Missing parameters CSV: {params_path}")

    params_df = pd.read_csv(params_path)
    for col in ("Name", "set_id"):
        if col not in params_df.columns:
            raise SystemExit(f"CSV must include column {col!r}: {params_path}")

    catalog_df = read_catalog()
    catalog_by_name = catalog_df.set_index("Name")
    catalog_names = catalog_df["Name"].astype(str).tolist()
    resampled_stems = path_ordered_resampled_force_csv_stems(catalog_df, project_root=_PROJECT_ROOT)

    rows_out: list[dict[str, Any]] = []

    for _, prow in params_df.iterrows():
        sid = str(prow["Name"]).strip()
        set_id = prow.get("set_id", "?")
        if sid not in resampled_stems:
            print(f"  skip {sid}: no resampled force_deformation.csv or not in catalog pipeline")
            continue

        csv_path = resolve_resampled_force_deformation_csv(sid, _PROJECT_ROOT)
        if csv_path is None or not csv_path.is_file():
            print(f"  skip {sid}: missing resampled data")
            continue

        df = pd.read_csv(csv_path)
        if "Force[kip]" not in df.columns or "Deformation[in]" not in df.columns:
            print(f"  skip {sid}: missing Force[kip] or Deformation[in]")
            continue

        D_exp = df["Deformation[in]"].to_numpy(dtype=float)
        F_exp = df["Force[kip]"].to_numpy(dtype=float)

        loaded = load_cycle_points_resampled(sid)
        points, _segments = loaded if loaded is not None else find_cycle_points(df)
        _mse_weights, amp_meta = build_amplitude_weights(
            D_exp,
            points,
            p=loss.amplitude_weight_power,
            eps=loss.amplitude_weight_eps,
            debug_partition=DEBUG_PARTITION,
            use_amplitude_weights=use_amp_w,
        )
        s_f_ref = force_scale_s_f(F_exp)
        s_d_ref = deformation_scale_s_d(D_exp)
        s_e_ref = energy_scale_s_e(D_exp, F_exp)
        n_cycles = len(amp_meta)

        p_y_catalog = load_p_y_kip_catalog(
            _PROJECT_ROOT,
            sid,
            float(prow["fyp"]),
            float(prow["A_sc"]),
        )
        cm = catalog_metrics_fields(sid, catalog_by_name)
        exp_landmark_cache: dict = {}

        try:
            F_sim = np.asarray(run_simulation(D_exp, **_row_to_sim_params(prow)), dtype=float)
        except Exception as exc:
            print(f"  {sid} set {set_id}: simulation failed: {exc}")
            continue

        if F_sim.shape != F_exp.shape:
            print(f"  {sid} set {set_id}: length mismatch sim vs exp")
            continue

        bd = simulate_and_loss_breakdown(
            D_exp,
            F_exp,
            F_sim,
            amp_meta,
            s_d=s_d_ref,
            loss=loss,
            fy_ksi=float(prow["fyp"]),
            a_sc=float(prow["A_sc"]),
            L_T=float(prow["L_T"]),
            L_y=float(prow["L_y"]),
            A_t=float(prow["A_t"]),
            E_ksi=float(prow["E"]),
            exp_landmark_cache=exp_landmark_cache,
        )
        if bd is None:
            print(f"  {sid} set {set_id}: loss breakdown failed")
            continue

        jtot = bd.j_total
        cloud = compute_unordered_cloud_metrics(D_exp, F_exp, D_exp, F_sim)
        mi = _metrics_dict_for_breakdown(bd, loss, "initial")
        mf = _metrics_dict_for_breakdown(bd, loss, "final")

        save_simulated_force_history_csv(
            sim_dir,
            sid,
            set_id,
            D_exp,
            F_exp,
            F_sim,
        )

        norm_xy_half = None
        try:
            plot_force_def_overlays(
                sid,
                D_exp,
                F_exp,
                F_sim,
                fy_ksi=float(prow["fyp"]),
                A_c_in2=float(prow["A_sc"]),
                L_y_in=float(prow["L_y"]),
                set_id=set_id,
                out_dir=overlays_dir,
                norm_xy_half=norm_xy_half,
                numerical_color=COLOR_NUMERICAL_COHORT,
                show_numerical_curve=True,
            )
        except Exception as exc:
            print(f"  {sid} set {set_id}: overlay plot failed: {exc}")

        rows_out.append(
            {
                "Name": sid,
                "set_id": set_id,
                "specimen_weight": 1.0,
                "contributes_to_aggregate": True,
                **cm,
                "weight_config": "params_eval",
                "calibration_stage": "params_eval",
                "aggregate_by_set_id": False,
                **mi,
                **mf,
                **_loss_weight_snapshot(loss),
                "S_F": s_f_ref,
                "S_D": s_d_ref,
                "S_E": s_e_ref,
                "P_y_ref": p_y_catalog,
                "n_cycles": n_cycles,
                "success": jtot < FAILURE_PENALTY * 0.5,
            }
        )
        print(f"  {sid} set {set_id}: J_total={jtot:.6g}  J_binenv={cloud.J_binenv:.6g}")

    metrics_dataframe(rows_out).to_csv(metrics_path, index=False)
    print(f"Wrote metrics: {metrics_path} ({len(rows_out)} rows)")

    idx = _discover_simulated_index(sim_dir)
    if idx:
        for set_id in sorted(idx.keys()):
            specimen_names = _order_specimens(idx[set_id], catalog_names)
            ok = plot_combined_for_set(
                int(set_id),
                specimen_names,
                overlays_dir,
                params_path,
                sim_dir,
                catalog_df,
                grid_cols=int(args.grid_cols),
                stage_weight_fn=None,
            )
            if ok:
                print(f"Wrote combined overlay: {overlays_dir / f'set{set_id}_combined_force_def_norm.png'}")
    else:
        print("No simulated_force/*_simulated.csv found; skipping combined montages.")


if __name__ == "__main__":
    main()
