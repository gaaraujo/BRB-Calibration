"""
Calibrate and plot one specimen using ``input.csv`` in this folder (SteelMPF only).

Reads brace geometry and nominal yield from ``config/calibration/BRB-Specimens.csv``.
SteelMPF seeds, optimizer bounds, and loss weights come from ``input.csv`` (default:
``scripts/calibrate_single/input.csv``). Experimental F-u from ``--force-deformation``.

Typical (from repository root)::

    python scripts/calibrate_single/calibrate_one_specimen.py STF01

Omit ``--force-deformation`` to use ``data/resampled/{Name}/force_deformation.csv`` when it exists.
With ``--prepare-data``, raw data must live under ``data/raw/{Name}/``; postprocess writes the
standard filtered/resampled tree, then calibration reads the resampled CSV (created if missing).

Apparent ``b_p`` / ``b_n`` diagnostics (slope overlays + segment histograms) are written under
``results/plots/calibration/single_specimen/{Name}/apparent_b/``.

Cycle partitioning, landmark overlays, cycle-weight maps, and per-cycle energy panels are written
under ``.../single_specimen/{Name}/cycles/`` (same figures as ``plot_cycle_landmarks_debug.py`` and
``plot_cycle_energy_debug.py`` in the main pipeline).
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
sys.path.insert(0, str(_SCRIPTS / "postprocess"))
sys.path.insert(0, str(_SCRIPT_DIR))

from calibrate.amplitude_mse_partition import (  # noqa: E402
    build_amplitude_weights,
    energy_scale_s_e,
)
from calibrate.calibration_io import metrics_dataframe  # noqa: E402
from calibrate.calibration_loss_settings import CalibrationLossSettings  # noqa: E402
from calibrate.cycle_feature_loss import (  # noqa: E402
    deformation_scale_s_d,
    load_p_y_kip_catalog,
)
from calibrate.extract_bn_bp import extract_bn_bp_one_specimen, get_b_lists_one_specimen  # noqa: E402
from calibrate.optimize_brb_mse import (  # noqa: E402
    DEBUG_PARTITION,
    FAILURE_PENALTY,
    _loss_weight_snapshot,
    _metrics_dict_for_breakdown,
    force_scale_s_f,
    optimize_one_specimen,
    plot_cycle_weight_hysteresis,
    save_simulated_force_history,
    save_simulated_force_history_csv,
)
from calibrate.plot_params_vs_filtered import run_one_specimen  # noqa: E402
from calibrate.specimen_weights import catalog_metrics_fields  # noqa: E402
from cycle_points import find_cycle_points, load_cycle_points_resampled, run_specimen  # noqa: E402
from filter_force import (  # noqa: E402
    _process_digitized_unordered,
    _process_path_ordered,
)
from load_input import (  # noqa: E402
    STEEL_MODEL,
    SingleCalibrateInput,
    load_single_calibrate_input,
)
from resample_filtered import (  # noqa: E402
    process_specimen,
    process_specimen_digitized_unordered,
)
from specimen_catalog import (  # noqa: E402
    get_specimen_record,
    list_names_digitized_unordered,
    read_catalog,
    resampled_force_deformation_csv,
    resolve_resampled_force_deformation_csv,
    uses_unordered_inputs,
)

DEFAULT_INPUT = _SCRIPT_DIR / "input.csv"
RESULTS_SINGLE = _PROJECT_ROOT / "results" / "calibration" / "single_specimen"
PLOTS_SINGLE = _PROJECT_ROOT / "results" / "plots" / "calibration" / "single_specimen"


def _finite_or(default: float, value: object) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _apparent_b_medians(
    specimen: str,
    df: pd.DataFrame,
    points: list[dict],
    cat_row: pd.Series,
    cfg: SingleCalibrateInput,
) -> tuple[float, float]:
    stats = extract_bn_bp_one_specimen(
        specimen,
        df,
        points,
        float(cat_row["L_T_in"]),
        float(cat_row["L_y_in"]),
        float(cat_row["A_c_in2"]),
        float(cat_row["A_t_in2"]),
        float(cat_row["f_yc_ksi"]),
    )
    b_p = _finite_or(cfg.default_b_p, stats.get("b_p_median", np.nan))
    b_n = _finite_or(cfg.default_b_n, stats.get("b_n_median", np.nan))
    return b_p, b_n


def _plot_apparent_b(specimen: str, cat_row: pd.Series, *, plots_base: Path) -> None:
    """Write per-specimen apparent-b slope overlay and segment histogram under ``plots_base/apparent_b/``."""
    from calibrate.plot_b_histograms_and_scatter import plot_histogram_one_specimen  # noqa: E402
    from calibrate.plot_b_slopes import (  # noqa: E402
        plot_one_digitized_unordered,
        plot_one_specimen,
    )

    apparent_root = plots_base / "apparent_b"
    slopes_dir = apparent_root / "b_slopes"
    hist_dir = apparent_root / "b_histograms"

    catalog = read_catalog()
    rec = get_specimen_record(specimen, catalog)
    is_unordered = uses_unordered_inputs(rec) or specimen in list_names_digitized_unordered(catalog)

    if not is_unordered:
        plot_one_specimen(specimen, cat_row, slopes_dir)
        slopes_png = slopes_dir / f"{specimen}.png"
        if slopes_png.is_file():
            print(f"  Wrote apparent-b slopes: {slopes_png}")
        else:
            print(f"  Skipped apparent-b slopes (no resampled F-u for {specimen!r})")
    elif plot_one_digitized_unordered(specimen, cat_row, slopes_dir):
        print(f"  Wrote apparent-b slopes: {slopes_dir / f'{specimen}.png'}")
    else:
        print(f"  Skipped apparent-b slopes (no digitized envelope for {specimen!r})")

    b_n_list, b_p_list = get_b_lists_one_specimen(specimen)
    if b_n_list or b_p_list:
        plot_histogram_one_specimen(specimen, b_n_list, b_p_list, hist_dir)
        print(f"  Wrote apparent-b histogram: {hist_dir / f'{specimen}.png'}")
    else:
        print(f"  Skipped apparent-b histogram (no segment b values for {specimen!r})")


def _plot_cycle_debug(
    specimen: str,
    set_id: int,
    D_exp: np.ndarray,
    F_exp: np.ndarray,
    F_sim: np.ndarray,
    amp_meta: list[dict],
    pointwise_weights: np.ndarray,
    prow: pd.Series,
    cat_row: pd.Series,
    *,
    plots_base: Path,
) -> None:
    """Cycle weights, J_feat landmarks, and per-cycle energy panels (path-ordered specimens only)."""
    from calibrate.cycle_feature_loss import LANDMARK_EXP_CSV_COLUMNS  # noqa: E402
    from calibrate.plot_cycle_energy_debug import plot_specimen_cycles  # noqa: E402
    from calibrate.plot_cycle_landmarks_debug import plot_landmark_overlay  # noqa: E402

    cycles_dir = plots_base / "cycles"
    cycles_dir.mkdir(parents=True, exist_ok=True)
    f_yc = float(cat_row["f_yc_ksi"])
    A_c = float(cat_row["A_c_in2"])
    L_y = float(cat_row["L_y_in"])

    plot_cycle_weight_hysteresis(
        specimen,
        set_id,
        D_exp,
        F_exp,
        pointwise_weights,
        amp_meta,
        cycles_dir,
        f_yc=f_yc,
        A_c=A_c,
        L_y=L_y,
    )
    print(f"  Wrote cycle weights: {cycles_dir / f'{specimen}_set{set_id}_cycle_weights.png'}")

    landmarks_png = cycles_dir / f"{specimen}_set{set_id}_landmarks.png"
    exp_csv_rows: list[dict] = []
    plot_landmark_overlay(
        specimen,
        set_id,
        D_exp,
        F_exp,
        F_sim,
        amp_meta,
        prow,
        landmarks_png,
        f_yc=f_yc,
        A_c=A_c,
        L_y=L_y,
        exp_csv_rows=exp_csv_rows,
    )
    print(f"  Wrote landmarks: {landmarks_png}")
    if exp_csv_rows:
        landmarks_csv = cycles_dir / f"{specimen}_set{set_id}_landmarks_exp.csv"
        pd.DataFrame(exp_csv_rows).reindex(columns=LANDMARK_EXP_CSV_COLUMNS).to_csv(
            landmarks_csv, index=False
        )
        print(f"  Wrote landmarks CSV: {landmarks_csv}")

    energy_png = cycles_dir / f"{specimen}_set{set_id}_cycle_energy_debug.png"
    plot_specimen_cycles(
        specimen,
        set_id,
        D_exp,
        F_exp,
        F_sim,
        amp_meta,
        energy_png,
        f_yc=f_yc,
        A_c=A_c,
        L_y=L_y,
    )
    print(f"  Wrote cycle energy: {energy_png}")


def _parameter_row(
    specimen: str,
    cat_row: pd.Series,
    cfg: SingleCalibrateInput,
    *,
    b_p: float,
    b_n: float,
) -> pd.Series:
    fy = float(cat_row["f_yc_ksi"])
    row = {
        "ID": int(cat_row["ID"]),
        "Name": specimen,
        "set_id": cfg.set_id,
        "steel_model": STEEL_MODEL,
        "L_T": float(cat_row["L_T_in"]),
        "L_y": float(cat_row["L_y_in"]),
        "A_sc": float(cat_row["A_c_in2"]),
        "A_t": float(cat_row["A_t_in2"]),
        "fyp": fy,
        "fyn": fy,
        "b_p": b_p,
        "b_n": b_n,
        **cfg.steel_seeds,
    }
    return pd.Series(row)


def _bounds_for_active(
    active: list[str], cfg: SingleCalibrateInput
) -> dict[str, tuple[float, float]]:
    missing = [p for p in active if p not in cfg.param_bounds]
    if missing:
        raise ValueError(
            f"No bounds in input.csv for optimized parameters: {missing}"
        )
    return {p: cfg.param_bounds[p] for p in active}


def _validate_specimen(specimen: str) -> tuple[pd.DataFrame, pd.Series]:
    catalog = read_catalog()
    by_name = catalog.set_index("Name")
    if specimen not in by_name.index:
        raise SystemExit(
            f"Unknown specimen {specimen!r}. Add it to config/calibration/BRB-Specimens.csv."
        )
    return catalog, by_name.loc[specimen]


def _resolve_force_deformation_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_absolute():
        raise SystemExit(f"--force-deformation must be an absolute path; got {path!r}")
    return resolved


def _default_force_deformation_path(specimen: str, *, prepare_data: bool) -> Path:
    """Default F-u CSV: resampled layout (same as optimize_brb_mse / plot_params_vs_filtered)."""
    found = resolve_resampled_force_deformation_csv(specimen, _PROJECT_ROOT)
    if found is not None:
        return found.resolve()
    canonical = resampled_force_deformation_csv(specimen, _PROJECT_ROOT).resolve()
    if prepare_data:
        return canonical
    raise SystemExit(
        f"No resampled force_deformation.csv for {specimen!r} "
        f"(expected {canonical}). "
        "Run with --prepare-data or pass --force-deformation."
    )


def _load_force_deformation_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise SystemExit(f"Force-deformation CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "Force[kip]" not in df.columns or "Deformation[in]" not in df.columns:
        raise SystemExit(f"{csv_path}: missing Force[kip] or Deformation[in] columns")
    return df


def _prepare_specimen_data(specimen: str, *, e_ksi: float) -> None:
    """Run cycle_points -> filter_force -> resample_filtered for one catalog specimen."""
    catalog = read_catalog()
    rec = get_specimen_record(specimen, catalog)
    catalog_by_name = catalog.set_index("Name")

    print(f"  Preparing data for {specimen!r}...")
    cp = run_specimen(specimen, save=True, overwrite=True)
    if cp is None:
        raise SystemExit(
            f"cycle_points failed for {specimen!r}: no valid raw F-u under data/raw/{specimen}/"
        )
    points, segments, wrote = cp
    print(
        f"    cycle_points: {len(points)} points, {len(segments)} segments"
        + (" (wrote JSON)" if wrote else " (JSON unchanged)")
    )

    if uses_unordered_inputs(rec):
        _process_digitized_unordered(specimen, catalog)
    else:
        _process_path_ordered(specimen, catalog)
    print("    filter_force: done")

    e_by_name = {specimen: e_ksi}
    if uses_unordered_inputs(rec):
        process_specimen_digitized_unordered(specimen, catalog_by_name, e_by_name, catalog)
    else:
        process_specimen(specimen, catalog_by_name, e_by_name, catalog)
    print("    resample_filtered: done")


def _cycle_points_for_csv(specimen: str, df: pd.DataFrame) -> list[dict]:
    loaded = load_cycle_points_resampled(specimen)
    if loaded is not None:
        points, _segments = loaded
        if len(points) > 0:
            return points
    return find_cycle_points(df)[0]


def calibrate_and_plot(
    specimen: str,
    force_deformation_csv: Path,
    cfg: SingleCalibrateInput,
    *,
    prepare_data: bool = False,
    out_dir: Path | None = None,
    plots_dir: Path | None = None,
    use_amplitude_weights: bool | None = None,
) -> Path:
    catalog, cat_row = _validate_specimen(specimen)

    csv_path = _resolve_force_deformation_path(force_deformation_csv)
    if prepare_data:
        _prepare_specimen_data(specimen, e_ksi=float(cfg.steel_seeds["E"]))

    df = _load_force_deformation_csv(csv_path)
    print(f"  Using force-deformation: {csv_path}")
    print(f"  Input settings: set_id={cfg.set_id}, steel_model={STEEL_MODEL}")

    D_exp = df["Deformation[in]"].to_numpy(dtype=float)
    F_exp = df["Force[kip]"].to_numpy(dtype=float)
    points = _cycle_points_for_csv(specimen, df)

    b_p, b_n = _apparent_b_medians(specimen, df, points, cat_row, cfg)
    print(f"  Apparent b seeds: b_p={b_p:.6g} (median), b_n={b_n:.6g} (median)")

    overlay_dir = plots_dir or (PLOTS_SINGLE / specimen)
    _plot_apparent_b(specimen, cat_row, plots_base=overlay_dir)

    prow = _parameter_row(specimen, cat_row, cfg, b_p=b_p, b_n=b_n)
    loss = cfg.loss
    if use_amplitude_weights is not None:
        loss = CalibrationLossSettings(
            w_feat_l2=cfg.loss.w_feat_l2,
            w_feat_l1=cfg.loss.w_feat_l1,
            w_energy_l2=cfg.loss.w_energy_l2,
            w_energy_l1=cfg.loss.w_energy_l1,
            w_unordered_binenv_l2=cfg.loss.w_unordered_binenv_l2,
            w_unordered_binenv_l1=cfg.loss.w_unordered_binenv_l1,
            use_amplitude_weights=use_amplitude_weights,
            amplitude_weight_power=cfg.loss.amplitude_weight_power,
            amplitude_weight_eps=cfg.loss.amplitude_weight_eps,
        )

    active = list(cfg.optimize_params)
    bounds = _bounds_for_active(active, cfg)

    use_amp_w = loss.use_amplitude_weights
    mse_weights, amp_meta = build_amplitude_weights(
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
    p_y_ref = load_p_y_kip_catalog(
        _PROJECT_ROOT,
        specimen,
        float(prow["fyp"]),
        float(prow["A_sc"]),
    )

    print(f"  Optimizing: {', '.join(active)}")
    out_row, bd_initial, bd_final, _F0, F_sim_final = optimize_one_specimen(
        specimen,
        prow,
        D_exp,
        F_exp,
        amp_meta,
        active,
        bounds,
        p_y_ref=p_y_ref,
        s_d=s_d_ref,
        loss=loss,
    )

    if bd_final is None:
        raise SystemExit(f"Optimization failed for {specimen!r} (simulation or loss breakdown).")

    specimen_out = out_dir or (RESULTS_SINGLE / specimen)
    specimen_out.mkdir(parents=True, exist_ok=True)
    params_path = specimen_out / "parameters.csv"
    pd.DataFrame([out_row]).to_csv(params_path, index=False)

    sim_dir = specimen_out / "parameters_simulated_force"
    save_simulated_force_history(
        sim_dir, specimen, cfg.set_id, D_exp, F_exp, F_sim_final
    )
    save_simulated_force_history_csv(
        sim_dir, specimen, cfg.set_id, D_exp, F_exp, F_sim_final
    )

    catalog_by_name = catalog.set_index("Name")
    mi = _metrics_dict_for_breakdown(bd_initial, loss, "initial") if bd_initial else {}
    mf = _metrics_dict_for_breakdown(bd_final, loss, "final")
    metrics_path = specimen_out / "parameters_metrics.csv"
    metrics_dataframe(
        [
            {
                "Name": specimen,
                "set_id": cfg.set_id,
                "specimen_weight": 1.0,
                "contributes_to_aggregate": True,
                **catalog_metrics_fields(specimen, catalog_by_name),
                "weight_config": "single_specimen",
                "calibration_stage": "optimize",
                "aggregate_by_set_id": False,
                **mi,
                **mf,
                **_loss_weight_snapshot(loss),
                "S_F": s_f_ref,
                "S_D": s_d_ref,
                "S_E": s_e_ref,
                "P_y_ref": p_y_ref,
                "n_cycles": len(amp_meta),
                "success": mf["final_J_total"] < FAILURE_PENALTY * 0.5,
            }
        ]
    ).to_csv(metrics_path, index=False)

    jtot = mf["final_J_total"]
    print(
        f"  {specimen} set {cfg.set_id}: J_total={jtot:.6g}  "
        f"J_feat_L2={mf['final_J_feat_raw']:.6g}"
    )

    overlay_dir.mkdir(parents=True, exist_ok=True)

    catalog = read_catalog()
    if not uses_unordered_inputs(get_specimen_record(specimen, catalog)):
        _plot_cycle_debug(
            specimen,
            int(cfg.set_id),
            D_exp,
            F_exp,
            np.asarray(F_sim_final, dtype=float),
            amp_meta,
            mse_weights,
            out_row,
            cat_row,
            plots_base=overlay_dir,
        )
    else:
        print(f"  Skipped cycle/landmark debug (digitized unordered specimen {specimen!r})")

    params_df = pd.read_csv(params_path)
    run_one_specimen(
        specimen,
        params_df,
        cat_row,
        overlay_dir,
        norm_xy_half=None,
        override_bp=None,
        override_bn=None,
        force_deformation_csv=csv_path,
    )
    print(f"  Wrote overlays under {overlay_dir}")
    return params_path


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Individual L-BFGS-B calibration + overlays for one specimen (SteelMPF). "
            "Edit scripts/calibrate_single/input.csv for seeds, bounds, and loss weights."
        ),
    )
    p.add_argument(
        "specimen",
        nargs="?",
        default=None,
        help="Specimen Name (e.g. STF01). Same as --specimen.",
    )
    p.add_argument(
        "--specimen",
        dest="specimen_flag",
        type=str,
        default=None,
        help="Specimen Name (alternative to positional argument).",
    )
    p.add_argument(
        "--force-deformation",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Absolute path to force_deformation.csv (default: "
            "data/resampled/{Name}/force_deformation.csv when present; "
            "with --prepare-data, that path is created if missing)."
        ),
    )
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Calibration input CSV (default: {DEFAULT_INPUT})",
    )
    p.add_argument(
        "--prepare-data",
        action="store_true",
        help=(
            "Run cycle_points, filter_force, and resample_filtered for the specimen first "
            "(requires data/raw/{Name}/). Then read the resampled force_deformation.csv."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Output directory (default: {RESULTS_SINGLE}/<Name>/)",
    )
    p.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help=f"Overlay PNG directory (default: {PLOTS_SINGLE}/<Name>/)",
    )
    p.add_argument(
        "--amplitude-weights",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override input.csv use_amplitude_weights for J_feat cycle weights.",
    )
    args = p.parse_args()

    specimen = args.specimen_flag or args.specimen
    if not specimen:
        p.error("specimen Name is required (positional or --specimen).")

    specimen = str(specimen).strip()
    input_path = Path(args.input).expanduser().resolve()
    try:
        cfg = load_single_calibrate_input(input_path)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    prepare_data = bool(args.prepare_data)
    if args.force_deformation is None:
        force_csv = _default_force_deformation_path(specimen, prepare_data=prepare_data)
    else:
        force_csv = args.force_deformation

    print(f"  Loaded input: {input_path}")
    params_path = calibrate_and_plot(
        specimen,
        force_csv,
        cfg,
        prepare_data=prepare_data,
        out_dir=args.out_dir,
        plots_dir=args.plots_dir,
        use_amplitude_weights=args.amplitude_weights,
    )
    out_dir = params_path.parent
    plots_dir = args.plots_dir or (PLOTS_SINGLE / specimen)
    force_resolved = force_csv.expanduser().resolve()
    print(
        f"\nDone: {specimen}\n"
        f"  Input:      {input_path}\n"
        f"  Force-u:    {force_resolved}\n"
        f"  Parameters: {params_path}\n"
        f"  Metrics:    {out_dir / 'parameters_metrics.csv'}\n"
        f"  Sim CSV:    {out_dir / 'parameters_simulated_force'}\n"
        f"  Overlays:   {plots_dir}\n"
        f"  Apparent b: {plots_dir / 'apparent_b'}\n"
        f"  Cycles:     {plots_dir / 'cycles'}"
    )


if __name__ == "__main__":
    main()
