"""
Debug: overlay J_feat landmark points on experimental vs simulated hysteresis.

Twelve landmarks per weight cycle (yield/peak/subpath construction; ``F_thr = f_y A_sc``).
For non–F=0 slots, exp/sim pairs share the same displacement-grid index (nearest vertex to each
target ``d_e`` in the slot path window). F=0 extremal slots keep interpolated crossings on each curve.
Circles are experimental, triangles simulated, with a connector when both exist. Optional CSV: per-cycle experimental ``d_k,f_k`` (grid-snapped), simulated ``d_sim_k,f_sim_k``,
``w_c``, ``j_feat_l2_mean`` / ``j_feat_l1_mean`` (calibration ``J_feat``), and ``n_jfeat_slots`` /
``jfeat_contributes``.

Same inputs as optimize_brb_mse / plot_cycle_energy_debug (resampled CSV, cycle points,
parameters CSV). Requires OpenSees for run_simulation. Yield displacement ``Dy`` for landmark
gating is computed from the params row: ``(fyp/E_hat)*L_T`` via ``yield_displacement_dy_in``,
with fallback ``(fyp/E)*L_y`` when that is not finite.

Example:
  python scripts/calibrate/plot_cycle_landmarks_debug.py --specimen PC160
  python scripts/calibrate/plot_cycle_landmarks_debug.py --params results/calibration/individual_optimize/optimized_brb_parameters.csv --sim-cache
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_SCRIPTS = _PROJECT_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS / "postprocess"))

from calibrate.pipeline_log import line, run_banner, section  # noqa: E402
from calibrate.amplitude_mse_partition import build_amplitude_weights  # noqa: E402
from calibrate.optimize_brb_mse import AMPLITUDE_WEIGHTS_ARG_HELP, force_scale_s_f  # noqa: E402
from calibrate.debug_sim_cache import (  # noqa: E402
    save_fsim_cache,
    try_load_cached_fsim,
)
from calibrate.cycle_feature_loss import (  # noqa: E402
    LANDMARK_EXP_CSV_COLUMNS,
    N_LANDMARK_SLOTS,
    deformation_scale_s_d,
    extract_cycle_landmarks,
    jfeat_means_from_paired_landmarks,
    landmark_exp_row_dict,
    pair_sim_cycle_landmarks,
    yield_displacement_dy_in,
)
from model.corotruss import run_simulation  # noqa: E402
from postprocess.cycle_points import find_cycle_points, load_cycle_points_resampled  # noqa: E402
from postprocess.plot_dimensions import (  # noqa: E402
    COLOR_EXPERIMENTAL,
    SAVE_DPI,
    SINGLE_FIGSIZE_IN,
    configure_matplotlib_style,
    style_axes_spines_and_ticks,
)
from postprocess.plot_specimens import (  # noqa: E402
    NORM_FORCE_LABEL,
    NORM_STRAIN_LABEL,
    apply_normalized_fu_axes,
    set_symmetric_axes,
)
from specimen_catalog import (  # noqa: E402
    path_ordered_resampled_force_csv_stems,
    read_catalog,
    resolve_resampled_force_deformation_csv,
)
DEFAULT_PARAMS = (
    _PROJECT_ROOT / "results" / "calibration" / "individual_optimize" / "optimized_brb_parameters.csv"
)
OUT_DIR_DEFAULT = (
    _PROJECT_ROOT / "results" / "plots" / "calibration" / "individual_optimize" / "debug_landmarks"
)

AMPLITUDE_WEIGHT_POWER = 2.0
AMPLITUDE_WEIGHT_EPS = 0.05
DEBUG_PARTITION = False

COLOR_SIMULATED = "black"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
configure_matplotlib_style()


def _meta_cycle_weight_w_c(meta_row: dict) -> float:
    w = float(meta_row.get("w_c", 1.0))
    if not np.isfinite(w) or w <= 0.0:
        return 1.0
    return w


def _row_to_sim_params(prow: pd.Series) -> dict:
    """Map params CSV row to run_simulation keyword arguments."""
    keys = ("L_T", "L_y", "A_sc", "A_t", "fyp", "fyn", "E", "b_p", "b_n", "R0", "cR1", "cR2", "a1", "a2", "a3", "a4")
    return {k: float(prow[k]) for k in keys}


def plot_landmark_overlay(
    specimen_id: str,
    set_id: int | str,
    D_exp: np.ndarray,
    F_exp: np.ndarray,
    F_sim: np.ndarray,
    meta: list[dict],
    params_row: pd.Series,
    out_path: Path,
    *,
    f_yc: float,
    A_c: float,
    L_y: float,
    exp_csv_rows: list[dict] | None = None,
) -> None:
    """Debug plot of cycle landmarks on F–u. If ``exp_csv_rows`` is a list, append one row dict per cycle."""
    fyA = float(f_yc) * float(A_c)
    L_yf = float(L_y)
    if fyA <= 0 or L_yf <= 0 or not np.isfinite(fyA) or not np.isfinite(L_yf):
        fyA, L_yf = 1.0, 1.0
    D_n = np.asarray(D_exp, dtype=float) / L_yf
    F_e_n = np.asarray(F_exp, dtype=float) / fyA
    F_s_n = np.asarray(F_sim, dtype=float) / fyA
    fyp = float(params_row["fyp"])
    a_sc = float(params_row["A_sc"])
    E_ksi = float(params_row["E"])
    L_T_in = float(params_row["L_T"])
    L_y_in_row = float(params_row["L_y"])
    A_t = float(params_row["A_t"])
    dy_in = yield_displacement_dy_in(
        fy_ksi=fyp,
        E_ksi=E_ksi,
        L_T_in=L_T_in,
        L_y_in=L_y_in_row,
        A_sc_in2=a_sc,
        A_t_in2=A_t,
    )
    if not (np.isfinite(dy_in) and dy_in > 0.0) and np.isfinite(E_ksi) and E_ksi > 0.0:
        if np.isfinite(L_y_in_row) and L_y_in_row > 0.0:
            dy_in = float((fyp / E_ksi) * L_y_in_row)

    s_f = float(force_scale_s_f(F_exp))
    s_d = float(deformation_scale_s_d(D_exp))

    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE_IN)
    ax.plot(
        D_n,
        F_e_n,
        color=COLOR_EXPERIMENTAL,
        linewidth=0.9,
        alpha=0.95,
    )
    ax.plot(
        D_n,
        F_s_n,
        color=COLOR_SIMULATED,
        linewidth=0.9,
        linestyle="--",
        alpha=0.95,
    )

    cmap = plt.get_cmap("tab10")
    for k, m in enumerate(meta):
        s, e = int(m["start"]), int(m["end"])
        if e <= s:
            le = [None] * N_LANDMARK_SLOTS
            ls = [None] * N_LANDMARK_SLOTS
            if exp_csv_rows is not None:
                exp_csv_rows.append(
                    landmark_exp_row_dict(
                        specimen_id,
                        set_id,
                        k,
                        m,
                        le,
                        fy_ksi=fyp,
                        a_sc=a_sc,
                        ls=ls,
                        w_c=_meta_cycle_weight_w_c(m),
                        j_feat_l2_mean=float("nan"),
                        j_feat_l1_mean=float("nan"),
                        n_jfeat_slots=0,
                        jfeat_contributes=False,
                    )
                )
            continue
        color = cmap(k % 10)

        le = extract_cycle_landmarks(
            D_exp, F_exp, s, e, fy_ksi=fyp, a_sc=a_sc, dy_in=dy_in
        )
        ls, le_m = pair_sim_cycle_landmarks(
            D_exp, F_exp, F_sim, s, e, le, fy_ksi=fyp, a_sc=a_sc
        )

        j2, j1, n_jf = jfeat_means_from_paired_landmarks(le_m, ls, s_f, s_d)
        w_c = _meta_cycle_weight_w_c(m)

        if exp_csv_rows is not None:
            exp_csv_rows.append(
                landmark_exp_row_dict(
                    specimen_id,
                    set_id,
                    k,
                    m,
                    le_m,
                    fy_ksi=fyp,
                    a_sc=a_sc,
                    ls=ls,
                    w_c=w_c,
                    j_feat_l2_mean=j2,
                    j_feat_l1_mean=j1,
                    n_jfeat_slots=n_jf,
                    jfeat_contributes=n_jf > 0,
                )
            )

        for slot in range(N_LANDMARK_SLOTS):
            label = str(slot + 1)
            if le_m[slot] is not None and ls[slot] is not None:
                d_e, f_e = le_m[slot]
                d_s, f_s = ls[slot]
                if all(np.isfinite(x) for x in (d_e, f_e, d_s, f_s)):
                    ax.plot(
                        [d_e / L_yf, d_s / L_yf],
                        [f_e / fyA, f_s / fyA],
                        color=color,
                        linestyle="-",
                        linewidth=0.65,
                        alpha=0.55,
                        zorder=5,
                        solid_capstyle="round",
                    )
            if le_m[slot] is not None:
                d, f = le_m[slot]
                ax.scatter(
                    [d / L_yf],
                    [f / fyA],
                    c=[color],
                    s=18,
                    marker="o",
                    edgecolors="0.2",
                    linewidths=0.35,
                    zorder=6,
                )
                ax.annotate(
                    label,
                    (d / L_yf, f / fyA),
                    textcoords="offset points",
                    xytext=(3, 2),
                    fontsize=4.25,
                    color=color,
                    alpha=0.95,
                )
            if ls[slot] is not None:
                d, f = ls[slot]
                ax.scatter(
                    [d / L_yf],
                    [f / fyA],
                    c=[color],
                    s=22,
                    marker="^",
                    edgecolors="0.2",
                    linewidths=0.35,
                    zorder=6,
                )
                ax.annotate(
                    label,
                    (d / L_yf, f / fyA),
                    textcoords="offset points",
                    xytext=(-5, -7),
                    fontsize=4.25,
                    color=color,
                    alpha=0.95,
                )

    set_symmetric_axes(ax, D_n, np.concatenate([F_e_n, F_s_n]))
    ax.set_xlabel(NORM_STRAIN_LABEL)
    ax.set_ylabel(NORM_FORCE_LABEL)
    apply_normalized_fu_axes(ax)
    ax.axhline(0.0, color="k", linewidth=0.45, alpha=0.45)
    ax.grid(True, alpha=0.28)
    legend_handles = [
        Line2D([0], [0], color=COLOR_EXPERIMENTAL, lw=0.85, label="Exp. (hysteresis)"),
        Line2D([0], [0], color=COLOR_SIMULATED, lw=0.85, ls="--", label="Sim. (hysteresis)"),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markerfacecolor=COLOR_EXPERIMENTAL,
            markeredgecolor="0.2",
            markeredgewidth=0.35,
            markersize=4.0,
            label="Exp. landmark",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="^",
            markerfacecolor=COLOR_SIMULATED,
            markeredgecolor="0.2",
            markeredgewidth=0.35,
            markersize=4.25,
            label="Sim. landmark",
        ),
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=5,
        handlelength=1.5,
        handletextpad=0.45,
        borderpad=0.3,
        frameon=True,
    )
    style_axes_spines_and_ticks(ax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path,
        dpi=SAVE_DPI,
        bbox_inches="tight",
        bbox_extra_artists=(leg,),
        pad_inches=0.08,
    )
    plt.close(fig)


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description="Overlay cycle landmark points (J_feat) on exp vs sim hysteresis.",
    )
    p.add_argument("--specimen", type=str, default=None, help="Single Name; default: all with resampled + params.")
    p.add_argument("--params", type=str, default=str(DEFAULT_PARAMS), help="Parameters CSV.")
    p.add_argument("--output-dir", type=str, default=str(OUT_DIR_DEFAULT), help="Directory for PNGs.")
    p.add_argument(
        "--no-landmark-csv",
        action="store_true",
        help="Do not write landmark CSV (exp + sim columns) next to each PNG.",
    )
    p.add_argument(
        "--sim-cache",
        action="store_true",
        help=(
            "Reuse OpenSees F_sim from output-dir/_sim_cache/ when params CSV row is unchanged "
            "(mtime + fingerprint); otherwise simulate and refresh cache."
        ),
    )
    p.add_argument(
        "--amplitude-weights",
        action="store_true",
        help=AMPLITUDE_WEIGHTS_ARG_HELP,
    )
    args = p.parse_args()

    run_banner("plot_cycle_landmarks_debug.py")
    section("Per-specimen landmark debug (PNG per set_id)")

    params_path = Path(args.params)
    out_dir = Path(args.output_dir)
    params_df = pd.read_csv(params_path)
    if "Name" not in params_df.columns:
        raise SystemExit(f"No Name column in {params_path}")

    catalog = read_catalog()
    available = sorted(
        set(params_df["Name"].astype(str))
        & path_ordered_resampled_force_csv_stems(catalog, project_root=_PROJECT_ROOT)
    )
    if not available:
        raise SystemExit(
            "No specimens with both params CSV and data/resampled/{Name}/force_deformation.csv (path-ordered)."
        )

    specimens = [args.specimen] if args.specimen else available
    if args.specimen and args.specimen not in available:
        raise SystemExit(f"Specimen {args.specimen!r} not in available set.")

    for sid in specimens:
        csv_path = resolve_resampled_force_deformation_csv(sid, _PROJECT_ROOT)
        if csv_path is None or not csv_path.is_file():
            line(f"skip {sid}: missing resampled force_deformation.csv")
            continue
        df = pd.read_csv(csv_path)
        if "Force[kip]" not in df.columns or "Deformation[in]" not in df.columns:
            line(f"skip {sid}: missing columns")
            continue

        D_exp = df["Deformation[in]"].to_numpy(dtype=float)
        F_exp = df["Force[kip]"].to_numpy(dtype=float)
        loaded = load_cycle_points_resampled(sid)
        points, _ = loaded if loaded is not None else find_cycle_points(df)
        _w, meta = build_amplitude_weights(
            D_exp,
            points,
            p=AMPLITUDE_WEIGHT_POWER,
            eps=AMPLITUDE_WEIGHT_EPS,
            debug_partition=DEBUG_PARTITION,
            use_amplitude_weights=bool(args.amplitude_weights),
        )

        rows = params_df[params_df["Name"].astype(str) == sid]
        if rows.empty:
            print(f"Skip {sid}: no row in parameters CSV")
            continue

        for _, prow in rows.iterrows():
            set_id = prow.get("set_id", "?")
            sim_kw = _row_to_sim_params(prow)
            F_sim = None
            if args.sim_cache:
                F_sim = try_load_cached_fsim(out_dir, sid, set_id, params_path, sim_kw)
                if F_sim is not None:
                    line(f"{sid} set {set_id}: using cached F_sim (--sim-cache)")
            if F_sim is None:
                try:
                    F_sim = np.asarray(run_simulation(D_exp, **sim_kw), dtype=float)
                except Exception as e:
                    line(f"{sid} set {set_id}: simulation failed ({e})")
                    continue
                if args.sim_cache:
                    save_fsim_cache(out_dir, sid, set_id, params_path, sim_kw, F_sim)
            if F_sim.shape != F_exp.shape:
                line(f"{sid} set {set_id}: length mismatch sim vs exp")
                continue

            out_png = out_dir / f"{sid}_set{set_id}_landmarks.png"
            crow = catalog.set_index("Name").loc[sid]
            exp_csv_rows: list[dict] | None = None if args.no_landmark_csv else []
            plot_landmark_overlay(
                sid,
                set_id,
                D_exp,
                F_exp,
                F_sim,
                meta,
                prow,
                out_png,
                f_yc=float(crow["f_yc_ksi"]),
                A_c=float(crow["A_c_in2"]),
                L_y=float(crow["L_y_in"]),
                exp_csv_rows=exp_csv_rows,
            )
            line(f"wrote  {out_png}")
            if exp_csv_rows is not None and exp_csv_rows:
                csv_path_out = out_dir / f"{sid}_set{set_id}_landmarks_exp.csv"
                pd.DataFrame(exp_csv_rows).reindex(columns=LANDMARK_EXP_CSV_COLUMNS).to_csv(
                    csv_path_out, index=False
                )
                line(f"wrote  {csv_path_out}")


if __name__ == "__main__":
    main()
