"""
Debug: per-specimen multi-panel figure -- one subplot per weight cycle.

For each cycle c, plots axial P vs δ on the resampled segment [start:end), shades the signed
trapezoidal work ∫ P dδ (matplotlib fill_between between P and the δ-axis), and annotates
cycle weight w_c, amplitude A_c, E_c = |∫ P dδ| normalized by S_E = (P_max−P_min)(δ_max−δ_min)
on the full experiment (same definition as energy_scale_s_e / J_E).

Requires resampled CSVs and cycle points (JSON or auto from find_cycle_points), same as
optimize_brb_mse. Runs one simulation per specimen row (set_id) using parameters CSV.

Example:
  python scripts/calibrate/plot_cycle_energy_debug.py --specimen PC160
  python scripts/calibrate/plot_cycle_energy_debug.py --params results/calibration/individual_optimize/optimized_brb_parameters.csv --sim-cache
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_SCRIPTS = _PROJECT_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS / "postprocess"))

from calibrate.pipeline_log import line, run_banner, section  # noqa: E402
from calibrate.optimize_brb_mse import (  # noqa: E402
    AMPLITUDE_WEIGHTS_ARG_HELP,
    run_simulation_kwargs_from_prow,
)
from calibrate.amplitude_mse_partition import (  # noqa: E402
    build_amplitude_weights,
    cycle_abs_trapz_work,
    energy_scale_s_e,
)
from calibrate.debug_sim_cache import (  # noqa: E402
    save_fsim_cache,
    try_load_cached_fsim,
)
from model.corotruss import run_simulation  # noqa: E402
from postprocess.cycle_points import find_cycle_points, load_cycle_points_resampled  # noqa: E402
from postprocess.plot_dimensions import (  # noqa: E402
    AXES_SPINE_LINEWIDTH,
    COLOR_EXPERIMENTAL,
    HYSTERESIS_LINEWIDTH_SCALE,
    SAVE_DPI,
    configure_matplotlib_style,
    figsize_for_grid,
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
    _PROJECT_ROOT / "results" / "plots" / "calibration" / "individual_optimize" / "debug_energy"
)

AMPLITUDE_WEIGHT_POWER = 2.0
AMPLITUDE_WEIGHT_EPS = 0.05
DEBUG_PARTITION = False

COLOR_SIMULATED = "black"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
configure_matplotlib_style()


def plot_specimen_cycles(
    specimen_id: str,
    set_id: int | str,
    D_exp: np.ndarray,
    F_exp: np.ndarray,
    F_sim: np.ndarray,
    meta: list[dict],
    out_path: Path,
    *,
    f_yc: float,
    A_c: float,
    L_y: float,
    ncols: int = 4,
) -> None:
    """Per-cycle energy panels for one specimen."""
    s_e = energy_scale_s_e(D_exp, F_exp)
    fyA = float(f_yc) * float(A_c)
    L_yf = float(L_y)
    if fyA <= 0 or L_yf <= 0 or not np.isfinite(fyA) or not np.isfinite(L_yf):
        fyA, L_yf = 1.0, 1.0
    if not meta:
        return

    n_c = len(meta)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n_c / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize_for_grid(nrows, ncols),
        layout="constrained",
        squeeze=False,
    )

    for k, m in enumerate(meta):
        r, c = divmod(k, ncols)
        ax = axes[r][c]
        s, e = int(m["start"]), int(m["end"])
        d = np.asarray(D_exp[s:e], dtype=float)
        f_e = np.asarray(F_exp[s:e], dtype=float)
        f_s = np.asarray(F_sim[s:e], dtype=float)
        d_n = d / L_yf
        f_e_n = f_e / fyA
        f_s_n = f_s / fyA

        if len(d_n) >= 2:
            ax.fill_between(
                d_n,
                0.0,
                f_e_n,
                alpha=0.35,
                color=COLOR_EXPERIMENTAL,
                label=r"$\int P_{\mathrm{exp}}\,\mathrm{d}\delta$ (signed)",
            )
        ax.plot(
            d_n,
            f_e_n,
            color=COLOR_EXPERIMENTAL,
            linewidth=1.0 * HYSTERESIS_LINEWIDTH_SCALE,
            label=r"$P_{\mathrm{exp}}$",
        )
        ax.plot(
            d_n,
            f_s_n,
            color=COLOR_SIMULATED,
            linewidth=0.9 * HYSTERESIS_LINEWIDTH_SCALE,
            linestyle="--",
            label="P_sim",
        )
        ax.axhline(0.0, color="k", linewidth=AXES_SPINE_LINEWIDTH, alpha=0.5)
        ax.axvline(0.0, color="k", linewidth=AXES_SPINE_LINEWIDTH, alpha=0.5)

        e_e = cycle_abs_trapz_work(D_exp, F_exp, s, e)
        e_s = cycle_abs_trapz_work(D_exp, F_sim, s, e)
        ne = e_e / s_e if s_e > 0 else float("nan")
        ns = e_s / s_e if s_e > 0 else float("nan")

        kind = m.get("kind", "")
        w_c = float(m.get("w_c", 1.0))
        a_c = m.get("amp")
        a_str = f"{float(a_c):.4g}" if a_c is not None else "--"
        ax.set_title(
            f"c{k} ({kind})  w_c={w_c:.4g}  A_c={a_str}\n"
            f"E_exp/S_E={ne:.4g}  E_sim/S_E={ns:.4g}",
            fontsize=8,
        )
        set_symmetric_axes(ax, d_n, np.concatenate([f_e_n, f_s_n]))
        ax.set_xlabel(NORM_STRAIN_LABEL, fontsize=7)
        ax.set_ylabel(NORM_FORCE_LABEL, fontsize=7)
        apply_normalized_fu_axes(ax, pct_decimals=0)
        ax.grid(True, alpha=0.25)
        style_axes_spines_and_ticks(ax)
        ax.tick_params(labelsize=6)

    for j in range(n_c, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=SAVE_DPI)
    plt.close(fig)


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description="Debug plots: one subplot per cycle, shaded signed ∫ P dδ, title shows E/S_E and w_c.",
    )
    p.add_argument("--specimen", type=str, default=None, help="Single Name; default: all with resampled + params.")
    p.add_argument("--params", type=str, default=str(DEFAULT_PARAMS), help="Parameters CSV.")
    p.add_argument("--output-dir", type=str, default=str(OUT_DIR_DEFAULT), help="Directory for PNGs.")
    p.add_argument("--ncols", type=int, default=4, help="Subplot columns per figure.")
    p.add_argument(
        "--sim-cache",
        action="store_true",
        help=(
            "Reuse OpenSees F_sim from output-dir/_sim_cache/ when params CSV row is unchanged; "
            "otherwise simulate and refresh cache (same layout as plot_cycle_landmarks_debug)."
        ),
    )
    p.add_argument(
        "--amplitude-weights",
        action="store_true",
        help=AMPLITUDE_WEIGHTS_ARG_HELP,
    )
    args = p.parse_args()

    run_banner("plot_cycle_energy_debug.py")
    section("Per-specimen cycle energy debug (PNG per set_id)")

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
            print(f"Skip {sid}: missing resampled force_deformation.csv")
            continue
        df = pd.read_csv(csv_path)
        if "Force[kip]" not in df.columns or "Deformation[in]" not in df.columns:
            print(f"Skip {sid}: missing columns")
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
        for _, prow in rows.iterrows():
            set_id = prow.get("set_id", "?")
            sm, sim_kw = run_simulation_kwargs_from_prow(prow)
            F_sim = None
            if args.sim_cache:
                F_sim = try_load_cached_fsim(out_dir, sid, set_id, params_path, sim_kw)
                if F_sim is not None:
                    line(f"{sid} set {set_id}: using cached F_sim (--sim-cache)")
            if F_sim is None:
                try:
                    F_sim = np.asarray(run_simulation(D_exp, steel_model=sm, **sim_kw), dtype=float)
                except Exception as e:
                    line(f"{sid} set {set_id}: simulation failed ({e})")
                    continue
                if args.sim_cache:
                    save_fsim_cache(out_dir, sid, set_id, params_path, sim_kw, F_sim)
            if F_sim.shape != F_exp.shape:
                line(f"{sid} set {set_id}: length mismatch sim vs exp")
                continue

            out_png = out_dir / f"{sid}_set{set_id}_cycle_energy_debug.png"
            crow = catalog.set_index("Name").loc[sid]
            plot_specimen_cycles(
                sid,
                set_id,
                D_exp,
                F_exp,
                F_sim,
                meta,
                out_png,
                f_yc=float(crow["f_yc_ksi"]),
                A_c=float(crow["A_c_in2"]),
                L_y=float(crow["L_y_in"]),
                ncols=args.ncols,
            )
            line(f"wrote  {out_png}")


if __name__ == "__main__":
    main()
