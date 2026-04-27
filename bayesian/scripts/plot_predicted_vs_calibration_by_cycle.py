"""
Overlay experimental vs simulated force–deformation with **one subplot per weight cycle**
(``start``/``end`` index spans from ``config/cycle_meta.json``).

Same normalization and line styling as ``plot_predicted_vs_calibration.py`` (δ/L_y [%], P/(f_y A_sc));
**shared** symmetric x/y limits over the full trace (all panels use the same axes). Between curves, **red** fill where
numerical $P/(f_y A_{sc})$ is **above** experimental, **blue** where it is **below** (same sample index along the path;
abscissa is experimental $\\delta/L_y$). Subplot grid is **4** columns; **rows** = ``ceil(n_cycles / 4)``. Runs ``model.run_analysis()`` and writes ``predicted_force.csv``.

  python scripts/plot_predicted_vs_calibration_by_cycle.py
  python scripts/plot_predicted_vs_calibration_by_cycle.py -o predicted_vs_calibration_by_cycle.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_pp = _REPO_ROOT / "scripts" / "postprocess"
if str(_pp) not in sys.path:
    sys.path.insert(0, str(_pp))

from model import run_analysis

from lib.landmark_vector import load_cycle_meta_json
from lib.specimen_config import load_specimen_config, resolve_path

from plot_dimensions import LINEWIDTH_HYSTERESIS_OVERLAY_EXPERIMENTAL

COLOR_EXPERIMENTAL = "#B0B0B0"
COLOR_SIMULATED = "#001F3F"
# Numerical − experimental in normalized force: red if numerical higher, blue if lower.
COLOR_GAP_POS = "#D62728"
COLOR_GAP_NEG = "#1F77B4"
HATCH_GAP = '||||'
FILL_GAP_ALPHA = 0.35
LINEWIDTH = LINEWIDTH_HYSTERESIS_OVERLAY_EXPERIMENTAL
SPINE_LINEWIDTH = 0.6
SAVE_DPI = 300
PANEL_W_IN = 2.35
PANEL_H_IN = 2.05
NCOLS = 4

TEXT_PT = 10

_RC_PLOT = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "sans-serif"],
    "mathtext.fontset": "dejavusans",
    "axes.labelsize": TEXT_PT,
    "axes.titlesize": TEXT_PT,
    "xtick.labelsize": TEXT_PT,
    "ytick.labelsize": TEXT_PT,
    "legend.fontsize": TEXT_PT,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "xtick.bottom": True,
    "ytick.right": True,
    "ytick.left": True,
}

NORM_STRAIN_LABEL = r"Axial strain, $\delta/L_y$ [%]"
NORM_FORCE_LABEL = r"Axial force, $P/\left(f_y A_{sc}\right)$"


def _fraction_to_percent_formatter(*, decimals: int = 2) -> mticker.FuncFormatter:
    def _fmt(x: float, _pos: int) -> str:
        if not np.isfinite(x):
            return ""
        p = 100.0 * x
        if decimals <= 0:
            return f"{p:.0f}"
        return f"{p:.{decimals}f}"

    return mticker.FuncFormatter(_fmt)


def _fill_signed_force_gap(
    ax: plt.Axes,
    x: np.ndarray,
    y_e: np.ndarray,
    y_p: np.ndarray,
) -> None:
    """Fill between experimental and numerical normalized force at shared indices; x = δ/L_y (experimental grid)."""
    x = np.asarray(x, dtype=float)
    y_e = np.asarray(y_e, dtype=float)
    y_p = np.asarray(y_p, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y_e) & np.isfinite(y_p)
    if not np.any(ok):
        return
    x = x[ok]
    y_e = y_e[ok]
    y_p = y_p[ok]
    kw = {"interpolate": True, "linewidth": 0.0, "zorder": 1}
    ax.fill_between(
        x,
        y_e,
        y_p,
        where=(y_p >= y_e),
        facecolor=to_rgba(COLOR_GAP_POS, FILL_GAP_ALPHA),
        hatch=HATCH_GAP,
        edgecolor=to_rgba(COLOR_GAP_POS, 1.0),
        **kw,
    )
    ax.fill_between(
        x,
        y_e,
        y_p,
        where=(y_p < y_e),
        facecolor=to_rgba(COLOR_GAP_NEG, FILL_GAP_ALPHA),
        hatch=HATCH_GAP,
        edgecolor=to_rgba(COLOR_GAP_NEG, 1.0),
        **kw,
    )


def _symmetric_limits(u: np.ndarray, v: np.ndarray, *, pad: float = 0.05) -> tuple[float, float]:
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    m = float(np.nanmax(np.abs(np.concatenate([u[np.isfinite(u)], v[np.isfinite(v)]]))))
    if not np.isfinite(m) or m <= 0.0:
        return -1.0, 1.0
    w = m * (1.0 + pad)
    return -w, w


def main() -> None:
    default_cfg = _ROOT / "config" / "specimen_config.yaml"
    p = argparse.ArgumentParser(
        description="Plot calibration vs predicted F–u with one panel per cycle_meta span.",
    )
    p.add_argument("--config", type=Path, default=default_cfg)
    p.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Experimental CSV (default: paths.force_deformation from config)",
    )
    p.add_argument(
        "--displacement",
        type=Path,
        default=_ROOT / "target_displacement.csv",
        help="Deformation history CSV [in]; path passed to model.run_analysis()",
    )
    p.add_argument(
        "--predicted",
        type=Path,
        default=_ROOT / "predicted_force.csv",
        help="Write simulated forces here (one row, comma-separated)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_ROOT / "predicted_vs_calibration_by_cycle.png",
    )
    args = p.parse_args()

    cfg = load_specimen_config(args.config)
    base = args.config.parent
    cal_path = args.calibration or resolve_path(cfg, "force_deformation", base)
    meta_path = resolve_path(cfg, "cycle_meta", base)

    fy = float(cfg["fy_ksi"])
    A_sc = float(cfg["A_sc_in2"])
    L_y = float(cfg["L_y_in"])
    fy_a = fy * A_sc
    if fy_a <= 0.0 or L_y <= 0.0:
        raise ValueError("fy * A_sc and L_y must be positive")

    meta, _, _, n_meta = load_cycle_meta_json(meta_path)
    if not meta:
        raise ValueError(f"no cycles in {meta_path}")

    df_e = pd.read_csv(cal_path)
    for col in ("Deformation[in]", "Force[kip]"):
        if col not in df_e.columns:
            raise ValueError(f"Calibration CSV needs column {col!r}")

    D_e = df_e["Deformation[in]"].to_numpy(dtype=float)
    F_e = df_e["Force[kip]"].to_numpy(dtype=float)

    disp_path = args.displacement.expanduser().resolve()
    D_p, F_p = run_analysis(disp_path)
    D_p = np.asarray(D_p, dtype=float).reshape(-1)
    F_p = np.asarray(F_p, dtype=float).reshape(-1)
    if D_p.shape != F_p.shape:
        raise ValueError(
            f"model force length {F_p.size} != displacement length {D_p.size} (from {disp_path})"
        )
    if n_meta is not None and len(D_e) != n_meta:
        raise ValueError(f"n_meta={n_meta} len(D_e)={len(D_e)}")
    if D_e.shape != D_p.shape:
        raise ValueError(
            f"predicted length {D_p.size} != experimental length {D_e.size}"
        )

    pred_path = args.predicted.expanduser().resolve()
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(pred_path, F_p[np.newaxis, :], delimiter=",", fmt="%.17g")

    x_e_full = D_e / L_y
    y_e_full = F_e / fy_a
    x_p_full = D_p / L_y
    y_p_full = F_p / fy_a
    gx_lo, gx_hi = _symmetric_limits(
        np.concatenate([x_e_full[np.isfinite(x_e_full)], x_p_full[np.isfinite(x_p_full)]]),
        np.array([0.0]),
    )
    gy_lo, gy_hi = _symmetric_limits(
        np.concatenate([y_e_full[np.isfinite(y_e_full)], y_p_full[np.isfinite(y_p_full)]]),
        np.array([0.0]),
    )

    n_cycles = len(meta)
    ncols = NCOLS
    nrows = max(1, (n_cycles + ncols - 1) // ncols)
    fig_w = ncols * PANEL_W_IN
    fig_h = nrows * PANEL_H_IN + 0.45

    fmt_pct = _fraction_to_percent_formatter(decimals=2)
    legend_handles: tuple | None = None
    legend_labels: tuple | None = None

    with plt.rc_context(_RC_PLOT):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(fig_w, fig_h),
            layout="constrained",
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        flat = axes.flatten()

        for i, m in enumerate(meta):
            ax = flat[i]
            s, e = int(m["start"]), int(m["end"])
            if e <= s or s < 0 or e > len(D_e):
                ax.set_title(f"Cycle {i}", fontsize=TEXT_PT, pad=2)
                ax.text(0.5, 0.5, "invalid span", ha="center", va="center", fontsize=TEXT_PT)
                ax.set_axis_off()
                continue

            sl = slice(s, e)
            x_e = D_e[sl] / L_y
            y_e = F_e[sl] / fy_a
            x_p = D_p[sl] / L_y
            y_p = F_p[sl] / fy_a

            _fill_signed_force_gap(ax, x_e, y_e, y_p)

            (ln_e,) = ax.plot(
                x_e,
                y_e,
                color=COLOR_EXPERIMENTAL,
                alpha=0.95,
                linewidth=LINEWIDTH,
                linestyle="-",
                label="Experimental",
                zorder=3,
            )
            (ln_p,) = ax.plot(
                x_p,
                y_p,
                color=COLOR_SIMULATED,
                alpha=0.95,
                linewidth=LINEWIDTH,
                linestyle="--",
                label="Numerical",
                zorder=3,
            )
            if legend_handles is None:
                legend_handles = (ln_e, ln_p)
                legend_labels = (ln_e.get_label(), ln_p.get_label())

            ax.xaxis.set_major_formatter(fmt_pct)
            ax.axhline(0, color="k", linewidth=SPINE_LINEWIDTH)
            ax.axvline(0, color="k", linewidth=SPINE_LINEWIDTH)
            ax.set_title(f"Cycle {i}", fontsize=TEXT_PT, pad=2)
            ax.tick_params(axis="both", which="both", direction="in", labelsize=TEXT_PT)

            for spine in ax.spines.values():
                spine.set_linewidth(SPINE_LINEWIDTH)

        for j in range(n_cycles, len(flat)):
            flat[j].set_visible(False)

        for ax in flat[:n_cycles]:
            ax.set_xlim(gx_lo, gx_hi)
            ax.set_ylim(gy_lo, gy_hi)

        fig.supxlabel(NORM_STRAIN_LABEL, fontsize=TEXT_PT)
        fig.supylabel(NORM_FORCE_LABEL, fontsize=TEXT_PT)

        if legend_handles and legend_labels:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="outside upper center",
                ncol=2,
                fontsize=TEXT_PT,
                frameon=False,
            )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=SAVE_DPI)
        plt.close(fig)

    print(f"Wrote {pred_path}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
