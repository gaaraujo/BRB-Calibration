"""
Overlay resampled hysteresis with plastic hardening fits, backward-extended plastic lines,
elastic asymptotes from the **latest** opposite-side cycle peak (``min_def`` / ``max_def``)
before each $b$-fit, the $\\sigma_0$ intersection, $\\sigma_0^{\\mathrm{eq}}$ from the plastic line
intersected with the elastic asymptote from **global** max compressive / tensile deformation, and the
plastic line $\\cap$ $F=k_{\\mathrm{init}}u$ through the origin with $k_{\\mathrm{init}}=\\hat{E}A_{sc}/L_T$
(same initial stiffness as $b=k_{\\mathrm{sh}}/k_{\\mathrm{init}}$; segment + marker).

Writes one PNG per path-ordered specimen under ``results/plots/apparent_b/sig0_slopes/`` by default.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme()
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_SCRIPTS = _PROJECT_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS / "postprocess"))
sys.path.insert(0, str(_SCRIPTS))

from plot_dimensions import (  # noqa: E402
    AXES_SPINE_LINEWIDTH_SINGLE_AX,
    COLOR_EXPERIMENTAL,
    HYSTERESIS_LINEWIDTH_SCALE,
    LEGEND_FONT_SIZE_SINGLE_AX_PT,
    SAVE_DPI,
    SINGLE_FIGSIZE_IN,
    configure_matplotlib_style,
    single_axis_style_context,
    style_single_axis_spines,
)
configure_matplotlib_style()
from extract_bn_bp import (  # noqa: E402
    CATALOG_PATH,
    get_sig0_overlay_segments_one_specimen,
    get_specimens_with_resampled,
)
from postprocess.plot_specimens import (  # noqa: E402
    NORM_FORCE_LABEL,
    NORM_STRAIN_LABEL,
    apply_normalized_fu_axes,
    normalize,
    set_symmetric_axes,
)
from load_raw import load_raw_valid  # noqa: E402
from specimen_catalog import read_catalog, resolve_resampled_force_deformation_csv  # noqa: E402

_APPARENT_B = _PROJECT_ROOT / "results" / "plots" / "apparent_b"
PLOTS_SIG0_SLOPES = _APPARENT_B / "sig0_slopes"

COLOR_TENSION = "#CC3311"
COLOR_COMPRESSION = "#0077BB"
COLOR_ELASTIC = "#2ca02c"
COLOR_SIG0 = "#9467bd"
COLOR_EQUIV_SIG0 = "#FF7F0E"
COLOR_ORIGIN_INTER = "#17BECF"
# Marker area scale vs prior defaults (matplotlib ``s`` is points²).
_SIG0_MARKER_SCALE = 0.30
# Plastic / extended plastic / elastic construction lines vs prior linewidths.
_SIG0_TANGENT_LINEWIDTH_SCALE = 0.6


def plot_one_specimen_sig0(specimen_id: str, catalog_row: pd.Series, out_dir: Path) -> None:
    """Normalized F–u with plastic / elastic constructions and $\\sigma_0$ markers."""
    rpath = resolve_resampled_force_deformation_csv(specimen_id, _PROJECT_ROOT)
    if rpath is None or not rpath.is_file():
        return
    df = pd.read_csv(rpath)
    if "Force[kip]" not in df.columns or "Deformation[in]" not in df.columns:
        return
    u = df["Deformation[in]"].values.astype(float, copy=False)
    F = df["Force[kip]"].values.astype(float, copy=False)
    n = len(u)
    if n == 0:
        return

    L_y = float(catalog_row["L_y_in"])
    A_sc = float(catalog_row["A_c_in2"])
    fy = float(catalog_row["f_yc_ksi"])
    fy_A = fy * A_sc
    if L_y <= 0 or fy_A <= 0:
        return

    u_norm = u / L_y
    F_norm = F / fy_A

    segs = get_sig0_overlay_segments_one_specimen(specimen_id)
    if not segs:
        return

    tlw = _SIG0_TANGENT_LINEWIDTH_SCALE * HYSTERESIS_LINEWIDTH_SCALE
    with single_axis_style_context():
        fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE_IN, layout="constrained")
        ax.plot(
            u_norm,
            F_norm,
            color=COLOR_EXPERIMENTAL,
            linewidth=1.2 * HYSTERESIS_LINEWIDTH_SCALE,
            label="Experimental",
            zorder=1,
        )

        for seg in segs:
            is_tension = bool(seg["is_tension"])
            color_p = COLOR_TENSION if is_tension else COLOR_COMPRESSION
            u_fit = np.asarray(seg["u_fit"], dtype=float) / L_y
            F_fit = np.asarray(seg["F_fit_line"], dtype=float) / fy_A
            u_pl = np.asarray(seg["u_plastic_long"], dtype=float) / L_y
            F_pl = np.asarray(seg["F_plastic_long"], dtype=float) / fy_A
            u_el = np.asarray(seg["u_elastic_long"], dtype=float) / L_y
            F_el = np.asarray(seg["F_elastic_long"], dtype=float) / fy_A

            ax.plot(u_fit, F_fit, color=color_p, linewidth=1.6 * tlw, linestyle="-", alpha=0.95, zorder=2)
            ax.plot(u_pl, F_pl, color=color_p, linewidth=1.1 * tlw, linestyle="--", alpha=0.65, zorder=2)
            mfin = np.isfinite(F_el)
            if np.any(mfin):
                ax.plot(
                    u_el[mfin],
                    F_el[mfin],
                    color=COLOR_ELASTIC,
                    linewidth=1.1 * tlw,
                    linestyle=":",
                    alpha=0.85,
                    zorder=2,
                )

            dbg = seg.get("dbg") or {}
            if dbg:
                ur = float(dbg["u_r"]) / L_y
                Fr = float(dbg["F_r"]) / fy_A
                us = float(dbg["u_star"]) / L_y
                Fs = float(dbg["F_star"]) / fy_A
                ax.scatter(
                    [ur],
                    [Fr],
                    s=36 * _SIG0_MARKER_SCALE,
                    facecolors="none",
                    edgecolors=COLOR_ELASTIC,
                    marker="^",
                    linewidths=0.55,
                    zorder=4,
                )
                ax.scatter(
                    [us],
                    [Fs],
                    s=42 * _SIG0_MARKER_SCALE,
                    facecolors="none",
                    edgecolors=COLOR_SIG0,
                    marker="o",
                    linewidths=0.6,
                    zorder=5,
                )

            dbg_eq = seg.get("dbg_equiv") or {}
            if dbg_eq:
                ueq = float(dbg_eq["u_star"]) / L_y
                Feq = float(dbg_eq["F_star"]) / fy_A
                ax.scatter(
                    [ueq],
                    [Feq],
                    s=46 * _SIG0_MARKER_SCALE,
                    facecolors="none",
                    edgecolors=COLOR_EQUIV_SIG0,
                    marker="s",
                    linewidths=0.65,
                    zorder=6,
                )

            dbg_og = seg.get("dbg_origin") or {}
            if dbg_og:
                uo = float(dbg_og["u_star"]) / L_y
                Fo = float(dbg_og["F_star"]) / fy_A
                ax.plot(
                    [0.0, uo],
                    [0.0, Fo],
                    color=COLOR_ORIGIN_INTER,
                    linestyle="-",
                    linewidth=1.0 * tlw,
                    alpha=0.55,
                    zorder=3,
                )
                ax.scatter(
                    [uo],
                    [Fo],
                    s=40 * _SIG0_MARKER_SCALE,
                    facecolors="none",
                    edgecolors=COLOR_ORIGIN_INTER,
                    marker="D",
                    linewidths=0.6,
                    zorder=7,
                )

        ax.plot([], [], color=COLOR_TENSION, linestyle="-", linewidth=1.5 * tlw, label=r"$b_p$ fit")
        ax.plot([], [], color=COLOR_COMPRESSION, linestyle="-", linewidth=1.5 * tlw, label=r"$b_n$ fit")
        ax.plot(
            [],
            [],
            color=COLOR_ELASTIC,
            linestyle=":",
            linewidth=1.2 * tlw,
            label=r"$k_{\mathrm{init}}$ (peak)",
        )
        ax.scatter(
            [],
            [],
            facecolors="none",
            edgecolors=COLOR_SIG0,
            s=40 * _SIG0_MARKER_SCALE,
            marker="o",
            linewidths=0.6,
            label=r"$\sigma_0$",
        )
        ax.scatter(
            [],
            [],
            facecolors="none",
            edgecolors=COLOR_EQUIV_SIG0,
            s=44 * _SIG0_MARKER_SCALE,
            marker="s",
            linewidths=0.65,
            label=r"$\sigma_0^{\mathrm{eq}}$",
        )
        ax.plot(
            [],
            [],
            color=COLOR_ORIGIN_INTER,
            linestyle="-",
            linewidth=1.1 * tlw,
            alpha=0.75,
            marker="D",
            markersize=5.2,
            markerfacecolor="none",
            markeredgecolor=COLOR_ORIGIN_INTER,
            markeredgewidth=0.6,
            label=r"$b \cap$ origin",
        )

        ax.set_xlabel(NORM_STRAIN_LABEL)
        ax.set_ylabel(NORM_FORCE_LABEL)
        raw_df = load_raw_valid(specimen_id)
        if raw_df is not None and "Force[kip]" in raw_df.columns and "Deformation[in]" in raw_df.columns:
            raw_n = normalize(raw_df, fy, A_sc, L_y)
            filtered_n = normalize(df, fy, A_sc, L_y)
            x_all = np.concatenate([raw_n["Deformation_norm"].values, filtered_n["Deformation_norm"].values])
            y_all = np.concatenate([raw_n["Force_norm"].values, filtered_n["Force_norm"].values])
            set_symmetric_axes(ax, x_all, y_all)
        else:
            set_symmetric_axes(ax, u_norm, F_norm)
        apply_normalized_fu_axes(ax, pct_decimals=2)
        h, lab = ax.get_legend_handles_labels()
        fig.legend(
            h,
            lab,
            loc="outside upper center",
            ncol=3,
            fontsize=LEGEND_FONT_SIZE_SINGLE_AX_PT,
            frameon=False,
        )
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linewidth=AXES_SPINE_LINEWIDTH_SINGLE_AX)
        ax.axvline(0, color="k", linewidth=AXES_SPINE_LINEWIDTH_SINGLE_AX)
        style_single_axis_spines(ax)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{specimen_id}.png", dpi=SAVE_DPI)
        plt.close(fig)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Plot σ0 construction overlays (plastic + elastic asymptotes).")
    p.add_argument("--specimen", type=str, default=None, help="Single specimen Name; default: all resampled")
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output directory (default: results/plots/apparent_b/sig0_slopes)",
    )
    args = p.parse_args()
    out_dir = Path(args.output).resolve() if args.output else PLOTS_SIG0_SLOPES
    catalog = read_catalog(CATALOG_PATH)
    catalog_by_name = catalog.set_index("Name")

    if args.specimen:
        sid = args.specimen
        if sid not in catalog_by_name.index:
            print(f"Specimen {sid!r} not in catalog.")
            return
        plot_one_specimen_sig0(sid, catalog_by_name.loc[sid], out_dir)
        print(f"Wrote {out_dir}")
        return

    specimens = get_specimens_with_resampled()
    for sid in specimens:
        plot_one_specimen_sig0(sid, catalog_by_name.loc[sid], out_dir)
    if specimens:
        print(f"Wrote {out_dir} ({len(specimens)} specimen(s))")
    else:
        print("No specimens with resampled data found.")


if __name__ == "__main__":
    main()
