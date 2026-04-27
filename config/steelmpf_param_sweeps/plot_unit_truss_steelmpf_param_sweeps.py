r"""
BRB corotruss (``model.corotruss.run_simulation``): SteelMPF + geometry factor Q.

**Specimen selection** (edit ``SPECIMEN_NAME_INPUT``, ``GENERALIZED_SET_ID`` below): leave ``SPECIMEN_NAME_INPUT``
blank to use the editable **Manual BRB defaults** block (``E``, ``FY``, ``L_y``, geometry ratios, ``DEFAULT_STEEL``).
Otherwise geometry and reference yield ``f_yc`` come from ``config/calibration/BRB-Specimens.csv`` and steel plus ``E``
from ``results/calibration/generalized_optimize/generalized_brb_parameters.csv`` for that name and ``set_id`` when
that file exists and contains a matching row; otherwise steel uses ``_GENERIC_STEEL_FALLBACK`` with ``fyp``/``fyn``
from the catalog. When ``OPTIMIZED_BRB_PARAMETERS_SET_ID`` is set, a matching **steelmpf** row in
``results/calibration/individual_optimize/optimized_brb_parameters.csv`` overwrites classic SteelMPF fields only
(``E``, ``b_p``, ``b_n``, ``R0``, ``cR1``, ``cR2``, ``a1``–``a4``, ``fyp``, ``fyn``; see ``_apply_optimized_brb_parameters_steel``).
Optional **``SPECIMEN_STEEL_OVERRIDES``** merges last.
Rows in ``generalized_brb_parameters.csv`` may omit ``fup_ratio``, ``fun_ratio``, and ``Ru0``; those default to
4, 4, and 5 when building ``DEFAULT_STEEL``.
Experimental peak ``n_peak = max|δ|/(L_y ε_y)`` uses the same specimen catalog row (resolved F--u CSV).

Core-referenced displacement ``u = n * eps_y * L_y`` (horizontal axis ``delta / (L_y * eps_y)``, i.e. ``n``).
``ε_y`` and the σ/f_y axis use catalog ``f_yc_ksi`` and generalized ``E``; the simulation uses generalized
``fyp``/``fyn`` (warns if they differ much from catalog). Three **load protocols** (subfolders ``load_*``):
``n_half = n_peak/2``.

1. ``n_half, -n_half, n_peak, -n_peak, 0``
2. ``n_peak, -n_peak, n_peak, -n_peak, 0``
3. ``n_peak, -n_peak, n_half, -n_half, 0``

If experimental F--u cannot be resolved, falls back to ``n_peak=10``, ``n_half=5``.

**Geometry ratios** (same interpretation as before: ``L_y/L_T`` is yielding fraction, ``L_T = L_y / (L_y/L_T)``):

- Default brace for **material** sweeps: ``L_y/L_T`` = ``DEFAULT_LY_OVER_LT``, ``A_t/A_sc`` = ``DEFAULT_AT_OVER_ASC``.
- One figure **sweep_LyLt.png**: vary ``L_y/L_T`` over ``LY_OVER_LT_VALUES``; ``A_t/A_sc`` fixed at default.
- One figure **sweep_AtAsc.png**: vary ``A_t/A_sc`` over ``AT_OVER_ASC_VALUES``; ``L_y/L_T`` fixed at default.

SteelMPF: varies one parameter at a time (``STEEL_PARAM_EXPLORATIONS``) at default geometry, including
post-``a4`` parameters ``fup_ratio``, ``fun_ratio``, and ``Ru0`` (via ``model.corotruss``: ``-ult`` then ``fup``, ``fun``, ``Ru0``).
``sweep_Ru0.png`` pins ``f_{\mathrm{up}}/f_{\mathrm{yp}}`` to ``RU0_SWEEP_FUP_RATIO`` (default 2.0) on every displacement driver.

On every figure (for each load protocol): **experimental** ``F``--``δ`` from the same resolved CSV as the peak-strain
logic, plotted as subsampled scatter (``σ/f_y`` vs ``δ/(L_y ε_y)``) for reference alongside the swept model curves.
The test ``δ`` sequence differs from the synthetic protocol; both are shown in the same normalized axes.

Plots use matplotlib mathtext on axes and legends. Each sweep draws curves with line styles ``-``, ``--``, and
``-.`` (cycled if there are more than three). Footer annotations use ``$...$`` for math.

Run from repository root (optional ``--out-dir`` only; set specimen in the script)::

    python config/steelmpf_param_sweeps/plot_unit_truss_steelmpf_param_sweeps.py

Each sweep subfolder writes a prescribed-$\delta$ diagnostic with ``x`` markers: by default
``displacement_vs_cumulative_deformation.png`` ($\delta$ vs cumulative $\sum|\Delta\delta|$). Subfolders listed in
``DISPLACEMENT_STEP_INDEX_SUBDIRS`` write ``displacement_vs_step.png`` ($\delta$ vs zero-based step index) instead.
``resampled_eval_history`` plots **only** that subfolder's drive (same as ``load_*``); no second history on the
cumulative-deformation diagnostic. ``resampled_eval_history`` also writes ``sweep_a3_fup_ratio_<tag>.png``:
``a3`` sweep at fixed ``f_{\mathrm{up}}/f_{\mathrm{yp}}`` (``RESAMPLED_EVAL_SWEEP_A3_EXTRA_FUP_RATIO``), using
``RESAMPLED_EVAL_SWEEP_A3_EXTRA_VALUES`` (through 0.08 by default), not the shorter ``sweep_a3.png`` grid.
If ``RESAMPLED_EVAL_SWEEP_A3_EXTRA_BP_BN`` is set, a second PNG uses the same sweep with that ``(b_p, b_n)``.

PNG layout: ``<out-dir>/<protocol>/sweep_<name>.png`` (steel keys, ``LyLt``, ``AtAsc``, ``sweep_Dsample.png``,
``sweep_prependZero.png`` compares explicit ``\delta=0`` prepend vs none (diagnostic); except ``resampled_eval_history``
(no ``Dsample``, no prepend sweep; drive is raw
CSV). ``Dsample`` varies ``f`` in ``D_\mathrm{sample}=f\,L_y\,\varepsilon_y``
(inch) over ``DSAMPLE_FRAC_VALUES``. Once per synthetic
protocol (peaks from experimental F--u), plus:

- ``<out-dir>/resampled_eval_history/`` — full ``Deformation[in]`` series as in
  ``config/test/run_eval_params_metrics.py``.

Production resampled drives use a leading ``0.1\,D_\mathrm{sample}`` (brace-based); ``sweep_prependZero.png`` alone
still uses a true ``\delta=0`` prepend for A/B testing.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_SCRIPTS = _PROJECT_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS / "postprocess"))

from calibrate.deformation_history_drive import PREPEND_LEAD_FRAC  # noqa: E402
from calibrate.digitized_unordered_eval_lib import load_digitized_unordered_series  # noqa: E402
from calibrate.resample_experiment import d_sampling_from_brace_params  # noqa: E402
from model.brace_geometry import compute_Q  # noqa: E402
from model.corotruss import run_simulation  # noqa: E402
from postprocess.plot_dimensions import (  # noqa: E402
    HYSTERESIS_LINEWIDTH_SCALE,
    SAVE_DPI,
    configure_matplotlib_style,
)
from specimen_catalog import (  # noqa: E402
    DEFORMATION_IN_COL,
    FORCE_KIP_COL,
    list_names_digitized_unordered,
    max_abs_strain_delta_over_Ly,
    path_ordered_resampled_force_csv_stems,
    read_catalog,
    resolve_force_deformation_csv_for_max_strain,
    resolve_resampled_force_deformation_csv,
)

configure_matplotlib_style()

# Stress–strain hysteresis loops (σ/f_y vs δ/(L_y ε_y)): match repo hysteresis stroke scale.
_STEELMPF_HYST_LOOP_LW = 1.2 * HYSTERESIS_LINEWIDTH_SCALE

# --- Edit specimen / calibration row (not CLI) ---
# Non-blank: load ``L_y``, ``A_sc``, ``f_yc`` from BRB-Specimens.csv and steel from generalized CSV
# (``GENERALIZED_SET_ID`` / ``GENERALIZED_CSV_OVERRIDE``). Blank ``""``: use the manual defaults below.
SPECIMEN_NAME_INPUT = "PC3SB"
GENERALIZED_SET_ID = 3
# ``None`` uses ``_GENERALIZED_PARAMS_PATH``; set a ``Path`` to override the CSV location.
GENERALIZED_CSV_OVERRIDE: Path | None = None
# If not ``None``, merge steel from ``optimized_brb_parameters.csv`` for ``(Name, set_id)`` steelmpf when present.
OPTIMIZED_BRB_PARAMETERS_SET_ID: int | None = 3
# ``None`` uses ``_OPTIMIZED_BRB_PARAMETERS_PATH`` below.
OPTIMIZED_BRB_PARAMETERS_CSV: Path | None = None

# After loading a specimen row, merge these into ``DEFAULT_STEEL`` (overrides generalized CSV for these keys).
# Leave empty ``{}`` to use the generalized row as-is. If you override ``E``, also set module ``E`` and ``EPS_Y``.
SPECIMEN_STEEL_OVERRIDES: dict[str, float] = {}

# --- Manual BRB defaults (ksi, in, in²; used only when SPECIMEN_NAME_INPUT is blank — edit here) ---
E = 29000.0
FY = 44.522
L_Y = 120.0
A_SC = 11.2
EPS_Y = FY / E
SPECIMEN_NAME: str = ""
DEFAULT_LY_OVER_LT = L_Y / 215.0
DEFAULT_AT_OVER_ASC = 33.6 / A_SC
DEFAULT_STEEL: dict[str, float] = {
    "fyp": FY,
    "fyn": FY,
    "E": E,
    "b_p": 0.0207186,
    "b_n": 0.0377999,
    "R0": 20.0,
    "cR1": 0.864087,
    "cR2": 0.169528,
    "a1": 0.0381144,
    "a2": 1.0,
    "a3": 0.0347035,
    "a4": 1.0,
    "fup_ratio": 4.0,
    "fun_ratio": 4.0,
    "Ru0": 5.0,
}

_FALLBACK_N_PEAK = 10.0
_CATALOG_PATH = _PROJECT_ROOT / "config" / "calibration" / "BRB-Specimens.csv"
_GENERALIZED_PARAMS_PATH = (
    _PROJECT_ROOT / "results" / "calibration" / "generalized_optimize" / "generalized_brb_parameters.csv"
)
_OPTIMIZED_BRB_PARAMETERS_PATH = (
    _PROJECT_ROOT / "results" / "calibration" / "individual_optimize" / "optimized_brb_parameters.csv"
)
# SteelMPF when ``generalized_brb_parameters.csv`` is missing or has no row for (Name, set_id).
# Strengths ``fyp``/``fyn`` are taken from catalog ``f_yc_ksi`` in that case; geometry always from the catalog.
_GENERIC_STEEL_FALLBACK: dict[str, float] = {
    "E": 29000.0,
    "b_p": 0.02,
    "b_n": 0.02,
    "R0": 20.0,
    "cR1": 0.925,
    "cR2": 0.15,
    "a1": 0.04,
    "a2": 1.0,
    "a3": 0.04,
    "a4": 1.0,
    "fup_ratio": 4.0,
    "fun_ratio": 4.0,
    "Ru0": 5.0,
}
_GEOMETRY_TOL_IN = 0.05
# Subsample experimental points for scatter (full series can be 1e4+).
EXPERIMENTAL_SCATTER_MAX_POINTS = 2500

# Cumulative deformation figure: mark each prescribed step with ``x`` (dense series use small markers).
_CUM_DEF_MARKER_KW: dict[str, object] = {
    "marker": "x",
    "markersize": 2.5,
    "markevery": 1,
    "markeredgewidth": 0.85,
}

# Output folder for sweeps driven by the same ``Deformation[in]`` as ``run_eval_params_metrics.py``.
RESAMPLED_EVAL_HISTORY_SUBDIR = "resampled_eval_history"
# Separate ``sweep_a3_fup_ratio_<tag>.png`` in ``resampled_eval_history`` only (``a3`` sweep at this ``fup_ratio``).
RESAMPLED_EVAL_SWEEP_A3_EXTRA_FUP_RATIO: float = 2.0
# ``a3`` values for that figure only (extends through 0.08; other sweeps still use ``STEEL_PARAM_EXPLORATIONS``).
RESAMPLED_EVAL_SWEEP_A3_EXTRA_VALUES: tuple[float, ...] = (0.0, 0.02, 0.04, 0.06, 0.08)
# If not ``None``, also write ``sweep_a3_fup_ratio_<tag>_bp<...>_bn<...>.png`` (same ``a3`` grid + ``fup_ratio``).
RESAMPLED_EVAL_SWEEP_A3_EXTRA_BP_BN: tuple[float, float] | None = (0.005, 0.02)

# Displacement increment per OpenSees step (approx.): ``D_\mathrm{sample} = f\,L_y\,\varepsilon_y`` (inch).
DSAMPLE_FRAC_VALUES: tuple[float, ...] = (0.20, 0.50, 1.0)

# Subfolders that write ``displacement_vs_step.png`` (``\\delta`` vs step index) instead of cumulative deformation.
DISPLACEMENT_STEP_INDEX_SUBDIRS: frozenset[str] = frozenset({"load_peak_-peak_half_-half_0"})

# Fixed σ/f_y limits on sweep hysteresis figures (stable y-scale).
SWEEP_SIGMA_OVER_FY_YLIM: tuple[float, float] = (-2.5, 2.5)


def _set_sweep_sigma_over_fy_ylim(ax: plt.Axes) -> None:
    ax.set_ylim(SWEEP_SIGMA_OVER_FY_YLIM)


@dataclass(frozen=True)
class LoadProtocol:
    """Piecewise path from 0 through waypoint multipliers n (u = n * eps_y * L_y), last target 0."""

    subdir: str
    waypoint_mults: tuple[float, ...]


# Individual-optimize columns merged into sweeps: classic SteelMPF only (no ``deltap``/``deltan``/``cb*``, no ``fup_ratio``/``fun_ratio``/``Ru0``).
_OPTIMIZED_STEEL_KEYS: tuple[str, ...] = (
    "E",
    "b_p",
    "b_n",
    "R0",
    "cR1",
    "cR2",
    "a1",
    "a2",
    "a3",
    "a4",
    "fyp",
    "fyn",
)


def _apply_optimized_brb_parameters_steel(
    steel: dict[str, float],
    *,
    specimen_name: str,
    set_id: int | None,
    csv_path: Path | None = None,
) -> tuple[dict[str, float], int]:
    """
    Overlay ``optimized_brb_parameters.csv`` steel columns onto ``steel`` for a steelmpf row.

    Only keys present and finite in the CSV are copied. Does not read decay deltas, ``cb*``, or ``fup_ratio`` /
    ``fun_ratio`` / ``Ru0`` — those stay from generalized defaults or generic fallback.

    Returns ``(updated_steel, n_keys_merged)`` for logging (``n_keys_merged`` is 0 if nothing was applied).
    """
    if set_id is None:
        return steel, 0
    path = Path(csv_path).expanduser().resolve() if csv_path is not None else _OPTIMIZED_BRB_PARAMETERS_PATH
    if not path.is_file():
        return steel, 0
    try:
        odf = pd.read_csv(path, comment="#", skipinitialspace=True)
    except (OSError, ValueError, pd.errors.ParserError):
        return steel, 0
    odf.columns = odf.columns.astype(str).str.strip()
    need = ("Name", "set_id", "steel_model")
    if not all(c in odf.columns for c in need):
        return steel, 0
    sm = odf["steel_model"].astype(str).str.strip().str.lower()
    sub = odf[
        (odf["Name"].astype(str) == specimen_name)
        & (odf["set_id"].astype(int) == int(set_id))
        & (sm.isin(("steelmpf", "steel_mpf", "mpf")))
    ]
    if sub.empty:
        print(
            f"Note: no steelmpf row in {path.name} for Name={specimen_name!r}, set_id={set_id}; "
            "keeping generalized/generic steel."
        )
        return steel, 0
    row = sub.iloc[0]
    out = dict(steel)
    merged: list[str] = []
    for k in _OPTIMIZED_STEEL_KEYS:
        if k not in row.index or pd.isna(row[k]):
            continue
        try:
            out[k] = float(row[k])
        except (TypeError, ValueError):
            continue
        merged.append(k)
    if merged:
        print(f"Steel from {path.name} (set_id={set_id}): {', '.join(merged)}")
    return out, len(merged)


def configure_specimen(
    name: str,
    set_id: int,
    *,
    generalized_csv: Path | None = None,
) -> None:
    """
    If ``name`` is blank (after strip), keep the module-level manual defaults (``E``, ``FY``, ``L_Y``, ``A_SC``,
    ``DEFAULT_LY_OVER_LT``, ``DEFAULT_AT_OVER_ASC``, ``DEFAULT_STEEL``) and only refresh ``EPS_Y = FY / E``.

    Otherwise load ``name`` from BRB-Specimens.csv (geometry, ``f_yc_ksi``) and the matching generalized row.
    If the generalized CSV is missing or has no row for this specimen and ``set_id``, steel uses
    ``_GENERIC_STEEL_FALLBACK`` with ``fyp``/``fyn`` = catalog ``f_yc_ksi``.

    Sets module globals: ``SPECIMEN_NAME``, ``L_Y``, ``A_SC``, ``FY`` (catalog), ``E``, ``EPS_Y``, ``DEFAULT_STEEL``,
    ``DEFAULT_LY_OVER_LT``, ``DEFAULT_AT_OVER_ASC``. Invoked once at import (after this definition) and may be called
    again to switch specimen in the same interpreter.
    """
    global E, FY, L_Y, A_SC, EPS_Y, SPECIMEN_NAME, DEFAULT_STEEL, DEFAULT_LY_OVER_LT, DEFAULT_AT_OVER_ASC

    SPECIMEN_NAME = str(name).strip()
    if not SPECIMEN_NAME:
        EPS_Y = FY / E
        print(
            "Using manual geometry/steel defaults (SPECIMEN_NAME_INPUT blank; edit the "
            "'Manual BRB defaults' block)."
        )
        return

    cat = read_catalog(_CATALOG_PATH)
    sel = cat[cat["Name"].astype(str) == SPECIMEN_NAME]
    if sel.empty:
        raise SystemExit(f"Specimen {SPECIMEN_NAME!r} not found in {_CATALOG_PATH}")

    r = sel.iloc[0]
    L_Y = float(r["L_y_in"])
    A_SC = float(r["A_c_in2"])
    a_t_cat = float(r["A_t_in2"])
    l_t_cat = float(r["L_T_in"])
    FY = float(r["f_yc_ksi"])

    gpath = Path(generalized_csv).expanduser().resolve() if generalized_csv else _GENERALIZED_PARAMS_PATH
    sid = int(set_id)
    use_generic_steel = False
    g: pd.Series | None = None

    if not gpath.is_file():
        print(f"Warning: generalized parameters CSV not found ({gpath}); using generic SteelMPF (_GENERIC_STEEL_FALLBACK).")
        use_generic_steel = True
    else:
        gdf = pd.read_csv(gpath)
        for req in ("Name", "set_id", "E", "fyp", "fyn", "b_p", "b_n", "R0", "cR1", "cR2", "a1", "a2", "a3", "a4"):
            if req not in gdf.columns:
                raise SystemExit(f"{gpath} missing required column {req!r}")

        gr = gdf[(gdf["Name"].astype(str) == SPECIMEN_NAME) & (gdf["set_id"].astype(int) == sid)]
        if gr.empty:
            print(
                f"Warning: no generalized row for Name={SPECIMEN_NAME!r}, set_id={sid} in {gpath.name}; "
                "using generic SteelMPF (_GENERIC_STEEL_FALLBACK)."
            )
            use_generic_steel = True
        else:
            g = gr.iloc[0]

    if use_generic_steel:
        fb = dict(_GENERIC_STEEL_FALLBACK)
        E = float(fb["E"])
        EPS_Y = FY / E
        fy_cat = float(FY)
        DEFAULT_STEEL = {
            "fyp": fy_cat,
            "fyn": fy_cat,
            **fb,
        }
        steel_note = "generic fallback (_GENERIC_STEEL_FALLBACK)"
    else:
        assert g is not None
        E = float(g["E"])
        EPS_Y = FY / E

        fyp_g = float(g["fyp"])
        fyn_g = float(g["fyn"])
        if abs(fyp_g - FY) > 0.02 or abs(fyn_g - FY) > 0.02:
            print(
                f"Note: catalog f_yc_ksi={FY:g} vs generalized fyp/fyn={fyp_g:g}/{fyn_g:g}; "
                "using catalog FY for ε_y and σ/f_y axis; simulation uses generalized fyp/fyn."
            )

        DEFAULT_STEEL = {
            "fyp": fyp_g,
            "fyn": fyn_g,
            "E": E,
            "b_p": float(g["b_p"]),
            "b_n": float(g["b_n"]),
            "R0": float(g["R0"]),
            "cR1": float(g["cR1"]),
            "cR2": float(g["cR2"]),
            "a1": float(g["a1"]),
            "a2": float(g["a2"]),
            "a3": float(g["a3"]),
            "a4": float(g["a4"]),
        }
        for _dk, _dv in (("fup_ratio", 4.0), ("fun_ratio", 4.0), ("Ru0", 5.0)):
            if _dk not in g.index or pd.isna(g[_dk]):
                DEFAULT_STEEL[_dk] = _dv
            else:
                DEFAULT_STEEL[_dk] = float(g[_dk])

        steel_note = gpath.name

    DEFAULT_LY_OVER_LT = L_Y / l_t_cat
    DEFAULT_AT_OVER_ASC = a_t_cat / A_SC

    DEFAULT_STEEL, n_opt_steel = _apply_optimized_brb_parameters_steel(
        DEFAULT_STEEL,
        specimen_name=SPECIMEN_NAME,
        set_id=OPTIMIZED_BRB_PARAMETERS_SET_ID,
        csv_path=OPTIMIZED_BRB_PARAMETERS_CSV,
    )
    if n_opt_steel > 0 and OPTIMIZED_BRB_PARAMETERS_SET_ID is not None:
        opt_src = (
            Path(OPTIMIZED_BRB_PARAMETERS_CSV).expanduser().resolve()
            if OPTIMIZED_BRB_PARAMETERS_CSV is not None
            else _OPTIMIZED_BRB_PARAMETERS_PATH
        )
        steel_note = (
            f"{steel_note} + {opt_src.name} (individual optimize, set_id={OPTIMIZED_BRB_PARAMETERS_SET_ID})"
        )

    if not use_generic_steel and g is not None:
        for col, cat_val in (
            ("L_y", L_Y),
            ("L_T", l_t_cat),
            ("A_sc", A_SC),
            ("A_t", a_t_cat),
        ):
            gv = float(g[col])
            if abs(gv - cat_val) > _GEOMETRY_TOL_IN:
                print(
                    f"Warning: generalized {col}={gv:g} vs catalog {cat_val:g} for {SPECIMEN_NAME!r} "
                    f"(tolerance {_GEOMETRY_TOL_IN} in)."
                )

    print(
        f"Specimen {SPECIMEN_NAME!r}, set_id={sid}: L_y={L_Y:g} in, A_sc={A_SC:g} in², "
        f"f_yc={FY:g} ksi, steel from {steel_note}"
    )


configure_specimen(
    SPECIMEN_NAME_INPUT,
    GENERALIZED_SET_ID,
    generalized_csv=GENERALIZED_CSV_OVERRIDE,
)

if SPECIMEN_STEEL_OVERRIDES:
    for _ok, _ov in SPECIMEN_STEEL_OVERRIDES.items():
        DEFAULT_STEEL[_ok] = float(_ov)


def strain_peak_mults_from_specimen() -> tuple[float, float, float]:
    """
    Return (n_half, n_peak, max_abs_delta_over_Ly).

    n_peak matches max experimental |δ|/L_y divided by eps_y = f_y/E (same normalization as plot axis).
    n_half = n_peak / 2.
    """
    cat = read_catalog(_CATALOG_PATH)
    doly = max_abs_strain_delta_over_Ly(
        SPECIMEN_NAME,
        cat,
        project_root=_PROJECT_ROOT,
        ly_in=L_Y,
    )
    if doly is None or doly <= 0.0 or not np.isfinite(doly):
        n_peak = float(_FALLBACK_N_PEAK)
        print(
            f"Warning: could not get max |delta|/L_y for {SPECIMEN_NAME!r}; "
            f"using fallback n_peak={n_peak}, n_half={n_peak / 2.0}."
        )
        return n_peak / 2.0, n_peak, float("nan")

    n_peak = float(doly / EPS_Y)
    n_half = n_peak / 2.0
    print(
        f"{SPECIMEN_NAME}: max|delta|/L_y = {doly:.6g} in/in -> "
        f"n_peak = delta/(L_y*eps_y) = {n_peak:.4g}, n_half = {n_half:.4g}"
    )
    return n_half, n_peak, doly


def load_experimental_norm_scatter(
    *,
    max_points: int = EXPERIMENTAL_SCATTER_MAX_POINTS,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Experimental ``(δ/(L_y ε_y), σ/f_y)`` using catalog ``L_y``, ``f_yc``, ``A_c`` and the same F--u CSV
    as ``max_abs_strain_delta_over_Ly`` / ``resolve_force_deformation_csv_for_max_strain``.
    """
    if not SPECIMEN_NAME:
        return None
    cat = read_catalog(_CATALOG_PATH)
    path = resolve_force_deformation_csv_for_max_strain(
        SPECIMEN_NAME,
        cat,
        project_root=_PROJECT_ROOT,
    )
    if path is None or not path.is_file():
        return None
    try:
        df = pd.read_csv(path, usecols=[DEFORMATION_IN_COL, FORCE_KIP_COL])
    except ValueError:
        df = pd.read_csv(path)
        if DEFORMATION_IN_COL not in df.columns or FORCE_KIP_COL not in df.columns:
            return None
    u = pd.to_numeric(df[DEFORMATION_IN_COL], errors="coerce").to_numpy(dtype=float)
    f_kip = pd.to_numeric(df[FORCE_KIP_COL], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(u) & np.isfinite(f_kip)
    u, f_kip = u[ok], f_kip[ok]
    if u.size == 0:
        return None
    denom = float(L_Y * EPS_Y)
    if denom <= 0.0 or not np.isfinite(denom):
        return None
    x = u / denom
    stress_ksi = f_kip / float(A_SC)
    y = stress_ksi / float(FY)
    n = int(x.size)
    if n > max_points:
        idx = np.unique(np.linspace(0, n - 1, max_points, dtype=float).astype(int))
        x, y = x[idx], y[idx]
    return x, y


def load_eval_driving_displacement() -> tuple[np.ndarray, np.ndarray] | None:
    """
    Driving displacement aligned with ``config/test/run_eval_params_metrics.py``:

    - **Path-ordered** specimens: ``Deformation[in]`` from ``resolve_resampled_force_deformation_csv``.
    - **Digitized unordered**: ``load_digitized_unordered_series`` (pipeline resampled ``deformation_history``
      when present, same defaults as that script).
    """
    if not SPECIMEN_NAME:
        return None
    cat = read_catalog(_CATALOG_PATH)
    name = SPECIMEN_NAME
    root = _PROJECT_ROOT

    resampled_stems = path_ordered_resampled_force_csv_stems(cat, project_root=root)
    unordered_ok = set(list_names_digitized_unordered(cat, project_root=root))

    if name in resampled_stems:
        csv_path = resolve_resampled_force_deformation_csv(name, root)
        if csv_path is None or not csv_path.is_file():
            print(f"Warning: eval driving history: missing resampled force_deformation for {name!r}.")
            return None
        df = pd.read_csv(csv_path)
        if DEFORMATION_IN_COL not in df.columns:
            print(f"Warning: eval driving history: no {DEFORMATION_IN_COL!r} in {csv_path}.")
            return None
        D = pd.to_numeric(df[DEFORMATION_IN_COL], errors="coerce").to_numpy(dtype=float)
        D = D[np.isfinite(D)]
    elif name in unordered_ok:
        sel = cat[cat["Name"].astype(str) == name]
        if sel.empty:
            return None
        cat_row = sel.iloc[0]
        steel_row: dict[str, float] = {
            **DEFAULT_STEEL,
            "L_T": float(L_Y / DEFAULT_LY_OVER_LT),
            "L_y": float(L_Y),
            "A_sc": float(A_SC),
            "A_t": float(DEFAULT_AT_OVER_ASC * A_SC),
        }
        loaded = load_digitized_unordered_series(
            name,
            root,
            steel_row=steel_row,
            catalog_row=cat_row,
            prepare_drive=True,
            use_pipeline_resampled_drive=True,
        )
        if loaded is None:
            print(f"Warning: eval driving history: could not load digitized-unordered series for {name!r}.")
            return None
        D_arr, _, _ = loaded
        D = np.asarray(D_arr, dtype=float)
        D = D[np.isfinite(D)]
    else:
        print(
            f"Warning: eval driving history: {name!r} is not in path-ordered resampled set "
            f"and not digitized-unordered eligible (see run_eval_params_metrics.py)."
        )
        return None

    if D.size == 0:
        return None
    ly_eps = float(L_Y) * float(EPS_Y)
    if ly_eps <= 0.0 or not np.isfinite(ly_eps):
        return None
    strain_norm_ly = D / ly_eps
    return D, strain_norm_ly


def make_load_protocols(n_half: float, n_peak: float) -> tuple[LoadProtocol, ...]:
    nh, np_ = float(n_half), float(n_peak)
    return (
        LoadProtocol(
            "load_half_-half_peak_-peak_0",
            (nh, -nh, np_, -np_, 0.0),
        ),
        LoadProtocol(
            "load_peak_-peak_peak_-peak_0",
            (np_, -np_, np_, -np_, 0.0),
        ),
        LoadProtocol(
            "load_peak_-peak_half_-half_0",
            (np_, -np_, nh, -nh, 0.0),
        ),
    )


# With a non-blank specimen, ``configure_specimen`` overwrites ``DEFAULT_*`` / ``DEFAULT_STEEL`` at import.

# L_y/L_T (yielding length / total brace length). L_T = L_Y / fraction.
LY_OVER_LT_VALUES: tuple[float, ...] = (0.5, 0.6, 0.70)
# A_t / A_sc
AT_OVER_ASC_VALUES: tuple[float, ...] = (3.0, 5.0, 10.0)

# Matplotlib linestyle for each curve in a sweep (solid, dashed, dash-dot).
SWEEP_LINE_STYLES: tuple[str, ...] = ("-", "--", "-.")

# When sweeping ``Ru0`` only: hold ``fup_ratio`` at this value (``fun_ratio`` still from ``DEFAULT_STEEL``).
RU0_SWEEP_FUP_RATIO = 2.0

STEEL_PARAM_EXPLORATIONS: dict[str, list[float]] = {
    "R0": [5.0, 10.0, 20.0],
    "cR1": [0.875, 0.90, 0.925],
    "cR2": [0.0015, 0.15, 0.30],
    "a1": [0.0, 0.02, 0.04],
    "a2": [1.0, 5.0, 10.0],
    "a3": [0.0, 0.02, 0.04],
    "a4": [1.0, 5.0, 10.0],
    "fup_ratio": [1.5, 1.75, 2.0, 4.0],
    "fun_ratio": [1.5, 1.75, 2.0, 4.0],
    "Ru0": [1.0, 2.0, 3.0, 5.0, 10.0],
}


def _steel_math_name(param: str) -> str:
    """Matplotlib mathtext body (no outer $) for OpenSees steel keys."""
    return {
        "R0": r"R_0",
        "cR1": r"c_{\mathrm{R1}}",
        "cR2": r"c_{\mathrm{R2}}",
        "a1": r"a_1",
        "a2": r"a_2",
        "a3": r"a_3",
        "a4": r"a_4",
        "fup_ratio": r"f_{\mathrm{up}}/f_{\mathrm{yp}}",
        "fun_ratio": r"f_{\mathrm{un}}/f_{\mathrm{yn}}",
        "Ru0": r"R_{\mathrm{u0}}",
    }.get(param, param)


def _default_L_T_A_t() -> tuple[float, float]:
    L_T = L_Y / DEFAULT_LY_OVER_LT
    A_t = DEFAULT_AT_OVER_ASC * A_SC
    return L_T, A_t


def _specimen_d_sampling_inches(*, u_fallback: np.ndarray | None = None) -> float:
    r"""``D_\mathrm{sample}`` [in] from brace geometry (Dy/4), same rule as ``resample_filtered``."""
    L_T, A_t = _default_L_T_A_t()
    uf = u_fallback if u_fallback is not None else np.array([0.0], dtype=float)
    return float(
        d_sampling_from_brace_params(
            fyp_ksi=FY,
            L_T_in=L_T,
            L_y_in=L_Y,
            A_sc_in2=A_SC,
            A_t_in2=A_t,
            E_ksi=E,
            u_fallback=uf,
        )
    )


def _fmt_legend_val(param: str, v: float) -> str:
    if param == "R0":
        return f"{v:.0f}"
    if param == "cR1":
        return f"{v:.3f}"
    if param == "cR2":
        return f"{v:.4f}".rstrip("0").rstrip(".") if v < 0.01 else f"{v:g}"
    if param in ("a1", "a3"):
        return f"{v:.2f}"
    if param in ("a2", "a4"):
        return f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:g}"
    if param in ("fup_ratio", "fun_ratio"):
        return f"{v:.2f}"
    if param == "Ru0":
        return f"{v:.2f}" if abs(v - round(v)) >= 1e-9 else f"{v:.0f}"
    return f"{v:g}"


def build_displacement_path(
    waypoints_u: list[float],
    steps: list[int],
    *,
    prepend_lead: bool = True,
    d_sample_in: float | None = None,
    prepend_zero: bool = False,
) -> np.ndarray:
    if len(steps) != len(waypoints_u):
        raise ValueError("steps and waypoints_u must have the same length.")
    start = 0.0
    legs: list[np.ndarray] = []
    for target, n in zip(waypoints_u, steps):
        if n < 0:
            raise ValueError("Each segment needs n >= 0 steps.")
        if n == 0:
            seg = np.array([], dtype=float)
        else:
            seg = np.linspace(start, target, n + 1)[1:]
        legs.append(seg)
        start = target
    out = np.concatenate(legs)
    if prepend_zero:
        # Legacy explicit leading δ=0 (diagnostic only; SteelMPF history quirk).
        if out.size == 0:
            return np.array([0.0])
        if not np.isclose(out[0], 0.0, rtol=0.0, atol=1e-12):
            out = np.concatenate((np.array([0.0], dtype=float), out))
        return out
    if prepend_lead:
        if d_sample_in is None or float(d_sample_in) <= 0.0 or not np.isfinite(float(d_sample_in)):
            raise ValueError("prepend_lead requires positive finite d_sample_in.")
        lead = float(PREPEND_LEAD_FRAC) * float(d_sample_in)
        if out.size == 0:
            return np.array([lead])
        out = out.astype(float, copy=True)
        if np.isclose(out[0], 0.0, rtol=0.0, atol=1e-12):
            out[0] = lead
        elif not np.isclose(out[0], lead, rtol=0.0, atol=1e-12):
            out = np.concatenate((np.array([lead], dtype=float), out))
        return out
    if out.size == 0:
        return np.array([0.0])
    return out


def _waypoints_u_from_mults(mult_t: tuple[float, ...]) -> list[float]:
    scale = EPS_Y * L_Y
    return [float(m) * scale for m in mult_t]


def _steps_for_waypoints(
    waypoints_u: list[float],
    *,
    uref_displacement: float,
    steps_per_uref: int = 220,
    minimum: int = 50,
) -> list[int]:
    """Scale step counts so a displacement span of ``uref_displacement`` gets ~``steps_per_uref`` steps."""
    if uref_displacement <= 0.0 or not np.isfinite(uref_displacement):
        uref_displacement = float(_FALLBACK_N_PEAK * EPS_Y * L_Y)
    start = 0.0
    out: list[int] = []
    for target in waypoints_u:
        span = abs(target - start)
        n = max(minimum, int(round(steps_per_uref * span / uref_displacement)))
        out.append(n)
        start = target
    return out


def _steps_for_dsample_displacement(
    waypoints_u: list[float],
    d_sample_in: float,
    *,
    minimum: int = 0,
) -> list[int]:
    """``ceil(span / d_sample)`` steps per leg (at least ``minimum``; default 0). Zero-span legs use 0 steps."""
    if d_sample_in <= 0.0 or not np.isfinite(d_sample_in):
        raise ValueError("d_sample_in must be positive and finite.")
    start = 0.0
    out: list[int] = []
    for target in waypoints_u:
        span = abs(target - start)
        n_raw = int(np.ceil(span / d_sample_in)) if span > 0.0 else 0
        n = max(minimum, n_raw)
        out.append(n)
        start = target
    return out


def displacement_strain_from_waypoints_dsample(
    waypoints_u: list[float],
    d_sample_in: float,
) -> tuple[np.ndarray, np.ndarray]:
    steps = _steps_for_dsample_displacement(waypoints_u, d_sample_in)
    disp = build_displacement_path(waypoints_u, steps, d_sample_in=d_sample_in)
    strain_norm_ly = disp / (L_Y * EPS_Y)
    return disp, strain_norm_ly


def displacement_strain_for_protocol(
    proto: LoadProtocol,
    *,
    uref_displacement: float,
    d_sample_in: float,
) -> tuple[np.ndarray, np.ndarray]:
    wu = _waypoints_u_from_mults(proto.waypoint_mults)
    steps = _steps_for_waypoints(wu, uref_displacement=uref_displacement)
    disp = build_displacement_path(wu, steps, d_sample_in=d_sample_in)
    strain_norm_ly = disp / (L_Y * EPS_Y)
    return disp, strain_norm_ly


def _footer_geometry_line(
    *,
    L_T: float,
    L_y: float,
    ly_over_lt: float,
    A_sc: float,
    A_t: float,
    at_over_asc: float,
) -> str:
    Q = compute_Q(L_T, L_y, A_sc, A_t)
    return (
        rf"$L_T={L_T:.1f}\,\mathrm{{in}}$, $L_y={L_y:.1f}\,\mathrm{{in}}$, "
        rf"$L_y/L_T={ly_over_lt:.2f}$, $A_{{\mathrm{{sc}}}}={A_sc:g}\,\mathrm{{in}}^2$, "
        rf"$A_t/A_{{\mathrm{{sc}}}}={at_over_asc:g}$, $Q={Q:.4f}$"
    )


def _place_sweep_footer_three_lines(
    fig: plt.Figure,
    line1: str,
    line2: str,
    line3: str,
    *,
    right: float = 0.98,
    bottom: float = 0.30,
) -> None:
    kw = dict(ha="center", fontsize=6.5, color="0.35")
    fig.text(0.5, 0.104, line1, **kw)
    fig.text(0.5, 0.068, line2, **kw)
    fig.text(0.5, 0.032, line3, ha="center", fontsize=6.5, color="0.35")
    top = 0.92 if (SPECIMEN_NAME or "").strip() else 0.96
    fig.subplots_adjust(left=0.11, right=right, top=top, bottom=bottom)


def _apply_specimen_axes_title(ax: plt.Axes) -> None:
    """Axes title from ``SPECIMEN_NAME`` when a catalog specimen is configured (manual mode: no title)."""
    name = (SPECIMEN_NAME or "").strip()
    if not name:
        return
    ax.set_title(name, fontsize=10, pad=4)


def _format_steel_key_chunk(d: dict[str, float], key: str) -> str:
    sym = _steel_math_name(key)
    val = d[key]
    if key == "R0":
        return rf"${sym}={val:.0f}$"
    if key in ("cR1", "a1", "a3"):
        return rf"${sym}={val:.3f}$"
    if key == "cR2":
        return rf"${sym}={val:.4g}$"
    if key in ("a2", "a4"):
        return rf"${sym}={val:.1f}$"
    if key in ("fup_ratio", "fun_ratio"):
        return rf"${sym}={val:.2f}$"
    if key == "Ru0":
        return rf"${sym}={val:.2f}$"
    return rf"${sym}={val:.1f}$"


def _footer_steel_lines_through_cr2(
    steel_exclude: str | None,
    *,
    steel_ref: dict[str, float] | None = None,
) -> tuple[str, str]:
    """Footer lines 2–3: through ``cR2``, then ``a1`` … ``a4`` (swept key omitted when given)."""
    d = DEFAULT_STEEL if steel_ref is None else steel_ref
    line2: list[str] = [
        rf"$E={d['E']:.0f}$",
        rf"$f_y={d['fyp']:.0f}$",
        rf"$b_p={d['b_p']:.3f}$",
        rf"$b_n={d['b_n']:.3f}$",
    ]
    line3: list[str] = []
    for key in ("R0", "cR1", "cR2"):
        if steel_exclude is not None and key == steel_exclude:
            continue
        line2.append(_format_steel_key_chunk(d, key))
    for key in ("a1", "a2", "a3", "a4"):
        if steel_exclude is not None and key == steel_exclude:
            continue
        line3.append(_format_steel_key_chunk(d, key))
    for key in ("fup_ratio", "fun_ratio", "Ru0"):
        if steel_exclude is not None and key == steel_exclude:
            continue
        line3.append(_format_steel_key_chunk(d, key))
    return "; ".join(line2), "; ".join(line3)


def _footer_steel_line(steel_exclude: str | None, *, steel_ref: dict[str, float] | None = None) -> str:
    a, b = _footer_steel_lines_through_cr2(steel_exclude, steel_ref=steel_ref)
    return f"{a}; {b}"


def run_brb(
    displacement: np.ndarray,
    *,
    L_T: float,
    L_y: float,
    A_sc: float,
    A_t: float,
    steel: dict[str, float],
) -> np.ndarray:
    return run_simulation(
        displacement,
        L_T=L_T,
        L_y=L_y,
        A_sc=A_sc,
        A_t=A_t,
        fyp=steel["fyp"],
        fyn=steel["fyn"],
        E=steel["E"],
        b_p=steel["b_p"],
        b_n=steel["b_n"],
        R0=steel["R0"],
        cR1=steel["cR1"],
        cR2=steel["cR2"],
        a1=steel["a1"],
        a2=steel["a2"],
        a3=steel["a3"],
        a4=steel["a4"],
        fup_ratio=float(steel.get("fup_ratio", 4.0)),
        fun_ratio=float(steel.get("fun_ratio", 4.0)),
        Ru0=float(steel.get("Ru0", 5.0)),
    )


def _steel_baseline_for_steel_sweep(_param: str) -> dict[str, float]:
    """Copy ``DEFAULT_STEEL`` for one-at-a-time steel sweeps."""
    return dict(DEFAULT_STEEL)


def _plot_experimental_reference_scatter(
    ax: plt.Axes,
    exp_xy: tuple[np.ndarray, np.ndarray] | None,
    *,
    zorder: float = 2.0,
) -> None:
    if exp_xy is None:
        return
    xe, ye = exp_xy
    if xe.size == 0:
        return
    ax.scatter(
        xe,
        ye,
        s=5,
        c="0.4",
        alpha=0.35,
        linewidths=0,
        label=r"$\mathrm{Test}$",
        zorder=zorder,
        rasterized=True,
    )


def plot_sweep_figure(
    out_path: Path,
    param: str,
    values: Iterable[float],
    disp: np.ndarray,
    strain_norm_ly: np.ndarray,
    *,
    L_T: float,
    L_y: float,
    ly_over_lt: float,
    A_sc: float,
    A_t: float,
    at_over_asc: float,
    exp_xy: tuple[np.ndarray, np.ndarray] | None,
    fixed_steel_overrides: dict[str, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    _plot_experimental_reference_scatter(ax, exp_xy, zorder=2.0)
    sym = _steel_math_name(param)
    base_steel = {**_steel_baseline_for_steel_sweep(param), **(fixed_steel_overrides or {})}
    for v, ls in zip(values, cycle(SWEEP_LINE_STYLES)):
        steel = {**base_steel, param: float(v)}
        force = run_brb(
            disp,
            L_T=L_T,
            L_y=L_y,
            A_sc=A_sc,
            A_t=A_t,
            steel=steel,
        )
        stress = force / A_sc
        lbl = rf"${sym} = {_fmt_legend_val(param, v)}$"
        ax.plot(strain_norm_ly, stress / FY, label=lbl, linewidth=_STEELMPF_HYST_LOOP_LW, linestyle=ls, zorder=3.0)

    ax.set_xlabel(r"$\delta / (L_y\,\varepsilon_y)$")
    ax.set_ylabel(r"$\sigma / f_y$")
    ax.axhline(0.0, color="0.75", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="0.75", linewidth=0.8, zorder=0)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.35, linewidth=0.6)
    _set_sweep_sigma_over_fy_ylim(ax)
    _apply_specimen_axes_title(ax)

    line1 = _footer_geometry_line(
        L_T=L_T,
        L_y=L_y,
        ly_over_lt=ly_over_lt,
        A_sc=A_sc,
        A_t=A_t,
        at_over_asc=at_over_asc,
    )
    line2, line3 = _footer_steel_lines_through_cr2(param, steel_ref=base_steel)
    _place_sweep_footer_three_lines(fig, line1, line2, line3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=SAVE_DPI)
    plt.close(fig)


def plot_ly_lt_sweep_figure(
    out_path: Path,
    disp: np.ndarray,
    strain_norm_ly: np.ndarray,
    exp_xy: tuple[np.ndarray, np.ndarray] | None,
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    _plot_experimental_reference_scatter(ax, exp_xy, zorder=2.0)
    A_t = DEFAULT_AT_OVER_ASC * A_SC
    for ly_over_lt, ls in zip(LY_OVER_LT_VALUES, cycle(SWEEP_LINE_STYLES)):
        L_T = L_Y / ly_over_lt
        force = run_brb(
            disp,
            L_T=L_T,
            L_y=L_Y,
            A_sc=A_SC,
            A_t=A_t,
            steel=DEFAULT_STEEL,
        )
        stress = force / A_SC
        lbl = rf"$L_y/L_T = {ly_over_lt:.2f}$"
        ax.plot(strain_norm_ly, stress / FY, label=lbl, linewidth=_STEELMPF_HYST_LOOP_LW, linestyle=ls, zorder=3.0)

    ax.set_xlabel(r"$\delta / (L_y\,\varepsilon_y)$")
    ax.set_ylabel(r"$\sigma / f_y$")
    ax.axhline(0.0, color="0.75", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="0.75", linewidth=0.8, zorder=0)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.35, linewidth=0.6)
    _set_sweep_sigma_over_fy_ylim(ax)
    _apply_specimen_axes_title(ax)

    line1 = (
        rf"$L_y={L_Y:.1f}\,\mathrm{{in}}$, $A_{{\mathrm{{sc}}}}={A_SC:g}\,\mathrm{{in}}^2$, "
        rf"$A_t/A_{{\mathrm{{sc}}}}={DEFAULT_AT_OVER_ASC:g}$"
    )
    line2, line3 = _footer_steel_lines_through_cr2(None)
    _place_sweep_footer_three_lines(fig, line1, line2, line3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=SAVE_DPI)
    plt.close(fig)


def plot_at_asc_sweep_figure(
    out_path: Path,
    disp: np.ndarray,
    strain_norm_ly: np.ndarray,
    exp_xy: tuple[np.ndarray, np.ndarray] | None,
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    _plot_experimental_reference_scatter(ax, exp_xy, zorder=2.0)
    L_T = L_Y / DEFAULT_LY_OVER_LT
    for at_over_asc, ls in zip(AT_OVER_ASC_VALUES, cycle(SWEEP_LINE_STYLES)):
        A_t = at_over_asc * A_SC
        force = run_brb(
            disp,
            L_T=L_T,
            L_y=L_Y,
            A_sc=A_SC,
            A_t=A_t,
            steel=DEFAULT_STEEL,
        )
        stress = force / A_SC
        lbl = rf"$A_t/A_{{\mathrm{{sc}}}} = {at_over_asc:g}$"
        ax.plot(strain_norm_ly, stress / FY, label=lbl, linewidth=_STEELMPF_HYST_LOOP_LW, linestyle=ls, zorder=3.0)

    ax.set_xlabel(r"$\delta / (L_y\,\varepsilon_y)$")
    ax.set_ylabel(r"$\sigma / f_y$")
    ax.axhline(0.0, color="0.75", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="0.75", linewidth=0.8, zorder=0)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.35, linewidth=0.6)
    _set_sweep_sigma_over_fy_ylim(ax)
    _apply_specimen_axes_title(ax)

    line1 = (
        rf"$L_T={L_T:.1f}\,\mathrm{{in}}$, $L_y={L_Y:.1f}\,\mathrm{{in}}$, "
        rf"$L_y/L_T={DEFAULT_LY_OVER_LT:.2f}$, $A_{{\mathrm{{sc}}}}={A_SC:g}\,\mathrm{{in}}^2$"
    )
    line2, line3 = _footer_steel_lines_through_cr2(None)
    _place_sweep_footer_three_lines(fig, line1, line2, line3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=SAVE_DPI)
    plt.close(fig)


def plot_prepend_zero_sweep_figure(
    out_path: Path,
    waypoints_u: list[float],
    steps: list[int],
    *,
    L_T: float,
    L_y: float,
    ly_over_lt: float,
    A_sc: float,
    A_t: float,
    at_over_asc: float,
    exp_xy: tuple[np.ndarray, np.ndarray] | None,
) -> None:
    r"""
    Same drive geometry and ``np.linspace`` leg counts; compare legacy ``prepend_zero=True`` (explicit leading
    ``\delta=0``) vs ``prepend_lead=False`` (production-style path without that extra sample).
    """
    disp_yes = build_displacement_path(waypoints_u, steps, prepend_zero=True)
    disp_no = build_displacement_path(waypoints_u, steps, prepend_lead=False)
    sn_yes = disp_yes / (float(L_y) * float(EPS_Y))
    sn_no = disp_no / (float(L_y) * float(EPS_Y))
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    _plot_experimental_reference_scatter(ax, exp_xy, zorder=2.0)
    for disp_i, sn_i, lbl, ls in (
        (
            disp_yes,
            sn_yes,
            r"$\mathrm{Prepend}\ \delta{=}0$",
            "-",
        ),
        (
            disp_no,
            sn_no,
            r"$\mathrm{No\ prepend}$",
            "--",
        ),
    ):
        force = run_brb(
            disp_i,
            L_T=L_T,
            L_y=L_y,
            A_sc=A_sc,
            A_t=A_t,
            steel=DEFAULT_STEEL,
        )
        stress = force / A_sc
        ax.plot(sn_i, stress / FY, label=lbl, linewidth=_STEELMPF_HYST_LOOP_LW, linestyle=ls, zorder=3.0)

    ax.set_xlabel(r"$\delta / (L_y\,\varepsilon_y)$")
    ax.set_ylabel(r"$\sigma / f_y$")
    ax.axhline(0.0, color="0.75", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="0.75", linewidth=0.8, zorder=0)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.35, linewidth=0.6)
    _set_sweep_sigma_over_fy_ylim(ax)
    _apply_specimen_axes_title(ax)

    line1 = _footer_geometry_line(
        L_T=L_T,
        L_y=L_y,
        ly_over_lt=ly_over_lt,
        A_sc=A_sc,
        A_t=A_t,
        at_over_asc=at_over_asc,
    )
    line2, line3 = _footer_steel_lines_through_cr2(None)
    _place_sweep_footer_three_lines(fig, line1, line2, line3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=SAVE_DPI)
    plt.close(fig)


def plot_dsample_sweep_figure(
    out_path: Path,
    waypoints_u: list[float],
    d_sample_fracs: tuple[float, ...],
    *,
    L_T: float,
    L_y: float,
    ly_over_lt: float,
    A_sc: float,
    A_t: float,
    at_over_asc: float,
    exp_xy: tuple[np.ndarray, np.ndarray] | None,
) -> None:
    r"""
    Vary ``D_\mathrm{sample} = f\,L_y\,\varepsilon_y`` (inch) for fixed piecewise targets ``waypoints_u``.
    """
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    _plot_experimental_reference_scatter(ax, exp_xy, zorder=2.0)
    ly_eps = float(L_y) * float(EPS_Y)
    for frac in d_sample_fracs:
        d_in = float(frac) * ly_eps
        disp_i, strain_i = displacement_strain_from_waypoints_dsample(waypoints_u, d_in)
        force = run_brb(
            disp_i,
            L_T=L_T,
            L_y=L_y,
            A_sc=A_sc,
            A_t=A_t,
            steel=DEFAULT_STEEL,
        )
        stress = force / A_sc
        lbl = rf"$D_\mathrm{{sample}}/(L_y\,\varepsilon_y) = {float(frac):.2f}$"
        ax.plot(
            strain_i,
            stress / FY,
            label=lbl,
            linewidth=_STEELMPF_HYST_LOOP_LW,
            linestyle="-",
            zorder=3.0,
            **_CUM_DEF_MARKER_KW,
        )

    ax.set_xlabel(r"$\delta / (L_y\,\varepsilon_y)$")
    ax.set_ylabel(r"$\sigma / f_y$")
    ax.axhline(0.0, color="0.75", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="0.75", linewidth=0.8, zorder=0)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=7,
        borderaxespad=0.0,
        frameon=True,
    )
    ax.grid(True, alpha=0.35, linewidth=0.6)
    _set_sweep_sigma_over_fy_ylim(ax)
    _apply_specimen_axes_title(ax)

    line1 = _footer_geometry_line(
        L_T=L_T,
        L_y=L_y,
        ly_over_lt=ly_over_lt,
        A_sc=A_sc,
        A_t=A_t,
        at_over_asc=at_over_asc,
    )
    line2, line3 = _footer_steel_lines_through_cr2(None)
    _place_sweep_footer_three_lines(fig, line1, line2, line3, right=0.78, bottom=0.30)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def _cumulative_deformation_vs_disp(disp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(cum_path_in, delta_in)`` with virtual start at ``\\delta=0``."""
    d = np.asarray(disp, dtype=float).ravel()
    if d.size == 0:
        return np.array([]), np.array([])
    d0 = np.concatenate((np.array([0.0], dtype=float), d))
    inc = np.abs(np.diff(d0))
    cum = np.cumsum(inc)
    return cum, d


def plot_displacement_vs_cumulative_deformation(
    out_path: Path,
    disp: np.ndarray,
    *,
    disp_underlay: np.ndarray | None = None,
    disp_resampled_eval_ontop: np.ndarray | None = None,
    primary_label: str = r"$\mathrm{This\ drive}$",
    primary_emphasis: bool = False,
) -> None:
    """
    Prescribed ``\\delta`` vs cumulative ``\\sum|\\Delta\\delta|`` from ``\\delta=0`` (in).

    ``disp`` is the folder's drive. Optional ``disp_underlay`` (e.g. peaks+linspace) is drawn first (dashed).
    Optional ``disp_resampled_eval_ontop`` draws the full eval history last (highest z-order). If that series
    equals ``disp``, it is skipped. ``primary_emphasis`` uses the same dark styling as the on-top eval trace for
    the primary line (used for ``resampled_eval_history``).
    """
    d = np.asarray(disp, dtype=float).ravel()
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    if d.size == 0:
        ax.text(0.5, 0.5, "empty displacement", ha="center", va="center", transform=ax.transAxes)
    else:
        has_under = disp_underlay is not None and np.asarray(disp_underlay).size > 0
        has_top = disp_resampled_eval_ontop is not None and np.asarray(disp_resampled_eval_ontop).size > 0

        if has_under:
            u = np.asarray(disp_underlay, dtype=float).ravel()
            cx, uy = _cumulative_deformation_vs_disp(u)
            ax.plot(
                cx,
                uy,
                color="0.65",
                linewidth=1.0,
                linestyle="--",
                label=r"$\mathrm{Peaks + linspace}$",
                zorder=2,
                alpha=0.9,
                **_CUM_DEF_MARKER_KW,
            )

        cx, dy = _cumulative_deformation_vs_disp(d)
        if primary_emphasis:
            p_color, p_lw, p_z = "0.15", 1.35, 5
        else:
            p_color, p_lw, p_z = "C0", 1.0, 3
        ax.plot(
            cx,
            dy,
            color=p_color,
            linewidth=p_lw,
            label=primary_label,
            zorder=p_z,
            **_CUM_DEF_MARKER_KW,
        )

        plotted_resampled_top = False
        if has_top:
            t = np.asarray(disp_resampled_eval_ontop, dtype=float).ravel()
            ctx, ty = _cumulative_deformation_vs_disp(t)
            same_as_primary = t.size == d.size and np.allclose(t, d, rtol=1e-12, atol=1e-12, equal_nan=True)
            if not same_as_primary:
                ax.plot(
                    ctx,
                    ty,
                    color="0.15",
                    linewidth=1.35,
                    label=r"$\mathrm{Resampled\ eval}$",
                    zorder=6,
                    **_CUM_DEF_MARKER_KW,
                )
                plotted_resampled_top = True

        if has_under or plotted_resampled_top:
            ax.legend(loc="best", fontsize=7)

    ax.set_xlabel(r"Cumulative deformation $\sum|\Delta\delta|$ (in)")
    ax.set_ylabel(r"Displacement $\delta$ (in)")
    ax.axhline(0.0, color="0.75", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="0.75", linewidth=0.8, zorder=0)
    ax.grid(True, alpha=0.35, linewidth=0.6)
    _apply_specimen_axes_title(ax)
    top = 0.90 if (SPECIMEN_NAME or "").strip() else 0.94
    fig.subplots_adjust(left=0.12, right=0.98, top=top, bottom=0.14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=SAVE_DPI)
    plt.close(fig)


def plot_displacement_vs_step(
    out_path: Path,
    disp: np.ndarray,
    *,
    disp_underlay: np.ndarray | None = None,
    disp_resampled_eval_ontop: np.ndarray | None = None,
    primary_label: str = r"$\mathrm{This\ drive}$",
    primary_emphasis: bool = False,
) -> None:
    """
    Prescribed ``\\delta`` vs zero-based step index (same optional underlay / on-top series as the cumulative figure).
    """
    d = np.asarray(disp, dtype=float).ravel()
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    if d.size == 0:
        ax.text(0.5, 0.5, "empty displacement", ha="center", va="center", transform=ax.transAxes)
    else:
        has_under = disp_underlay is not None and np.asarray(disp_underlay).size > 0
        has_top = disp_resampled_eval_ontop is not None and np.asarray(disp_resampled_eval_ontop).size > 0

        if has_under:
            u = np.asarray(disp_underlay, dtype=float).ravel()
            ax.plot(
                np.arange(u.size, dtype=float),
                u,
                color="0.65",
                linewidth=1.0,
                linestyle="--",
                label=r"$\mathrm{Peaks + linspace}$",
                zorder=2,
                alpha=0.9,
                **_CUM_DEF_MARKER_KW,
            )

        if primary_emphasis:
            p_color, p_lw, p_z = "0.15", 1.35, 5
        else:
            p_color, p_lw, p_z = "C0", 1.0, 3
        ax.plot(
            np.arange(d.size, dtype=float),
            d,
            color=p_color,
            linewidth=p_lw,
            label=primary_label,
            zorder=p_z,
            **_CUM_DEF_MARKER_KW,
        )

        plotted_resampled_top = False
        if has_top:
            t = np.asarray(disp_resampled_eval_ontop, dtype=float).ravel()
            same_as_primary = t.size == d.size and np.allclose(t, d, rtol=1e-12, atol=1e-12, equal_nan=True)
            if not same_as_primary:
                ax.plot(
                    np.arange(t.size, dtype=float),
                    t,
                    color="0.15",
                    linewidth=1.35,
                    label=r"$\mathrm{Resampled\ eval}$",
                    zorder=6,
                    **_CUM_DEF_MARKER_KW,
                )
                plotted_resampled_top = True

        if has_under or plotted_resampled_top:
            ax.legend(loc="best", fontsize=7)

    ax.set_xlabel("Step")
    ax.set_ylabel(r"Displacement $\delta$ (in)")
    ax.axhline(0.0, color="0.75", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="0.75", linewidth=0.8, zorder=0)
    ax.grid(True, alpha=0.35, linewidth=0.6)
    _apply_specimen_axes_title(ax)
    top = 0.90 if (SPECIMEN_NAME or "").strip() else 0.94
    fig.subplots_adjust(left=0.12, right=0.98, top=top, bottom=0.14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=SAVE_DPI)
    plt.close(fig)


def write_sweep_pngs_for_displacement(
    out_dir: Path,
    subdir: str,
    disp: np.ndarray,
    strain_norm_ly: np.ndarray,
    *,
    L_T_d: float,
    A_t_d: float,
    ly_d: float,
    at_d: float,
    exp_xy: tuple[np.ndarray, np.ndarray] | None,
    disp_cum_underlay: np.ndarray | None = None,
    disp_cum_resampled_eval_ontop: np.ndarray | None = None,
    disp_cum_primary_label: str = r"$\mathrm{This\ drive}$",
    disp_cum_primary_emphasis: bool = False,
    waypoints_u_for_dsample: list[float] | None = None,
    include_dsample_sweep: bool = True,
    prepend_zero_compare: tuple[list[float], list[int]] | None = None,
) -> None:
    pdir = out_dir / subdir
    if subdir in DISPLACEMENT_STEP_INDEX_SUBDIRS:
        p_disp = pdir / "displacement_vs_step.png"
        plot_displacement_vs_step(
            p_disp,
            disp,
            disp_underlay=disp_cum_underlay,
            disp_resampled_eval_ontop=disp_cum_resampled_eval_ontop,
            primary_label=disp_cum_primary_label,
            primary_emphasis=disp_cum_primary_emphasis,
        )
        print(f"Wrote {p_disp}")
    else:
        p_disp = pdir / "displacement_vs_cumulative_deformation.png"
        plot_displacement_vs_cumulative_deformation(
            p_disp,
            disp,
            disp_underlay=disp_cum_underlay,
            disp_resampled_eval_ontop=disp_cum_resampled_eval_ontop,
            primary_label=disp_cum_primary_label,
            primary_emphasis=disp_cum_primary_emphasis,
        )
        print(f"Wrote {p_disp}")

    for param, values in STEEL_PARAM_EXPLORATIONS.items():
        out_path = pdir / f"sweep_{param}.png"
        ru0_fup: dict[str, float] | None = (
            {"fup_ratio": float(RU0_SWEEP_FUP_RATIO)} if param == "Ru0" else None
        )
        plot_sweep_figure(
            out_path,
            param,
            values,
            disp,
            strain_norm_ly,
            L_T=L_T_d,
            L_y=L_Y,
            ly_over_lt=ly_d,
            A_sc=A_SC,
            A_t=A_t_d,
            at_over_asc=at_d,
            exp_xy=exp_xy,
            fixed_steel_overrides=ru0_fup,
        )
        print(f"Wrote {out_path}")

    if subdir == RESAMPLED_EVAL_HISTORY_SUBDIR and RESAMPLED_EVAL_SWEEP_A3_EXTRA_VALUES:
        fr = float(RESAMPLED_EVAL_SWEEP_A3_EXTRA_FUP_RATIO)
        fr_tag = str(int(fr)) if abs(fr - round(fr)) < 1e-9 else str(fr).replace(".", "p")

        def _sweep_fname_float_token(x: float) -> str:
            t = f"{float(x):.4f}".rstrip("0").rstrip(".")
            return t.replace(".", "p").replace("-", "m")

        a3_extra_scenarios: list[tuple[str, dict[str, float]]] = [
            (f"sweep_a3_fup_ratio_{fr_tag}.png", {"fup_ratio": fr}),
        ]
        if RESAMPLED_EVAL_SWEEP_A3_EXTRA_BP_BN is not None:
            bp, bn = (float(RESAMPLED_EVAL_SWEEP_A3_EXTRA_BP_BN[0]), float(RESAMPLED_EVAL_SWEEP_A3_EXTRA_BP_BN[1]))
            a3_extra_scenarios.append(
                (
                    f"sweep_a3_fup_ratio_{fr_tag}_bp{_sweep_fname_float_token(bp)}_bn{_sweep_fname_float_token(bn)}.png",
                    {"fup_ratio": fr, "b_p": bp, "b_n": bn},
                )
            )

        for fname, overrides in a3_extra_scenarios:
            out_a3_fup = pdir / fname
            plot_sweep_figure(
                out_a3_fup,
                "a3",
                RESAMPLED_EVAL_SWEEP_A3_EXTRA_VALUES,
                disp,
                strain_norm_ly,
                L_T=L_T_d,
                L_y=L_Y,
                ly_over_lt=ly_d,
                A_sc=A_SC,
                A_t=A_t_d,
                at_over_asc=at_d,
                exp_xy=exp_xy,
                fixed_steel_overrides=overrides,
            )
            print(f"Wrote {out_a3_fup}")

    p_ly = pdir / "sweep_LyLt.png"
    plot_ly_lt_sweep_figure(p_ly, disp, strain_norm_ly, exp_xy)
    print(f"Wrote {p_ly}")

    p_at = pdir / "sweep_AtAsc.png"
    plot_at_asc_sweep_figure(p_at, disp, strain_norm_ly, exp_xy)
    print(f"Wrote {p_at}")

    if include_dsample_sweep and waypoints_u_for_dsample is not None:
        p_ds = pdir / "sweep_Dsample.png"
        plot_dsample_sweep_figure(
            p_ds,
            waypoints_u_for_dsample,
            DSAMPLE_FRAC_VALUES,
            L_T=L_T_d,
            L_y=L_Y,
            ly_over_lt=ly_d,
            A_sc=A_SC,
            A_t=A_t_d,
            at_over_asc=at_d,
            exp_xy=exp_xy,
        )
        print(f"Wrote {p_ds}")

    if prepend_zero_compare is not None:
        wu_pz, st_pz = prepend_zero_compare
        p_pz = pdir / "sweep_prependZero.png"
        plot_prepend_zero_sweep_figure(
            p_pz,
            wu_pz,
            st_pz,
            L_T=L_T_d,
            L_y=L_Y,
            ly_over_lt=ly_d,
            A_sc=A_SC,
            A_t=A_t_d,
            at_over_asc=at_d,
            exp_xy=exp_xy,
        )
        print(f"Wrote {p_pz}")


def main() -> None:
    p = argparse.ArgumentParser(description="BRB corotruss SteelMPF sweep figures.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_SCRIPT_DIR,
        help="Root directory; each load protocol writes under a subfolder.",
    )
    args = p.parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()

    n_half, n_peak, _ = strain_peak_mults_from_specimen()
    load_protocols = make_load_protocols(n_half, n_peak)
    uref_displacement = n_peak * EPS_Y * L_Y

    L_T_d, A_t_d = _default_L_T_A_t()
    ly_d = DEFAULT_LY_OVER_LT
    at_d = DEFAULT_AT_OVER_ASC
    d_samp_spec = _specimen_d_sampling_inches(u_fallback=np.array([uref_displacement], dtype=float))

    exp_xy = load_experimental_norm_scatter()
    if exp_xy is None:
        print(
            f"Warning: no experimental scatter for {SPECIMEN_NAME!r} "
            "(missing resolved F--u CSV or required columns)."
        )
    else:
        print(f"Experimental scatter overlay: {exp_xy[0].size} points (subsampled).")

    for proto in load_protocols:
        disp, strain_norm_ly = displacement_strain_for_protocol(
            proto,
            uref_displacement=uref_displacement,
            d_sample_in=d_samp_spec,
        )
        wu_proto = _waypoints_u_from_mults(proto.waypoint_mults)
        st_proto = _steps_for_waypoints(wu_proto, uref_displacement=uref_displacement)
        write_sweep_pngs_for_displacement(
            out_dir,
            proto.subdir,
            disp,
            strain_norm_ly,
            L_T_d=L_T_d,
            A_t_d=A_t_d,
            ly_d=ly_d,
            at_d=at_d,
            exp_xy=exp_xy,
            waypoints_u_for_dsample=wu_proto,
            include_dsample_sweep=True,
            prepend_zero_compare=(wu_proto, st_proto),
        )

    eval_hist = load_eval_driving_displacement()
    if eval_hist is not None:
        disp_e, sn_e = eval_hist
        print(
            f"Eval driving history ({RESAMPLED_EVAL_HISTORY_SUBDIR}): {disp_e.size} deformation samples."
        )
        write_sweep_pngs_for_displacement(
            out_dir,
            RESAMPLED_EVAL_HISTORY_SUBDIR,
            disp_e,
            sn_e,
            L_T_d=L_T_d,
            A_t_d=A_t_d,
            ly_d=ly_d,
            at_d=at_d,
            exp_xy=exp_xy,
            disp_cum_primary_label=r"$\mathrm{Resampled\ eval}$",
            disp_cum_primary_emphasis=True,
            include_dsample_sweep=False,
        )


if __name__ == "__main__":
    main()
