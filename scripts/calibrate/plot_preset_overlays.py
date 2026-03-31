"""
Preset **combined** normalized overlays (``set{k}_combined_force_def_norm.png``) with fixed ``b_p`` /
``b_n`` and steel from ``set_id_settings.csv``.

Writes numerical ``{Name}_set{k}_simulated.csv`` under ``results/calibration/individual_optimize/initial_params_simulated_force/``,
a snapshot parameters CSV ``initial_params_overlay_parameters.csv``, then builds one montage per ``set_id``
(same layout as ``plot_compare_calibration_overlays.py`` for individual optimize). Does **not** write per-specimen PNGs.

Typical (matches ``run.ps1`` / ``run.sh`` before L-BFGS)::

    python scripts/calibrate/plot_preset_overlays.py --set-id-settings config/calibration/set_id_settings.csv

Or use an existing parameters CSV instead of rebuilding from seeds::

    python scripts/calibrate/plot_preset_overlays.py --params results/calibration/individual_optimize/initial_brb_parameters.csv

Requires: postprocess through ``extract_bn_bp.py`` (``specimen_apparent_bn_bp.csv``) and
resampled ``data/resampled/{Name}/force_deformation.csv`` for path-ordered specimens with
``individual_optimize=true``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_SCRIPTS = _PROJECT_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from calibrate.build_initial_brb_parameters import (  # noqa: E402
    CATALOG_PATH,
    DEFAULT_BN_BP_PATH,
    OUT_COLS,
    build_initial_rows,
    load_initial_brb_seeds,
)
from calibrate.calibration_paths import (  # noqa: E402
    INITIAL_PARAMS_OVERLAY_PARAMETERS_PATH,
    INITIAL_PARAMS_SIMULATED_FORCE_DIR,
    PLOTS_INDIVIDUAL_OPTIMIZE,
    PLOTS_INITIAL_PARAMS_OVERLAYS,
    SET_ID_SETTINGS_CSV,
)
from calibrate.plot_compare_calibration_overlays import (  # noqa: E402
    _discover_simulated_index,
    _order_specimens,
    plot_combined_for_set,
)
from calibrate.plot_params_vs_filtered import run_multi_specimen_simulated_csvs  # noqa: E402
from specimen_catalog import read_catalog  # noqa: E402

DEFAULT_PRESET_BP = 0.007
DEFAULT_PRESET_BN = 0.020


def _params_from_seeds(
    *,
    set_id_settings_path: Path,
    catalog_path: Path,
    bn_bp_path: Path,
) -> pd.DataFrame:
    if not bn_bp_path.is_file():
        raise SystemExit(f"Missing {bn_bp_path}; run extract_bn_bp.py after resample_filtered.py.")
    catalog = pd.read_csv(catalog_path)
    bn_bp = pd.read_csv(bn_bp_path)
    seeds = load_initial_brb_seeds(set_id_settings_path)
    rows = build_initial_rows(catalog, bn_bp, seeds=seeds)
    if not rows:
        raise SystemExit("No rows after catalog merge; check BRB-Specimens.csv.")
    out = pd.DataFrame(rows)
    return out[OUT_COLS]


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Preset combined normalized overlays (one PNG per set_id) with fixed b_p/b_n; "
            "steel from --set-id-settings or --params CSV."
        ),
    )
    p.add_argument(
        "--set-id-settings",
        type=Path,
        default=SET_ID_SETTINGS_CSV,
        help=(
            "set_id_settings.csv (ignored if --params is set). "
            f"Default: {SET_ID_SETTINGS_CSV}."
        ),
    )
    p.add_argument(
        "--params",
        type=Path,
        default=None,
        help=(
            "Optional: use this parameters CSV instead of rebuilding from --set-id-settings "
            "(e.g. results/calibration/individual_optimize/initial_brb_parameters.csv)."
        ),
    )
    p.add_argument(
        "--catalog",
        type=Path,
        default=CATALOG_PATH,
        help="BRB-Specimens.csv (only used when building from --set-id-settings).",
    )
    p.add_argument(
        "--bn-bp",
        type=Path,
        default=DEFAULT_BN_BP_PATH,
        help="specimen_apparent_bn_bp.csv from extract_bn_bp.py (only used with --set-id-settings).",
    )
    p.add_argument(
        "--override-bp",
        type=float,
        default=DEFAULT_PRESET_BP,
        metavar="VAL",
        help=f"Fixed b_p for every simulation (default: {DEFAULT_PRESET_BP:g}).",
    )
    p.add_argument(
        "--override-bn",
        type=float,
        default=DEFAULT_PRESET_BN,
        metavar="VAL",
        help=f"Fixed b_n for every simulation (default: {DEFAULT_PRESET_BN:g}).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=PLOTS_INITIAL_PARAMS_OVERLAYS.name,
        help=(
            f"Subfolder under {PLOTS_INDIVIDUAL_OPTIMIZE.relative_to(_PROJECT_ROOT)}/ "
            f"(default: {PLOTS_INITIAL_PARAMS_OVERLAYS.name!r})."
        ),
    )
    p.add_argument(
        "--specimen",
        type=str,
        default=None,
        help="If set, only this specimen Name (must be individual_optimize with resampled data).",
    )
    args = p.parse_args()

    plots_dir = PLOTS_INDIVIDUAL_OPTIMIZE / args.output_dir

    params_df: pd.DataFrame
    label: str | Path

    if args.params is not None:
        params_path = Path(args.params).expanduser().resolve()
        if not params_path.is_file():
            raise SystemExit(f"Parameters CSV not found: {params_path}")
        params_df = pd.read_csv(params_path)
        label = params_path
        print(f"Parameters from {params_path}")
    else:
        settings_path = Path(args.set_id_settings).expanduser().resolve()
        if not settings_path.is_file():
            raise SystemExit(f"set_id settings CSV not found: {settings_path}")
        catalog_path = Path(args.catalog).expanduser().resolve()
        bn_bp_path = Path(args.bn_bp).expanduser().resolve()
        print(f"Building parameters from set_id settings: {settings_path}")
        params_df = _params_from_seeds(
            set_id_settings_path=settings_path,
            catalog_path=catalog_path,
            bn_bp_path=bn_bp_path,
        )
        label = f"{settings_path} (built)"

    obp = float(args.override_bp)
    obn = float(args.override_bn)
    params_df = params_df.copy()
    params_df["b_p"] = obp
    params_df["b_n"] = obn

    sim_dir = INITIAL_PARAMS_SIMULATED_FORCE_DIR
    if sim_dir.is_dir():
        for f in sim_dir.glob("*_set*_simulated.csv"):
            try:
                f.unlink()
            except OSError:
                pass

    INITIAL_PARAMS_OVERLAY_PARAMETERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    params_df.to_csv(INITIAL_PARAMS_OVERLAY_PARAMETERS_PATH, index=False)
    print(f"Wrote {INITIAL_PARAMS_OVERLAY_PARAMETERS_PATH}")

    run_multi_specimen_simulated_csvs(
        params_df,
        sim_dir,
        params_path_label=label,
        specimen=args.specimen,
        override_bp=obp,
        override_bn=obn,
    )

    catalog = read_catalog(CATALOG_PATH)
    catalog_names = catalog["Name"].astype(str).tolist()

    idx = _discover_simulated_index(sim_dir)
    if not idx:
        print(f"No *_simulated.csv under {sim_dir}; combined figures skipped.")
        return

    any_written = False
    for set_id in sorted(idx.keys()):
        specimen_names = _order_specimens(idx[set_id], catalog_names)
        if plot_combined_for_set(
            set_id,
            specimen_names,
            plots_dir,
            INITIAL_PARAMS_OVERLAY_PARAMETERS_PATH,
            sim_dir,
            catalog,
            grid_cols=3,
            stage_weight_fn=None,
        ):
            any_written = True
            print(
                f"Wrote {plots_dir / f'set{set_id}_combined_force_def_norm.png'} "
                f"({len(specimen_names)} specimens, 3-column grid)"
            )

    if any_written:
        print(f"Done. Combined PNGs under {plots_dir}")
    else:
        print("No combined figures written.")


if __name__ == "__main__":
    main()
