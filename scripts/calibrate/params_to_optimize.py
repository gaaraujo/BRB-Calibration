"""
SteelMPF / Steel4 BRB parameters updated by L-BFGS-B in ``optimize_brb_mse`` / ``optimize_generalized_brb_mse``.

Import this module from lightweight scripts (e.g. ``report_calibration_param_tables``) to avoid
pulling OpenSees via ``optimize_brb_mse``.
"""
from __future__ import annotations

from calibrate.steel_model import (
    STEEL4_ISO_KEYS,
    STEEL4_SIM_KEYS,
    STEEL_MODEL_STEEL4,
    STEELMPF_SIM_KEYS,
    normalize_steel_model,
)

# Default subset optimized by L-BFGS-B. Override per ``set_id`` via ``optimize_params`` in
# ``config/calibration/set_id_settings.csv`` (see ``set_id_optimize_params``).
# Alternate choices (keep one active):
# PARAMS_TO_OPTIMIZE = ["fyp", "fyn", "b_p", "b_n", "R0", "cR1", "cR2", "a1", "a3"]
# PARAMS_TO_OPTIMIZE = ["b_p", "b_n", "R0", "cR1", "cR2", "a1", "a3"]
PARAMS_TO_OPTIMIZE = ["R0", "cR1", "cR2", "a1", "a3"]
# PARAMS_TO_OPTIMIZE = ["a1", "a3"]

# Widest parameter-row float block (union of SteelMPF and Steel4 simulation keys).
SIM_PARAMS_FROM_ROW: tuple[str, ...] = tuple(
    dict.fromkeys([*STEELMPF_SIM_KEYS, *STEEL4_SIM_KEYS])
)

# When a parameters CSV omits optional columns, fill before simulation / optimization.
SIM_PARAM_FILL_DEFAULTS: dict[str, float] = {
    "E": 29000.0,
    "R0": 20.0,
    "cR1": 0.925,
    "cR2": 0.15,
    "a1": 0.04,
    "a2": 1.0,
    "a3": 0.04,
    "a4": 1.0,
    "b_ip": 0.01,
    "rho_ip": 2.0,
    "b_lp": 0.001,
    "R_i": 20.0,
    "l_yp": 0.01,
    "b_ic": 0.01,
    "rho_ic": 2.0,
    "b_lc": 0.001,
}

# Columns in ``summary_statistics/calibration_parameter_summary_<steel_model>*.csv``.
# Models are **not** mixed in one table: Steel4 adds ``STEEL4_ISO_KEYS``; do not compare rollups across models.
PARAMS_IN_SUMMARY_TABLES_STEELMPF: tuple[str, ...] = (
    "b_p",
    "b_n",
    *PARAMS_TO_OPTIMIZE,
    "a2",
    "a4",
)
PARAMS_IN_SUMMARY_TABLES_STEEL4: tuple[str, ...] = (*PARAMS_IN_SUMMARY_TABLES_STEELMPF, *STEEL4_ISO_KEYS)


def params_in_summary_tables_for_steel_model(steel_model: object) -> list[str]:
    """Parameter columns to tabulate for Markdown/CSV summaries for one ``steel_model``."""
    sm = normalize_steel_model(steel_model)
    if sm == STEEL_MODEL_STEEL4:
        return list(PARAMS_IN_SUMMARY_TABLES_STEEL4)
    return list(PARAMS_IN_SUMMARY_TABLES_STEELMPF)


# Union of columns (widest); prefer ``params_in_summary_tables_for_steel_model`` for new code.
PARAMS_IN_SUMMARY_TABLES: tuple[str, ...] = tuple(dict.fromkeys([*PARAMS_IN_SUMMARY_TABLES_STEEL4]))
