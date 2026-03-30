"""
SteelMPF / BRB parameters updated by L-BFGS-B in ``optimize_brb_mse`` / ``optimize_generalized_brb_mse``.

Import this module from lightweight scripts (e.g. ``report_calibration_param_tables``) to avoid
pulling OpenSees via ``optimize_brb_mse``.
"""
from __future__ import annotations

# Alternate choices (keep one active):
# PARAMS_TO_OPTIMIZE = ["fyp", "fyn", "b_p", "b_n", "R0", "cR1", "cR2", "a1", "a3"]
# PARAMS_TO_OPTIMIZE = ["b_p", "b_n", "R0", "cR1", "cR2", "a1", "a3"]
PARAMS_TO_OPTIMIZE = ["R0", "cR1", "cR2", "a1", "a3"]
# PARAMS_TO_OPTIMIZE = ["a1", "a3"]

# Extra SteelMPF columns included in summary_statistics/calibration_parameter_summary_*.csv (not necessarily optimized).
# Order matches SteelMPF material input (see ``model/corotruss.py``): b_p, b_n, then R0…a3, then a2, a4.
PARAMS_IN_SUMMARY_TABLES = ["b_p", "b_n", *PARAMS_TO_OPTIMIZE, "a2", "a4"]
