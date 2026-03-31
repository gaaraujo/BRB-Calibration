"""Shared column order for calibration metrics CSVs (optimize, averaged, generalized)."""
from __future__ import annotations

import pandas as pd

# Loss / reporting block (matches optimize_brb_mse metrics row keys).
METRICS_CORE_COLUMNS: tuple[str, ...] = (
    "initial_J_feat_raw",
    "final_J_feat_raw",
    "initial_J_feat_l1_raw",
    "final_J_feat_l1_raw",
    "initial_J_E_raw",
    "final_J_E_raw",
    "initial_J_E_l1_raw",
    "final_J_E_l1_raw",
    "initial_unordered_J_binenv",
    "final_unordered_J_binenv",
    "initial_unordered_J_binenv_l1",
    "final_unordered_J_binenv_l1",
    "initial_J_total",
    "final_J_total",
    "W_FEAT_L2",
    "W_FEAT_L1",
    "W_ENERGY_L2",
    "W_ENERGY_L1",
    "W_UNORDERED_BINENV_L2",
    "W_UNORDERED_BINENV_L1",
    "S_F",
    "S_D",
    "S_E",
    "P_y_ref",
    "n_cycles",
    "success",
)

METRICS_LEADING_COLUMNS: tuple[str, ...] = (
    "Name",
    "set_id",
    "specimen_weight",
    "contributes_to_aggregate",
    "individual_optimize",
    "weight_config",
    "calibration_stage",
    "aggregate_by_set_id",
)

METRICS_ALL_COLUMNS: tuple[str, ...] = METRICS_LEADING_COLUMNS + METRICS_CORE_COLUMNS


def metrics_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Build metrics DataFrame with a fixed column order (missing keys -> NaN)."""
    df = pd.DataFrame(rows)
    for c in METRICS_ALL_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[list(METRICS_ALL_COLUMNS)]
