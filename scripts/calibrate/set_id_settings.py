"""
Unified per-``set_id`` configuration.

Source of truth: ``config/calibration/set_id_settings.csv`` (one row per set_id).

This file merges steel seeds / b_p,b_n sourcing and per-set optimize/loss settings into one CSV.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from calibrate.calibration_loss_settings import (
    CalibrationLossSettings,
    DEFAULT_CALIBRATION_LOSS_SETTINGS,
    calibration_loss_settings_from_partial_dict,
)
from calibrate.calibration_paths import SET_ID_SETTINGS_CSV
from calibrate.set_id_optimize_params import _parse_optimize_params_cell


@dataclass(frozen=True)
class InitialBrbSeedRow:
    set_id: int
    steel: dict[str, float]
    b_p_spec: float | str
    b_n_spec: float | str


def load_set_id_settings(path: Path | None = None) -> pd.DataFrame:
    """
    Read the unified per-set_id settings CSV.

    Validates presence/uniqueness of `set_id` and normalizes it to int.
    """
    csv_path = Path(path).expanduser().resolve() if path is not None else SET_ID_SETTINGS_CSV
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing set_id settings CSV: {csv_path}")
    df = pd.read_csv(csv_path, comment="#")
    if df.empty:
        raise ValueError(f"{csv_path}: no data rows")
    if "set_id" not in df.columns:
        raise ValueError(f"{csv_path}: expected column 'set_id'; got {list(df.columns)}")

    sid = pd.to_numeric(df["set_id"], errors="coerce")
    if sid.isna().any():
        bad = df.loc[sid.isna(), "set_id"].tolist()
        raise ValueError(f"{csv_path}: invalid set_id value(s): {bad}")
    sid_int = sid.astype(int)
    if (sid_int < 1).any():
        bad = df.loc[sid_int < 1, "set_id"].tolist()
        raise ValueError(f"{csv_path}: set_id must be >= 1; got {bad}")
    if sid_int.duplicated().any():
        dups = sorted(set(sid_int[sid_int.duplicated()].tolist()))
        raise ValueError(f"{csv_path}: duplicate set_id(s): {dups}")

    out = df.copy()
    out["set_id"] = sid_int
    out = out.sort_values("set_id")
    return out


LOSS_SETTINGS_KEYS: tuple[str, ...] = (
    "w_feat_l2",
    "w_feat_l1",
    "w_energy_l2",
    "w_energy_l1",
    "w_unordered_binenv_l2",
    "w_unordered_binenv_l1",
    "use_amplitude_weights",
    "amplitude_weight_power",
    "amplitude_weight_eps",
)


def load_set_id_optimize_and_loss(
    path: Path | None = None,
) -> tuple[dict[int, list[str]], dict[int, CalibrationLossSettings]]:
    """
    Return:
    - `set_id -> optimize_params` list (may be empty if column missing/blank for all rows)
    - `set_id -> CalibrationLossSettings` (partial per-row overrides; missing cells use defaults)
    """
    df = load_set_id_settings(path)
    csv_path = Path(path).expanduser().resolve() if path is not None else SET_ID_SETTINGS_CSV

    opt: dict[int, list[str]] = {}
    loss: dict[int, CalibrationLossSettings] = {}
    has_opt_col = "optimize_params" in df.columns
    for idx, row in df.iterrows():
        sid = int(row["set_id"])
        if has_opt_col and pd.notna(row.get("optimize_params")):
            raw = row.get("optimize_params")
            if str(raw).strip() and str(raw).strip().lower() != "nan":
                opt[sid] = _parse_optimize_params_cell(raw, path=csv_path, set_id=sid)

        kv: dict[str, object] = {}
        for k in LOSS_SETTINGS_KEYS:
            if k in df.columns:
                kv[k] = row.get(k)
        loss[sid] = calibration_loss_settings_from_partial_dict(
            {str(k).lower(): v for k, v in kv.items()},
            default=DEFAULT_CALIBRATION_LOSS_SETTINGS,
        )
    return opt, loss


def resolve_set_id_settings_csv_arg(path: Path | None) -> Path:
    """CLI helper: if arg is None, return default; always returns resolved Path."""
    return (Path(path).expanduser().resolve() if path is not None else SET_ID_SETTINGS_CSV).resolve()

