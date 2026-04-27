"""
Calibration objective weights and cycle-weighting knobs.

These settings define the weighted objective:

.. code-block:: text

    J_total = Σ_k w_k * metric_k

where each ``metric_k`` is a **raw** (unweighted) diagnostic; ``w_k`` are the ``w_*`` weights
(``w_feat_l2``, ``w_feat_l1``, ``w_energy_l2``, ``w_energy_l1``, ``w_unordered_binenv_l2``,
``w_unordered_binenv_l1``).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from collections.abc import Mapping


@dataclass(frozen=True)
class CalibrationLossSettings:
    """Weights on raw metrics: feat, energy, and binned cloud envelope (L2/L1 each)."""

    w_feat_l2: float
    w_feat_l1: float
    w_energy_l2: float
    w_energy_l1: float
    w_unordered_binenv_l2: float
    w_unordered_binenv_l1: float
    use_amplitude_weights: bool
    amplitude_weight_power: float
    amplitude_weight_eps: float


DEFAULT_CALIBRATION_LOSS_SETTINGS = CalibrationLossSettings(
    w_feat_l2=0.7,
    w_feat_l1=0.0,
    w_energy_l2=0.3,
    w_energy_l1=0.0,
    w_unordered_binenv_l2=0.0,
    w_unordered_binenv_l1=0.0,
    use_amplitude_weights=False,
    amplitude_weight_power=2.0,
    amplitude_weight_eps=0.05,
)


def parse_bool_cell(v: object) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if v == 1 or v == 1.0:
            return True
        if v == 0 or v == 0.0:
            return False
    if hasattr(v, "item"):
        try:
            it = v.item()
            if isinstance(it, bool):
                return it
            if isinstance(it, (int, float)):
                if it == 1:
                    return True
                if it == 0:
                    return False
        except Exception:
            pass
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off", ""):
        return False
    raise ValueError(f"expected boolean-like value, got {v!r}")


def calibration_loss_settings_from_partial_dict(
    kv: Mapping[str, object] | None,
    *,
    default: CalibrationLossSettings = DEFAULT_CALIBRATION_LOSS_SETTINGS,
) -> CalibrationLossSettings:
    """
    Build settings from a partial mapping of ``name -> value``.

    Missing / NaN values fall back to ``default``. Intended for per-``set_id`` rows in
    ``config/calibration/set_id_settings.csv`` where some columns may be left blank.
    """
    if not kv:
        return default

    def get_raw(key: str) -> object | None:
        return kv.get(key.lower())

    def opt_float(name: str, dflt: float) -> float:
        v = get_raw(name)
        if v is None:
            return dflt
        if isinstance(v, (float, np.floating)) and np.isnan(v):
            return dflt
        if pd.isna(v):
            return dflt
        return float(v)

    def opt_bool(name: str, dflt: bool) -> bool:
        v = get_raw(name)
        if v is None:
            return dflt
        if pd.isna(v):
            return dflt
        return parse_bool_cell(v)

    return CalibrationLossSettings(
        w_feat_l2=opt_float("w_feat_l2", default.w_feat_l2),
        w_feat_l1=opt_float("w_feat_l1", default.w_feat_l1),
        w_energy_l2=opt_float("w_energy_l2", default.w_energy_l2),
        w_energy_l1=opt_float("w_energy_l1", default.w_energy_l1),
        w_unordered_binenv_l2=opt_float("w_unordered_binenv_l2", default.w_unordered_binenv_l2),
        w_unordered_binenv_l1=opt_float("w_unordered_binenv_l1", default.w_unordered_binenv_l1),
        use_amplitude_weights=opt_bool("use_amplitude_weights", default.use_amplitude_weights),
        amplitude_weight_power=opt_float("amplitude_weight_power", default.amplitude_weight_power),
        amplitude_weight_eps=opt_float("amplitude_weight_eps", default.amplitude_weight_eps),
    )


def _loss_settings_table_to_kv(df: pd.DataFrame) -> dict[str, object]:
    """
    Build ``name.lower() -> cell`` from either:

    - **Transposed (two columns):** header like ``setting,value`` / ``name,value`` / ``key,value``;
      each data row is one parameter.
    - **Wide:** column names are settings, first data row has values (any column count >= 2;
      for exactly two columns, the header must **not** be a transposed-style pair, so e.g.
      ``w_feat_l2,w_energy_l2`` + one row still works).
    """
    if df.empty:
        return {}
    ncols = int(df.shape[1])
    if ncols == 2:
        c0 = str(df.columns[0]).strip().lower()
        c1 = str(df.columns[1]).strip().lower()
        long_header = c0 in ("setting", "name", "key", "parameter") or c1 == "value"
        if long_header:
            kv: dict[str, object] = {}
            for _, r in df.iterrows():
                raw_k = r.iloc[0]
                if pd.isna(raw_k):
                    continue
                k = str(raw_k).strip().lower()
                if not k or k.startswith("#"):
                    continue
                kv[k] = r.iloc[1]
            return kv
        row = df.iloc[0]
        return {
            str(df.columns[0]).strip().lower(): row.iloc[0],
            str(df.columns[1]).strip().lower(): row.iloc[1],
        }
    if ncols > 2:
        row = df.iloc[0]
        return {str(c).strip().lower(): row[c] for c in df.columns}
    return {}


def load_calibration_loss_settings(path: Path | None) -> CalibrationLossSettings:
    """
    Read settings from a transposed ``setting,value`` table or a **wide** one-row CSV.

    Required: ``use_amplitude_weights``, ``w_feat_l2``, ``w_energy_l2``.

    Optional weights (default 0): ``w_feat_l1``, ``w_energy_l1``, ``w_unordered_binenv_l2``,
    ``w_unordered_binenv_l1``.

    Optional: ``amplitude_weight_power``, ``amplitude_weight_eps``.

    Unknown setting keys in the CSV are ignored.
    """
    if path is None or not path.is_file():
        return DEFAULT_CALIBRATION_LOSS_SETTINGS

    df = pd.read_csv(path, comment="#", skipinitialspace=True)
    df.columns = df.columns.astype(str).str.strip()
    if df.empty:
        return DEFAULT_CALIBRATION_LOSS_SETTINGS

    kv = _loss_settings_table_to_kv(df)
    if not kv:
        return DEFAULT_CALIBRATION_LOSS_SETTINGS

    def get_raw(key: str) -> object | None:
        return kv.get(key.lower())

    def opt_float(name: str, default: float) -> float:
        v = get_raw(name)
        if v is None:
            return default
        if isinstance(v, (float, np.floating)) and np.isnan(v):
            return default
        if pd.isna(v):
            return default
        return float(v)

    uaw_raw = get_raw("use_amplitude_weights")
    if uaw_raw is None:
        raise ValueError(f"{path}: missing use_amplitude_weights (have keys {sorted(kv)!r})")
    uaw = parse_bool_cell(uaw_raw)
    p = opt_float("amplitude_weight_power", DEFAULT_CALIBRATION_LOSS_SETTINGS.amplitude_weight_power)
    eps = opt_float("amplitude_weight_eps", DEFAULT_CALIBRATION_LOSS_SETTINGS.amplitude_weight_eps)

    v_wf2 = get_raw("w_feat_l2")
    if v_wf2 is None or pd.isna(v_wf2):
        raise ValueError(f"{path}: missing w_feat_l2 (have keys {sorted(kv)!r})")
    w_feat_l2 = float(v_wf2)

    v_we2 = get_raw("w_energy_l2")
    if v_we2 is None or pd.isna(v_we2):
        raise ValueError(f"{path}: missing w_energy_l2 (have keys {sorted(kv)!r})")
    w_energy_l2 = float(v_we2)

    w_feat_l1 = opt_float("w_feat_l1", 0.0)
    w_energy_l1 = opt_float("w_energy_l1", 0.0)
    w_unordered_binenv_l2 = opt_float("w_unordered_binenv_l2", 0.0)
    w_unordered_binenv_l1 = opt_float("w_unordered_binenv_l1", 0.0)

    return CalibrationLossSettings(
        w_feat_l2=w_feat_l2,
        w_feat_l1=w_feat_l1,
        w_energy_l2=w_energy_l2,
        w_energy_l1=w_energy_l1,
        w_unordered_binenv_l2=w_unordered_binenv_l2,
        w_unordered_binenv_l1=w_unordered_binenv_l1,
        use_amplitude_weights=uaw,
        amplitude_weight_power=p,
        amplitude_weight_eps=eps,
    )
