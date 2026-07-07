"""Load ``input.csv`` for ``calibrate_one_specimen.py`` (SteelMPF only)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from calibrate.calibration_loss_settings import (  # noqa: E402
    CalibrationLossSettings,
    parse_bool_cell,
)
from calibrate.steel_model import STEEL_MODEL_STEELMPF  # noqa: E402

STEEL_MODEL = STEEL_MODEL_STEELMPF

# SteelMPF seed columns in input.csv (stock model through a4; no ultimate-strength tail).
STEELMPF_SEED_KEYS = (
    "E",
    "R0",
    "cR1",
    "cR2",
    "a1",
    "a2",
    "a3",
    "a4",
)


@dataclass(frozen=True)
class SingleCalibrateInput:
    set_id: int
    optimize_params: list[str]
    default_b_p: float
    default_b_n: float
    steel_seeds: dict[str, float]
    loss: CalibrationLossSettings
    param_bounds: dict[str, tuple[float, float]]


def _parse_optimize_params(raw: object) -> list[str]:
    s = str(raw).strip().strip('"').strip("'")
    if not s:
        raise ValueError("meta.optimize_params is empty")
    return [p.strip() for p in s.split(",") if p.strip()]


def _parse_bound(raw: object, *, param: str) -> tuple[float, float]:
    parts = [x.strip() for x in str(raw).split(",")]
    if len(parts) != 2:
        raise ValueError(
            f"bound.{param}: expected 'lower,upper', got {raw!r}"
        )
    lo, hi = float(parts[0]), float(parts[1])
    if not (hi > lo):
        raise ValueError(f"bound.{param}: upper must exceed lower (got {lo}, {hi})")
    return lo, hi


def load_single_calibrate_input(path: Path) -> SingleCalibrateInput:
    """Read SteelMPF ``input.csv`` (section,key,value) from ``scripts/calibrate_single/``."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Missing calibration input: {p}")

    df = pd.read_csv(p, comment="#", skipinitialspace=True)
    for col in ("section", "key", "value"):
        if col not in df.columns:
            raise ValueError(f"{p}: expected columns section,key,value (got {list(df.columns)})")

    meta: dict[str, object] = {}
    steel: dict[str, float] = {}
    loss_kv: dict[str, object] = {}
    bounds: dict[str, tuple[float, float]] = {}

    for _, row in df.iterrows():
        section = str(row["section"]).strip().lower()
        key = str(row["key"]).strip()
        value = row["value"]
        if section == "meta":
            meta[key] = value
        elif section == "steel":
            steel[key] = float(value)
        elif section == "loss":
            loss_kv[key.lower()] = value
        elif section == "bound":
            bounds[key] = _parse_bound(value, param=key)
        else:
            raise ValueError(f"{p}: unknown section {section!r} (use meta|steel|loss|bound)")

    for req in ("set_id", "optimize_params", "default_b_p", "default_b_n"):
        if req not in meta:
            raise ValueError(f"{p}: missing meta.{req}")

    missing_steel = [k for k in STEELMPF_SEED_KEYS if k not in steel]
    if missing_steel:
        raise ValueError(f"{p}: missing steel rows: {missing_steel}")

    optimize_params = _parse_optimize_params(meta["optimize_params"])
    missing_bounds = [pname for pname in optimize_params if pname not in bounds]
    if missing_bounds:
        raise ValueError(
            f"{p}: optimize_params {missing_bounds} have no bound.* rows in input.csv"
        )

    loss = CalibrationLossSettings(
        w_feat_l2=float(loss_kv.get("w_feat_l2", 1.0)),
        w_feat_l1=float(loss_kv.get("w_feat_l1", 0.0)),
        w_energy_l2=float(loss_kv.get("w_energy_l2", 0.0)),
        w_energy_l1=float(loss_kv.get("w_energy_l1", 0.0)),
        w_unordered_binenv_l2=float(loss_kv.get("w_unordered_binenv_l2", 0.0)),
        w_unordered_binenv_l1=float(loss_kv.get("w_unordered_binenv_l1", 0.0)),
        use_amplitude_weights=parse_bool_cell(loss_kv.get("use_amplitude_weights", True)),
        amplitude_weight_power=float(loss_kv.get("amplitude_weight_power", 2.0)),
        amplitude_weight_eps=float(loss_kv.get("amplitude_weight_eps", 0.05)),
    )

    return SingleCalibrateInput(
        set_id=int(meta["set_id"]),
        optimize_params=optimize_params,
        default_b_p=float(meta["default_b_p"]),
        default_b_n=float(meta["default_b_n"]),
        steel_seeds=steel,
        loss=loss,
        param_bounds=bounds,
    )
