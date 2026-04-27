"""Steel material kind for calibration: SteelMPF or Steel4."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


def clamp_steel4_isotropic_slopes(sim_kw: dict[str, float]) -> dict[str, float]:
    """
    Steel4 ``-iso`` late-plateau slopes do not exceed the initial isotropic slopes on each side:

    ``b_lp <= b_ip`` and ``b_lc <= b_ic`` (applied with ``min`` so equality is allowed).

    Returns a copy of ``sim_kw``; unknown keys are ignored.
    """
    out = dict(sim_kw)
    if "b_lp" in out and "b_ip" in out:
        out["b_lp"] = min(float(out["b_lp"]), float(out["b_ip"]))
    if "b_lc" in out and "b_ic" in out:
        out["b_lc"] = min(float(out["b_lc"]), float(out["b_ic"]))
    return out


STEEL_MODEL_STEELMPF = "steelmpf"
STEEL_MODEL_STEEL4 = "steel4"

STEEL_MODEL_CHOICES: frozenset[str] = frozenset({STEEL_MODEL_STEELMPF, STEEL_MODEL_STEEL4})

# Order for filesystem subfolders (e.g. ``.../overlays_best_l1_l2/<name>/``, ``optimal_params_vs_geometry/<name>/``).
STEEL_MODEL_DIR_ORDER: tuple[str, ...] = (STEEL_MODEL_STEELMPF, STEEL_MODEL_STEEL4)


def ordered_steel_model_subdirs(models: set[str]) -> list[str]:
    """Stable ordering of normalized ``steel_model`` values for per-model output directories."""
    ordered = [m for m in STEEL_MODEL_DIR_ORDER if m in models]
    ordered.extend(sorted(models - set(ordered)))
    return ordered


# Steel4 `-iso` branch (CSV columns); `R_i` is passed twice to OpenSees (tension + compression).
STEEL4_ISO_KEYS: tuple[str, ...] = ("b_ip", "rho_ip", "b_lp", "R_i", "l_yp", "b_ic", "rho_ic", "b_lc")

# Subset used by ``clamp_steel4_isotropic_slopes`` / CSV precision for small slope magnitudes.
STEEL4_ISO_SLOPE_KEYS: tuple[str, ...] = ("b_ip", "b_lp", "b_ic", "b_lc")

# SteelMPF-only isotropic + decay parameters (ignored when simulating Steel4).
STEELMPF_ISO_KEYS: tuple[str, ...] = ("a1", "a2", "a3", "a4", "fup_ratio", "fun_ratio", "Ru0")

# Shared kinematic / modulus seeds in CSV for both models.
SHARED_STEEL_KEYS: tuple[str, ...] = ("E", "R0", "cR1", "cR2")


def normalize_steel_model(raw: object) -> str:
    """Return ``steelmpf`` or ``steel4``. Missing / blank / NaN defaults to ``steelmpf``."""
    if raw is None:
        return STEEL_MODEL_STEELMPF
    try:
        import pandas as pd

        if isinstance(raw, (float, int)) and not isinstance(raw, bool):
            if isinstance(raw, float) and pd.isna(raw):
                return STEEL_MODEL_STEELMPF
        if pd.isna(raw):
            return STEEL_MODEL_STEELMPF
    except Exception:
        pass
    s = str(raw).strip().lower()
    if not s or s == "nan":
        return STEEL_MODEL_STEELMPF
    if s == STEEL_MODEL_STEEL4 or s == "steel_4":
        return STEEL_MODEL_STEEL4
    if s in (STEEL_MODEL_STEELMPF, "steel_mpf", "mpf"):
        return STEEL_MODEL_STEELMPF
    raise ValueError(f"steel_model must be one of {sorted(STEEL_MODEL_CHOICES)}; got {raw!r}")


def sync_steel4_isotropic_slopes_in_output_row(row: MutableMapping[str, Any]) -> None:
    """
    After optimization / merge, set ``b_lp`` / ``b_lc`` to values ``run_simulation`` applies.

    Same rules as ``clamp_steel4_isotropic_slopes`` so parameter CSV rows match OpenSees.
    """
    sm = normalize_steel_model(row.get("steel_model"))
    if sm != STEEL_MODEL_STEEL4:
        return
    ks = STEEL4_ISO_SLOPE_KEYS
    if not all(k in row for k in ks):
        return
    try:
        import pandas as pd
    except ImportError:
        pd = None

    def _missing(v: object) -> bool:
        if pd is not None and pd.isna(v):
            return True
        return v is None

    parts: dict[str, float] = {}
    for k in ks:
        v = row[k]
        if _missing(v):
            return
        parts[k] = float(v)

    clamped = clamp_steel4_isotropic_slopes(parts)
    row["b_lp"] = clamped["b_lp"]
    row["b_lc"] = clamped["b_lc"]
