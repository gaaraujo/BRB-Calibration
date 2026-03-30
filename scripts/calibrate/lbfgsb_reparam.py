"""
L-BFGS-B internal reparameterization: finite box -> z in [0, 1], else raw p.

Used by optimize_brb_mse.optimize_one_specimen. Tested without OpenSees.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def prepare_lbfgsb_parameterization(
    params_to_optimize: list[str],
    bounds_dict: dict[str, tuple[float, float]],
    prow: pd.Series,
    *,
    specimen_hint: str = "",
) -> tuple[list[bool], list[float], list[float], np.ndarray, list[tuple[float, float]]]:
    """
    Build optimizer vector (z in [0,1] or raw p), SciPy bounds, and degeneracy checks.

    Returns
    -------
    use_normalized, L, U, x0, scipy_bounds
        For index i, if use_normalized[i] then L[i], U[i] are the box and x0[i] is z0;
        otherwise x0[i] is raw p0 and L[i]/U[i] are unused (0.0 placeholders).
    """
    use_normalized: list[bool] = []
    Ls: list[float] = []
    Us: list[float] = []
    x0_list: list[float] = []
    scipy_bounds: list[tuple[float, float]] = []

    prefix = f"{specimen_hint}: " if specimen_hint else ""

    for name in params_to_optimize:
        lo, hi = bounds_dict.get(name, (-np.inf, np.inf))
        lo_f = float(lo)
        hi_f = float(hi)
        p0 = float(prow[name])

        if np.isfinite(lo_f) and np.isfinite(hi_f):
            if hi_f < lo_f:
                raise ValueError(
                    f"{prefix}BOUNDS[{name!r}]: upper < lower ({hi_f} < {lo_f})"
                )
            if hi_f == lo_f:
                raise ValueError(
                    f"{prefix}BOUNDS[{name!r}]: upper == lower ({hi_f}); "
                    "remove from PARAMS_TO_OPTIMIZE or fix bounds"
                )
            p_clip = float(np.clip(p0, lo_f, hi_f))
            if p_clip != p0:
                warnings.warn(
                    f"{prefix}Initial {name}={p0} outside [{lo_f}, {hi_f}]; "
                    f"clipped to {p_clip} before optimization.",
                    UserWarning,
                    stacklevel=2,
                )
            span = hi_f - lo_f
            z0 = (p_clip - lo_f) / span
            use_normalized.append(True)
            Ls.append(lo_f)
            Us.append(hi_f)
            x0_list.append(z0)
            scipy_bounds.append((0.0, 1.0))
        else:
            use_normalized.append(False)
            Ls.append(0.0)
            Us.append(0.0)
            x0_list.append(p0)
            scipy_bounds.append((lo_f, hi_f))

    return use_normalized, Ls, Us, np.array(x0_list, dtype=float), scipy_bounds


def variable_params_from_optimizer_x(
    x: np.ndarray,
    params_to_optimize: list[str],
    use_normalized: list[bool],
    Ls: list[float],
    Us: list[float],
) -> dict[str, float]:
    """Map SciPy vector (mixed z and raw p) to physical parameters for the simulator."""
    out: dict[str, float] = {}
    for i, name in enumerate(params_to_optimize):
        if use_normalized[i]:
            out[name] = Ls[i] + (Us[i] - Ls[i]) * float(x[i])
        else:
            out[name] = float(x[i])
    return out
