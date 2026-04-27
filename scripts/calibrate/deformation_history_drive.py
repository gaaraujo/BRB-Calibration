"""
Prepare digitized scatter-cloud **deformation drives** (``deformation_history.csv``).

Raw digitized histories often contain small displacement reversals from measurement noise.
We (1) optionally median-filter, (2) **Ramer-Douglas-Peucker** simplify the polyline
``(s, u)`` where ``s`` is cumulative |Deltau| along the recorded order (both in inches), then
(3) **resample** at uniform spacing in cumulative |Deltau| (same rule as ``resample_segment_along_u_path``),
then **prepend** a leading sample ``0.1\,D_\mathrm{sample}`` (same ``D_\mathrm{sample}`` as resampling;
small offset from zero as a SteelMPF workaround).

This yields a drive that moves in clear half-cycles (monotonic between turning points) without
tiny noise loops, while staying close to the original path in the (s,u) plane.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.ndimage import median_filter

from calibrate.resample_experiment import d_sampling_from_brace_params, resample_segment_along_u_path

# Default RDP tolerance [in]: suppress perpendicular deviations smaller than this in (s,u).
DEFAULT_RDP_EPSILON_IN = 0.005

# Leading drive sample = this fraction × D_sampling [in] (temporary workaround for SteelMPF zero-step history).
PREPEND_LEAD_FRAC = 0.1


def prepend_lead_deformation(u: np.ndarray, d_sampling_in: float) -> np.ndarray:
    """
    Prepend one sample ``Deformation[in] = PREPEND_LEAD_FRAC * d_sampling_in`` at the start of the drive.

    Uses the same ``d_sampling_in`` spacing [in] as uniform |Δu| resampling (or brace-based D_sampling).
    """
    u = np.asarray(u, dtype=float).reshape(-1)
    lead = float(PREPEND_LEAD_FRAC) * float(d_sampling_in)
    return np.concatenate([[lead], u])


def _point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance from point ``p`` to segment ``a``-``b`` (2D)."""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-30:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def rdp_simplify_displacement(u: np.ndarray, epsilon_in: float) -> np.ndarray:
    """
    Ramer-Douglas-Peucker on points ``(s, u)`` with ``s`` = cumulative sum of |Deltau|.

    Both coordinates use inches so Euclidean ``epsilon_in`` is a physical tolerance on
    geometric deviation from the simplified polyline.
    """
    u = np.asarray(u, dtype=float).reshape(-1)
    n = u.size
    if n <= 2 or epsilon_in <= 0.0:
        return u.copy()
    du = np.abs(np.diff(u))
    s = np.concatenate([[0.0], np.cumsum(du)])
    points = np.column_stack([s, u])

    keep = {0, n - 1}
    stack = [(0, n - 1)]
    while stack:
        start, end = stack.pop()
        if end - start <= 1:
            continue
        pa, pb = points[start], points[end]
        max_d = 0.0
        max_i = start
        for i in range(start + 1, end):
            d = _point_to_segment_distance(points[i], pa, pb)
            if d > max_d:
                max_d = d
                max_i = i
        if max_d <= epsilon_in:
            continue
        keep.add(max_i)
        stack.append((start, max_i))
        stack.append((max_i, end))

    idx = np.asarray(sorted(keep), dtype=int)
    return u[idx]


def prepare_deformation_drive(
    u: np.ndarray,
    *,
    rdp_epsilon_in: float = DEFAULT_RDP_EPSILON_IN,
    median_kernel: int = 0,
    d_sampling_in: float | None = None,
    resample: bool = True,
    brace: dict[str, Any] | None = None,
    u_fallback: np.ndarray | None = None,
) -> np.ndarray:
    """
    Filter (optional median), RDP-simplify, then uniform |Deltau| resampling along index order.

    Parameters
    ----------
    u
        Raw ``Deformation[in]`` samples in drive order.
    rdp_epsilon_in
        RDP tolerance [in]. ``<= 0`` skips RDP (still may resample).
    median_kernel
        If > 1 and odd, apply ``scipy.ndimage.median_filter`` along the series first.
    d_sampling_in
        Target spacing in cumulative |Deltau| for resampling [in]. If ``None``, use
        ``d_sampling_from_brace_params`` when ``brace`` has all keys, else
        ``max(|u|)/100`` after simplification (with a small warning if brace incomplete).
    resample
        If False, return after median + RDP without uniform resampling.
    brace
        Optional dict with ``fyp_ksi``, ``L_T_in``, ``L_y_in``, ``A_sc_in2``, ``A_t_in2``, ``E_ksi``
        for ``d_sampling_from_brace_params`` (Dy/10 convention).
    u_fallback
        Array passed to ``d_sampling_from_brace_params`` when Dy is invalid (usually same as ``u``).

    After all steps, **prepend** ``PREPEND_LEAD_FRAC * D_\mathrm{sample}`` (see ``prepend_lead_deformation``).
    """
    u = np.asarray(u, dtype=float).reshape(-1)
    uf = u_fallback if u_fallback is not None else u

    def _d_sp_for_lead() -> float:
        if d_sampling_in is not None and float(d_sampling_in) > 0.0:
            return float(d_sampling_in)
        if brace is not None:
            try:
                return d_sampling_from_brace_params(
                    fyp_ksi=float(brace["fyp_ksi"]),
                    L_T_in=float(brace["L_T_in"]),
                    L_y_in=float(brace["L_y_in"]),
                    A_sc_in2=float(brace["A_sc_in2"]),
                    A_t_in2=float(brace["A_t_in2"]),
                    E_ksi=float(brace["E_ksi"]),
                    u_fallback=uf,
                )
            except (KeyError, TypeError, ValueError):
                umax = float(np.nanmax(np.abs(u))) if u.size else float(np.nanmax(np.abs(uf)))
                d_sp = umax / 100.0 if umax > 0 else 1e-6
                warnings.warn(
                    "prepare_deformation_drive: incomplete brace dict; "
                    f"using d_sampling = max(|u|)/100 ~ {d_sp:.6g} in.",
                    UserWarning,
                    stacklevel=2,
                )
                return d_sp
        umax = float(np.nanmax(np.abs(u))) if u.size else float(np.nanmax(np.abs(uf)))
        d_sp = umax / 100.0 if umax > 0 else 1e-6
        warnings.warn(
            "prepare_deformation_drive: no d_sampling_in or brace; "
            f"using max(|u|)/100 ~ {d_sp:.6g} in.",
            UserWarning,
            stacklevel=2,
        )
        return d_sp

    if u.size == 0:
        return np.array([float(PREPEND_LEAD_FRAC) * _d_sp_for_lead()], dtype=float)

    if median_kernel > 1:
        k = int(median_kernel)
        if k % 2 == 0:
            k += 1
        u = median_filter(u, size=k, mode="nearest")

    if rdp_epsilon_in > 0.0:
        u = rdp_simplify_displacement(u, rdp_epsilon_in)

    if u.size <= 1:
        return prepend_lead_deformation(u, _d_sp_for_lead())

    if not resample:
        return prepend_lead_deformation(u, _d_sp_for_lead())

    if d_sampling_in is not None and float(d_sampling_in) > 0.0:
        d_sp = float(d_sampling_in)
    elif brace is not None:
        try:
            d_sp = d_sampling_from_brace_params(
                fyp_ksi=float(brace["fyp_ksi"]),
                L_T_in=float(brace["L_T_in"]),
                L_y_in=float(brace["L_y_in"]),
                A_sc_in2=float(brace["A_sc_in2"]),
                A_t_in2=float(brace["A_t_in2"]),
                E_ksi=float(brace["E_ksi"]),
                u_fallback=uf,
            )
        except (KeyError, TypeError, ValueError):
            umax = float(np.nanmax(np.abs(u)))
            d_sp = umax / 100.0 if umax > 0 else 1e-6
            warnings.warn(
                "prepare_deformation_drive: incomplete brace dict; "
                f"using d_sampling = max(|u|)/100 ~ {d_sp:.6g} in.",
                UserWarning,
                stacklevel=2,
            )
    else:
        umax = float(np.nanmax(np.abs(u)))
        d_sp = umax / 100.0 if umax > 0 else 1e-6
        warnings.warn(
            "prepare_deformation_drive: no d_sampling_in or brace; "
            f"using max(|u|)/100 ~ {d_sp:.6g} in.",
            UserWarning,
            stacklevel=2,
        )

    u_out, _ = resample_segment_along_u_path(u, u, d_sp)
    return prepend_lead_deformation(u_out, d_sp)
