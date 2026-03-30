"""
Apparent b_p / b_n from an **unordered** digitized F-u cloud.

**Plastic mask (scatter cloud):** only ``|F| / (f_yc A_sc) > PLASTIC_NORMALIZED_FORCE_MIN`` plus finite ``u``, ``F`` -- no elastic-line
filter (unlike segment-based ``extract_bn_bp``, which also requires ``|u| > |u_el|``).

Tension (F >= 0) and compression (F < 0) plastic subsets are each reduced to a **per-u-bin
envelope** (max F per bin for tension, min F per bin for compression), then a single linear
F vs u fit yields k_sh and b = k_sh / k_init (clipped), matching the segment-based pipeline.

Use :func:`compute_envelope_bn_unordered` when you need masks / envelope coordinates for plotting.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model.brace_geometry import compute_Q

E_ksi = 29000.0
MIN_DEF_RANGE_FRAC = 0.005
MIN_DEF_RANGE_ABS_IN = 0.001
B_CLIP = (0.0, 0.2)
# First point in hardening fit: normalized force |F|/(f_y A_sc) must exceed this (same as resampled segments in extract_bn_bp).
PLASTIC_NORMALIZED_FORCE_MIN = 1.1


def plastic_point_mask(
    u: np.ndarray,
    F: np.ndarray,
    *,
    fy_A: float,
) -> np.ndarray:
    """Plastic if |F|/(f_y A_sc) > PLASTIC_NORMALIZED_FORCE_MIN and u, F are finite."""
    u = np.asarray(u, dtype=float)
    F = np.asarray(F, dtype=float)
    if fy_A <= 0 or not np.isfinite(fy_A):
        return np.zeros_like(u, dtype=bool)
    m_force = np.abs(F) / fy_A > PLASTIC_NORMALIZED_FORCE_MIN
    finite = np.isfinite(u) & np.isfinite(F)
    return m_force & finite


def _bin_extrema_envelope(
    u: np.ndarray,
    F: np.ndarray,
    n_bins: int,
    mode: str,
    *,
    global_row_index: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One (u,F) per bin: max F (tension) or min F (compression).

    If ``global_row_index`` is provided (same length as u, F), also return the original
    row index in the full cloud for each envelope vertex.
    """
    u = np.asarray(u, dtype=float)
    F = np.asarray(F, dtype=float)
    lo, hi = float(np.nanmin(u)), float(np.nanmax(u))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.array([]), np.array([]), np.array([], dtype=int)
    edges = np.linspace(lo, hi, int(n_bins) + 1)
    idx = np.digitize(np.clip(u, lo, hi), edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    ue: list[float] = []
    fe: list[float] = []
    gi: list[int] = []
    for b in range(n_bins):
        m = idx == b
        if not np.any(m):
            continue
        sub = np.where(m)[0]
        if mode == "max":
            j = int(sub[np.argmax(F[m])])
        else:
            j = int(sub[np.argmin(F[m])])
        ue.append(float(u[j]))
        fe.append(float(F[j]))
        if global_row_index is not None:
            gi.append(int(global_row_index[j]))
    ge = np.asarray(gi, dtype=int) if global_row_index is not None else np.array([], dtype=int)
    return np.asarray(ue, dtype=float), np.asarray(fe, dtype=float), ge


def _b_linear_fit(
    u: np.ndarray,
    F: np.ndarray,
    *,
    kinit: float,
    L_y: float,
) -> float:
    """Apparent Menegotto–Pinto ``b`` from slope of F vs u in normalized space (clipped)."""
    if len(u) < 2 or kinit <= 0:
        return float("nan")
    min_range = MIN_DEF_RANGE_FRAC * L_y if L_y > 0 else MIN_DEF_RANGE_ABS_IN
    span = float(np.ptp(u))
    if span < min_range:
        return float("nan")
    cov = float(np.cov(u, F)[0, 1])
    var_u = float(np.var(u))
    if var_u <= 1e-20:
        return float("nan")
    ksh = cov / var_u
    b = ksh / kinit
    return float(np.clip(b, B_CLIP[0], B_CLIP[1]))


@dataclass(frozen=True)
class CloudEnvelopeBnDiagnostics:
    """Plastic mask and per-bin envelope vertices used for cloud b_p / b_n (full-array length)."""

    b_p: float
    b_n: float
    plastic_mask: np.ndarray
    tension_envelope_mask: np.ndarray
    compression_envelope_mask: np.ndarray
    u_tension_env: np.ndarray
    F_tension_env: np.ndarray
    u_compression_env: np.ndarray
    F_compression_env: np.ndarray
    n_bins: int


def compute_envelope_bn_unordered(
    u: np.ndarray,
    F: np.ndarray,
    *,
    L_T: float,
    L_y: float,
    A_sc: float,
    A_t: float,
    f_yc: float,
    E_ksi_val: float = E_ksi,
    n_bins: int | None = None,
) -> CloudEnvelopeBnDiagnostics:
    """
    Same fitting as :func:`envelope_bn_from_unordered`, plus boolean masks and envelope coordinates
    for visualization (aligned with ``plot_b_slopes`` tension/compression colors).
    """
    u = np.asarray(u, dtype=float)
    F = np.asarray(F, dtype=float)
    n = len(u)
    plastic_mask = np.zeros(n, dtype=bool)
    tension_envelope_mask = np.zeros(n, dtype=bool)
    compression_envelope_mask = np.zeros(n, dtype=bool)

    Q = compute_Q(L_T, L_y, A_sc, A_t)
    E_hat = Q * float(E_ksi_val)
    fy_A = float(f_yc) * float(A_sc)
    kinit = E_hat * A_sc / L_T if L_T > 0 else 0.0

    mask = plastic_point_mask(u, F, fy_A=fy_A)
    plastic_mask[:] = mask
    row_plastic = np.nonzero(mask)[0]
    u_p = u[mask]
    F_p = F[mask]
    if len(u_p) < 4:
        return CloudEnvelopeBnDiagnostics(
            b_p=float("nan"),
            b_n=float("nan"),
            plastic_mask=plastic_mask,
            tension_envelope_mask=tension_envelope_mask,
            compression_envelope_mask=compression_envelope_mask,
            u_tension_env=np.array([]),
            F_tension_env=np.array([]),
            u_compression_env=np.array([]),
            F_compression_env=np.array([]),
            n_bins=0,
        )

    nb = n_bins if n_bins is not None else max(8, int(np.sqrt(len(u_p))))
    nb = max(4, min(nb, len(u_p) // 2))

    mt = F_p >= 0.0
    mc = F_p < 0.0
    ut, Ft, row_t = u_p[mt], F_p[mt], row_plastic[mt]
    uc, Fc, row_c = u_p[mc], F_p[mc], row_plastic[mc]

    b_p = float("nan")
    eu_t = eF_t = np.array([])
    if len(ut) >= 4:
        eu_t, eF_t, gi_t = _bin_extrema_envelope(ut, Ft, nb, "max", global_row_index=row_t)
        b_p = _b_linear_fit(eu_t, eF_t, kinit=kinit, L_y=L_y)
        for g in gi_t:
            tension_envelope_mask[g] = True

    b_n = float("nan")
    eu_c = eF_c = np.array([])
    if len(uc) >= 4:
        eu_c, eF_c, gi_c = _bin_extrema_envelope(uc, Fc, nb, "min", global_row_index=row_c)
        b_n = _b_linear_fit(eu_c, eF_c, kinit=kinit, L_y=L_y)
        for g in gi_c:
            compression_envelope_mask[g] = True

    return CloudEnvelopeBnDiagnostics(
        b_p=b_p,
        b_n=b_n,
        plastic_mask=plastic_mask,
        tension_envelope_mask=tension_envelope_mask,
        compression_envelope_mask=compression_envelope_mask,
        u_tension_env=eu_t,
        F_tension_env=eF_t,
        u_compression_env=eu_c,
        F_compression_env=eF_c,
        n_bins=nb,
    )


def envelope_bn_from_unordered(
    u: np.ndarray,
    F: np.ndarray,
    *,
    L_T: float,
    L_y: float,
    A_sc: float,
    A_t: float,
    f_yc: float,
    E_ksi_val: float = E_ksi,
    n_bins: int | None = None,
) -> tuple[float, float]:
    """
    Return (b_p, b_n) from cloud arrays. NaN if insufficient plastic / envelope points.

    Tension uses plastic points with F >= 0; compression with F < 0.
    """
    d = compute_envelope_bn_unordered(
        u,
        F,
        L_T=L_T,
        L_y=L_y,
        A_sc=A_sc,
        A_t=A_t,
        f_yc=f_yc,
        E_ksi_val=E_ksi_val,
        n_bins=n_bins,
    )
    return (d.b_p, d.b_n)
