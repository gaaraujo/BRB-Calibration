"""
Path-ordered calibration metrics matching the main repo's ``optimize_brb_mse`` breakdown.

- **J_feat** (L2/L1): cycle-amplitude-weighted mean of per-cycle mean landmark errors
  (same slot pairing as ``jfeat_landmarks`` / ``landmark_vector``).
- **J_E** (L2/L1): mean over weight cycles of normalized dissipated-energy error; ``S_E = S_D S_F``.
- **J_binenv** (L2/L1): binned upper/lower force-band mismatch (``digitized_unordered_eval_lib``).

Vendored numerics are kept local so ``bayesian/`` stays standalone (see repo
``amplitude_mse_partition.py``, ``digitized_unordered_eval_lib.py``).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .jfeat_landmarks import (
    N_LANDMARK_SLOTS,
    _slot_error_combined_l1,
    _slot_error_combined_sq,
    extract_cycle_landmarks,
    pair_sim_cycle_landmarks,
)

FAILURE_PENALTY = 1e6
DEFAULT_BINENV_N_BINS = 32


def _integrate_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))  # type: ignore[attr-defined]


def _cycle_signed_trapz_work(D: np.ndarray, F: np.ndarray, start: int, end: int) -> float:
    if end <= start:
        return 0.0
    d = np.asarray(D[start:end], dtype=float)
    f = np.asarray(F[start:end], dtype=float)
    if len(d) < 2:
        return 0.0
    return _integrate_trapezoid(f, d)


def _cycle_abs_trapz_work(D: np.ndarray, F: np.ndarray, start: int, end: int) -> float:
    return abs(_cycle_signed_trapz_work(D, F, start, end))


def energy_scale_s_e(D_exp: np.ndarray, F_exp: np.ndarray) -> float:
    d = np.asarray(D_exp, dtype=float)
    f = np.asarray(F_exp, dtype=float)
    du = float(np.nanmax(d) - np.nanmin(d))
    df = float(np.nanmax(f) - np.nanmin(f))
    s_e = du * df
    if not np.isfinite(s_e) or s_e <= 0.0:
        return 1.0
    return s_e


def energy_mse_cycles(
    D_exp: np.ndarray,
    F_exp: np.ndarray,
    F_sim: np.ndarray,
    meta: list[dict],
    *,
    failure_penalty: float = FAILURE_PENALTY,
) -> float:
    D_exp = np.asarray(D_exp, dtype=float)
    F_exp = np.asarray(F_exp, dtype=float)
    F_sim = np.asarray(F_sim, dtype=float)
    if F_exp.shape != F_sim.shape or D_exp.shape != F_exp.shape:
        return failure_penalty
    if not meta:
        return 0.0
    s_e = energy_scale_s_e(D_exp, F_exp)
    if s_e <= 0.0 or not np.isfinite(s_e):
        s_e = 1.0
    numer = 0.0
    denom = 0.0
    for m in meta:
        s, e = int(m["start"]), int(m["end"])
        if e <= s:
            continue
        e_exp = _cycle_abs_trapz_work(D_exp, F_exp, s, e)
        e_sim = _cycle_abs_trapz_work(D_exp, F_sim, s, e)
        diff = e_sim - e_exp
        numer += diff * diff
        denom += 1.0
    if not np.isfinite(denom) or denom <= 0.0:
        return failure_penalty
    return float((numer / denom) / (s_e * s_e))


def energy_mae_cycles(
    D_exp: np.ndarray,
    F_exp: np.ndarray,
    F_sim: np.ndarray,
    meta: list[dict],
    *,
    failure_penalty: float = FAILURE_PENALTY,
) -> float:
    D_exp = np.asarray(D_exp, dtype=float)
    F_exp = np.asarray(F_exp, dtype=float)
    F_sim = np.asarray(F_sim, dtype=float)
    if F_exp.shape != F_sim.shape or D_exp.shape != F_exp.shape:
        return failure_penalty
    if not meta:
        return 0.0
    s_e = energy_scale_s_e(D_exp, F_exp)
    if s_e <= 0.0 or not np.isfinite(s_e):
        s_e = 1.0
    numer = 0.0
    denom = 0.0
    for m in meta:
        s, e = int(m["start"]), int(m["end"])
        if e <= s:
            continue
        e_exp = _cycle_abs_trapz_work(D_exp, F_exp, s, e)
        e_sim = _cycle_abs_trapz_work(D_exp, F_sim, s, e)
        numer += abs(e_sim - e_exp)
        denom += 1.0
    if not np.isfinite(denom) or denom <= 0.0:
        return failure_penalty
    return float((numer / denom) / s_e)


def _compute_binned_binenv_l2(
    exp_xy_raw: np.ndarray,
    num_xy_raw: np.ndarray,
    s_f: float,
    *,
    n_bins: int = DEFAULT_BINENV_N_BINS,
    scale_eps: float = 1e-12,
) -> float:
    if exp_xy_raw.shape[0] == 0 or num_xy_raw.shape[0] == 0:
        return float("nan")
    if not np.isfinite(s_f) or abs(s_f) <= scale_eps:
        return float("nan")
    u_e = exp_xy_raw[:, 0]
    f_e = exp_xy_raw[:, 1]
    u_n = num_xy_raw[:, 0]
    f_n = num_xy_raw[:, 1]
    u_min = float(np.min(u_e))
    u_max = float(np.max(u_e))
    if not np.isfinite(u_min) or not np.isfinite(u_max) or u_max <= u_min:
        return float("nan")
    edges = np.linspace(u_min, u_max, int(n_bins) + 1)
    errs: list[float] = []
    for b in range(int(n_bins)):
        lo, hi = float(edges[b]), float(edges[b + 1])
        if b == int(n_bins) - 1:
            m_e = (u_e >= lo) & (u_e <= hi)
            m_n = (u_n >= lo) & (u_n <= hi)
        else:
            m_e = (u_e >= lo) & (u_e < hi)
            m_n = (u_n >= lo) & (u_n < hi)
        if int(np.count_nonzero(m_e)) < 1 or int(np.count_nonzero(m_n)) < 1:
            continue
        f_up_e = float(np.max(f_e[m_e]))
        f_lo_e = float(np.min(f_e[m_e]))
        f_up_n = float(np.max(f_n[m_n]))
        f_lo_n = float(np.min(f_n[m_n]))
        du = (f_up_e - f_up_n) / s_f
        dl = (f_lo_e - f_lo_n) / s_f
        errs.append(0.5 * (du * du + dl * dl))
    if not errs:
        return float("nan")
    return float(np.mean(errs))


def _compute_binned_binenv_l1(
    exp_xy_raw: np.ndarray,
    num_xy_raw: np.ndarray,
    s_f: float,
    *,
    n_bins: int = DEFAULT_BINENV_N_BINS,
    scale_eps: float = 1e-12,
) -> float:
    if exp_xy_raw.shape[0] == 0 or num_xy_raw.shape[0] == 0:
        return float("nan")
    if not np.isfinite(s_f) or abs(s_f) <= scale_eps:
        return float("nan")
    u_e = exp_xy_raw[:, 0]
    f_e = exp_xy_raw[:, 1]
    u_n = num_xy_raw[:, 0]
    f_n = num_xy_raw[:, 1]
    u_min = float(np.min(u_e))
    u_max = float(np.max(u_e))
    if not np.isfinite(u_min) or not np.isfinite(u_max) or u_max <= u_min:
        return float("nan")
    edges = np.linspace(u_min, u_max, int(n_bins) + 1)
    errs: list[float] = []
    for b in range(int(n_bins)):
        lo, hi = float(edges[b]), float(edges[b + 1])
        if b == int(n_bins) - 1:
            m_e = (u_e >= lo) & (u_e <= hi)
            m_n = (u_n >= lo) & (u_n <= hi)
        else:
            m_e = (u_e >= lo) & (u_e < hi)
            m_n = (u_n >= lo) & (u_n < hi)
        if int(np.count_nonzero(m_e)) < 1 or int(np.count_nonzero(m_n)) < 1:
            continue
        f_up_e = float(np.max(f_e[m_e]))
        f_lo_e = float(np.min(f_e[m_e]))
        f_up_n = float(np.max(f_n[m_n]))
        f_lo_n = float(np.min(f_n[m_n]))
        du = abs(f_up_e - f_up_n) / s_f
        dl = abs(f_lo_e - f_lo_n) / s_f
        errs.append(0.5 * (du + dl))
    if not errs:
        return float("nan")
    return float(np.mean(errs))


def compute_unordered_binenv_metrics(
    u_exp: np.ndarray,
    F_exp: np.ndarray,
    D_num: np.ndarray,
    F_num: np.ndarray,
    *,
    scale_eps: float = 1e-12,
    n_binenv_bins: int = DEFAULT_BINENV_N_BINS,
) -> tuple[float, float]:
    exp_xy_raw = np.column_stack([np.asarray(u_exp, dtype=float), np.asarray(F_exp, dtype=float)])
    num_xy_raw = np.column_stack([np.asarray(D_num, dtype=float), np.asarray(F_num, dtype=float)])
    exp_mask = np.isfinite(exp_xy_raw).all(axis=1)
    num_mask = np.isfinite(num_xy_raw).all(axis=1)
    exp_xy_raw = exp_xy_raw[exp_mask]
    num_xy_raw = num_xy_raw[num_mask]
    if exp_xy_raw.shape[0] == 0 or num_xy_raw.shape[0] == 0:
        return float("nan"), float("nan")
    s_f = float(np.max(exp_xy_raw[:, 1]) - np.min(exp_xy_raw[:, 1]))
    if not np.isfinite(s_f) or abs(s_f) <= scale_eps:
        return float("nan"), float("nan")
    j2 = _compute_binned_binenv_l2(
        exp_xy_raw, num_xy_raw, s_f, n_bins=n_binenv_bins, scale_eps=scale_eps
    )
    j1 = _compute_binned_binenv_l1(
        exp_xy_raw, num_xy_raw, s_f, n_bins=n_binenv_bins, scale_eps=scale_eps
    )
    return float(j2), float(j1)


def jfeat_weighted_l2_l1(
    D: np.ndarray,
    F_exp: np.ndarray,
    F_sim: np.ndarray,
    meta: list[dict],
    *,
    fy: float,
    a_sc: float,
    dy: float | None,
    s_f: float,
    s_d: float,
) -> tuple[float, float]:
    """``sum_c w_c * bar_e_c / sum_c w_c`` for L2 and L1 (contributing cycles only)."""
    D = np.asarray(D, dtype=float)
    F_exp = np.asarray(F_exp, dtype=float)
    F_sim = np.asarray(F_sim, dtype=float)
    if not meta:
        warnings.warn("jfeat_weighted_l2_l1: empty meta; returning NaN", UserWarning, stacklevel=2)
        return float("nan"), float("nan")

    w_num_l2 = 0.0
    w_den_l2 = 0.0
    w_num_l1 = 0.0
    w_den_l1 = 0.0

    for m in meta:
        s, e = int(m["start"]), int(m["end"])
        w_c = float(m.get("w_c", 1.0))
        if not np.isfinite(w_c) or w_c <= 0.0:
            w_c = 1.0
        if e <= s:
            continue

        le = extract_cycle_landmarks(D, F_exp, s, e, fy=fy, a_sc=a_sc, dy=dy)
        ls, le_m, _ = pair_sim_cycle_landmarks(
            D, F_exp, F_sim, s, e, le, fy=fy, a_sc=a_sc, geometry_f=F_exp
        )

        errs_l2: list[float] = []
        errs_l1: list[float] = []
        for i in range(N_LANDMARK_SLOTS):
            ee2 = _slot_error_combined_sq(le_m[i], ls[i], s_f, s_d)
            ee1 = _slot_error_combined_l1(le_m[i], ls[i], s_f, s_d)
            if ee2 is not None:
                errs_l2.append(ee2)
            if ee1 is not None:
                errs_l1.append(ee1)

        if not errs_l2:
            continue
        bar_l2 = float(sum(errs_l2) / len(errs_l2))
        w_num_l2 += w_c * bar_l2
        w_den_l2 += w_c
        if errs_l1:
            bar_l1 = float(sum(errs_l1) / len(errs_l1))
            w_num_l1 += w_c * bar_l1
            w_den_l1 += w_c

    j_l2 = float(w_num_l2 / w_den_l2) if w_den_l2 > 0.0 and np.isfinite(w_den_l2) else float("nan")
    j_l1 = float(w_num_l1 / w_den_l1) if w_den_l1 > 0.0 and np.isfinite(w_den_l1) else float("nan")
    if not np.isfinite(j_l2):
        warnings.warn(
            "jfeat_weighted_l2_l1: no L2-contributing cycles; returning NaN for L2", UserWarning, stacklevel=2
        )
    return j_l2, j_l1


def _cycle_delta_max_in(D: np.ndarray, m: dict) -> float:
    """Max |δ| in the cycle [in]: ``meta['amp']`` if present, else from ``D[start:end]``."""
    raw = m.get("amp")
    if raw is not None:
        try:
            v = float(raw)
            if np.isfinite(v):
                return v
        except (TypeError, ValueError):
            pass
    s, e = int(m["start"]), int(m["end"])
    if e <= s:
        return float("nan")
    seg = np.asarray(D[s:e], dtype=float)
    if seg.size == 0:
        return float("nan")
    return float(np.max(np.abs(seg)))


def compute_per_cycle_metric_rows(
    D: np.ndarray,
    F_exp: np.ndarray,
    F_sim: np.ndarray,
    meta: list[dict],
    *,
    fy: float,
    a_sc: float,
    dy: float | None,
    s_d: float,
    s_f: float,
    l_y_in: float,
) -> list[dict[str, float | int]]:
    """
    One dict per ``meta`` row: amplitude ``delta_in``, ``strain_pct`` = ``100 * delta_in / L_y``,
    ``w_c``, per-cycle mean landmark errors (``j_feat_l2_mean``, ``j_feat_l1_mean``), and
    energy terms ``j_e_l2`` = ``(ΔE_c)^2 / S_E^2``, ``j_e_l1`` = ``|ΔE_c| / S_E`` (global ``S_E``).

    The mean of ``j_e_l2`` / ``j_e_l1`` over cycles with ``end > start`` matches
    :func:`energy_mse_cycles` / :func:`energy_mae_cycles`.
    """
    D = np.asarray(D, dtype=float)
    F_exp = np.asarray(F_exp, dtype=float)
    F_sim = np.asarray(F_sim, dtype=float)
    s_e = energy_scale_s_e(D, F_exp)
    inv_se2 = 1.0 / (s_e * s_e) if s_e > 0.0 and np.isfinite(s_e) else 0.0
    inv_se = 1.0 / s_e if s_e > 0.0 and np.isfinite(s_e) else 0.0

    out: list[dict[str, float | int]] = []
    for i, m in enumerate(meta):
        s, e = int(m["start"]), int(m["end"])
        w_c = float(m.get("w_c", 1.0))
        if not np.isfinite(w_c) or w_c <= 0.0:
            w_c = 1.0

        delta_in = _cycle_delta_max_in(D, m)
        strain_pct = (
            100.0 * delta_in / l_y_in if l_y_in > 0.0 and np.isfinite(delta_in) else float("nan")
        )

        if e <= s:
            out.append(
                {
                    "cycle_id": int(i),
                    "start": s,
                    "end": e,
                    "delta_in": float(delta_in),
                    "strain_pct": float(strain_pct),
                    "w_c": w_c,
                    "j_feat_l2_mean": float("nan"),
                    "j_feat_l1_mean": float("nan"),
                    "j_e_l2": float("nan"),
                    "j_e_l1": float("nan"),
                }
            )
            continue

        le = extract_cycle_landmarks(D, F_exp, s, e, fy=fy, a_sc=a_sc, dy=dy)
        ls, le_m, _ = pair_sim_cycle_landmarks(
            D, F_exp, F_sim, s, e, le, fy=fy, a_sc=a_sc, geometry_f=F_exp
        )

        errs_l2: list[float] = []
        errs_l1: list[float] = []
        for slot in range(N_LANDMARK_SLOTS):
            ee2 = _slot_error_combined_sq(le_m[slot], ls[slot], s_f, s_d)
            ee1 = _slot_error_combined_l1(le_m[slot], ls[slot], s_f, s_d)
            if ee2 is not None:
                errs_l2.append(ee2)
            if ee1 is not None:
                errs_l1.append(ee1)

        jf2 = float(sum(errs_l2) / len(errs_l2)) if errs_l2 else float("nan")
        jf1 = float(sum(errs_l1) / len(errs_l1)) if errs_l1 else float("nan")

        e_exp = _cycle_abs_trapz_work(D, F_exp, s, e)
        e_sim = _cycle_abs_trapz_work(D, F_sim, s, e)
        diff = e_sim - e_exp
        je2 = float(diff * diff * inv_se2)
        je1 = float(abs(diff) * inv_se)

        out.append(
            {
                "cycle_id": int(i),
                "start": s,
                "end": e,
                "delta_in": float(delta_in),
                "strain_pct": float(strain_pct),
                "w_c": w_c,
                "j_feat_l2_mean": jf2,
                "j_feat_l1_mean": jf1,
                "j_e_l2": je2,
                "j_e_l1": je1,
            }
        )
    return out


@dataclass(frozen=True)
class RepoErrorMetrics:
    """Raw metrics aligned with ``optimized_brb_parameters_metrics.csv`` naming (``final_*`` / ``initial_*``)."""

    j_feat_l2: float
    j_feat_l1: float
    j_e_l2: float
    j_e_l1: float
    binenv_l2: float
    binenv_l1: float
    s_d: float
    s_f: float
    s_e: float


def compute_repo_style_metrics(
    D: np.ndarray,
    F_exp: np.ndarray,
    F_sim: np.ndarray,
    meta: list[dict],
    *,
    fy: float,
    a_sc: float,
    dy: float | None,
    s_d: float,
    s_f: float,
    n_binenv_bins: int = DEFAULT_BINENV_N_BINS,
    failure_penalty: float = FAILURE_PENALTY,
) -> RepoErrorMetrics:
    D = np.asarray(D, dtype=float)
    F_exp = np.asarray(F_exp, dtype=float)
    F_sim = np.asarray(F_sim, dtype=float)
    if D.shape != F_exp.shape or D.shape != F_sim.shape:
        raise ValueError(
            f"D, F_exp, F_sim must have same shape; got {D.shape}, {F_exp.shape}, {F_sim.shape}"
        )

    j_feat_l2, j_feat_l1 = jfeat_weighted_l2_l1(
        D, F_exp, F_sim, meta, fy=fy, a_sc=a_sc, dy=dy, s_f=s_f, s_d=s_d
    )

    j_e_l2 = energy_mse_cycles(D, F_exp, F_sim, meta, failure_penalty=failure_penalty)
    j_e_l1 = energy_mae_cycles(D, F_exp, F_sim, meta, failure_penalty=failure_penalty)

    binenv_l2, binenv_l1 = compute_unordered_binenv_metrics(
        D, F_exp, D, F_sim, n_binenv_bins=n_binenv_bins
    )

    s_e = energy_scale_s_e(D, F_exp)

    return RepoErrorMetrics(
        j_feat_l2=float(j_feat_l2),
        j_feat_l1=float(j_feat_l1),
        j_e_l2=float(j_e_l2),
        j_e_l1=float(j_e_l1),
        binenv_l2=float(binenv_l2),
        binenv_l1=float(binenv_l1),
        s_d=float(s_d),
        s_f=float(s_f),
        s_e=float(s_e),
    )
