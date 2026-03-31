"""
Cycle-based partitions (zero_def anchors) and amplitude weights for force MSE.

Weight **cycles** group two consecutive zero-to-zero index spans: [z_{2k}, z_{2k+2})
when possible; incomplete head/tail are separate cycles. Imported by optimize_brb_mse.py;
tests import this module only to avoid corotruss/opensees.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def _integrate_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal rule int y dx; NumPy 2.0+ uses ``trapezoid``, older uses ``trapz``."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))  # type: ignore[attr-defined]


def w_from_amplitude(
    A_c: float,
    A_max: float,
    *,
    p: float,
    eps: float,
) -> float:
    """Edit amplitude weighting here: maps cycle amplitude A_c to cycle weight w_c given global A_max."""
    if not np.isfinite(A_max) or A_max <= 0.0:
        A_max = 1.0
    if not np.isfinite(A_c) or A_c < 0.0:
        A_c = 0.0
    return float((A_c / A_max) ** p + eps)


def zero_def_indices_from_points(points: list[dict], n: int) -> list[int]:
    """Sorted unique sample indices where type contains 'zero_def', clamped to [0, n)."""
    z: list[int] = []
    for p in points:
        idx = p.get("idx")
        if idx is None:
            continue
        ptype = str(p.get("type", ""))
        if "zero_def" in ptype:
            i = int(idx)
            if 0 <= i < n:
                z.append(i)
    return sorted(set(z))


def verify_partition_coverage(ranges: list[dict], n: int) -> bool:
    """True if ranges disjointly cover every index 0..n-1 exactly once."""
    if n <= 0:
        return True
    cov = np.zeros(n, dtype=np.int32)
    for r in ranges:
        s, e = int(r["start"]), int(r["end"])
        if s < 0 or e > n or s >= e:
            return False
        cov[s:e] += 1
    return bool(np.all(cov == 1))


def build_cycle_weight_ranges(
    n: int,
    points: list[dict],
    *,
    debug_partition: bool = False,
) -> list[dict]:
    """
    Partition 0..n-1 into disjoint **weight cycles** from sorted zero_def indices z_0..z_{K-1}.

    - **Full cycle** when z[i], z[i+1], z[i+2] exist: half-open [z[i], z[i+2]); advance i by 2.
    - **Incomplete head** [0, z_0) if z_0 > 0.
    - **Incomplete tail** [z[i], n) for remaining anchor index z[i] when no further pair closes.
    - If no zero_def: single incomplete [0, n).
    """
    if n <= 0:
        return []

    z = zero_def_indices_from_points(points, n)
    ranges: list[dict] = []

    if not z:
        ranges.append(
            {"start": 0, "end": n, "kind": "incomplete", "incomplete": True}
        )
    else:
        if z[0] > 0:
            ranges.append(
                {
                    "start": 0,
                    "end": z[0],
                    "kind": "incomplete_head",
                    "incomplete": True,
                }
            )
        i = 0
        while i + 2 < len(z):
            ranges.append(
                {
                    "start": z[i],
                    "end": z[i + 2],
                    "kind": "full_cycle",
                    "incomplete": False,
                }
            )
            i += 2
        if i < len(z):
            ranges.append(
                {
                    "start": z[i],
                    "end": n,
                    "kind": "incomplete_tail",
                    "incomplete": True,
                }
            )

    ranges = [r for r in ranges if r["end"] > r["start"]]

    if not ranges:
        ranges.append(
            {"start": 0, "end": n, "kind": "incomplete", "incomplete": True}
        )

    if debug_partition:
        assert verify_partition_coverage(ranges, n), (ranges, n)

    return ranges


def build_amplitude_weights(
    deformation: np.ndarray,
    points: list[dict],
    *,
    p: float = 2.0,
    eps: float = 0.05,
    debug_partition: bool = False,
    use_amplitude_weights: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    """
    Pointwise weights from per-cycle |u| amplitude over ``build_cycle_weight_ranges``;
    meta lists cycles with amp, w_c, kind, incomplete.
    If ``use_amplitude_weights`` is False (default), w_c = 1 for every cycle (amp still filled for diagnostics).
    """
    u = np.asarray(deformation, dtype=float)
    n = len(u)
    if n == 0:
        return np.array([], dtype=float), []

    ranges = build_cycle_weight_ranges(n, points, debug_partition=debug_partition)

    if debug_partition and not verify_partition_coverage(ranges, n):
        raise AssertionError(f"partition failed coverage check: {ranges!r} n={n}")

    amps: list[float] = []
    for r in ranges:
        s, e = int(r["start"]), int(r["end"])
        if e <= s:
            amps.append(0.0)
        else:
            amps.append(float(np.max(np.abs(u[s:e]))))

    A_max = max(amps) if amps else 0.0
    if A_max <= 0.0 or not np.isfinite(A_max):
        A_max = 1.0

    weights = np.ones(n, dtype=float)
    meta: list[dict] = []

    for r, A_c in zip(ranges, amps):
        s, e = int(r["start"]), int(r["end"])
        if e <= s:
            continue
        if use_amplitude_weights:
            w_c = w_from_amplitude(A_c, A_max, p=p, eps=eps)
        else:
            w_c = 1.0
        weights[s:e] = w_c
        rr = dict(r)
        rr["amp"] = A_c
        rr["w_c"] = w_c
        meta.append(rr)

    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        warnings.warn(
            "build_amplitude_weights: non-finite or non-positive weights; using ones.",
            UserWarning,
            stacklevel=2,
        )
        weights = np.ones(n, dtype=float)

    return weights, meta


def cycle_signed_trapz_work(D: np.ndarray, F: np.ndarray, start: int, end: int) -> float:
    """int F du along half-open index range [start, end) via trapezoidal rule; E_c = |this|."""
    if end <= start:
        return 0.0
    d = np.asarray(D[start:end], dtype=float)
    f = np.asarray(F[start:end], dtype=float)
    if len(d) < 2:
        return 0.0
    return _integrate_trapezoid(f, d)


def cycle_abs_trapz_work(D: np.ndarray, F: np.ndarray, start: int, end: int) -> float:
    """|int F du| along half-open index range [start, end) via trapezoidal rule (same D for exp/sim)."""
    return abs(cycle_signed_trapz_work(D, F, start, end))


def energy_scale_s_e(D_exp: np.ndarray, F_exp: np.ndarray) -> float:
    """S_E = (F_max - F_min) * (u_max - u_min) on experiment; fallback 1.0."""
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
    failure_penalty: float,
) -> float:
    """
    J_E = mean_c (E_c^sim - E_c^exp)^2 / S_E^2 with E_c = |int F du| per cycle (trapezoidal).

    Each weight cycle counts equally; amplitude weights w_c apply only to J_feat.
    """
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
        e_exp = cycle_abs_trapz_work(D_exp, F_exp, s, e)
        e_sim = cycle_abs_trapz_work(D_exp, F_sim, s, e)
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
    failure_penalty: float,
) -> float:
    """
    L1 analogue of ``energy_mse_cycles``: mean per cycle of |E_c^sim - E_c^exp| / S_E
    with E_c = |∫ F du| (trapezoidal).
    """
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
        e_exp = cycle_abs_trapz_work(D_exp, F_exp, s, e)
        e_sim = cycle_abs_trapz_work(D_exp, F_sim, s, e)
        numer += abs(e_sim - e_exp)
        denom += 1.0

    if not np.isfinite(denom) or denom <= 0.0:
        return failure_penalty
    return float((numer / denom) / s_e)


def meta_to_dataframe(meta: list[dict]) -> pd.DataFrame:
    """Flatten cycle meta for CSV export (cycle_id = row index)."""
    rows: list[dict] = []
    for i, m in enumerate(meta):
        rows.append(
            {
                "cycle_id": i,
                "start": m.get("start"),
                "end": m.get("end"),
                "kind": m.get("kind"),
                "incomplete": m.get("incomplete"),
                "amp": m.get("amp"),
                "w_c": m.get("w_c"),
            }
        )
    return pd.DataFrame(rows)
