"""
Cycle-weighted landmark loss J_feat for prescribed (D, F) hysteresis.

Twelve landmarks per weight cycle ``[s, e)`` on the experimental polyline: tension/compression
first yield (|F| > f_y A_sc), global max/min force, F at D=0 on each yield-to-peak subpath,
two mid-D points per side, then D at F=0 unloading after each peak. Simulation pairs at the
same experimental D for slots 1–10 (force error / S_F); slots 11–12 compare unload D only / S_D.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Apparent plastic onset elsewhere (e.g. extract_bn_bp): |F| >= ratio * f_y * A_sc.
# Landmarks use strict f_y * A_sc for yield picks (no 1.1 factor).
PLASTIC_STRESS_RATIO_FY = 1.1


def deformation_scale_s_d(D: np.ndarray) -> float:
    """S_D = max(D) - min(D); fallback 1.0."""
    d = np.asarray(D, dtype=float)
    r = float(np.nanmax(d) - np.nanmin(d))
    if not np.isfinite(r) or r <= 0.0:
        return 1.0
    return r


def landmark_force_threshold_kip(fy_ksi: float, a_sc: float) -> float:
    """Yield force threshold [kip] = f_y * A_sc (landmark gating, no 1.1 factor)."""
    return float(fy_ksi) * float(a_sc)


def _interp_edge(
    da: float, db: float, fa: float, fb: float, target_f: float
) -> tuple[float, float] | None:
    """If target_f is bracketed on the edge, return (d_star, f_star) with f_star=target_f."""
    if not (math.isfinite(da) and math.isfinite(db) and math.isfinite(fa) and math.isfinite(fb)):
        return None
    lo_f, hi_f = (fa, fb) if fa <= fb else (fb, fa)
    if target_f < lo_f or target_f > hi_f:
        return None
    if fb == fa:
        return None
    frac = (target_f - fa) / (fb - fa)
    d_star = da + frac * (db - da)
    return (float(d_star), float(target_f))


def _first_f_level_crossing(
    D: np.ndarray,
    F: np.ndarray,
    e: int,
    k_min: int,
    level: float,
) -> tuple[tuple[float, float], int] | None:
    """First edge [k,k+1) with k >= k_min, k+1 < e, bracketing F=level. Returns ((d,f), k_gate)."""
    k_min = max(0, k_min)
    for k in range(k_min, e - 1):
        da, db = float(D[k]), float(D[k + 1])
        fa, fb = float(F[k]), float(F[k + 1])
        if fa == level:
            return ((da, level), k + 1)
        if fb == level:
            return ((db, level), k + 2)
        if fa * fb < 0.0 and level == 0.0:
            if fb == fa:
                continue
            frac = -fa / (fb - fa)
            d_star = da + frac * (db - da)
            return ((float(d_star), 0.0), k + 1)
        hit = _interp_edge(da, db, fa, fb, level)
        if hit is not None:
            return (hit, k + 1)
    return None


def _d_zero_cross_subpath(
    D: np.ndarray,
    F: np.ndarray,
    e: int,
    i_lo: int,
    i_hi_vert: int,
) -> tuple[float, float] | None:
    """
    First D=0 crossing on edges between vertices ``i_lo .. i_hi_vert`` inclusive.
    Returns ``(0.0, f_star)`` interpolated on the crossing edge.
    """
    if i_hi_vert <= i_lo or i_lo >= e:
        return None
    i_hi_vert = min(i_hi_vert, e - 1)
    for k in range(i_lo, i_hi_vert):
        if k + 1 >= e:
            break
        da, db = float(D[k]), float(D[k + 1])
        fa, fb = float(F[k]), float(F[k + 1])
        if not all(math.isfinite(x) for x in (da, db, fa, fb)):
            continue
        if da == 0.0:
            return (0.0, float(fa))
        if db == 0.0:
            return (0.0, float(fb))
        if da * db < 0.0:
            if db == da:
                continue
            frac = -da / (db - da)
            f_star = fa + frac * (fb - fa)
            return (0.0, float(f_star))
    return None


def _interp_f_at_deformation_on_path(
    D: np.ndarray,
    F: np.ndarray,
    s: int,
    e_path: int,
    d_star: float,
    *,
    hint_idx: int,
) -> float | None:
    """
    Linearly interpolate ``F`` at deformation ``d_star`` along the polyline
    ``(D[j],F[j]) → (D[j+1],F[j+1])`` for ``j ∈ [s, e_path-2]``. If ``d_star`` lies on
    several edges (non-monotonic ``D``), pick the edge whose ``j`` is closest to ``hint_idx``.
    """
    if not np.isfinite(d_star) or e_path <= s + 1:
        return None
    best_j: int | None = None
    best_dist = 10**18
    for j in range(s, e_path - 1):
        da, db = float(D[j]), float(D[j + 1])
        fa, fb = float(F[j]), float(F[j + 1])
        if not all(np.isfinite(x) for x in (da, db, fa, fb)):
            continue
        lo, hi = (da, db) if da <= db else (db, da)
        if lo - 1e-15 <= d_star <= hi + 1e-15:
            dist = abs(j - int(hint_idx))
            if dist < best_dist:
                best_dist = dist
                best_j = j
    if best_j is None:
        ih = int(np.clip(int(hint_idx), s, e_path - 1))
        d_h = float(D[ih])
        if np.isfinite(d_h) and abs(d_h - d_star) <= 1e-9 * (abs(d_star) + 1.0):
            fh = float(F[ih])
            return fh if np.isfinite(fh) else None
        return None
    j = best_j
    da, db = float(D[j]), float(D[j + 1])
    fa, fb = float(F[j]), float(F[j + 1])
    if da == db:
        return float(fa) if abs(d_star - da) <= 1e-15 * (abs(da) + 1.0) else None
    t = (d_star - da) / (db - da)
    return float(fa + t * (fb - fa))


N_LANDMARK_SLOTS = 12
# Slots 1–10 (indices 0..9): compare force at experimental D. Slots 11–12 (10,11): D at F=0 unload.
LANDMARK_SLOTS_FORCE_METRIC = frozenset(range(10))
LANDMARK_SLOTS_DISP_METRIC = frozenset({10, 11})


def extract_cycle_landmarks(
    D: np.ndarray,
    F: np.ndarray,
    s: int,
    e: int,
    *,
    fy_ksi: float,
    a_sc: float,
) -> list[tuple[float, float] | None]:
    """
    Twelve experimental landmarks on ``[s, e)``.

    Order (1-based labels in docs): (1) tension yield, (2) compression yield, (3) max F,
    (4) min F, (5–6) F at D=0 on tension / compression yield→peak subpaths, (7–8) mid-D
    tension pair, (9–10) mid-D compression pair, (11–12) D at F=0 after max tension / min
    compression.

    Yield: first index with ``F > F_thr`` in ``[s, i_max]`` (tension) and first with
    ``F < -F_thr`` in ``[s, i_min]`` (compression), ``F_thr = f_y * A_sc`` [kip].
    """
    out: list[tuple[float, float] | None] = [None] * N_LANDMARK_SLOTS
    if e <= s:
        return out

    D = np.asarray(D, dtype=float)
    F = np.asarray(F, dtype=float)
    F_thr = landmark_force_threshold_kip(fy_ksi, a_sc)
    if not (np.isfinite(F_thr) and F_thr > 0.0):
        return out

    fseg = F[s:e]
    if len(fseg) == 0:
        return out
    rel_max = int(np.argmax(fseg))
    rel_min = int(np.argmin(fseg))
    i_max = s + rel_max
    i_min = s + rel_min

    i_yt: int | None = None
    for i in range(s, i_max + 1):
        if F[i] > F_thr:
            i_yt = i
            break

    i_yc: int | None = None
    for i in range(s, i_min + 1):
        if F[i] < -F_thr:
            i_yc = i
            break

    out[2] = (float(D[i_max]), float(F[i_max]))
    out[3] = (float(D[i_min]), float(F[i_min]))

    if i_yt is not None:
        out[0] = (float(D[i_yt]), float(F[i_yt]))
        e_t = i_max + 1
        hint_tm = (i_yt + i_max) // 2
        zt = _d_zero_cross_subpath(D, F, e, i_yt, i_max)
        if zt is not None:
            out[4] = zt
        d_yt = float(D[i_yt])
        d_mt = float(D[i_max])
        d_mid_a = 0.5 * (d_yt + 0.0)
        d_mid_b = 0.5 * (0.0 + d_mt)
        fa = _interp_f_at_deformation_on_path(D, F, i_yt, e_t, d_mid_a, hint_idx=i_yt)
        if fa is not None and np.isfinite(fa):
            out[6] = (d_mid_a, float(fa))
        fb = _interp_f_at_deformation_on_path(D, F, i_yt, e_t, d_mid_b, hint_idx=hint_tm)
        if fb is not None and np.isfinite(fb):
            out[7] = (d_mid_b, float(fb))
        if i_max < e - 1:
            hu = _first_f_level_crossing(D, F, e, i_max, 0.0)
            if hu is not None:
                out[10] = (float(hu[0][0]), 0.0)

    if i_yc is not None:
        out[1] = (float(D[i_yc]), float(F[i_yc]))
        e_c = i_min + 1
        hint_cm = (i_yc + i_min) // 2
        zc = _d_zero_cross_subpath(D, F, e, i_yc, i_min)
        if zc is not None:
            out[5] = zc
        d_yc = float(D[i_yc])
        d_mc = float(D[i_min])
        d_mid_ca = 0.5 * (d_yc + 0.0)
        d_mid_cb = 0.5 * (0.0 + d_mc)
        fca = _interp_f_at_deformation_on_path(D, F, i_yc, e_c, d_mid_ca, hint_idx=i_yc)
        if fca is not None and np.isfinite(fca):
            out[8] = (d_mid_ca, float(fca))
        fcb = _interp_f_at_deformation_on_path(D, F, i_yc, e_c, d_mid_cb, hint_idx=hint_cm)
        if fcb is not None and np.isfinite(fcb):
            out[9] = (d_mid_cb, float(fcb))
        if i_min < e - 1:
            hc = _first_f_level_crossing(D, F, e, i_min, 0.0)
            if hc is not None:
                out[11] = (float(hc[0][0]), 0.0)

    return out


def cycle_landmarks_experiment_cached(
    D: np.ndarray,
    F_exp: np.ndarray,
    s: int,
    e: int,
    fy_ksi: float,
    a_sc: float,
    cache: dict | None,
) -> list[tuple[float, float] | None]:
    """Cached ``extract_cycle_landmarks`` for experiment; keys use ``id(D)``, ``id(F_exp)``."""
    if cache is None:
        return extract_cycle_landmarks(
            D, F_exp, s, e, fy_ksi=fy_ksi, a_sc=a_sc
        )
    key = (id(D), id(F_exp), int(s), int(e), float(fy_ksi), float(a_sc))
    hit = cache.get(key)
    if hit is not None:
        return hit
    le = extract_cycle_landmarks(D, F_exp, s, e, fy_ksi=fy_ksi, a_sc=a_sc)
    cache[key] = le
    return le


def pair_sim_cycle_landmarks(
    D: np.ndarray,
    F_sim: np.ndarray,
    s: int,
    e: int,
    le: list[tuple[float, float] | None],
    *,
    fy_ksi: float,
    a_sc: float,
) -> list[tuple[float, float] | None]:
    """
    Build simulated landmarks paired to ``le``: slots 0–9 use experimental D and ``F_sim``
    interpolated on ``(D, F_sim)``; slots 10–11 use unload D from ``F_sim`` after peak indices
    (same geometry as experiment).
    """
    ls: list[tuple[float, float] | None] = [None] * N_LANDMARK_SLOTS
    if e <= s:
        return ls

    D = np.asarray(D, dtype=float)
    F_sim = np.asarray(F_sim, dtype=float)
    F_thr = landmark_force_threshold_kip(fy_ksi, a_sc)

    fseg = F_sim[s:e]
    if len(fseg) == 0:
        return ls
    i_max = s + int(np.argmax(fseg))
    i_min = s + int(np.argmin(fseg))

    i_yt: int | None = None
    for i in range(s, i_max + 1):
        if F_sim[i] > F_thr:
            i_yt = i
            break
    i_yc: int | None = None
    for i in range(s, i_min + 1):
        if F_sim[i] < -F_thr:
            i_yc = i
            break

    for slot in range(10):
        if le[slot] is None:
            continue
        d_e, _ = le[slot]
        if not np.isfinite(d_e):
            continue
        hint = s
        if slot == 0 and i_yt is not None:
            hint = i_yt
        elif slot == 1 and i_yc is not None:
            hint = i_yc
        elif slot == 2:
            hint = i_max
        elif slot == 3:
            hint = i_min
        elif slot in (4, 6, 7) and i_yt is not None:
            hint = (i_yt + i_max) // 2
        elif slot in (5, 8, 9) and i_yc is not None:
            hint = (i_yc + i_min) // 2
        f_s = _interp_f_at_deformation_on_path(D, F_sim, s, e, float(d_e), hint_idx=hint)
        if f_s is not None and np.isfinite(f_s):
            ls[slot] = (float(d_e), float(f_s))

    if le[10] is not None and i_max < e - 1:
        hu = _first_f_level_crossing(D, F_sim, e, i_max, 0.0)
        if hu is not None:
            ls[10] = (float(hu[0][0]), 0.0)
    if le[11] is not None and i_min < e - 1:
        hc = _first_f_level_crossing(D, F_sim, e, i_min, 0.0)
        if hc is not None:
            ls[11] = (float(hc[0][0]), 0.0)

    return ls


def plastic_mask_full_cycle(
    F_exp: np.ndarray,
    s: int,
    e: int,
    fy_A: float,
    *,
    ratio: float = PLASTIC_STRESS_RATIO_FY,
) -> np.ndarray:
    """Full-length mask: true on ``[s,e)`` where ``|F_exp| >= ratio * f_y A_sc``."""
    n = len(F_exp)
    out = np.zeros(n, dtype=bool)
    if e <= s or not (np.isfinite(fy_A) and fy_A > 0.0):
        return out
    thr = float(ratio) * float(fy_A)
    f = np.asarray(F_exp[s:e], dtype=float)
    m = np.isfinite(f) & (np.abs(f) >= thr)
    for ti, gi in enumerate(range(s, e)):
        if ti < len(m):
            out[gi] = bool(m[ti])
    return out


def _slot_error_force_sq(
    p_exp: tuple[float, float] | None,
    p_sim: tuple[float, float] | None,
    s_f: float,
) -> float | None:
    if p_exp is None or p_sim is None:
        return None
    _, fe = p_exp
    _, fs = p_sim
    inv_f = 1.0 / s_f if s_f > 0.0 and math.isfinite(s_f) else 1.0
    if not (np.isfinite(fe) and np.isfinite(fs)):
        return None
    return (fs - fe) ** 2 * inv_f**2


def _slot_error_force_l1(
    p_exp: tuple[float, float] | None,
    p_sim: tuple[float, float] | None,
    s_f: float,
) -> float | None:
    if p_exp is None or p_sim is None:
        return None
    _, fe = p_exp
    _, fs = p_sim
    inv_f = 1.0 / s_f if s_f > 0.0 and math.isfinite(s_f) else 1.0
    if not (np.isfinite(fe) and np.isfinite(fs)):
        return None
    return abs(fs - fe) * inv_f


def _slot_error_disp_sq(
    p_exp: tuple[float, float] | None,
    p_sim: tuple[float, float] | None,
    s_d: float,
) -> float | None:
    if p_exp is None or p_sim is None:
        return None
    de, _ = p_exp
    ds, _ = p_sim
    inv_d = 1.0 / s_d if s_d > 0.0 and math.isfinite(s_d) else 1.0
    if not (np.isfinite(de) and np.isfinite(ds)):
        return None
    return (ds - de) ** 2 * inv_d**2


def _slot_error_disp_l1(
    p_exp: tuple[float, float] | None,
    p_sim: tuple[float, float] | None,
    s_d: float,
) -> float | None:
    if p_exp is None or p_sim is None:
        return None
    de, _ = p_exp
    ds, _ = p_sim
    inv_d = 1.0 / s_d if s_d > 0.0 and math.isfinite(s_d) else 1.0
    if not (np.isfinite(de) and np.isfinite(ds)):
        return None
    return abs(ds - de) * inv_d


def feature_mse_cycles(
    D: np.ndarray,
    F_exp: np.ndarray,
    F_sim: np.ndarray,
    meta: list[dict],
    *,
    s_d: float,
    s_f: float,
    fy_ksi: float,
    A_sc: float,
    exp_landmark_cache: dict | None = None,
) -> float:
    """
    Cycle-weighted mean of per-cycle mean squared landmark error: force slots use
    ``((F_sim - F_exp)/S_F)^2`` at experimental D; displacement slots use ``((D_sim - D_exp)/S_D)^2``.
    """
    D = np.asarray(D, dtype=float)
    F_exp = np.asarray(F_exp, dtype=float)
    F_sim = np.asarray(F_sim, dtype=float)
    if F_exp.shape != F_sim.shape or D.shape != F_exp.shape:
        return 0.0
    if not meta:
        return 0.0

    numer = 0.0
    denom = 0.0
    for m in meta:
        s, e = int(m["start"]), int(m["end"])
        if e <= s:
            continue
        w_c = float(m.get("w_c", 1.0))
        if not np.isfinite(w_c) or w_c <= 0.0:
            w_c = 1.0

        le = cycle_landmarks_experiment_cached(
            D, F_exp, s, e, fy_ksi, A_sc, exp_landmark_cache
        )
        ls = pair_sim_cycle_landmarks(
            D, F_sim, s, e, le, fy_ksi=fy_ksi, a_sc=A_sc
        )

        errs: list[float] = []
        for i in range(N_LANDMARK_SLOTS):
            if i in LANDMARK_SLOTS_FORCE_METRIC:
                ee = _slot_error_force_sq(le[i], ls[i], s_f)
            else:
                ee = _slot_error_disp_sq(le[i], ls[i], s_d)
            if ee is not None:
                errs.append(ee)

        m_c = len(errs)
        if m_c == 0:
            continue
        bar_e = float(sum(errs) / m_c)
        numer += w_c * bar_e
        denom += w_c

    if denom <= 0.0 or not np.isfinite(denom):
        warnings.warn(
            "feature_mse_cycles: no contributing cycles; returning 0.0",
            UserWarning,
            stacklevel=2,
        )
        return 0.0
    return float(numer / denom)


def feature_mae_cycles(
    D: np.ndarray,
    F_exp: np.ndarray,
    F_sim: np.ndarray,
    meta: list[dict],
    *,
    s_d: float,
    s_f: float,
    fy_ksi: float,
    A_sc: float,
    exp_landmark_cache: dict | None = None,
) -> float:
    """Same weighting as ``feature_mse_cycles``; per-slot L1: |ΔF|/S_F or |ΔD|/S_D."""
    D = np.asarray(D, dtype=float)
    F_exp = np.asarray(F_exp, dtype=float)
    F_sim = np.asarray(F_sim, dtype=float)
    if F_exp.shape != F_sim.shape or D.shape != F_exp.shape:
        return 0.0
    if not meta:
        return 0.0

    numer = 0.0
    denom = 0.0
    for m in meta:
        s, e = int(m["start"]), int(m["end"])
        if e <= s:
            continue
        w_c = float(m.get("w_c", 1.0))
        if not np.isfinite(w_c) or w_c <= 0.0:
            w_c = 1.0

        le = cycle_landmarks_experiment_cached(
            D, F_exp, s, e, fy_ksi, A_sc, exp_landmark_cache
        )
        ls = pair_sim_cycle_landmarks(
            D, F_sim, s, e, le, fy_ksi=fy_ksi, a_sc=A_sc
        )

        errs: list[float] = []
        for i in range(N_LANDMARK_SLOTS):
            if i in LANDMARK_SLOTS_FORCE_METRIC:
                ee = _slot_error_force_l1(le[i], ls[i], s_f)
            else:
                ee = _slot_error_disp_l1(le[i], ls[i], s_d)
            if ee is not None:
                errs.append(ee)

        m_c = len(errs)
        if m_c == 0:
            continue
        bar_e = float(sum(errs) / m_c)
        numer += w_c * bar_e
        denom += w_c

    if denom <= 0.0 or not np.isfinite(denom):
        warnings.warn(
            "feature_mae_cycles: no contributing cycles; returning 0.0",
            UserWarning,
            stacklevel=2,
        )
        return 0.0
    return float(numer / denom)


def load_p_y_kip_catalog(project_root: Path, name: str, fallback_fyp_ksi: float, a_sc: float) -> float:
    """Nominal yield force P_y [kip] = f_yc_ksi * A_c_in2 from BRB-Specimens.csv if possible, else fyp * A_sc."""
    cat = Path(project_root) / "config" / "calibration" / "BRB-Specimens.csv"
    if cat.is_file():
        try:
            df = pd.read_csv(cat)
            row = df[df["Name"].astype(str) == str(name)]
            if not row.empty:
                fyc = float(row.iloc[0]["f_yc_ksi"])
                ac = float(row.iloc[0]["A_c_in2"])
                fy = fyc * ac
                if np.isfinite(fy) and fy > 0.0:
                    return fy
        except (KeyError, ValueError, TypeError):
            pass
    return float(fallback_fyp_ksi * a_sc)


LANDMARK_EXP_CSV_COLUMNS: list[str] = (
    ["Name", "set_id", "cycle_id", "start", "end", "kind", "incomplete", "F_thr_kip"]
    + [c for k in range(1, N_LANDMARK_SLOTS + 1) for c in (f"d_{k}", f"f_{k}")]
)


def landmark_exp_row_dict(
    specimen_id: str,
    set_id: int | str,
    cycle_id: int,
    meta_row: dict,
    le: list[tuple[float, float] | None],
    *,
    fy_ksi: float,
    a_sc: float,
) -> dict:
    """One CSV row dict for experimental landmarks (NaN for missing slots)."""
    F_thr = landmark_force_threshold_kip(fy_ksi, a_sc)
    row: dict = {
        "Name": specimen_id,
        "set_id": set_id,
        "cycle_id": cycle_id,
        "start": meta_row.get("start"),
        "end": meta_row.get("end"),
        "kind": meta_row.get("kind"),
        "incomplete": meta_row.get("incomplete"),
        "F_thr_kip": F_thr,
    }
    for k in range(N_LANDMARK_SLOTS):
        p = le[k]
        if p is None:
            row[f"d_{k + 1}"] = float("nan")
            row[f"f_{k + 1}"] = float("nan")
        else:
            d, f = p
            row[f"d_{k + 1}"] = d
            row[f"f_{k + 1}"] = f
    return row
