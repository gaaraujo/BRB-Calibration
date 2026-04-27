"""Flat weighted landmark vectors for calibration_data.csv and model results.out."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .jfeat_landmarks import (
    LANDMARK_SLOTS_F_LEVEL_PAIRING,
    N_LANDMARK_SLOTS,
    POST_PEAK_LANDMARK_FY_FACTOR,
    _all_f_level_crossings,
    _first_f_level_crossing,
    deformation_scale_s_d,
    extract_cycle_landmarks,
    landmark_force_threshold,
    pair_sim_cycle_landmarks,
    _slot_error_combined_sq,
)

# Cache schema version. Bumped to 2 when slots 13–14 (post-peak partial unload) were added,
# requiring per-cycle ``j``/``le_metric`` lists of length :data:`N_LANDMARK_SLOTS` and a
# new top-level ``f_thr`` (= ``f_y * A_sc``) used by the sim path.
LANDMARK_CACHE_VERSION = 2


def force_scale_s_f(F_exp: np.ndarray) -> float:
    f = np.asarray(F_exp, dtype=float)
    r = float(np.nanmax(f) - np.nanmin(f))
    if not np.isfinite(r) or r <= 0.0:
        return 1.0
    return r


def load_force_deformation_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    df = pd.read_csv(path)
    if "Deformation[in]" not in df.columns or "Force[kip]" not in df.columns:
        raise ValueError(f"Expected Deformation[in] and Force[kip] in {path}")
    D = df["Deformation[in]"].to_numpy(dtype=float)
    F = df["Force[kip]"].to_numpy(dtype=float)
    return D, F


def load_cycle_meta_json(path: Path) -> tuple[list[dict], float | None, float | None, int | None]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    meta = data["meta"]
    s_f = data.get("s_f")
    s_d = data.get("s_d")
    n = data.get("n")
    return meta, s_f, s_d, int(n) if n is not None else None


def _append_weighted_cycle(
    out: list[float],
    le_m: list[tuple[float, float] | None],
    ls: list[tuple[float, float] | None],
    w_c: float,
    s_f: float,
    s_d: float,
    *,
    from_experiment: bool,
) -> None:
    n_c = sum(
        1
        for slot in range(N_LANDMARK_SLOTS)
        if _slot_error_combined_sq(le_m[slot], ls[slot], s_f, s_d) is not None
    )
    if n_c == 0:
        return

    scale = math.sqrt(w_c / n_c)
    for slot in range(N_LANDMARK_SLOTS):
        if _slot_error_combined_sq(le_m[slot], ls[slot], s_f, s_d) is None:
            continue
        pt = le_m[slot] if from_experiment else ls[slot]
        if pt is None:
            continue
        d_v, f_v = float(pt[0]), float(pt[1])
        if not (np.isfinite(d_v) and np.isfinite(f_v)):
            continue
        out.append(scale * d_v / s_d)
        out.append(scale * f_v / s_f)


def weighted_landmark_vector_experiment(
    D: np.ndarray,
    F_exp: np.ndarray,
    meta: list[dict],
    *,
    fy: float,
    a_sc: float,
    dy: float | None,
    s_d: float | None = None,
    s_f: float | None = None,
) -> list[float]:
    """
    One flat row of weighted landmark (D, F) from the **experimental** curve at each paired grid
    index (same pairing rule as the model path). For ``calibration_data.csv``.
    """
    D = np.asarray(D, dtype=float)
    F_exp = np.asarray(F_exp, dtype=float)
    if s_d is None:
        s_d = deformation_scale_s_d(D)
    if s_f is None:
        s_f = force_scale_s_f(F_exp)

    out: list[float] = []
    for m in meta:
        s, e = int(m["start"]), int(m["end"])
        w_c = float(m.get("w_c", 1.0))
        if not np.isfinite(w_c) or w_c <= 0.0:
            w_c = 1.0
        if e <= s:
            continue

        le = extract_cycle_landmarks(
            D, F_exp, s, e, fy=fy, a_sc=a_sc, dy=dy
        )
        ls, le_m, _ = pair_sim_cycle_landmarks(
            D, F_exp, F_exp, s, e, le, fy=fy, a_sc=a_sc, geometry_f=F_exp
        )
        _append_weighted_cycle(out, le_m, ls, w_c, s_f, s_d, from_experiment=True)

    return out


def sum_w_c_contributing_cycles(
    D: np.ndarray,
    F_exp: np.ndarray,
    meta: list[dict],
    *,
    fy: float,
    a_sc: float,
    dy: float | None,
    s_f: float,
    s_d: float,
) -> float:
    """
    Sum of cycle weights ``w_c`` over cycles that emit at least one characteristic-point pair.

    Matches the denominator in :math:`J_{\\mathrm{feat}}^{(2)}` when using rows built by
    :func:`weighted_landmark_vector_experiment` / :func:`weighted_landmark_vector_model`.
    """
    D = np.asarray(D, dtype=float)
    F_exp = np.asarray(F_exp, dtype=float)
    total = 0.0
    for m in meta:
        s, e = int(m["start"]), int(m["end"])
        w_c = float(m.get("w_c", 1.0))
        if not np.isfinite(w_c) or w_c <= 0.0:
            w_c = 1.0
        if e <= s:
            continue

        le = extract_cycle_landmarks(D, F_exp, s, e, fy=fy, a_sc=a_sc, dy=dy)
        ls, le_m, _ = pair_sim_cycle_landmarks(
            D, F_exp, F_exp, s, e, le, fy=fy, a_sc=a_sc, geometry_f=F_exp
        )
        n_c = sum(
            1
            for slot in range(N_LANDMARK_SLOTS)
            if _slot_error_combined_sq(le_m[slot], ls[slot], s_f, s_d) is not None
        )
        if n_c > 0:
            total += w_c
    return float(total)


def jfeat_l2_squared(
    calibration_row: np.ndarray,
    prediction_row: np.ndarray,
    sum_w_c: float,
) -> float:
    r"""
    Characteristic-feature :math:`L_2` objective :math:`J_{\mathrm{feat}}^{(2)}` from weighted rows.

    For ``calibration_data.csv`` and ``results.out`` produced by this package,
    :math:`J_{\mathrm{feat}}^{(2)} = \|\mathbf{r} - \mathbf{c}\|_2^2 / \sum_c w_c` with the same
    :math:`\sum_c w_c` as :func:`sum_w_c_contributing_cycles`.
    """
    if sum_w_c <= 0.0 or not math.isfinite(sum_w_c):
        raise ValueError(f"sum_w_c must be finite and positive, got {sum_w_c}")
    c = np.asarray(calibration_row, dtype=float).ravel()
    r = np.asarray(prediction_row, dtype=float).ravel()
    if c.shape != r.shape:
        raise ValueError(f"length mismatch: calibration {c.size} vs prediction {r.size}")
    if c.size % 2 != 0:
        raise ValueError("expected even length (interleaved δ and P components per slot)")
    return float(np.sum((r - c) ** 2) / sum_w_c)


def build_landmark_feature_cache(
    D: np.ndarray,
    F_exp: np.ndarray,
    meta: list[dict],
    *,
    fy: float,
    a_sc: float,
    dy: float | None,
    s_d: float | None = None,
    s_f: float | None = None,
) -> dict:
    """
    Precompute per-cycle experimental ``le_metric`` and vertex indices ``j`` for the shared grid.

    Use with :func:`weighted_landmark_vector_model` so forward runs only need ``D`` and ``F_sim``.
    """
    D = np.asarray(D, dtype=float)
    F_exp = np.asarray(F_exp, dtype=float)
    if s_d is None:
        s_d = deformation_scale_s_d(D)
    if s_f is None:
        s_f = force_scale_s_f(F_exp)

    cycles: list[dict] = []
    for m in meta:
        s, e = int(m["start"]), int(m["end"])
        if e <= s:
            continue
        w_c = float(m.get("w_c", 1.0))
        if not np.isfinite(w_c) or w_c <= 0.0:
            w_c = 1.0

        le = extract_cycle_landmarks(
            D, F_exp, s, e, fy=fy, a_sc=a_sc, dy=dy
        )
        _, le_m, j_slot = pair_sim_cycle_landmarks(
            D, F_exp, F_exp, s, e, le, fy=fy, a_sc=a_sc, geometry_f=F_exp
        )

        le_serial: list[list[float] | None] = []
        for i in range(N_LANDMARK_SLOTS):
            p = le_m[i]
            if p is None:
                le_serial.append(None)
            else:
                le_serial.append([float(p[0]), float(p[1])])

        cycles.append(
            {
                "start": s,
                "end": e,
                "w_c": w_c,
                "j": [None if j is None else int(j) for j in j_slot],
                "le_metric": le_serial,
            }
        )

    return {
        "version": LANDMARK_CACHE_VERSION,
        "n": int(len(D)),
        "s_d": float(s_d),
        "s_f": float(s_f),
        "f_thr": float(landmark_force_threshold(fy, a_sc)),
        "cycles": cycles,
    }


def weighted_landmark_vector_model(
    D: np.ndarray,
    F_sim: np.ndarray,
    cache: dict,
) -> list[float]:
    """
    One flat row of weighted landmark (D, F) from **simulation**, using pairing frozen in ``cache``
    (from :func:`build_landmark_feature_cache`). For ``results.out``.
    """
    if int(cache.get("version", 0)) != LANDMARK_CACHE_VERSION:
        raise ValueError(
            f"landmark cache: expected version {LANDMARK_CACHE_VERSION}; "
            f"regenerate via scripts/precompute_landmark_cache.py"
        )

    D = np.asarray(D, dtype=float)
    F_sim = np.asarray(F_sim, dtype=float)
    n = int(cache["n"])
    if len(D) != n or len(F_sim) != n:
        raise ValueError(
            f"landmark cache expects length {n}; got D={len(D)}, F_sim={len(F_sim)}"
        )

    s_d = float(cache["s_d"])
    s_f = float(cache["s_f"])
    f_thr = float(cache["f_thr"])
    fp = float(POST_PEAK_LANDMARK_FY_FACTOR)
    out: list[float] = []

    for cyc in cache["cycles"]:
        s, e = int(cyc["start"]), int(cyc["end"])
        w_c = float(cyc.get("w_c", 1.0))
        if not np.isfinite(w_c) or w_c <= 0.0:
            w_c = 1.0
        if e <= s:
            continue

        j_slots: list[int | None] = cyc["j"]
        le_metric: list[tuple[float, float] | None] = []
        for p in cyc["le_metric"]:
            if p is None:
                le_metric.append(None)
            else:
                le_metric.append((float(p[0]), float(p[1])))
        if len(j_slots) != N_LANDMARK_SLOTS or len(le_metric) != N_LANDMARK_SLOTS:
            raise ValueError(
                f"landmark cache cycle [{s},{e}) expects {N_LANDMARK_SLOTS} slots; "
                f"got j={len(j_slots)}, le_metric={len(le_metric)}"
            )

        ls: list[tuple[float, float] | None] = [None] * N_LANDMARK_SLOTS
        for slot in range(N_LANDMARK_SLOTS):
            if slot in LANDMARK_SLOTS_F_LEVEL_PAIRING:
                continue
            j = j_slots[slot]
            if j is None:
                continue
            dj = float(D[j])
            fs = float(F_sim[j])
            if np.isfinite(dj) and np.isfinite(fs):
                ls[slot] = (dj, fs)

        if le_metric[6] is not None or le_metric[7] is not None:
            f0 = _all_f_level_crossings(D, F_sim, s, e, 0.0)
            if f0:
                if le_metric[6] is not None:
                    ls[6] = max(f0, key=lambda p: p[0])
                if le_metric[7] is not None:
                    ls[7] = min(f0, key=lambda p: p[0])

        # Slots 13–14: post-peak partial-unload crossings on F_sim. Anchors are sim-side
        # peaks computed at runtime (each curve searched after its own extremum).
        if le_metric[12] is not None or le_metric[13] is not None:
            fseg_sim = F_sim[s:e]
            if len(fseg_sim) > 0:
                i_max_sim = s + int(np.argmax(fseg_sim))
                i_min_sim = s + int(np.argmin(fseg_sim))
                if le_metric[12] is not None:
                    hit = _first_f_level_crossing(D, F_sim, e, i_max_sim, -fp * f_thr)
                    if hit is not None:
                        ls[12] = hit[0]
                if le_metric[13] is not None:
                    hit = _first_f_level_crossing(D, F_sim, e, i_min_sim, +fp * f_thr)
                    if hit is not None:
                        ls[13] = hit[0]

        _append_weighted_cycle(out, le_metric, ls, w_c, s_f, s_d, from_experiment=False)

    return out


def write_landmark_cache_json(path: Path, cache: dict) -> None:
    path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def read_landmark_cache_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_csv_single_row(path: Path, values: list[float]) -> None:
    path.write_text(",".join(str(v) for v in values), encoding="utf-8")
