"""
Identify characteristic points in force-deformation cycles for segment-based filtering.
Points: zero deformation, zero force, max deformation, min deformation.

Writes ``data/cycle_points_original/{Name}.json`` with ``n`` = trim-valid **raw** length (same row
count as ``data/filtered/{Name}/force_deformation.csv`` when filtering does not drop rows). ``filter_force.py`` and
``resample_filtered.py`` read from that folder only.

``resample_filtered.py`` writes remapped landmarks to ``data/cycle_points_resampled/{Name}.json``
(``n`` = resampled length; indices match ``data/resampled/``). Calibration and resampled plots load
from there via ``load_cycle_points_resampled``. There is no ``data/cycle_points/`` folder.

Segments are derived from points and ``n`` on load. You may edit the points array manually.
"""
from pathlib import Path

import json
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from load_raw import load_raw_valid
from specimen_catalog import list_names_for_cycle_points

FORCE_COL = "Force[kip]"
DEF_COL = "Deformation[in]"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
# Filtered-length grid (indices match trim-valid raw / filtered CSV row count).
CYCLE_POINTS_ORIGINAL_DIR = PROJECT_ROOT / "data" / "cycle_points_original"
# Post-resample grid (indices match ``data/resampled/{Name}/force_deformation.csv``).
CYCLE_POINTS_RESAMPLED_DIR = PROJECT_ROOT / "data" / "cycle_points_resampled"

# Order for local extrema (larger = smoother, fewer peaks)
EXTREMA_ORDER = 50
# Merge points within this many indices (same "event")
MERGE_RADIUS = 3
# Merge points at same physical location (same F, u within relative tolerance)
SAME_LOCATION_TOL = 1e-6  # relative tolerance for force and deformation to treat as same point


def _zero_crossings(series: pd.Series) -> list[int]:
    """Return indices of zero-crossings (index of point with smaller |value| at each crossing)."""
    v = series.values
    n = len(v)
    out = []
    for i in range(1, n):
        if v[i - 1] * v[i] <= 0 and (v[i - 1] != 0 or v[i] != 0):
            # Linear interpolation for approximate zero index, then round
            if v[i] == v[i - 1]:
                idx = i - 1
            else:
                frac = -v[i - 1] / (v[i] - v[i - 1])
                idx = int(round(i - 1 + frac))
            idx = max(0, min(idx, n - 1))
            out.append(idx)
    return out


def _local_extrema(series: pd.Series, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices of local max, indices of local min)."""
    v = series.values
    maxima = argrelextrema(v, np.greater_equal, order=order)[0]
    minima = argrelextrema(v, np.less_equal, order=order)[0]
    return maxima, minima


def _merge_nearby(points: list[tuple[int, str]], radius: int) -> list[tuple[int, str]]:
    """Merge points that are within radius indices; keep first index and combine types."""
    if not points:
        return []
    points = sorted(points, key=lambda p: p[0])
    merged = []
    cur_idx, cur_types = points[0][0], {points[0][1]}
    for idx, t in points[1:]:
        if idx <= cur_idx + radius:
            cur_types.add(t)
        else:
            merged.append((cur_idx, "+".join(sorted(cur_types))))
            cur_idx, cur_types = idx, {t}
    merged.append((cur_idx, "+".join(sorted(cur_types))))
    return merged


def _primary_type(t: str) -> str:
    """Single type for ordering: canonical cycle is d=0, dmax, F=0, d=0, dmin, F=0, d=0, ..."""
    if "max_def" in t:
        return "max_def"
    if "min_def" in t:
        return "min_def"
    if "zero_force" in t:
        return "zero_force"
    if "zero_def" in t:
        return "zero_def"
    return "other"


def _collapse_consecutive_same_type(
    points: list[tuple[int, str]], df: pd.DataFrame
) -> list[tuple[int, str]]:
    """
    When two or more consecutive points (by index) have the same primary type,
    keep only the 'true' one: max_def -> largest deformation; min_def -> smallest;
    zero_force -> smallest |F|; zero_def -> smallest |deformation|.
    """
    if not points:
        return []
    points = sorted(points, key=lambda p: p[0])
    out = []
    i = 0
    force = df[FORCE_COL]
    deformation = df[DEF_COL]
    while i < len(points):
        run = [points[i]]
        pt = _primary_type(points[i][1])
        while i + 1 < len(points) and _primary_type(points[i + 1][1]) == pt:
            run.append(points[i + 1])
            i += 1
        if pt == "max_def":
            best = max(run, key=lambda p: deformation.iloc[p[0]])
        elif pt == "min_def":
            best = min(run, key=lambda p: deformation.iloc[p[0]])
        elif pt == "zero_force":
            best = min(run, key=lambda p: abs(force.iloc[p[0]]))
        else:
            best = min(run, key=lambda p: abs(deformation.iloc[p[0]]))
        out.append(best)
        i += 1
    return out


def _collapse_same_location(
    points: list[tuple[int, str]], df: pd.DataFrame, rel_tol: float = 1e-9
) -> list[tuple[int, str]]:
    """
    Merge points that are at the same physical (force, deformation) location.
    Keeps the smaller index and combines types. Handles duplicate detections
    (e.g. same extreme reported as both max_def and min_def near end of signal).
    """
    if not points:
        return []
    force = df["Force[kip]"]
    deformation = df["Deformation[in]"]
    points = sorted(points, key=lambda p: p[0])
    out: list[tuple[int, str]] = []
    for idx, t in points:
        f, u = float(force.iloc[idx]), float(deformation.iloc[idx])
        merged = False
        for i, (out_idx, out_t) in enumerate(out):
            f0 = float(force.iloc[out_idx])
            u0 = float(deformation.iloc[out_idx])
            ref_f = max(abs(f), abs(f0), 1e-12)
            ref_u = max(abs(u), abs(u0), 1e-12)
            if abs(f - f0) <= rel_tol * ref_f and abs(u - u0) <= rel_tol * ref_u:
                types = set(out_t.split("+")) | set(t.split("+"))
                out[i] = (out_idx, "+".join(sorted(types)))
                merged = True
                break
        if not merged:
            out.append((idx, t))
    return sorted(out, key=lambda p: p[0])


def synthetic_fu_from_deformation(u: np.ndarray | pd.Series) -> pd.DataFrame:
    """
    Build a minimal (F, u) frame for ``find_cycle_points`` when only a prescribed displacement
    series is available (digitized deformation drive). Uses zero force so extrema and zero-u
    crossings follow the deformation path alone.
    """
    u = np.asarray(u, dtype=float).reshape(-1)
    return pd.DataFrame({DEF_COL: u, FORCE_COL: np.zeros_like(u)})


def find_cycle_points(df: pd.DataFrame) -> tuple[list[dict], list[tuple[int, int]]]:
    """
    Find characteristic points and segment boundaries.
    Returns (points with idx, type, force_kip, deformation_in; segments as (start, end) end-exclusive).
    """
    force = df[FORCE_COL]
    deformation = df[DEF_COL]

    points_with_type: list[tuple[int, str]] = []

    for i in _zero_crossings(deformation):
        points_with_type.append((i, "zero_def"))

    for i in _zero_crossings(force):
        points_with_type.append((i, "zero_force"))

    max_idx, min_idx = _local_extrema(deformation, EXTREMA_ORDER)
    for i in max_idx:
        points_with_type.append((int(i), "max_def"))
    for i in min_idx:
        points_with_type.append((int(i), "min_def"))

    # Sort by index and merge nearby
    points_merged = _merge_nearby(points_with_type, MERGE_RADIUS)
    # Collapse consecutive same primary type to one true point per cycle
    points_merged = _collapse_consecutive_same_type(points_merged, df)
    # Collapse points at same (force, deformation) so duplicate detections become one
    points_merged = _collapse_same_location(points_merged, df, rel_tol=SAME_LOCATION_TOL)

    force_vals = df[FORCE_COL]
    def_vals = df[DEF_COL]
    points_out = [
        {
            "idx": idx,
            "type": t,
            "force_kip": float(force_vals.iloc[idx]),
            "deformation_in": float(def_vals.iloc[idx]),
        }
        for idx, t in points_merged
    ]
    # Segment boundaries: characteristic points plus start/end of series (end exclusive for slicing)
    boundary_indices = sorted(set(p[0] for p in points_merged) | {0, len(df)})
    segments = [(boundary_indices[j], boundary_indices[j + 1]) for j in range(len(boundary_indices) - 1)]
    return points_out, segments


def run_specimen(
    specimen_id: str, save: bool = True, overwrite: bool = False
) -> tuple[list[dict], list[tuple[int, int]], bool] | None:
    """Load raw valid data, find cycle points and segments, optionally save to JSON. If the JSON already exists, skip writing unless overwrite=True. Returns (points, segments, wrote) or None if no data. wrote is True if the file was written."""
    df = load_raw_valid(specimen_id)
    if FORCE_COL not in df.columns or DEF_COL not in df.columns:
        return None
    points, segments = find_cycle_points(df)
    wrote = False
    if save:
        CYCLE_POINTS_ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
        out_path = CYCLE_POINTS_ORIGINAL_DIR / f"{specimen_id}.json"
        if out_path.exists() and not overwrite:
            return points, segments, False
        n = len(df)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"points": points, "n": n}, f, indent=2)
        wrote = True
    return points, segments, wrote


def _load_cycle_points_from_path(path: Path) -> tuple[list[dict], list[tuple[int, int]]] | None:
    """Load cycle JSON from path or return None."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    points = data.get("points", [])
    n = data.get("n")
    if n is None:
        return None
    n = int(n)
    boundary_indices = sorted(set(p["idx"] for p in points) | {0, n})
    segments = [(boundary_indices[j], boundary_indices[j + 1]) for j in range(len(boundary_indices) - 1)]
    return points, segments


def stored_cycle_points_grid_n(specimen_id: str) -> int | None:
    """``n`` from ``data/cycle_points_original/<id>.json`` (filtered-length grid), or None."""
    path = CYCLE_POINTS_ORIGINAL_DIR / f"{specimen_id}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    n = data.get("n")
    if n is None:
        return None
    return int(n)


def load_cycle_points_original(specimen_id: str) -> tuple[list[dict], list[tuple[int, int]]] | None:
    """Load cycle points on the **filtered** / trim-valid raw grid (``cycle_points_original``)."""
    return _load_cycle_points_from_path(CYCLE_POINTS_ORIGINAL_DIR / f"{specimen_id}.json")


def load_cycle_points_resampled(specimen_id: str) -> tuple[list[dict], list[tuple[int, int]]] | None:
    """Load cycle points on the **resampled** grid (``cycle_points_resampled``); use for calibration code."""
    return _load_cycle_points_from_path(CYCLE_POINTS_RESAMPLED_DIR / f"{specimen_id}.json")


def load_cycle_points_for_trimmed_length(specimen_id: str, n_trim: int) -> list[dict]:
    """Points list from ``cycle_points_original`` JSON whose stored ``n`` equals ``n_trim``."""
    path = CYCLE_POINTS_ORIGINAL_DIR / f"{specimen_id}.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if int(data.get("n", -1)) == n_trim:
        return list(data.get("points", []))
    return []


def main() -> None:
    """Generate cycle points for all specimens that have raw data. Skips existing JSON unless --overwrite."""
    import argparse
    parser = argparse.ArgumentParser(description="Detect cycle points; skip existing JSON unless --overwrite.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cycle-point JSON files")
    args = parser.parse_args()

    out_dir = CYCLE_POINTS_ORIGINAL_DIR
    names = list_names_for_cycle_points()
    if not names:
        print("No specimens with path_ordered primary F-u CSV (see BRB-Specimens.csv).")
        return
    for name in names:
        result = run_specimen(name, save=True, overwrite=args.overwrite)
        if result is None:
            print(f"Skip {name}: missing columns")
            continue
        points, segments, wrote = result
        if wrote:
            print(f"Wrote {out_dir.name}/{name}.json ({len(points)} points, {len(segments)} segments)")
        else:
            print(f"Skipped {out_dir.name}/{name}.json (exists; use --overwrite to replace)")


if __name__ == "__main__":
    main()
