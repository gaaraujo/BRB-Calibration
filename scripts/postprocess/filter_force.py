"""
Filter raw force-deformation data and write filtered outputs.

Uses ``load_raw_valid()`` for **path-ordered** rows so row trimming (``config/raw_trim_ranges.yaml``)
applies. When ``data/cycle_points_original/<id>.json`` exists, filters by segments (between
characteristic points).

Writes **per specimen** under ``data/filtered/{Name}/`` (mirrors ``data/raw/{Name}/``):

- ``force_deformation.csv`` -- ``Force[kip]``, ``Deformation[in]``
- ``deformation_history.csv`` -- ``Step`` (0..n-1), ``Deformation[in]`` (no lab time column)

**Digitized scatter-cloud** rows: **does not smooth the F-u cloud** -- filtered cloud is a copy of
raw. The **deformation drive** uses ``find_cycle_points`` on a synthetic F=0 series, writes
``data/cycle_points_original/{Name}.json`` with ``n`` = drive length, then **unless**
``skip_filter_resample=true`` applies Savitzky-Golay **inside each segment** (or global smooth if
segments are too short). Resampling then uses the same |Du| segment resampling as path-ordered specimens.

Catalog ``skip_filter_resample=true``: path-ordered **filtered** = trim-valid raw (no Savitzky-Golay);
digitized **drive** unsmoothed (still get cycle JSON + ``Step`` column); cloud still copied raw -> filtered.
Downstream resampling still runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from load_raw import load_raw_valid
from cycle_points import (
    CYCLE_POINTS_ORIGINAL_DIR,
    find_cycle_points,
    synthetic_fu_from_deformation,
    load_cycle_points_original,
)
from specimen_catalog import (
    DEFORMATION_HISTORY_CSV,
    FORCE_DEFORMATION_CSV,
    deformation_history_csv_path,
    force_deformation_unordered_csv_path,
    get_specimen_record,
    list_names_for_filter_outputs,
    read_catalog,
    uses_unordered_inputs,
    write_deformation_history_step_csv,
)

# Project root: parent of scripts/postprocess -> scripts -> root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
FILTERED_DIR = PROJECT_ROOT / "data" / "filtered"

# Savitzky-Golay: window length must be odd; polyorder < window_length
WINDOW_LENGTH = 11
POLYORDER = 3
# Deformation (displacement) often noisier; use longer window for smoother signal
DEF_WINDOW_LENGTH = 17
MIN_SEGMENT_POINTS = 11  # if any segment is shorter, skip segment-based filtering

FORCE_COL = "Force[kip]"
DEF_COL = "Deformation[in]"

# Normalized header tokens (strip edges, drop all whitespace, lowercased) must match these.
_FORCE_HEADER_KEY = "force[kip]"
_DEF_HEADER_KEY = "deformation[in]"


def _column_key(name: str) -> str:
    """Normalize a CSV header for comparison: BOM/edge strip, remove whitespace, lowercase."""
    s = str(name).strip().lstrip("\ufeff")
    return "".join(s.split()).lower()


def _read_table_csv(path: Path) -> pd.DataFrame:
    """Comma-separated CSV with UTF-8 BOM tolerated.

    ``skipinitialspace`` drops spaces right after each comma (e.g. ``0, 0.12``).
    """
    df = pd.read_csv(path, sep=",", encoding="utf-8-sig", skipinitialspace=True)
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df


def _to_float_stripped(s: pd.Series) -> pd.Series:
    """Parse floats from cell text after stripping ends (handles spaces inside quoted or object cells)."""
    return pd.to_numeric(s.astype(str).str.strip(), errors="raise")


def _canonicalize_force_def_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to FORCE_COL / DEF_COL when header key matches ``force[kip]`` / ``deformation[in]``."""
    renames: dict[str, str] = {}
    got_d = got_f = False
    for c in df.columns:
        cs = str(c)
        k = _column_key(cs)
        if k == _DEF_HEADER_KEY and not got_d:
            if cs != DEF_COL:
                renames[cs] = DEF_COL
            got_d = True
        elif k == _FORCE_HEADER_KEY and not got_f:
            if cs != FORCE_COL:
                renames[cs] = FORCE_COL
            got_f = True
    return df.rename(columns=renames) if renames else df


def filter_series(series: pd.Series, window: int = WINDOW_LENGTH, poly: int = POLYORDER) -> pd.Series:
    """Smooth series with Savitzky-Golay; use smaller window if series is short."""
    n = len(series)
    w = min(window, n if n % 2 else n - 1)
    if w < 3:
        return series
    p = min(poly, w - 1)
    return pd.Series(savgol_filter(series.values, w, p), index=series.index)


def filter_by_segments(df: pd.DataFrame, segments: list[tuple[int, int]]) -> pd.DataFrame:
    """Filter each segment separately and concatenate (avoids smoothing across segment boundaries)."""
    chunks = []
    for start, end in segments:
        if start >= end:
            continue
        chunk = df.iloc[start:end].copy()
        chunk[FORCE_COL] = filter_series(chunk[FORCE_COL])
        chunk[DEF_COL] = filter_series(chunk[DEF_COL], window=DEF_WINDOW_LENGTH)
        chunks.append(chunk)
    if not chunks:
        return df
    return pd.concat(chunks, ignore_index=True)


def _filter_deformation_only_by_segments(
    u: pd.Series, segments: list[tuple[int, int]], *, window: int = DEF_WINDOW_LENGTH
) -> np.ndarray:
    """Savitzky on displacement inside each [start, end) segment; concatenate (same length as u)."""
    parts: list[np.ndarray] = []
    for start, end in segments:
        if start >= end:
            continue
        chunk = u.iloc[start:end].astype(float)
        sm = filter_series(pd.Series(chunk.values), window=window)
        parts.append(sm.values)
    if not parts:
        return u.to_numpy(dtype=float)
    return np.concatenate(parts)


def _process_path_ordered(name: str, catalog: pd.DataFrame) -> None:
    """Savitzky + segment filter for path-ordered specimen."""
    df = load_raw_valid(name)
    if FORCE_COL not in df.columns or DEF_COL not in df.columns:
        print(f"Skip {name}: expected columns {FORCE_COL}, {DEF_COL}")
        return
    rec = get_specimen_record(name, catalog)
    if rec.skip_filter_resample:
        print(f"{name}: skip_filter_resample -> filtered = trim-valid raw (no Savitzky-Golay)")
    else:
        loaded = load_cycle_points_original(name)
        if loaded is not None:
            _points, segments = loaded
            min_len = min((end - start) for start, end in segments) if segments else 0
            if min_len < MIN_SEGMENT_POINTS:
                print(
                    f"Skip filtering {name}: min segment length {min_len} < {MIN_SEGMENT_POINTS}"
                )
            else:
                df = filter_by_segments(df, segments)
            if min_len < MIN_SEGMENT_POINTS:
                df[FORCE_COL] = filter_series(df[FORCE_COL])
                df[DEF_COL] = filter_series(df[DEF_COL], window=DEF_WINDOW_LENGTH)
        else:
            df[FORCE_COL] = filter_series(df[FORCE_COL])
            df[DEF_COL] = filter_series(df[DEF_COL], window=DEF_WINDOW_LENGTH)

    subdir = FILTERED_DIR / name
    subdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(subdir / FORCE_DEFORMATION_CSV, index=False)
    write_deformation_history_step_csv(subdir / DEFORMATION_HISTORY_CSV, df[DEF_COL].values)
    print(f"Wrote {subdir.relative_to(PROJECT_ROOT)}/{{force_deformation,deformation_history}}.csv")


def _process_digitized_unordered(name: str, catalog: pd.DataFrame) -> None:
    """Filter/resample pipeline for digitized unordered specimen."""
    rec = get_specimen_record(name, catalog)
    dh_path = deformation_history_csv_path(name, PROJECT_ROOT)
    fd_path = force_deformation_unordered_csv_path(name, PROJECT_ROOT)
    df_dh = _canonicalize_force_def_columns(_read_table_csv(dh_path))
    df_fd = _canonicalize_force_def_columns(_read_table_csv(fd_path))
    if DEF_COL not in df_dh.columns:
        print(
            f"Skip {name}: {dh_path.name} missing {DEF_COL} "
            f"(columns after normalize: {list(df_dh.columns)!r})"
        )
        return
    if DEF_COL not in df_fd.columns or FORCE_COL not in df_fd.columns:
        print(
            f"Skip {name}: {fd_path.name} missing {FORCE_COL} / {DEF_COL} "
            f"(columns after normalize: {list(df_fd.columns)!r})"
        )
        return

    df_dh = df_dh.copy()
    df_fd = df_fd.copy()
    df_dh[DEF_COL] = _to_float_stripped(df_dh[DEF_COL])
    df_fd[DEF_COL] = _to_float_stripped(df_fd[DEF_COL])
    df_fd[FORCE_COL] = _to_float_stripped(df_fd[FORCE_COL])

    u_dh = df_dh[DEF_COL]
    synth = synthetic_fu_from_deformation(u_dh)
    points, segments = find_cycle_points(synth)
    CYCLE_POINTS_ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
    cp_path = CYCLE_POINTS_ORIGINAL_DIR / f"{name}.json"
    n_drive = len(u_dh)
    with open(cp_path, "w", encoding="utf-8") as f:
        json.dump({"points": points, "n": n_drive}, f, indent=2)
    print(f"{name} (digitized): wrote {cp_path.relative_to(PROJECT_ROOT)} ({len(points)} points)")

    if rec.skip_filter_resample:
        u_filt = u_dh.to_numpy(dtype=float)
        print(f"{name} (digitized): skip_filter_resample -> drive not Savitzky-smoothed")
    else:
        min_len = min((end - start) for start, end in segments) if segments else 0
        if min_len < MIN_SEGMENT_POINTS:
            print(
                f"{name} (digitized): min segment length {min_len} < {MIN_SEGMENT_POINTS}; "
                "global smooth on drive"
            )
            u_filt = filter_series(pd.Series(u_dh.values), window=DEF_WINDOW_LENGTH).values
        else:
            u_filt = _filter_deformation_only_by_segments(
                pd.Series(u_dh.values), segments, window=DEF_WINDOW_LENGTH
            )

    subdir = FILTERED_DIR / name
    subdir.mkdir(parents=True, exist_ok=True)
    write_deformation_history_step_csv(subdir / DEFORMATION_HISTORY_CSV, u_filt)

    df_fd[[DEF_COL, FORCE_COL]].to_csv(subdir / FORCE_DEFORMATION_CSV, index=False)
    print(
        f"Wrote {subdir.relative_to(PROJECT_ROOT)}/{{force_deformation,deformation_history}}.csv "
        f"(cloud unsmoothed)"
    )


def main() -> None:
    """CLI entry point."""
    FILTERED_DIR.mkdir(parents=True, exist_ok=True)
    catalog = read_catalog()
    names = list_names_for_filter_outputs(catalog=catalog, project_root=PROJECT_ROOT)
    if not names:
        print("No specimens with path_ordered primary F-u and/or digitized cloud inputs on disk.")
        return
    for name in names:
        rec = get_specimen_record(name, catalog)
        if uses_unordered_inputs(rec):
            _process_digitized_unordered(name, catalog)
        else:
            _process_path_ordered(name, catalog)


if __name__ == "__main__":
    main()
