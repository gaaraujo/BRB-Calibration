"""
Load primary force-deformation CSV with optional per-specimen row trimming.

Uses ``config/raw_trim_ranges.yaml`` to replace excluded ranges with interpolated values so the
returned DataFrame keeps the same length and row index as the file. Ranges are 0-based, inclusive;
end null/None means end of file.

Which file opens for a given ``Name`` is decided in ``specimen_catalog.primary_f_u_csv_path``
(``data/raw/{Name}/force_deformation.csv`` for path-ordered rows). Digitized scatter-cloud rows (``digitized`` + ``path_ordered``
false) have no single series here--use the cloud loaders in ``calibrate/`` (see ``scripts/README.md``).
"""
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "raw_trim_ranges.yaml"

FORCE_COL = "Force[kip]"
DEF_COL = "Deformation[in]"


def _load_exclusion_config() -> dict:
    """Load YAML config; return {} if file missing or empty."""
    if not CONFIG_PATH.exists():
        return {}
    try:
        import yaml
        with open(CONFIG_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _resolve_primary_csv_path(specimen_id: str) -> Path:
    """Locate primary force_deformation CSV under data/raw."""
    from specimen_catalog import (
        get_specimen_record,
        primary_f_u_csv_path,
        read_catalog,
        uses_unordered_inputs,
    )

    cat = read_catalog()
    rec = get_specimen_record(specimen_id, cat)
    if uses_unordered_inputs(rec):
        raise ValueError(
            f"{specimen_id}: digitized scatter-cloud inputs have no single F-u raw; "
            "use data/raw/{Name}/deformation_history.csv and force_deformation.csv."
        )
    p = primary_f_u_csv_path(specimen_id, cat)
    if p is None or not p.is_file():
        raise FileNotFoundError(
            f"No primary F-u CSV for {specimen_id} (layout={rec.experimental_layout}). "
            "Expected data/raw/{Name}/force_deformation.csv (path-ordered catalog row)."
        )
    return p


def load_raw_valid(specimen_id: str) -> pd.DataFrame:
    """
    Load primary F-u CSV and replace excluded row ranges with interpolated values.
    Output has the same number of rows and indices (0..n-1) as the file.
    """
    raw_path = _resolve_primary_csv_path(specimen_id)
    df = pd.read_csv(raw_path).copy()
    config = _load_exclusion_config()
    spec_config = config.get(specimen_id) if isinstance(config.get(specimen_id), dict) else {}
    ranges = spec_config.get("exclude_ranges") or []
    if not ranges:
        return df
    n = len(df)
    force = df[FORCE_COL].values
    deformation = df[DEF_COL].values

    for r in ranges:
        start = int(r[0]) if len(r) > 0 else 0
        end_val = r[1] if len(r) > 1 else None
        start = max(0, min(start, n))
        end = (n - 1) if end_val is None else min(int(end_val), n - 1)
        end = max(0, end)
        if start > end:
            continue
        n_replace = end - start + 1

        if end_val is None:
            if start == 0:
                continue
            fill_force = force[start - 1]
            fill_def = deformation[start - 1]
            force[start : end + 1] = fill_force
            deformation[start : end + 1] = fill_def
        else:
            left_idx = start - 1 if start > 0 else end + 1
            right_idx = end + 1 if end + 1 < n else start - 1
            if left_idx < 0:
                left_idx = right_idx
            if right_idx >= n:
                right_idx = left_idx
            t = np.linspace(0, 1, n_replace + 2)[1:-1]
            force[start : end + 1] = (1 - t) * force[left_idx] + t * force[right_idx]
            deformation[start : end + 1] = (1 - t) * deformation[left_idx] + t * deformation[right_idx]

    df[FORCE_COL] = force
    df[DEF_COL] = deformation
    return df


def load_raw_full(specimen_id: str) -> pd.DataFrame | None:
    """Load primary F-u CSV without trim ranges. Returns None if not loadable (e.g. split layout)."""
    try:
        raw_path = _resolve_primary_csv_path(specimen_id)
    except (ValueError, FileNotFoundError):
        return None
    if not raw_path.is_file():
        return None
    return pd.read_csv(raw_path)
