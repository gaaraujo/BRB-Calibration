"""Reuse OpenSees F_sim arrays for debug plotting when parameters CSV is unchanged."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _fingerprint_sim_row(row: dict[str, float]) -> str:
    """Stable hash of steel/geometry fields used to invalidate cache."""
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _cache_dir(out_dir: Path) -> Path:
    """Subfolder under ``out_dir`` for cached ``F_sim`` arrays."""
    return out_dir / "_sim_cache"


def _fsim_path(out_dir: Path, specimen: str, set_id: Any) -> Path:
    """Path to ``.npy`` cache for simulated force."""
    return _cache_dir(out_dir) / f"{specimen}_set{set_id}_F_sim.npy"


def _manifest_path(out_dir: Path, specimen: str, set_id: Any) -> Path:
    """Sidecar JSON: params file mtime/size and fingerprint."""
    return _cache_dir(out_dir) / f"{specimen}_set{set_id}_manifest.json"


def try_load_cached_fsim(
    out_dir: Path,
    specimen: str,
    set_id: Any,
    params_path: Path,
    sim_row: dict[str, float],
) -> np.ndarray | None:
    """Return cached ``F_sim`` if manifest matches current parameters CSV and row fingerprint."""
    fpath = _fsim_path(out_dir, specimen, set_id)
    mpath = _manifest_path(out_dir, specimen, set_id)
    if not fpath.is_file() or not mpath.is_file():
        return None
    try:
        man = json.loads(mpath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        stat = params_path.stat()
    except OSError:
        return None
    if man.get("params_path") != str(params_path.resolve()):
        return None
    if man.get("params_mtime") != stat.st_mtime_ns or man.get("params_size") != stat.st_size:
        return None
    if man.get("fingerprint") != _fingerprint_sim_row(sim_row):
        return None
    arr = np.load(fpath)
    return np.asarray(arr, dtype=float)


def save_fsim_cache(
    out_dir: Path,
    specimen: str,
    set_id: Any,
    params_path: Path,
    sim_row: dict[str, float],
    F_sim: np.ndarray,
) -> None:
    """Write ``F_sim`` and manifest next to debug plots."""
    cdir = _cache_dir(out_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    fpath = _fsim_path(out_dir, specimen, set_id)
    mpath = _manifest_path(out_dir, specimen, set_id)
    np.save(fpath, np.asarray(F_sim, dtype=float))
    stat = params_path.stat()
    man = {
        "params_path": str(params_path.resolve()),
        "params_mtime": stat.st_mtime_ns,
        "params_size": stat.st_size,
        "fingerprint": _fingerprint_sim_row(sim_row),
    }
    mpath.write_text(json.dumps(man, indent=0) + "\n", encoding="utf-8")
