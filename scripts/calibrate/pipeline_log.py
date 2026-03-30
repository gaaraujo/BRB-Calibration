"""
Plain-text, redirect-friendly logging for calibration CLI scripts.

Uses flush=True on every line so tee/redirection interleaves cleanly. Section headers make
pipeline logs easy to scan in a file.
"""
from __future__ import annotations

from datetime import datetime, timezone

_WIDTH = 72


def _p(msg: str = "") -> None:
    """Print ``msg`` with flush (stdout)."""
    print(msg, flush=True)


def run_banner(script_name: str) -> None:
    """Print script title and UTC timestamp (call once at start of ``main()``)."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    _p("=" * _WIDTH)
    _p(f"  {script_name}")
    _p(f"  {ts}")
    _p("=" * _WIDTH)


def section(title: str) -> None:
    """Blank line plus a visible section heading."""
    _p("")
    _p(f"  --- {title} ---")


def kv(key: str, value: str) -> None:
    """Aligned key / value (paths and short strings)."""
    _p(f"  {key:<26} {value}")


def line(msg: str, *, indent: int = 2) -> None:
    """Body line with fixed indent (spaces)."""
    sp = " " * indent
    _p(f"{sp}{msg}")


def saved_artifacts(npz_name: str | None, csv_name: str | None) -> None:
    """One line listing NPZ/CSV saves when both exist."""
    parts = []
    if npz_name:
        parts.append(f"npz={npz_name}")
    if csv_name:
        parts.append(f"csv={csv_name}")
    if parts:
        line("saved  " + "  ".join(parts))
