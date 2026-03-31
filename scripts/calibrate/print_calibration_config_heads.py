"""Print ``config/calibration/*.csv`` as formatted tables (``#`` comment lines omitted)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_SCRIPTS = _PROJECT_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS / "postprocess"))

from calibrate.calibration_paths import (  # noqa: E402
    BRB_SPECIMENS_CSV,
    CALIBRATION_CONFIG_DIR,
    PARAM_LIMITS_CSV,
    SET_ID_SETTINGS_CSV,
)

_REPO_ROOT = CALIBRATION_CONFIG_DIR.parent.parent


def _print_table(path: Path, max_rows: int) -> None:
    try:
        rel = path.resolve().relative_to(_REPO_ROOT.resolve())
    except ValueError:
        rel = path
    print()
    print("-" * 72)
    print(f"  {rel}  (up to {max_rows} data rows; # comment lines skipped)")
    print("-" * 72)
    if not path.is_file():
        print(f"  (missing: {path})", file=sys.stderr)
        return
    try:
        df = pd.read_csv(path, comment="#")
    except pd.errors.EmptyDataError:
        print("  (no tabular rows after comment lines)")
        return
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as e:
        print(f"  (could not read as CSV: {e})", file=sys.stderr)
        return

    if len(df.columns) == 0:
        print("  (no tabular rows after comment lines)")
        return

    if df.empty:
        print("  (header only, no data rows)")
        print("  " + " | ".join(str(c) for c in df.columns))
        return

    n = len(df)
    view = df.head(max_rows)
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 240,
        "display.max_colwidth", 48,
        "display.expand_frame_repr", True,
    ):
        print(view.to_string(index=False))
    if n > max_rows:
        print(f"  ... ({n - max_rows} more rows)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--lines",
        type=int,
        default=25,
        metavar="N",
        help="max data rows to show per file (default: 25); # comment lines are never shown",
    )
    args = p.parse_args()
    n = max(1, args.lines)

    print()
    print("=" * 72)
    print("  Calibration input CSVs (tables)")
    print("=" * 72)

    for csv_path in (
        BRB_SPECIMENS_CSV,
        PARAM_LIMITS_CSV,
        SET_ID_SETTINGS_CSV,
    ):
        _print_table(csv_path, n)

    print()
    print("=" * 72)


if __name__ == "__main__":
    main()
