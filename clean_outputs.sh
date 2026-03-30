#!/usr/bin/env bash
# Wipe regenerated pipeline outputs. Never removes data/raw/ or config/calibration/*.csv
# (catalog, seeds, limits, loss settings); other top-level data/* folders (if any) are left untouched.
# Removes all of results/calibration/ (including individual_optimize/initial_brb_parameters.csv).
# Removes all contents of summary_statistics/ (entire folder cleared, directory kept if present).
# Run from repo root: ./clean_outputs.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

for d in data/filtered data/resampled data/cycle_points_original data/cycle_points_resampled; do
  if [[ -d "$ROOT/$d" ]]; then
    find "$ROOT/$d" -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null || true
  fi
done

rm -rf "$ROOT/results/plots"

if [[ -d "$ROOT/results/calibration" ]]; then
  rm -rf "$ROOT/results/calibration"/*
fi

SUM_DIR="$ROOT/summary_statistics"
if [[ -d "$SUM_DIR" ]]; then
  find "$SUM_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null || true
fi

echo "Clean complete (kept data/raw and config/calibration inputs; wiped regenerated data/* subdirs, results/calibration including individual_optimize/initial_brb_parameters.csv, and all of summary_statistics/)."
