# Wipe regenerated pipeline outputs. Never removes data/raw/ or config/calibration/*.csv
# (catalog, seeds, limits, loss settings); other top-level data/* folders (if any) are left untouched.
#
# Removes contents of:
#   data/filtered, data/resampled, data/cycle_points_original, data/cycle_points_resampled
#   results/plots/  (postprocess QA, apparent_b/, calibration/individual_optimize overlays,
#                    cycle_weights, debug_landmarks, debug_energy, calibration/single_specimen/, …)
#   results/calibration/  (individual_optimize, generalized_optimize, single_specimen,
#                          specimen_apparent_bn_bp.csv, calibration reports, …)
#   summary_statistics/
#
# Does not remove docs/readme snapshots, run_snapshots/, or repo-root demo PNGs.
# Run from repo root: .\clean_outputs.ps1

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

foreach ($rel in @("data\filtered", "data\resampled", "data\cycle_points_original", "data\cycle_points_resampled")) {
    $p = Join-Path $root $rel
    if (Test-Path $p) {
        Get-ChildItem -Path $p -Force | Remove-Item -Recurse -Force
    }
}

$plots = Join-Path $root "results\plots"
if (Test-Path $plots) {
    Remove-Item $plots -Recurse -Force
}

$cal = Join-Path $root "results\calibration"
if (Test-Path $cal) {
    Get-ChildItem -Path $cal -Force | Remove-Item -Recurse -Force
}

$sumDir = Join-Path $root "summary_statistics"
if (Test-Path $sumDir) {
    Get-ChildItem -Path $sumDir -Force | Remove-Item -Recurse -Force
}

Write-Host "Clean complete (kept data/raw and config/calibration inputs; wiped postprocess data/*, all of results/plots and results/calibration, and summary_statistics/)."
