# Wipe regenerated pipeline outputs. Never removes data/raw/ or config/calibration/*.csv
# (catalog, seeds, limits, loss settings); other top-level data/* folders (if any) are left untouched.
# Removes all of results/calibration/ (including individual_optimize/initial_brb_parameters.csv).
# Removes all contents of summary_statistics/ (entire folder cleared, directory kept if present).
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

Write-Host "Clean complete (kept data/raw and config/calibration inputs; wiped regenerated data/* subdirs, results/calibration including individual_optimize/initial_brb_parameters.csv, and all of summary_statistics/)."
