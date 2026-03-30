# Zip everything clean_outputs.ps1 would wipe, plus full results/plots and results/calibration.
# Run from repo root: .\archive_pipeline_outputs.ps1 [-Label myrun]
# Output: run_snapshots/<yyyyMMdd_HHmmss>_<Label>.zip - unzip at repo root to restore paths.

param(
    [string]$Label = "full",
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
if (-not $OutDir) {
    $OutDir = Join-Path $root "run_snapshots"
}

$sanitized = [regex]::Replace($Label, '[^\w\-.]+', '_').Trim('_')
if (-not $sanitized) { $sanitized = "full" }

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$zipName = "${stamp}_${sanitized}.zip"
$zipPath = Join-Path $OutDir $zipName
$stagingName = ".staging_${stamp}_$PID"
$staging = Join-Path $OutDir $stagingName

$dataDirs = @(
    "data\filtered",
    "data\resampled",
    "data\cycle_points_original",
    "data\cycle_points_resampled"
)
$rootFiles = @(
    "summary_statistics\calibration_parameter_summary.md",
    "summary_statistics\calibration_parameter_summary_generalized.csv",
    "summary_statistics\calibration_parameter_summary_individual.csv",
    "summary_statistics\calibration_parameter_summary_generalized_by_set.csv",
    "summary_statistics\calibration_parameter_summary_individual_by_set.csv",
    "summary_statistics\generalized_set_id_eval_summary.csv",
    "summary_statistics\generalized_unordered_J_binenv_summary_train.csv",
    "summary_statistics\generalized_unordered_J_binenv_summary_validation.csv"
)

$included = New-Object System.Collections.Generic.List[string]

function Try-GitHead {
    try {
        Push-Location $root
        $h = (& git rev-parse HEAD 2>$null)
        if ($LASTEXITCODE -eq 0 -and $h) { return $h.Trim() }
    } catch { }
    finally { Pop-Location }
    return $null
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
New-Item -ItemType Directory -Force -Path $staging | Out-Null

try {
    foreach ($rel in $dataDirs) {
        $src = Join-Path $root $rel
        if (-not (Test-Path -LiteralPath $src)) { continue }
        $parent = Split-Path -Parent $rel
        $leaf = Split-Path -Leaf $rel
        $destParent = Join-Path $staging $parent
        New-Item -ItemType Directory -Force -Path $destParent | Out-Null
        Copy-Item -LiteralPath $src -Destination (Join-Path $destParent $leaf) -Recurse -Force
        $included.Add($rel.Replace('\', '/'))
    }

    foreach ($rel in @("results\plots", "results\calibration")) {
        $src = Join-Path $root $rel
        if (-not (Test-Path -LiteralPath $src)) { continue }
        $parent = Split-Path -Parent $rel
        $leaf = Split-Path -Leaf $rel
        $destParent = Join-Path $staging $parent
        New-Item -ItemType Directory -Force -Path $destParent | Out-Null
        Copy-Item -LiteralPath $src -Destination (Join-Path $destParent $leaf) -Recurse -Force
        $included.Add($rel.Replace('\', '/'))
    }

    foreach ($rel in $rootFiles) {
        $src = Join-Path $root $rel
        if (-not (Test-Path -LiteralPath $src)) { continue }
        $dest = Join-Path $staging $rel
        $destParent = Split-Path -Parent $dest
        New-Item -ItemType Directory -Force -Path $destParent | Out-Null
        Copy-Item -LiteralPath $src -Destination $dest -Force
        $included.Add($rel.Replace('\', '/'))
    }

    if ($included.Count -eq 0) {
        Write-Error "Nothing to archive (no matching paths under repo root). Run the pipeline first or check paths."
    }

    $gitHead = Try-GitHead
    $manifestLines = @(
        "created_utc=$([DateTime]::UtcNow.ToString('o'))",
        "label=$sanitized",
        "git_head=$gitHead",
        "repo_relative_paths_included:"
    ) + ($included | ForEach-Object { "  $_" }) + @("  MANIFEST.txt")
    $manifestPath = Join-Path $staging "MANIFEST.txt"
    Set-Content -LiteralPath $manifestPath -Value $manifestLines -Encoding utf8
    $included.Add("MANIFEST.txt")

    $items = Get-ChildItem -LiteralPath $staging
    if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
    Compress-Archive -Path ($items | ForEach-Object { $_.FullName }) -DestinationPath $zipPath -CompressionLevel Optimal -Force

    Write-Host "Wrote $zipPath ($($included.Count) top-level entries / manifest paths)."
}
finally {
    if (Test-Path -LiteralPath $staging) {
        Remove-Item -LiteralPath $staging -Recurse -Force -ErrorAction SilentlyContinue
    }
}
