<#
.SYNOPSIS
  Replace the openseespy native module with a locally built opensees.pyd (backup first).

.DESCRIPTION
  Resolves the native extension without importing openseespy (so a broken .pyd/DLL can still be
  replaced). On Windows, targets site-packages\openseespywin. Backs up the .pyd, copies your
  Release\opensees.pyd, then copies Release\*.dll next to it.

  After installing the .pyd, copies all ``*.dll`` from the same Release folder into the directory that
  contains the installed ``opensees*.pyd`` (Windows often will not load the extension unless dependent
  DLLs sit next to it; ``PATH`` alone is not always enough).

.PARAMETER OpenSeesReleaseDir
  Folder containing your built opensees.pyd (default: sibling repo path).

.PARAMETER Python
  Python launcher (default: python).

.PARAMETER SkipDllCopy
  If set, do not copy ``*.dll`` from Release into the openseespywin folder (not recommended).

.EXAMPLE
  .\scripts\install_custom_opensees_pyd.ps1

.EXAMPLE
  .\scripts\install_custom_opensees_pyd.ps1 -OpenSeesReleaseDir D:\OpenSees\build\Release
#>
[CmdletBinding()]
param(
    [string] $OpenSeesReleaseDir = "",
    [string] $Python = "python",
    [switch] $SkipDllCopy
)

$ErrorActionPreference = "Stop"

if (-not $OpenSeesReleaseDir) {
    $sibling = Join-Path $PSScriptRoot "..\..\OpenSees\build\Release"
    if (Test-Path -LiteralPath $sibling) {
        $OpenSeesReleaseDir = (Resolve-Path -LiteralPath $sibling).Path
    } else {
        $OpenSeesReleaseDir = "C:\Users\garaujor\source\repos\OpenSees\build\Release"
    }
}
if (-not (Test-Path -LiteralPath $OpenSeesReleaseDir)) {
    Write-Error "OpenSees Release folder not found: $OpenSeesReleaseDir (pass -OpenSeesReleaseDir)"
}
$OpenSeesReleaseDir = (Resolve-Path -LiteralPath $OpenSeesReleaseDir).Path

$pydSource = Join-Path $OpenSeesReleaseDir "opensees.pyd"
if (-not (Test-Path -LiteralPath $pydSource)) {
    Write-Error "Source pyd not found: $pydSource"
}

# Multi-line python -c is unreliable from PowerShell; use a temp script.
$pyCode = @'
# No import of openseespy / openseespywin: a broken custom .pyd would crash before we find its path.
import importlib.metadata
import pathlib
import site
import sys


def _pick(pyds):
    if not pyds:
        return None
    pyds = sorted({str(pathlib.Path(p).resolve()) for p in pyds})
    winpkg = [p for p in pyds if "openseespywin" in p.replace("\\", "/").lower()]
    pool = winpkg if winpkg else pyds
    tagged = [p for p in pool if "cp" in pathlib.Path(p).name.lower()]
    return tagged[0] if tagged else pool[0]


def find_opensees_pyd():
    # 1) Wheel / dist metadata (no import)
    meta = []
    for dist_name in ("openseespywin", "openseespy"):
        try:
            dist = importlib.metadata.distribution(dist_name)
        except importlib.metadata.PackageNotFoundError:
            continue
        if dist.files:
            for file in dist.files:
                if str(file).lower().endswith(".pyd"):
                    try:
                        meta.append(dist.locate_file(file))
                    except (OSError, ValueError):
                        pass
    picked = _pick(meta)
    if picked:
        return str(pathlib.Path(picked).resolve())

    # 2) site-packages scan (conda when dist.files is empty)
    sp_roots = []
    if hasattr(site, "getsitepackages"):
        sp_roots.extend(site.getsitepackages())
    if hasattr(site, "getusersitepackages"):
        u = site.getusersitepackages()
        if u:
            sp_roots.append(u)
    sp_roots.append(pathlib.Path(sys.prefix) / "Lib" / "site-packages")
    for sp in sp_roots:
        for pkg_name in ("openseespywin", "openseespy"):
            pkg = pathlib.Path(sp) / pkg_name
            if pkg.is_dir():
                found = list(pkg.rglob("*.pyd"))
                picked = _pick(str(p) for p in found)
                if picked:
                    return picked

    raise SystemExit(
        "No .pyd found under openseespywin or openseespy (metadata + site-packages). "
        "Install openseespy / openseespywin in this env, or pass the .pyd path manually."
    )


print(find_opensees_pyd())
'@

$tmpPy = Join-Path $env:TEMP ("install_custom_opensees_pyd_{0}.py" -f [Guid]::NewGuid().ToString("N"))
try {
    Set-Content -LiteralPath $tmpPy -Value $pyCode -Encoding utf8
    $targetFile = & $Python $tmpPy 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to locate opensees .pyd. Output: $targetFile"
    }
}
finally {
    Remove-Item -LiteralPath $tmpPy -Force -ErrorAction SilentlyContinue
}

$targetFile = ($targetFile | Out-String).Trim()
if (-not $targetFile -or -not (Test-Path -LiteralPath $targetFile)) {
    Write-Error "Resolved pyd path missing: '$targetFile'"
}

$ext = [System.IO.Path]::GetExtension($targetFile).ToLowerInvariant()
if ($ext -ne ".pyd") {
    Write-Error "Resolved path is not a .pyd: $targetFile"
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$backup = "$targetFile.bak.$stamp"
Copy-Item -LiteralPath $targetFile -Destination $backup
Write-Host "Backed up: $backup"

Copy-Item -LiteralPath $pydSource -Destination $targetFile -Force
Write-Host "Installed: $pydSource -> $targetFile"

$targetDir = Split-Path -LiteralPath $targetFile
if (-not $SkipDllCopy) {
    $dlls = @(Get-ChildItem -LiteralPath $OpenSeesReleaseDir -Filter *.dll -File -ErrorAction SilentlyContinue)
    if ($dlls.Count -eq 0) {
        Write-Warning "No *.dll in $OpenSeesReleaseDir; if import fails with 'DLL load failed', add DLLs next to the .pyd or install the matching VC++ Redistributable."
    } else {
        foreach ($d in $dlls) {
            Copy-Item -LiteralPath $d.FullName -Destination (Join-Path $targetDir $d.Name) -Force
        }
        Write-Host "Copied $($dlls.Count) DLL(s) from Release -> $targetDir"
    }
} else {
    Write-Host "Skipped DLL copy (-SkipDllCopy). You may need PATH or sibling DLLs for import to succeed."
}

Write-Host ""
Write-Host "Optional: still prepend Release on PATH when running other tools, e.g.:"
Write-Host "  `$env:PATH = '$OpenSeesReleaseDir;' + `$env:PATH"
