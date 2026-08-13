param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = Join-Path $projectRoot "src"

Push-Location $projectRoot
try {
    & $PythonExe -m paris_avm.features.phase2
    if ($LASTEXITCODE -ne 0) { throw "Phase 2 feature build failed." }
    & $PythonExe -m paris_avm.modeling.benchmark_phase2
    if ($LASTEXITCODE -ne 0) { throw "Phase 2 benchmark failed." }
    & $PythonExe -m paris_avm.visualization.phase2
    if ($LASTEXITCODE -ne 0) { throw "Phase 2 visualization failed." }
}
finally {
    Pop-Location
}
