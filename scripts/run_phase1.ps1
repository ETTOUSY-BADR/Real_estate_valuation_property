param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = Join-Path $projectRoot "src"

Push-Location $projectRoot
try {
    & $PythonExe -m paris_avm.modeling.train_phase1
    if ($LASTEXITCODE -ne 0) { throw "Phase 1 training failed." }
    & $PythonExe -m paris_avm.modeling.benchmark_phase1
    if ($LASTEXITCODE -ne 0) { throw "Phase 1 benchmark failed." }
    & $PythonExe -m paris_avm.visualization.benchmark
    if ($LASTEXITCODE -ne 0) { throw "Phase 1 visualization failed." }
}
finally {
    Pop-Location
}
