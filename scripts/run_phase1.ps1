param(
    [string]$PythonExe = "C:\Users\badre\anaconda3\python.exe"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = Join-Path $projectRoot "src"

Push-Location $projectRoot
try {
    & $PythonExe -m paris_avm.modeling.train_phase1
    & $PythonExe -m paris_avm.modeling.benchmark_phase1
    & $PythonExe -m paris_avm.visualization.benchmark
    if ($LASTEXITCODE -ne 0) { throw "Phase 1 workflow failed." }
}
finally {
    Pop-Location
}
