param(
    [string]$PythonExe = "C:\Users\badre\anaconda3\python.exe",
    [switch]$SkipAcquisition,
    [switch]$SkipFeatureBuild,
    [switch]$SkipBenchmark
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = Join-Path $projectRoot "src"

Push-Location $projectRoot
try {
    if (-not $SkipAcquisition) {
        & $PythonExe -m paris_avm.data.acquire_phase3 --skip-dpe
        if ($LASTEXITCODE -ne 0) { throw "Phase 3 acquisition failed." }
    }
    if (-not $SkipFeatureBuild) {
        & $PythonExe -m paris_avm.features.phase3
        if ($LASTEXITCODE -ne 0) { throw "Phase 3 feature build failed." }
    }
    if (-not $SkipBenchmark) {
        & $PythonExe -m paris_avm.modeling.benchmark_phase3
        if ($LASTEXITCODE -ne 0) { throw "Phase 3 benchmark failed." }
    }
    & $PythonExe -m paris_avm.visualization.phase3
    if ($LASTEXITCODE -ne 0) { throw "Phase 3 visualization failed." }
    & powershell -ExecutionPolicy Bypass -File .\docs\paper\compile_phase3.ps1
    if ($LASTEXITCODE -ne 0) { throw "Phase 3 paper compilation failed." }
}
finally {
    Pop-Location
}
