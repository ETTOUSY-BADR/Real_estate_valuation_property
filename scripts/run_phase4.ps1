param(
    [string]$PythonExe = "python",
    [switch]$SkipAcquisition,
    [switch]$SkipFeatureBuild,
    [switch]$SkipPreflight,
    [switch]$SkipBenchmark,
    [switch]$SkipVisualization,
    [switch]$SkipPaper
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = Join-Path $projectRoot "src"

Push-Location $projectRoot
try {
    if (-not $SkipAcquisition) {
        & $PythonExe -m paris_avm.data.acquire_phase4
        if ($LASTEXITCODE -ne 0) { throw "Phase 4 national acquisition failed." }
    }
    if (-not $SkipFeatureBuild) {
        & $PythonExe -m paris_avm.features.phase4
        if ($LASTEXITCODE -ne 0) { throw "Phase 4 national feature build failed." }
    }
    if (-not $SkipPreflight) {
        & $PythonExe -m paris_avm.analysis.phase4_preflight
        if ($LASTEXITCODE -ne 0) { throw "Phase 4 national preflight analysis failed." }
    }
    if (-not $SkipBenchmark) {
        & $PythonExe -m paris_avm.modeling.benchmark_phase4
        if ($LASTEXITCODE -ne 0) { throw "Phase 4 national benchmark failed." }
    }
    if (-not $SkipVisualization) {
        & $PythonExe -m paris_avm.visualization.phase4
        if ($LASTEXITCODE -ne 0) { throw "Phase 4 report generation failed." }
        & $PythonExe -m paris_avm.reporting.phase4
        if ($LASTEXITCODE -ne 0) { throw "Phase 4 LaTeX generation failed." }
    }
    if (-not $SkipPaper) {
        & powershell -ExecutionPolicy Bypass -File .\docs\paper\compile_phase4.ps1
        if ($LASTEXITCODE -ne 0) { throw "Phase 4 PDF compilation failed." }
    }
}
finally {
    Pop-Location
}
