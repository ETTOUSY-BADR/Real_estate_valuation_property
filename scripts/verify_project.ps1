param(
    [string]$PythonExe = "C:\Users\badre\anaconda3\python.exe"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = Join-Path $projectRoot "src"
$env:PYTHONDONTWRITEBYTECODE = "1"

Push-Location $projectRoot
try {
    & $PythonExe -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw "Project verification failed." }
}
finally {
    Pop-Location
}
