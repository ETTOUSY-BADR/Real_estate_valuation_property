$ErrorActionPreference = "Stop"

$paperDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$miktexBin = "C:\Users\badre\AppData\Local\Programs\MiKTeX\miktex\bin\x64"
$pdflatex = Join-Path $miktexBin "pdflatex.exe"

if (-not (Test-Path $pdflatex)) {
    throw "pdflatex.exe was not found at $pdflatex. Install MiKTeX or update this script."
}

Push-Location $paperDirectory
try {
    & $pdflatex -interaction=nonstopmode -halt-on-error phase3_paper.tex
    if ($LASTEXITCODE -ne 0) { throw "First pdflatex pass failed." }
    & $pdflatex -interaction=nonstopmode -halt-on-error phase3_paper.tex
    if ($LASTEXITCODE -ne 0) { throw "Final pdflatex pass failed." }
    Write-Output "Compiled: $paperDirectory\phase3_paper.pdf"
}
finally {
    Pop-Location
}
