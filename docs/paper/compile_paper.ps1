$ErrorActionPreference = "Stop"

$paperDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$miktexBin = "C:\Users\badre\AppData\Local\Programs\MiKTeX\miktex\bin\x64"
$pdflatex = Join-Path $miktexBin "pdflatex.exe"
$bibtex = Join-Path $miktexBin "bibtex.exe"

if (-not (Test-Path $pdflatex)) {
    throw "pdflatex.exe was not found at $pdflatex. Install MiKTeX or update this script."
}
if (-not (Test-Path $bibtex)) {
    throw "bibtex.exe was not found at $bibtex. Install MiKTeX or update this script."
}

Push-Location $paperDirectory
try {
    & $pdflatex -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) { throw "First pdflatex pass failed." }
    & $bibtex main
    if ($LASTEXITCODE -ne 0) { throw "BibTeX pass failed." }
    & $pdflatex -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) { throw "Second pdflatex pass failed." }
    & $pdflatex -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) { throw "Final pdflatex pass failed." }
    Write-Output "Compiled: $paperDirectory\main.pdf"
}
finally {
    Pop-Location
}
