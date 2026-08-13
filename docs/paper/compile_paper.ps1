param(
    [string]$PdfLaTeX = "pdflatex",
    [string]$BibTeX = "bibtex"
)

$ErrorActionPreference = "Stop"
$paperDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Get-Command $PdfLaTeX -ErrorAction SilentlyContinue)) {
    throw "'$PdfLaTeX' was not found. Install a TeX distribution or pass -PdfLaTeX with its path."
}
if (-not (Get-Command $BibTeX -ErrorAction SilentlyContinue)) {
    throw "'$BibTeX' was not found. Install a TeX distribution or pass -BibTeX with its path."
}

Push-Location $paperDirectory
try {
    & $PdfLaTeX -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) { throw "First pdflatex pass failed." }
    & $BibTeX main
    if ($LASTEXITCODE -ne 0) { throw "BibTeX pass failed." }
    & $PdfLaTeX -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) { throw "Second pdflatex pass failed." }
    & $PdfLaTeX -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) { throw "Final pdflatex pass failed." }
    Write-Output "Compiled: $paperDirectory\main.pdf"
}
finally {
    Pop-Location
}
