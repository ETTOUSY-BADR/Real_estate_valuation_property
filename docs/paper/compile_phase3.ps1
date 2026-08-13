param(
    [string]$PdfLaTeX = "pdflatex"
)

$ErrorActionPreference = "Stop"
$paperDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Get-Command $PdfLaTeX -ErrorAction SilentlyContinue)) {
    throw "'$PdfLaTeX' was not found. Install a TeX distribution or pass -PdfLaTeX with its path."
}

Push-Location $paperDirectory
try {
    & $PdfLaTeX -interaction=nonstopmode -halt-on-error phase3_paper.tex
    if ($LASTEXITCODE -ne 0) { throw "First pdflatex pass failed." }
    & $PdfLaTeX -interaction=nonstopmode -halt-on-error phase3_paper.tex
    if ($LASTEXITCODE -ne 0) { throw "Final pdflatex pass failed." }
    Write-Output "Compiled: $paperDirectory\phase3_paper.pdf"
}
finally {
    Pop-Location
}
