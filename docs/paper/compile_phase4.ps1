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
    & $PdfLaTeX -interaction=nonstopmode -halt-on-error phase4_national_report.tex
    if ($LASTEXITCODE -ne 0) { throw "First Phase 4 pdflatex pass failed." }
    & $PdfLaTeX -interaction=nonstopmode -halt-on-error phase4_national_report.tex
    if ($LASTEXITCODE -ne 0) { throw "Second Phase 4 pdflatex pass failed." }
    & $PdfLaTeX -interaction=nonstopmode -halt-on-error phase4_national_report.tex
    if ($LASTEXITCODE -ne 0) { throw "Final Phase 4 pdflatex pass failed." }
    Write-Output "Compiled: $paperDirectory\phase4_national_report.pdf"
}
finally {
    Pop-Location
}
