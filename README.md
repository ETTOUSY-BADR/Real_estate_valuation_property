# Paris AVM — leakage-aware apartment valuation

An applied machine-learning project for Paris apartment valuation using French
DVF transactions enriched with BAN address identity, BDNB building attributes,
dated DPE records, public transport, schools, parks, services and strategic
noise maps.

The project follows a chronological research protocol:

- **2021–2023:** development training;
- **2024:** model and feature selection;
- **2025:** frozen final evaluation;
- **target:** total apartment transaction value in euros.

## Current result

The selected Phase 3 CatBoost model reaches **EUR 93,136 MAE** on 28,330 sales
from 2025, versus **EUR 96,428** for the Phase 2 reference. The paired reduction
is **EUR 3,293 (3.42%)**, with a 95% bootstrap interval of
**EUR 2,735–3,846**.

Entity resolution is near-complete: exact BAN address coverage is 99.49%, BDNB
building coverage is 99.997%, and 76.85% of all historical sales receive a DPE
that existed by the transaction date. The audited future-DPE leakage count is
zero.

## Repository structure

```text
.
├── configs/                 Project and temporal-split configuration
├── data/
│   ├── raw/dvf/             Immutable annual DVF source files
│   ├── bronze/              Versioned external source snapshots
│   ├── silver/              Entity-resolution audit tables
│   └── gold/                Model-ready feature tables
├── docs/
│   ├── paper/               LaTeX manuscripts and compiled PDFs
│   └── reference/           Source specification documents
├── models/                  Serialized fitted model artifacts
├── reports/                 Metrics, predictions, manifests and audits
├── scripts/                 Reproducible PowerShell workflows
├── src/paris_avm/           Installable Python package
├── tests/                   Fast artifact and leakage checks
├── visuals/                 Publication and diagnostic figures
├── pyproject.toml           Package and command metadata
└── requirements.txt         Runtime dependencies
```

## Environment setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

An editable installation exposes commands such as
`paris-avm-build-phase3`, `paris-avm-benchmark-phase3` and
`paris-avm-predict-phase3`. Without installation, modules can be run by setting:

```powershell
$env:PYTHONPATH = '.\src'
```

## Reproduce Phase 3

Run the complete acquisition, feature, benchmark, figure and paper workflow:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_phase3.ps1
```

When the source snapshots, Gold table and fitted benchmark already exist, only
regenerate the figures and paper:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_phase3.ps1 `
  -SkipAcquisition -SkipFeatureBuild -SkipBenchmark
```

Individual module commands are also available:

```powershell
$env:PYTHONPATH = '.\src'
python -m paris_avm.data.acquire_phase3 --skip-dpe
python -m paris_avm.features.phase3
python -m paris_avm.modeling.benchmark_phase3
python -m paris_avm.visualization.phase3
```

The `--skip-dpe` option is intentional: dated ADEME DPE rows and official
DPE-to-building relations are contained in the versioned BDNB Paris archive.

## Estimate one apartment

Phase 3 inference requires a deterministic DVF/FANTOIR address identity and a
valuation date within the model's evaluated 2025 period:

```powershell
$env:PYTHONPATH = '.\src'
python -m paris_avm.inference.phase3 `
  --surface 58 --rooms 3 --postal-code 75008 --commune-code 75108 `
  --street-code 2576 --address-number 29 `
  --latitude 48.878673 --longitude 2.302806 --lots 2 --date 2025-01-02
```

For repeated street numbers, `--address-suffix` accepts either compact codes or
common forms such as `bis`, `ter` and `quater`.

The estimator resolves BAN and BDNB identity, uses only DPE available by the
valuation date, verifies that supplied coordinates are consistent with the
resolved address, recomputes lagged comparable-sale evidence and warns that
most building/context attributes use a retrospective 2026 static snapshot.

## Verification

Run the fast project checks after making changes:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\verify_project.ps1
```

The checks validate Python imports, row identity, duplicate mutations, DPE
temporal leakage, model metadata, reported improvement and transfer artifacts.
In a fresh clone, checks that require generated data and model files are skipped
with an explicit message. Run the Phase 3 pipeline to enable the complete
artifact-integrity suite.

## Research deliverables

- [Phase 3 paper](docs/paper/phase3_paper.pdf)
- [Phase 3 technical report](reports/phase3/PHASE3_REPORT.md)

The source manifest, feature-quality contract, model comparison and publication
figures are generated locally under `reports/phase3/` and `visuals/phase3/`.
They are intentionally excluded from Git because they are reproducible outputs.

Earlier Phase 1 and Phase 2 experiments remain available in the same package,
with their reports, models and papers preserved for traceability.

## Important interpretation

Dated DPE and historical comparable-sale features are point-in-time. Current
BAN/BDNB physical fields, transport, amenities and the 2022 noise map are
retrospective reconstruction. Public data did not provide reliable apartment
floor or elevator fields, so both are explicitly unavailable rather than
fabricated. Predictions are statistical indications, not certified appraisals
or investment guarantees.
