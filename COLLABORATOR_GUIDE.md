# Collaborator handoff: trained models and results

This snapshot publishes all 15 locally saved model bundles, the generated
metrics/comparisons/predictions, diagnostic figures, and three prepared feature
tables. Raw source datasets, downloaded archives, caches, temporary files and
build logs remain excluded. Existing papers and source code remain available.

## Download

Install Git LFS before cloning, then run:

```powershell
git lfs install
git clone --branch feature/dvf-preprocessing https://github.com/ETTOUSY-BADR/Real_estate_valuation_property.git
cd Real_estate_valuation_property
git lfs pull
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

For an existing clone, switch to the published branch, pull normally, then run
`git lfs pull`. Model and Parquet contents use approximately 1.3 GiB of LFS
storage/download; allow additional disk space for the checkout and LFS cache.
A small text file beginning with `version https://git-lfs.github.com/spec/v1`
is an LFS pointer, not the actual model: complete `git lfs pull` first.

## Start here

| Purpose | Files |
| --- | --- |
| Selected Paris model | `models/phase3/phase3_selected_model.joblib` |
| Nationwide production model | `models/phase4/france_catboost.joblib` |
| Earlier models and baselines | `models/valuation_model.joblib`, `models/benchmark/`, `models/phase2/` |
| Prepared inference/history tables | `data/gold/phase2_sale_features.parquet`, `data/gold/phase3_sale_features.parquet`, `data/gold/phase4_france_sale_features.parquet` |
| Paris evaluation | `reports/phase3/phase3_results.json`, `phase3_model_comparison.csv`, `phase3_test_predictions.csv` |
| Nationwide evaluation | `reports/phase4/benchmark.json`, `benchmark_comparison.csv`, `test_predictions.parquet` |
| Feature contracts and provenance | `reports/preprocessing/`, `reports/phase2/feature_quality.json`, `reports/phase3/source_manifest.json`, phase-specific feature-quality reports |
| Visual summaries | `visuals/` and the PDFs in `docs/paper/` |

The joblib files contain fitted model bundles, not merely hyperparameters. For
these tree-based/scikit-learn models, these files are the equivalent of saved
trained weights. Load only trusted joblib files.

```powershell
.\.venv\Scripts\python.exe -m paris_avm.inference.phase3 --help
.\.venv\Scripts\python.exe -m paris_avm.inference.phase4 --help
```

Use the required property/date/address inputs shown by each command. The
published Gold tables supply historical comparables and feature lookup data.
Run from the repository root. `README.md` and `data/README.md` describe the
feature contracts and rebuild workflows. Full raw-data regeneration still
requires acquiring the source snapshots; they are not included in this handoff.
Historical absolute Windows paths in result JSON are provenance, not portable
file locations. Use the corresponding paths relative to the repository root.

## Evaluation model versus production model

The Phase 3 saved selected-model bundle records training on 2021-2024 and testing
on 2025. Inspect its embedded feature list and categorical feature definitions
before manually calling `predict`.

The Phase 4 `france_catboost.joblib` bundle is a **production refit on all available
rows, including 2025**. Its separate benchmark metrics and test predictions were
computed earlier using a model fitted only through 2024. Do not evaluate the
production refit on 2025 and call that an untouched holdout result. The earlier
Phase 4 evaluation estimator is not separately saved by the current workflow;
reproducing that fit requires rerunning the benchmark. Its predictions and metrics
are included here without retraining or modification.

Dependency ranges are recorded in `requirements.txt`; this handoff does not claim
to recover an exact training-environment lockfile. Serialized estimator
compatibility may depend on the original library versions. No models were
retrained to prepare this upload.

## Integrity

`collaboration/artifact_manifest.json` records each published model, feature
table, numeric report and figure with its byte size and SHA-256 digest. Verify
all entries after downloading with this PowerShell snippet from the repo root:

```powershell
$manifest = Get-Content collaboration/artifact_manifest.json -Raw | ConvertFrom-Json
foreach ($entry in $manifest.files) {
    $item = Get-Item -LiteralPath $entry.path
    $hash = (Get-FileHash -LiteralPath $entry.path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($item.Length -ne $entry.bytes -or $hash -ne $entry.sha256) {
        throw "Artifact mismatch: $($entry.path)"
    }
}
Write-Output 'All artifact checksums match.'
```

Inspect the saved evaluations without retraining. To change models, retain the
chronological development/validation/test separation and use new output
locations so that the published benchmark remains reproducible.