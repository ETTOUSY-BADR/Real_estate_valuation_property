# Phase 2 data contract

The original annual CSV files remain immutable source inputs. Their sizes and
SHA-256 hashes are recorded in `../reports/phase2/feature_quality.json`.

The analytical Gold table is `gold/phase2_sale_features.parquet`:

- one row per retained single-apartment sale mutation;
- date range: 2021-01-04 through 2025-12-31;
- 143,009 rows and 60 typed columns;
- geographic storage coordinates: WGS84 (`latitude`, `longitude`);
- metric spatial coordinates: Lambert-93 (`x_l93`, `y_l93`);
- H3 cells: resolutions 8, 9 and 10;
- historical comparable windows: up to 1,000 m and 730 days;
- strict information-availability lag: at least 90 days.

The source of truth for the complete schema, source hashes, row counts, feature
ranges and annual coverage is `../reports/phase2/feature_quality.json`.
The model contract contains 44 Phase 2 inputs; mutation, parcel and high-cardinality
address identifiers remain audit columns, while the current transaction's
`price_per_m2` is explicitly excluded from model inputs.

Rebuild the table from the project root with:

```powershell
$env:PYTHONPATH = '.\src'
python -m paris_avm.features.phase2
```

## Phase 3 data contract

`gold/phase3_sale_features.parquet` preserves the same 143,009 mutation rows
and adds BAN/BDNB/DPE, accessibility, amenity and noise features (141 columns
total). `silver/phase3_match_audit.parquet` records the BAN and building match
method, candidate counts and confidence for every transaction.

Raw snapshots are versioned under `bronze/`; exact source provenance and
checksums are in `../reports/phase3/source_manifest.json`. DPE fields are joined
only when the diagnostic date is no later than the sale. BAN, BDNB physical,
transport, amenity and noise fields are current/static retrospective context.
See `../reports/phase3/feature_quality.json` for match coverage, missingness and
the complete new-column contract.

## Complete preprocessing command

Build both contracts in sequence with:

```powershell
$env:PYTHONPATH = '.\src'
python -m paris_avm.features.preprocessing
```

The complete feature contract has 100 model inputs across property, time,
spatial, historical-comparable, building, point-in-time DPE and neighborhood
groups, including three externally resolved BAN/BDNB identity fields used by
the selected Phase 3 model. The run-level provenance report is written to
`../reports/preprocessing/run_summary.json`.

## Phase 4 nationwide contract

`gold/phase4_france_sale_features.parquet` is built from the compressed annual
files under `raw/dvf_france/`. It covers every department available in Geo-DVF,
uses spherical coordinates (including overseas departments), and computes
90-day-lagged nearby comparables independently within each department. Source
hashes, territorial coverage, cleaning counts, and comparable coverage are
written to `../reports/phase4/feature_quality.json`.
