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
