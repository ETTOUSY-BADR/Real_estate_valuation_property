# Phase 2 — Leakage-safe spatial and comparable-sale valuation

## Outcome

Phase 2 converts the cleaned 2021–2025 DVF apartment transactions into an
auditable Gold feature table and tests whether local historical market evidence
improves the Phase 1 valuation model.

The selected method is deliberately simple:

```text
Phase 2 estimate = 75% × Phase 1 LightGBM estimate
                 + 25% × (local historical median EUR/m² × surface)
```

The local median contains only apartment sales within 500 metres, occurring
90–365 days before the valuation date. The feature and the 75/25 weight were
selected on 2024 validation data. They were then frozen and evaluated on 28,330
sales from 2025.

| Metric | Phase 1 LightGBM | Phase 2 correction | Change |
|---|---:|---:|---:|
| MAE | EUR 96,980.69 | **EUR 96,428.47** | **−EUR 552.22** |
| MAE 95% bootstrap interval | EUR 94,935–99,014 | EUR 94,383–98,500 | — |
| Paired MAE reduction 95% interval | — | **EUR 333–743** | favors Phase 2 |
| Median absolute percentage error | 12.33% | **12.22%** | −0.11 points |
| Within ±20% | 71.41% | **71.69%** | +0.28 points |
| R² | **0.8619** | 0.8601 | −0.0018 |

This is a small improvement, not a breakthrough. It reduces typical absolute
and percentage errors, but slightly reduces R². The paired interval indicates
that the MAE improvement is consistent on this particular test cohort.

## Data engineering delivered

The reproducible feature build is implemented in `build_phase2_features.py`.
It:

1. validates and combines the five annual source files;
2. retains mutation, address, street, commune and parcel identifiers;
3. transforms WGS84 coordinates to Lambert-93 for metric distances;
4. creates H3 cells at resolutions 8, 9 and 10;
5. computes local transaction counts and median EUR/m² values over multiple
   spatial and temporal windows;
6. creates distance/time-weighted and size/room-similar comparable estimates;
7. enforces a 90-day information-availability lag for every comparable; and
8. writes a typed Parquet Gold table plus a machine-readable quality report.

Key artifacts:

- `data/gold/phase2_sale_features.parquet`: 143,009 rows, 60 columns;
- `reports/phase2/feature_quality.json`: hashes, counts, coverage and contract;
- `reports/phase2/phase2_test_predictions.csv`: row-level 2025 predictions;
- `reports/phase2/phase2_model_comparison.csv`: metrics and bootstrap intervals;
- `models/phase2/`: six fitted enhanced model artifacts.

Comparable coverage is 100% for the weighted estimator from 2022 onward. In
2025, the median property has 243 eligible prior sales within 500 metres and 836
within one kilometre over the trailing 365-day window.

## Benchmark design

The chronology is fixed:

- 2021–2023: development training (90,275 sales);
- 2024: model/configuration/weight selection (24,404 sales);
- 2021–2024: final refit (114,679 sales);
- 2025: final test only (28,330 sales).

LightGBM and XGBoost were tested with three target formulations:

- log total price;
- log price per square metre;
- log residual relative to the 500 m / 365-day local median.

Additional ablations tested spatial-only and comparable-only feature groups.
The benchmark also tested a validation-weighted six-model blend.

| Rank | Method | 2025 MAE | MdAPE | Within ±20% |
|---:|---|---:|---:|---:|
| 1 | Phase 1 + historical comparable correction | **EUR 96,428** | **12.22%** | **71.69%** |
| 2 | XGBoost, total-price target | EUR 96,914 | 12.43% | 71.18% |
| 3 | Phase 1 LightGBM reference | EUR 96,981 | 12.33% | 71.41% |
| 4 | XGBoost, EUR/m² target | EUR 96,996 | 12.61% | 70.73% |
| 5 | Six-model blend | EUR 97,183 | 12.52% | 70.89% |
| 6 | XGBoost, residual target | EUR 97,507 | 12.60% | 70.69% |
| 7 | LightGBM spatial-only ablation | EUR 98,232 | 12.72% | 70.31% |
| 8 | LightGBM comparable-only ablation | EUR 98,963 | 12.73% | 70.81% |
| 9 | LightGBM full total-price model | EUR 99,144 | 12.73% | 69.89% |

The full enriched boosters did not outperform Phase 1. More features increased
model freedom without providing the missing building-level characteristics.
The transparent correction worked because it adds a small amount of recent,
local price information while retaining the stable Phase 1 estimate.

## Reproduce Phase 2

```powershell
$env:PYTHONPATH = '.\src'
python -m paris_avm.features.phase2
python -m paris_avm.modeling.benchmark_phase2
python -m paris_avm.visualization.phase2
powershell -ExecutionPolicy Bypass -File .\docs\paper\compile_phase2.ps1
```

Estimate one property:

```powershell
$env:PYTHONPATH = '.\src'
python -m paris_avm.inference.phase2 `
  --surface 50 --rooms 2 --postal-code 75011 `
  --latitude 48.859 --longitude 2.379 `
  --address-number 25 --lots 2 --date 2025-12-31
```

## Interpretation and remaining limitations

- DVF does not describe floor, lift, condition, view, noise, renovation quality,
  energy performance or detailed building state.
- Gain importance is diagnostic, not causal evidence.
- 2025 is computationally held out here, but it was already inspected during
  Phase 1. A genuinely prospective 2026 or later cohort is needed before making
  a strong external-performance claim.
- The confidence interval measures sampling variation within the 2025 cohort;
  it does not cover future market-regime change.
- The current estimator is an automated statistical indication, not a certified
  appraisal or investment guarantee.

## Recommended Phase 3

The next useful step is not an attention mechanism. It is reliable external
data linkage: BAN address normalization, BDNB building attributes, DPE energy
ratings, cadastral/building age, floor/elevator information where legally
available, and distance to transit/amenities. Those features target the largest
known information gaps and should again be evaluated with chronological and
spatial holdouts.
