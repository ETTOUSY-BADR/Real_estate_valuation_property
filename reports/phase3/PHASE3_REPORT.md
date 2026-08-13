# Phase 3 — Building, energy, accessibility and neighborhood enrichment

## Outcome

Phase 3 links the 143,009-row Paris DVF Gold cohort to authoritative address,
building, energy, transport, school, green-space, shop/service and strategic
noise sources. The selected CatBoost model reduces 2025 MAE from **EUR
96,428.47** for the frozen Phase 2 comparable correction to **EUR 93,135.57**.
The reduction is **EUR 3,292.89 (3.41%)**, with a paired 95% bootstrap interval
of **EUR 2,735.12–3,845.97**.

| Method | 2025 MAE | MdAPE | Within ±20% | R² |
|---|---:|---:|---:|---:|
| CatBoost, full identity/enrichment | **EUR 93,136** | **11.70%** | **73.39%** | **0.8679** |
| LightGBM, full without identity | EUR 93,968 | 11.80% | 73.03% | 0.8661 |
| LightGBM, building only | EUR 94,571 | 11.90% | 72.28% | 0.8658 |
| LightGBM, dated DPE only | EUR 95,222 | 11.97% | 72.28% | 0.8637 |
| LightGBM, neighborhood context only | EUR 95,632 | 11.99% | 71.74% | 0.8627 |
| Frozen Phase 2 correction | EUR 96,428 | 12.22% | 71.69% | 0.8601 |

All model and feature choices use 2024 validation data. The selected candidates
are refitted on 2021–2024 and evaluated on the 28,330-row 2025 cohort.

## Publication paper and figures

The detailed standalone manuscript is `docs/paper/phase3_paper.tex`; the compiled
12-page deliverable is `docs/paper/phase3_paper.pdf`. It contains the full source and
entity-resolution methodology, temporal feature contract, ablations, paired
uncertainty, spatial/segment diagnostics, limitations and transfer package.

Publication figures are generated reproducibly by
`python -m paris_avm.visualization.phase3` and
stored in `visuals/phase3/`: the data architecture, executive result, prediction
diagnostics, spatial performance, enrichment profile and feature importance.

## Sources added

- Base Adresse Nationale: exact normalized address identity and coordinates;
- BDNB 2026-02.a: building identity, footprint, levels, height, construction
  year/period, dwelling/lot counts, wall/roof fields and building state;
- ADEME DPE data embedded in BDNB: labels, consumption, emissions, heating,
  construction and surface fields;
- Île-de-France Mobilités: distance and density by Metro, RER, train and tram;
- Ville de Paris: school and public green-space proximity/density;
- BDNB BPE: nearby shop/service equipment counts and distance;
- DRIEAT strategic noise maps: 2022 road, RATP-rail and SNCF-rail Lden bands.

Exact URLs, producer, licence, retrieval timestamp, byte size and SHA-256 are
recorded in `source_manifest.json`. Raw inputs are immutable in `data/bronze/`.

## Entity resolution and quality

BAN matching uses the exact commune + FANTOIR road code + address number +
suffix key. BAN-to-BDNB relations are ranked by the official relationship
reliability score; cadastral parcel linkage is the fallback.

| Quality check | Result |
|---|---:|
| Exact BAN match | 99.49% |
| Any BDNB building match | 99.997% |
| High-confidence BDNB match | 99.46% |
| DPE available by sale date | 76.85% |
| Future-DPE leakage rows | **0** |
| Gold rows after all joins | **143,009** |

The row-level audit is `data/silver/phase3_match_audit.parquet`. The Gold table
is `data/gold/phase3_sale_features.parquet` (141 columns).

## Temporal interpretation

DPE is point-in-time: only a diagnostic with
`date_etablissement_dpe <= date_mutation` can join a historical transaction.
Historical comparable features retain Phase 2's 90-day availability lag.

BAN, most BDNB physical fields, the 2022 noise map, current stations, schools,
parks and services are **retrospective static reconstruction**. They are useful
for reconstructing a building/location but do not constitute fully vintage-
correct historical data. For this reason:

- the dated-DPE ablation is reported separately;
- static building/context ablations are reported separately;
- the selected full model artifact carries a `retrospective_static_context`
  flag;
- a prospective 2026+ cohort remains necessary for a strong deployment claim.

## Requested fields that remain unavailable

The current public BAN/BDNB/DPE sources do not expose a reliable apartment
floor or elevator field for this cohort. The Gold contract includes explicit
`apartment_floor_available=0` and `elevator_source_available=0` indicators;
values are left missing rather than fabricated. Building level count is not a
substitute for apartment floor. Future floor/elevator enrichment requires a
lawful notarial, property-manager, copropriété or licensed listing source.

## Reproduce

```powershell
$env:PYTHONPATH = '.\src'
python -m paris_avm.data.acquire_phase3 --skip-dpe
python -m paris_avm.features.phase3
python -m paris_avm.modeling.benchmark_phase3
python -m paris_avm.visualization.phase3
powershell -ExecutionPolicy Bypass -File .\docs\paper\compile_phase3.ps1
```

The `--skip-dpe` flag is intentional: ADEME DPE rows and official DPE↔building
relations are already contained in the versioned Paris BDNB archive.

Single-property inference requires the DVF/FANTOIR street code so address
identity is deterministic. Valuation dates are restricted to the evaluated
2025 period; prospective use requires a newly validated model:

```powershell
$env:PYTHONPATH = '.\src'
python -m paris_avm.inference.phase3 `
  --surface 58 --rooms 3 --postal-code 75008 --commune-code 75108 `
  --street-code 2576 --address-number 29 `
  --latitude 48.878673 --longitude 2.302806 --date 2025-01-02
```

The command rejects supplied coordinates more than 250 m from the resolved
canonical address to prevent mixing one building with another neighborhood's
context and comparable sales.

This is an automated statistical indication, not a certified appraisal.
