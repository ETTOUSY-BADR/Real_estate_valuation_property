# Phase 4 — nationwide France

Phase 4 extends the project from Paris to one region-aware model covering every
department represented in the national geolocated DVF source. It deliberately
does not copy Paris-only transport, school, park, or noise variables into other
regions. The national contract instead uses property, calendar, global spatial,
territorial, H3, and strictly lagged comparable-sale features.

The evaluation protocol remains unchanged: 2021–2023 development, 2024 model
selection and interval calibration, and an untouched 2025 test. The production
artifact is then refitted on all available 2021–2025 observations.

The completed empirical run retained 1,850,486 sales across 97 departments and
17 regions. On 350,399 untouched 2025 transactions, the selected model records
EUR 41,936 MAE, 13.50% median APE and R² = 0.849, improving MAE by EUR 10,957
over the training-only hierarchical median baseline. The paired 95% interval for
that reduction is EUR 10,763–11,114.

The pre-training audit selected two descriptive department clusters from
2021–2023 profiles. Adding their label made 2024 validation MAE EUR 14 worse,
so the production model correctly excludes the cluster feature. Median
imputation and standardization are used only inside K-means; CatBoost receives
its numeric features without normalization. Terrain surface is preserved in the
Gold audit but excluded from modeling because it is 99.44% missing.

Geo-DVF does not contain open transaction records for Moselle (57), Bas-Rhin
(67), Haut-Rhin (68), or Mayotte (976). Phase 4 reports those gaps explicitly.
It must therefore be described as nationwide DVF coverage, not universal
coverage of every French department.

Run the pipeline from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_phase4.ps1
```

The five compressed national snapshots are large, and feature construction plus
CatBoost training is expected to be a long-running, memory-intensive workflow.
Use the individual module commands documented in the root README to resume a
completed stage.

The visualization stage generates nine national audit and benchmark figures.
`python -m paris_avm.reporting.phase4` then creates the detailed
`PHASE4_REPORT.md` and empirical LaTeX manuscript; compile the paper with
`docs/paper/compile_phase4.ps1`. Use `--allow-incomplete` on the reporting
command before training to produce an explicitly labeled methodology edition
without inventing results.
