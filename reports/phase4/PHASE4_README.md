# Phase 4 — nationwide France

Phase 4 extends the project from Paris to one region-aware model covering every
department represented in the national geolocated DVF source. It deliberately
does not copy Paris-only transport, school, park, or noise variables into other
regions. The national contract instead uses property, calendar, global spatial,
territorial, H3, and strictly lagged comparable-sale features.

The evaluation protocol remains unchanged: 2021–2023 development, 2024 model
selection and interval calibration, and an untouched 2025 test. The production
artifact is then refitted on all available 2021–2025 observations.

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

The final stage runs `python -m paris_avm.visualization.phase4`, generating the
detailed `PHASE4_REPORT.md` and the national coverage, performance, residual,
regional, feature-importance and comparable-density dashboards. Run it with
`--allow-incomplete` before training to produce an explicitly labeled
methodology report and artifact inventory.
