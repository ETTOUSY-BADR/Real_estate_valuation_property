"""Generate the artifact-backed Phase 4 nationwide LaTeX manuscript."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

from paris_avm.artifacts import write_atomic
from paris_avm.features.phase4 import (
    PHASE4_CATEGORICAL_FEATURES,
    PHASE4_MODEL_FEATURES,
    PHASE4_NUMERIC_FEATURES,
)
from paris_avm.geography import KNOWN_DVF_GAPS, REGION_NAMES
from paris_avm.paths import PROJECT_ROOT


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def latex_fraction_percent(value: object, digits: int = 2) -> str:
    """Format a 0--1 fraction as a LaTeX-safe percentage."""
    return f"{float(value) * 100:.{digits}f}\\%"


def metric_table(benchmark: dict[str, Any] | None) -> str:
    if benchmark is None:
        return r"""
\begin{statusbox}
\textbf{Benchmark pending.} The national Gold table and fitted model are not
present in this workspace. No accuracy number is asserted in this edition.
The registered comparison is Phase 4 CatBoost versus the hierarchical
training-only geographic median on the identical untouched 2025 transactions.
\end{statusbox}
"""
    model = benchmark["test_metrics"]
    baseline = benchmark["baseline_test_metrics"]
    bootstrap = benchmark.get("paired_bootstrap", {})
    reduction_interval = "not available"
    if bootstrap:
        reduction_interval = (
            f"EUR {bootstrap['reduction_ci95_lower_eur']:,.0f}--"
            f"{bootstrap['reduction_ci95_upper_eur']:,.0f}"
        )
    return rf"""
\begin{{table}}[H]
\centering
\caption{{Frozen 2025 national benchmark on identical apartment transactions.}}
\label{{tab:national-benchmark}}
\begin{{tabular}}{{lrrrr}}
\toprule
Method & MAE & RMSE & MdAPE & Within $\pm20\%$ \\
\midrule
Phase 4 CatBoost & EUR {model['mae_eur']:,.0f} & EUR {model['rmse_eur']:,.0f} & {model['median_absolute_percentage_error']:.2f}\% & {latex_fraction_percent(model['within_20_percent'])} \\
Hierarchical median baseline & EUR {baseline['mae_eur']:,.0f} & EUR {baseline['rmse_eur']:,.0f} & {baseline['median_absolute_percentage_error']:.2f}\% & {latex_fraction_percent(baseline['within_20_percent'])} \\
\bottomrule
\end{{tabular}}
\end{{table}}

The paired MAE reduction is EUR {benchmark['mae_reduction_vs_baseline_eur']:,.0f};
its deterministic paired-bootstrap 95\% interval is {reduction_interval}.
"""


def executive_status(benchmark: dict[str, Any] | None) -> str:
    if benchmark is None:
        return r"""
\begin{statusbox}
\textbf{Protocol edition.} The national Gold table and fitted model are not
present in this workspace. No accuracy number is asserted. The manuscript will
be regenerated from the frozen benchmark artifacts without manual score entry.
\end{statusbox}
"""
    model = benchmark["test_metrics"]
    return rf"""
\begin{{center}}
\metriccard{{EUR {model['mae_eur']:,.0f}}}{{2025 model MAE}} \hfill
\metriccard{{{model['median_absolute_percentage_error']:.2f}\%}}{{2025 median APE}} \hfill
\metriccard{{{latex_fraction_percent(model['within_20_percent'])}}}{{within $\pm 20\%$}} \hfill
\metriccard{{{latex_fraction_percent(benchmark['prediction_interval']['observed_test_coverage'])}}}{{90\% interval coverage}}
\end{{center}}
"""


def validation_table(trials: pd.DataFrame | None) -> str:
    if trials is None or trials.empty:
        return r"""
\begin{table}[H]
\centering
\caption{Predeclared CatBoost and clustering ablations for 2024 model selection.}
\begin{tabular}{cllll}
\toprule
Candidate & Depth & Loss & Cluster & Regularization \\
\midrule
1 & 8 & MAE & no & $\ell_2=8$ \\
2 & 9 & Huber ($\delta=1$) & no & $\ell_2=10$ \\
3 & 8 & MAE & yes & $\ell_2=8$ \\
4 & 9 & Huber ($\delta=1$) & yes & $\ell_2=10$ \\
\bottomrule
\end{tabular}
\end{table}
"""
    rows: list[str] = []
    for _, row in trials.sort_values("mae_eur").iterrows():
        try:
            configuration = json.loads(row["configuration"])
            description = (
                f"depth={configuration.get('depth')}, "
                f"loss={configuration.get('loss_function')}, "
                f"l2={configuration.get('l2_leaf_reg')}"
            )
        except (json.JSONDecodeError, TypeError):
            description = str(row["configuration"])
        cluster_value = row.get("market_cluster_feature", False)
        uses_cluster = (
            cluster_value.strip().lower() in {"true", "1", "yes"}
            if isinstance(cluster_value, str)
            else bool(cluster_value)
        )
        rows.append(
            f"{int(row['candidate'])} & {latex_escape(description)} & "
            f"{'yes' if uses_cluster else 'no'} & "
            f"{int(row.get('clipped_predictions', 0)):,} & "
            f"EUR {row['mae_eur']:,.0f} & {row['median_absolute_percentage_error']:.2f}\\% & "
            f"{latex_fraction_percent(row['within_20_percent'])} \\\\"
        )
    body = "\n".join(rows)
    return rf"""
\begin{{table}}[H]
\centering
\caption{{CatBoost candidate performance on the 2024 selection cohort.}}
\begin{{tabularx}}{{\textwidth}}{{cXcrrrr}}
\toprule
Candidate & Configuration & Cluster & Clipped & MAE & MdAPE & Within $\pm20\%$ \\
\midrule
{body}
\bottomrule
\end{{tabularx}}
\end{{table}}
"""


def validation_notes(trials: pd.DataFrame | None) -> str:
    if trials is None or trials.empty or "clipped_predictions" not in trials:
        return ""
    clipped = trials.loc[trials["clipped_predictions"].gt(0)]
    if clipped.empty:
        return "No validation prediction required output bounding."
    details = ", ".join(
        f"candidate {int(row['candidate'])}: {int(row['clipped_predictions']):,}"
        for _, row in clipped.sort_values("candidate").iterrows()
    )
    return (
        "The Huber candidates generated extreme inverse-log outputs. The registered "
        f"transaction-value bounds prevented non-finite metrics ({details} clipped predictions), "
        "and their resulting validation errors rejected them from selection."
    )


def preflight_section(
    preflight: dict[str, Any] | None,
    visual_dir: Path,
    trials: pd.DataFrame | None = None,
) -> str:
    figures = "\n".join(
        filter(
            None,
            [
                optional_figure(
                    visual_dir,
                    "phase4_preflight_data_audit.png",
                    "National pre-training audit: annual volume, price distribution, missingness and regional balance.",
                    "preflight-audit",
                ),
                optional_figure(
                    visual_dir,
                    "phase4_department_clusters.png",
                    "Development-only department market-structure clusters and their geographic distribution.",
                    "department-clusters",
                ),
            ],
        )
    )
    if preflight is None:
        return r"""
\begin{statusbox}
The empirical preflight is pending the national Gold table. The registered audit
covers temporal and territorial balance, feature missingness, lagged-comparable
coverage and development-only department clustering before model training.
\end{statusbox}
"""
    clustering = preflight["clustering"]
    ablation = (
        "The predictive ablation is pending."
    )
    if trials is not None and not trials.empty and "market_cluster_feature" in trials:
        flags = trials["market_cluster_feature"].map(
            lambda value: value if isinstance(value, bool) else str(value).lower() == "true"
        )
        base = trials.loc[~flags, "mae_eur"].min()
        clustered = trials.loc[flags, "mae_eur"].min()
        if pd.notna(base) and pd.notna(clustered):
            difference = float(clustered - base)
            decision = "not retained" if difference >= 0 else "retained"
            ablation = (
                f"The best clustered candidate records EUR {clustered:,.0f} validation MAE versus "
                f"EUR {base:,.0f} without clustering (difference EUR {difference:,.0f}); the "
                f"cluster feature is therefore {decision}."
            )
    return rf"""
The empirical preflight covers {preflight['rows']:,} retained transactions,
{preflight['departments']} represented departments and {preflight['regions']}
regions. Weighted lagged-comparable coverage is
{latex_fraction_percent(preflight['weighted_comparable_coverage'])}. The median transaction value is
EUR {preflight['median_transaction_value_eur']:,.0f} and median price per square
metre is EUR {preflight['median_price_per_m2_eur']:,.0f}.

K-means is fitted only to 2021--2023 department profiles. Inputs summarize
transaction volume, apartment structure and strictly lagged comparable evidence;
the current transaction target is excluded. Median imputation and standardization
apply only inside clustering, not to CatBoost. Silhouette selection retains
{clustering['selected_clusters']} clusters. These groups are descriptive until
the with/without-cluster 2024 ablation demonstrates predictive value.

{latex_escape(ablation)}

{figures}
"""


def regional_table(region: pd.DataFrame | None) -> str:
    if region is None or region.empty:
        return r"""
\begin{statusbox}
Regional and department metrics are pending the frozen 2025 evaluation. The
reporting contract requires at least 100 test sales before a territorial row is
published and at least 500 calibration sales before a region receives its own
empirical interval ratios.
\end{statusbox}
"""
    data = region.copy()
    if "region_name" not in data:
        data["region_name"] = data["code_region"].map(
            lambda value: REGION_NAMES.get(str(value), str(value))
        )
    rows = []
    for _, row in data.sort_values("mae_eur").iterrows():
        rows.append(
            f"{latex_escape(row['region_name'])} & {int(row['rows']):,} & "
            f"EUR {row['mae_eur']:,.0f} & {row['median_absolute_percentage_error']:.2f}\\% & "
            f"{latex_fraction_percent(row['within_20_percent'])} & {row['r2']:.4f} \\\\"
        )
    return rf"""
\begin{{longtable}}{{p{{4.1cm}}rrrrr}}
\caption{{Frozen-test performance by region.}}\label{{tab:regions}}\\
\toprule
Region & Rows & MAE & MdAPE & Within $\pm20\%$ & $R^2$ \\
\midrule
\endfirsthead
\toprule
Region & Rows & MAE & MdAPE & Within $\pm20\%$ & $R^2$ \\
\midrule
\endhead
{chr(10).join(rows)}
\bottomrule
\end{{longtable}}
"""


def artifact_table(paths: dict[str, Path]) -> str:
    rows = []
    for name, path in paths.items():
        status = "available" if path.exists() else r"\textbf{pending}"
        size = f"{path.stat().st_size:,}" if path.exists() else "---"
        rows.append(f"{latex_escape(name)} & {status} & {size} \\\\")
    return "\n".join(rows)


def optional_figure(visual_dir: Path, filename: str, caption: str, label: str) -> str:
    if not (visual_dir / filename).exists():
        return ""
    return rf"""
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{filename}}}
\caption{{{caption}}}
\label{{fig:{label}}}
\end{{figure}}
"""


def generate_latex(
    output: Path,
    *,
    benchmark_path: Path,
    trials_path: Path,
    region_path: Path,
    quality_path: Path,
    visual_dir: Path,
    artifact_paths: dict[str, Path],
    preflight_path: Path | None = None,
) -> str:
    benchmark = (
        json.loads(benchmark_path.read_text(encoding="utf-8"))
        if benchmark_path.exists()
        else None
    )
    trials = pd.read_csv(trials_path) if trials_path.exists() else None
    region = (
        pd.read_csv(region_path, dtype={"code_region": "string"})
        if region_path.exists()
        else None
    )
    quality = (
        json.loads(quality_path.read_text(encoding="utf-8"))
        if quality_path.exists()
        else None
    )
    preflight = (
        json.loads(preflight_path.read_text(encoding="utf-8"))
        if preflight_path is not None and preflight_path.exists()
        else None
    )
    empirical = benchmark is not None
    edition = "Empirical benchmark edition" if empirical else "Protocol and pre-benchmark edition"
    if empirical:
        model = benchmark["test_metrics"]
        abstract = rf"""This report evaluates one region-aware CatBoost automated valuation model
on {benchmark['rows']['test']:,} untouched 2025 apartment sales represented in nationwide Geo-DVF.
Development uses 2021--2023 and model selection uses 2024. The selected model reaches EUR
{model['mae_eur']:,.0f} MAE, {model['median_absolute_percentage_error']:.2f}\% median absolute
percentage error and $R^2={model['r2']:.4f}$. It improves MAE by EUR
{benchmark['mae_reduction_vs_baseline_eur']:,.0f} relative to a training-only hierarchical
geographic median baseline. All comparable-sale features enforce a minimum 90-day information lag."""
        coverage = rf"""The audited Gold table contains {quality['row_count']:,} sales from
{latex_escape(quality['date_min'])} through {latex_escape(quality['date_max'])}, representing
{len(quality['observed_departments'])} departments and {len(quality['observed_regions'])} regions.""" if quality else "The Gold cohort is recorded in the benchmark artifacts."
        decision_status = rf"""\begin{{itemize}}[leftmargin=1.5em]
\item The data, benchmark and reporting protocols are fully specified and tested.
\item The frozen 2025 result covers {benchmark['rows']['test']:,} transactions; no 2025 outcome influenced model selection.
\item Regional MAE is interpreted beside MdAPE, hit rate and test-row count.
\item The output is an automated indication, not a certified appraisal.
\end{{itemize}}"""
        conclusion_status = rf"""The completed frozen evaluation supports a national
2025 MAE of EUR {model['mae_eur']:,.0f} and a paired improvement of EUR
{benchmark['mae_reduction_vs_baseline_eur']:,.0f} over the training-only baseline.
The empirical claims in this edition are generated directly from the registered
benchmark artifacts; limitations remain especially important for sparse territories
and for applications beyond the 2025 evaluation period."""
    else:
        abstract = r"""This protocol defines a nationwide France apartment automated valuation
benchmark using annual geolocated DVF transactions from 2021--2025. It specifies the cohort,
global spatial representation, strictly lagged comparable-sale features, hierarchical baseline,
CatBoost candidates, chronological split, paired uncertainty analysis and territorial diagnostics
before national model results are available. This edition intentionally reports no accuracy number;
        rerunning the generator after the benchmark replaces pending sections with audited empirical tables."""
        coverage = "Territorial row counts are pending the national Gold feature build."
        decision_status = r"""\begin{itemize}[leftmargin=1.5em]
\item The data, benchmark and reporting protocols are fully specified and tested.
\item National accuracy may be claimed only after the 2025 frozen-test artifacts exist.
\item Regional MAE must be interpreted beside MdAPE, hit rate and test-row count.
\item The output is an automated indication, not a certified appraisal.
\end{itemize}"""
        conclusion_status = r"""Claims in this protocol edition remain limited to the
method and artifacts actually present. Rerunning the identical generator after the
full national benchmark completes creates the empirical edition."""

    figure_sections = "\n".join(
        filter(
            None,
            [
                optional_figure(visual_dir, "phase4_national_coverage.png", "National transaction footprint, retained sales by region and chronological cohorts.", "coverage"),
                optional_figure(visual_dir, "phase4_performance_dashboard.png", "Frozen-test model-versus-baseline benchmark and regional error heterogeneity.", "performance"),
                optional_figure(visual_dir, "phase4_prediction_diagnostics.png", "Out-of-time calibration, residual and value-band diagnostics.", "diagnostics"),
                optional_figure(visual_dir, "phase4_regional_performance.png", "Regional evidence and department-level robustness audit.", "regional"),
                optional_figure(visual_dir, "phase4_feature_importance.png", "CatBoost information usage by feature family and individual input.", "importance"),
                optional_figure(visual_dir, "phase4_comparable_coverage.png", "Lagged comparable-sale availability and density by region.", "comparables"),
            ],
        )
    )
    inventory = artifact_table(artifact_paths)
    created = datetime.now(timezone.utc).strftime("%d %B %Y")
    source = rf"""\documentclass[11pt,a4paper]{{article}}

\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage[margin=1.85cm]{{geometry}}
\usepackage{{amsmath}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{tabularx}}
\usepackage{{longtable}}
\usepackage{{graphicx}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}
\usepackage{{microtype}}
\usepackage{{caption}}
\usepackage{{float}}
\usepackage{{fancyhdr}}
\usepackage{{enumitem}}

\definecolor{{navy}}{{HTML}}{{102A43}}
\definecolor{{blue}}{{HTML}}{{1677B8}}
\definecolor{{green}}{{HTML}}{{2A9D6F}}
\definecolor{{light}}{{HTML}}{{F4F7FA}}
\definecolor{{muted}}{{HTML}}{{627D98}}
\hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue,
pdftitle={{Nationwide France Apartment Valuation: Phase 4 Benchmark}},pdfauthor={{Badr Ettousy}}}}
\graphicspath{{{{../../visuals/phase4/}}}}
\captionsetup{{font=small,labelfont=bf}}
\setlength{{\parskip}}{{0.38em}}
\setlength{{\parindent}}{{0pt}}
\renewcommand{{\arraystretch}}{{1.15}}
\pagestyle{{fancy}}
\fancyhf{{}}
\fancyhead[L]{{\small\textcolor{{muted}}{{France AVM --- Phase 4 national benchmark}}}}
\fancyhead[R]{{\small\textcolor{{muted}}{{Badr Ettousy}}}}
\fancyfoot[C]{{\thepage}}
\setlength{{\headheight}}{{14pt}}

\newenvironment{{statusbox}}{{\begin{{center}}\begin{{minipage}}{{0.92\textwidth}}\textcolor{{navy}}{{\rule{{\textwidth}}{{0.8pt}}}}\vspace{{0.5em}}}}{{\vspace{{0.5em}}\textcolor{{navy}}{{\rule{{\textwidth}}{{0.8pt}}}}\end{{minipage}}\end{{center}}}}
\newcommand{{\metriccard}}[2]{{\colorbox{{light}}{{\begin{{minipage}}[c][1.45cm][c]{{0.215\textwidth}}\centering\textcolor{{navy}}{{\Large\textbf{{#1}}}}\\[-0.1em]\small\textcolor{{muted}}{{#2}}\end{{minipage}}}}}}

\title{{\textcolor{{navy}}{{\textbf{{Nationwide France Apartment Valuation}}}}\\[0.25em]
\large Phase 4 Data Engineering, Leakage Controls and\\Chronological Model Benchmark}}
\author{{Badr Ettousy}}
\date{{{created}\\\small {edition}}}

\begin{{document}}
\maketitle

\begin{{abstract}}
{abstract}
\end{{abstract}}

\textbf{{Keywords:}} automated valuation model; Geo-DVF; France; CatBoost;
historical comparable sales; chronological validation; regional robustness.

\tableofcontents
\clearpage

\section{{Executive assessment}}

Phase 4 extends the earlier Paris research into one national apartment model. The
central design choice is comparability rather than superficial feature parity:
Paris-only transport, school, park, noise and building snapshots are not copied
into other regions without a nationally uniform source contract. The national
model instead uses DVF property attributes, transaction time, hierarchical
territorial identity, global geometry and point-in-time comparable-sale evidence.

{executive_status(benchmark)}

\subsection*{{Decision status}}
{decision_status}

\section{{Research objective and contributions}}

The target is total transaction value in euros for an ordinary apartment sale.
The study asks whether one model can learn heterogeneous French markets while
preserving local evidence and honest temporal evaluation. Its contributions are:
\begin{{enumerate}}[leftmargin=1.6em]
\item chunk-safe processing of the national annual Geo-DVF archives;
\item explicit support for Corsican and overseas territorial identifiers;
\item spherical H3 and Cartesian location features valid outside Lambert-93;
\item local comparables subject to a 90-day information-availability lag;
\item a training-only hierarchical postcode-to-national median baseline;
\item a 2021--2023/2024/2025 chronological benchmark with paired bootstrap uncertainty;
\item region and sufficiently sized department diagnostics;
\item atomic artifacts, checksums, a reusable production bundle and inference CLI.
\end{{enumerate}}

{optional_figure(visual_dir, 'phase4_national_pipeline.png', 'Phase 4 source-to-report architecture and chronological evaluation design.', 'architecture')}

\section{{Data source and analytical cohort}}

The source is the normalized and geolocated Geo-DVF annual full archive for
2021--2025. CSV files are streamed in bounded chunks. The last mutation in each
chunk is carried into the next chunk so a multi-line transaction is never
classified before all its property rows are available.

The cohort retains \emph{{Vente}} mutations containing exactly one apartment and
no house or commercial premises. Dependencies such as a cellar or parking place
may remain bundled in the mutation consideration. Required fields are date,
transaction value, apartment surface, commune and valid WGS84 coordinates.
Broad national quality bounds are 9--300 m$^2$, EUR 10,000--10,000,000 and EUR
100--30,000 per m$^2$.

{coverage}

Geo-DVF contains no open observations for departments {', '.join(KNOWN_DVF_GAPS)}.
``Nationwide'' therefore means every territory represented upstream, not every
French department without exception.

\section{{Feature engineering contract}}

The fitted contract contains {len(PHASE4_MODEL_FEATURES)} inputs:
{len(PHASE4_NUMERIC_FEATURES)} numeric and
{len(PHASE4_CATEGORICAL_FEATURES)} categorical variables. The current
transaction's target and price per square metre are excluded.

\begin{{table}}[H]
\centering
\caption{{Phase 4 feature families.}}
\begin{{tabularx}}{{\textwidth}}{{p{{3.4cm}}rX}}
\toprule
Family & Inputs & Role \\
\midrule
Property & 6 & Surface, rooms, address number, lots and scale/density transforms \\
Calendar & 4 & Seasonality and elapsed market time \\
Global location & 5 & WGS84 and unit-sphere Cartesian coordinates \\
Territorial identity & 7 & Region, department, commune, postcode and H3 r7/r8/r9 \\
Historical comparables & 23 & Lagged counts, medians, dispersion, similarity, age, distance and momentum \\
\bottomrule
\end{{tabularx}}
\end{{table}}

\subsection{{Global spatial representation}}

Lambert-93 is unsuitable for overseas departments. Longitude $\lambda$ and
latitude $\phi$ are therefore supplemented by unit-sphere coordinates
\begin{{equation}}
(x,y,z)=(\cos\phi\cos\lambda,\cos\phi\sin\lambda,\sin\phi),
\end{{equation}}
and H3 cells. Nearby transactions are queried using haversine distance within
each department.

\subsection{{Leakage-safe comparable sales}}

For target sale $i$ at date $t_i$, candidate $j$ is eligible only if
\begin{{equation}}
90 \le t_i-t_j \le 730\quad\text{{days}},\qquad d(i,j)\le1{{,}}000\text{{ m}}.
\end{{equation}}
Windows at 180, 365 and 730 days and radii at 250, 500 and 1,000 metres produce
counts and median prices per square metre. Exponential recency/distance weights,
surface and room similarity, effective sample size, dispersion and momentum
summarize local evidence. The current transaction cannot become its own input.

\begin{{table}}[H]
\centering
\caption{{Primary leakage and integrity controls.}}
\begin{{tabularx}}{{\textwidth}}{{p{{4.4cm}}X}}
\toprule
Risk & Enforced control \\
\midrule
Mutation target duplication & One row only after transaction-level local-type audit \\
Future comparable sales & Minimum 90-day age and positive chronological difference \\
Target-derived leakage & Current value and price/m$^2$ excluded from fitted inputs \\
Geographic projection failure & Haversine distances and global unit-sphere coordinates \\
Model-selection leakage & 2024 selection separated from untouched 2025 evaluation \\
Partial artifact corruption & Sibling \texttt{{.part}} writes followed by atomic promotion \\
Silent territorial overclaim & Upstream missing departments recorded in every report \\
\bottomrule
\end{{tabularx}}
\end{{table}}

\section{{Benchmark design}}

\subsection{{Pre-training audit and market clustering}}

{preflight_section(preflight, visual_dir, trials)}

Development uses 2021--2023. Two predeclared CatBoost configurations are each
evaluated with and without the market-cluster feature on 2024; no 2025 outcome
participates in selection. The winner is
refitted through 2024 and evaluated once on 2025. Only after evaluation is the
production model refitted on all 2021--2025 observations.

{validation_table(trials)}

{latex_escape(validation_notes(trials))}

The objective is fitted to $\log(1+y)$ and transformed back to non-negative
euros. The benchmark baseline calculates median historical price per square
metre by postcode and falls back successively to commune, department, region
and national median. Every lookup is learned exclusively from the corresponding
training period.

\subsection{{Metrics and paired uncertainty}}

The report publishes MAE, RMSE, $R^2$, MAPE, median absolute percentage error
and hit rates within $\pm10\%$ and $\pm20\%$. Because model and baseline predict
the same transactions, uncertainty is estimated from paired absolute-error
differences. A deterministic bootstrap uses 300 repetitions and at most 250,000
test rows to bound memory. The exact sample size is stored in the benchmark JSON.

\section{{Frozen national benchmark}}

{metric_table(benchmark)}

\section{{Territorial robustness}}

Raw euro MAE is expected to be higher in expensive markets. Regional assessment
therefore requires MdAPE, hit rates, $R^2$ and sample size beside MAE. Department
rows require at least 100 test transactions; small segments are suppressed
rather than presented as stable estimates.

{regional_table(region)}

\section{{Calibration and uncertainty}}

The empirical 90\% interval uses 2024 target-to-prediction ratios. A region gets
its own calibration ratios only with at least 500 observations; otherwise it
inherits the national interval. Observed 2025 coverage evaluates temporal
transfer but is not a formal conditional guarantee for a specific apartment.

\section{{Visual diagnostic suite}}

{figure_sections}

Figures are generated only from existing artifacts. Missing benchmark outputs
produce no placeholder chart and no synthetic metric. Feature importance records
predictive usage rather than a causal housing premium.

\section{{Reproducibility and artifact inventory}}

\begin{{longtable}}{{p{{5.8cm}}lr}}
\toprule
Artifact & Status & Bytes \\
\midrule
\endfirsthead
\toprule Artifact & Status & Bytes \\
\midrule
\endhead
{inventory}
\bottomrule
\end{{longtable}}

\begin{{verbatim}}
python -m paris_avm.data.acquire_phase4
python -m paris_avm.features.phase4
python -m paris_avm.analysis.phase4_preflight
python -m paris_avm.modeling.benchmark_phase4
python -m paris_avm.visualization.phase4
python -m paris_avm.reporting.phase4
powershell -ExecutionPolicy Bypass -File .\docs\paper\compile_phase4.ps1
\end{{verbatim}}

\section{{Limitations and responsible use}}

\begin{{itemize}}[leftmargin=1.5em]
\item DVF omits condition, renovation, floor, elevator, view and occupancy.
\item Mutation value can bundle dependencies or transaction-specific rights.
\item Departments 57, 67, 68 and 976 are absent from the source.
\item Sparse submarkets may have weak comparable coverage and unstable errors.
\item Administrative/H3 identity is predictive context, not a causal estimate.
\item A 2025 holdout does not establish performance in 2026 or later markets.
\item The interval is empirical and marginal, not a certified property guarantee.
\item Predictions are not notarial valuations, lending decisions or investment advice.
\end{{itemize}}

\section{{Conclusion}}

Phase 4 defines a scalable and auditable path from a Paris research model to a
national France AVM. Its strongest contribution is the combination of global
geography, local point-in-time market evidence, chronological evaluation and
territorial reporting. In this {edition.lower()}, claims remain limited to the
artifacts actually present. {conclusion_status}

\appendix
\section{{Machine-readable deliverables}}

The benchmark publishes JSON metrics, validation trials, a two-method comparison,
regional and department CSV tables, Parquet predictions with paired errors,
feature importance and a checksummed Joblib production bundle. The Markdown and
LaTeX reports consume these outputs without refitting or selecting a model.

\section{{Source references}}

\begin{{itemize}}[leftmargin=1.5em]
\item \href{{https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres-geolocalisees}}{{Geo-DVF dataset catalogue on data.gouv.fr}}
\item \href{{https://files.data.gouv.fr/geo-dvf/latest/csv/}}{{Geo-DVF annual file index}}
\item \href{{https://www.etalab.gouv.fr/licence-ouverte-open-licence/}}{{Open Licence 2.0}}
\end{{itemize}}

\end{{document}}
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(output, lambda temporary: temporary.write_text(source, encoding="utf-8"))
    print(f"Saved {output}")
    return source


def build_parser() -> argparse.ArgumentParser:
    root = PROJECT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "docs/paper/phase4_national_report.tex")
    parser.add_argument("--benchmark", type=Path, default=root / "reports/phase4/benchmark.json")
    parser.add_argument("--validation-trials", type=Path, default=root / "reports/phase4/validation_trials.csv")
    parser.add_argument("--region-performance", type=Path, default=root / "reports/phase4/region_performance.csv")
    parser.add_argument("--quality", type=Path, default=root / "reports/phase4/feature_quality.json")
    parser.add_argument("--preflight", type=Path, default=root / "reports/phase4/preflight_summary.json")
    parser.add_argument("--visual-dir", type=Path, default=root / "visuals/phase4")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = PROJECT_ROOT
    artifact_paths = {
        "Source manifest": root / "data/raw/dvf_france/manifest.json",
        "Gold feature table": root / "data/gold/phase4_france_sale_features.parquet",
        "Feature quality report": args.quality,
        "Preflight summary": args.preflight,
        "Department market clusters": root / "reports/phase4/department_market_clusters.csv",
        "Preflight Markdown report": root / "reports/phase4/PHASE4_PREFLIGHT_REPORT.md",
        "Benchmark JSON": args.benchmark,
        "Validation trials": args.validation_trials,
        "Benchmark comparison": root / "reports/phase4/benchmark_comparison.csv",
        "Frozen predictions": root / "reports/phase4/test_predictions.parquet",
        "Regional metrics": args.region_performance,
        "Department metrics": root / "reports/phase4/department_performance.csv",
        "Feature importance": root / "reports/phase4/feature_importance.csv",
        "Production model": root / "models/phase4/france_catboost.joblib",
    }
    generate_latex(
        args.output,
        benchmark_path=args.benchmark,
        trials_path=args.validation_trials,
        region_path=args.region_performance,
        quality_path=args.quality,
        visual_dir=args.visual_dir,
        artifact_paths=artifact_paths,
        preflight_path=args.preflight,
    )


if __name__ == "__main__":
    main()
