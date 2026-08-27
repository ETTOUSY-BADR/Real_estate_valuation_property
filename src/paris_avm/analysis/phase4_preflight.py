"""Audit and cluster the national Phase 4 Gold table before model training.

Department clusters are learned from 2021--2023 only.  They use structural and
strictly lagged comparable-sale summaries, never the current transaction value.
The analysis is diagnostic: the benchmark must still establish whether the
cluster feature adds out-of-time value.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from paris_avm.artifacts import write_atomic, write_csv_atomic, write_json_atomic
from paris_avm.features.phase4 import PHASE4_MODEL_FEATURES, validate_phase4_table
from paris_avm.geography import REGION_NAMES
from paris_avm.paths import PROJECT_ROOT


CLUSTER_INPUTS = (
    "log_transactions",
    "median_surface_m2",
    "median_rooms_per_10m2",
    "weighted_comparable_coverage",
    "log_median_comparable_ppm2",
    "log_median_local_count",
    "log_median_effective_comparables",
)


def department_profiles(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize leakage-safe market structure at department level."""
    required = {
        "code_departement",
        "code_region",
        "surface_reelle_bati",
        "rooms_per_10m2",
        "weighted_comp_ppm2",
        "local_365d_500m_count",
        "effective_comparable_count",
        "latitude",
        "longitude",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError("Cannot create department profiles; missing: " + ", ".join(missing))
    grouped = data.groupby("code_departement", observed=True)
    profile = grouped.agg(
        code_region=("code_region", "first"),
        transactions=("code_departement", "size"),
        median_surface_m2=("surface_reelle_bati", "median"),
        median_rooms_per_10m2=("rooms_per_10m2", "median"),
        median_comparable_ppm2=("weighted_comp_ppm2", "median"),
        median_local_count=("local_365d_500m_count", "median"),
        median_effective_comparables=("effective_comparable_count", "median"),
        centroid_latitude=("latitude", "median"),
        centroid_longitude=("longitude", "median"),
    ).reset_index()
    coverage = grouped["weighted_comp_ppm2"].apply(lambda values: float(values.notna().mean()))
    profile["weighted_comparable_coverage"] = profile["code_departement"].map(coverage)
    profile["region_name"] = profile["code_region"].map(
        lambda value: REGION_NAMES.get(str(value), str(value))
    )
    profile["log_transactions"] = np.log1p(profile["transactions"])
    profile["log_median_comparable_ppm2"] = np.log1p(profile["median_comparable_ppm2"])
    profile["log_median_local_count"] = np.log1p(profile["median_local_count"])
    profile["log_median_effective_comparables"] = np.log1p(
        profile["median_effective_comparables"]
    )
    return profile


def fit_department_clusters(
    data: pd.DataFrame,
    *,
    requested_clusters: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit deterministic department clusters and return profiles plus audit metadata."""
    profile = department_profiles(data)
    if len(profile) < 3:
        raise ValueError("At least three departments are required for clustering")
    matrix = profile.loc[:, CLUSTER_INPUTS].replace([np.inf, -np.inf], np.nan)
    imputation = matrix.median().fillna(0.0)
    matrix = matrix.fillna(imputation)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)

    maximum = min(8, len(profile) - 1)
    candidates = [requested_clusters] if requested_clusters else list(range(2, maximum + 1))
    candidates = [value for value in candidates if value is not None and 2 <= value <= maximum]
    if not candidates:
        raise ValueError(f"Cluster count must be between 2 and {maximum}")
    scores: dict[int, float] = {}
    fitted: dict[int, KMeans] = {}
    for clusters in candidates:
        model = KMeans(n_clusters=clusters, n_init=20, random_state=random_state)
        labels = model.fit_predict(scaled)
        scores[clusters] = float(silhouette_score(scaled, labels))
        fitted[clusters] = model
    selected = max(scores, key=lambda value: (scores[value], -value))
    raw_labels = fitted[selected].labels_

    ordering = (
        pd.DataFrame({"raw": raw_labels, "price": profile["median_comparable_ppm2"]})
        .groupby("raw", observed=True)["price"]
        .median()
        .sort_values(na_position="last")
        .index.tolist()
    )
    label_map = {raw: f"market_{position:02d}" for position, raw in enumerate(ordering, 1)}
    profile["market_cluster"] = [label_map[label] for label in raw_labels]
    profile = profile.sort_values(["market_cluster", "code_departement"]).reset_index(drop=True)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fit_period": "through 2023",
        "target_used": False,
        "cluster_inputs": list(CLUSTER_INPUTS),
        "standardization_scope": "department profiles only; CatBoost inputs are not standardized",
        "departments": int(len(profile)),
        "selected_clusters": int(selected),
        "silhouette_scores": {str(key): round(value, 6) for key, value in scores.items()},
        "imputation_medians": {key: float(value) for key, value in imputation.items()},
        "cluster_sizes": {
            str(key): int(value)
            for key, value in profile["market_cluster"].value_counts().sort_index().items()
        },
    }
    return profile, metadata


def audit_summary(data: pd.DataFrame, profiles: pd.DataFrame, clustering: dict[str, Any]) -> dict[str, Any]:
    missingness = data.loc[:, PHASE4_MODEL_FEATURES].isna().mean().sort_values(ascending=False)
    year_counts = data.groupby("year", observed=True).size()
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "empirical preflight complete",
        "rows": int(len(data)),
        "date_min": data["date_mutation"].min().date().isoformat(),
        "date_max": data["date_mutation"].max().date().isoformat(),
        "departments": int(data["code_departement"].nunique()),
        "regions": int(data["code_region"].nunique()),
        "rows_by_year": {str(key): int(value) for key, value in year_counts.items()},
        "median_transaction_value_eur": round(float(data["valeur_fonciere"].median()), 2),
        "median_price_per_m2_eur": round(float(data["price_per_m2"].median()), 2),
        "weighted_comparable_coverage": round(float(data["weighted_comp_ppm2"].notna().mean()), 6),
        "preprocessing_decisions": {
            "surface_terrain_missingness": round(float(data["surface_terrain"].isna().mean()), 6),
            "surface_terrain_model_status": "excluded after preflight",
            "catboost_numeric_scaling": "not applied",
            "kmeans_profile_scaling": "median imputation followed by StandardScaler fitted on development only",
        },
        "highest_missingness": {
            str(key): round(float(value), 6) for key, value in missingness.head(12).items()
        },
        "clustering": clustering,
        "cluster_profile_rows": int(len(profiles)),
        "interpretation": (
            "Clusters are descriptive market-structure groups learned from development-period "
            "features. They are not administrative boundaries, causal market definitions, or "
            "evidence of predictive improvement until the 2024 ablation is evaluated."
        ),
    }


def draw_preflight(data: pd.DataFrame, profiles: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    outputs: list[Path] = []

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    annual = data.groupby("year", observed=True).size()
    axes[0, 0].bar(annual.index.astype(str), annual.values / 1_000, color="#1677B8")
    axes[0, 0].set_title("Retained apartment sales by year")
    axes[0, 0].set_ylabel("Rows (thousands)")
    capped = data["price_per_m2"].clip(upper=data["price_per_m2"].quantile(0.99))
    axes[0, 1].hist(capped, bins=50, color="#2A9D6F", alpha=0.9)
    axes[0, 1].set_title("Price per m² (capped at p99 for display)")
    axes[0, 1].set_xlabel("EUR per m²")
    missing = data.loc[:, PHASE4_MODEL_FEATURES].isna().mean().nlargest(10).sort_values()
    axes[1, 0].barh(missing.index, missing.values * 100, color="#E09F3E")
    axes[1, 0].set_title("Highest model-feature missingness")
    axes[1, 0].set_xlabel("Missing (%)")
    region_rows = data.groupby("code_region", observed=True).size().sort_values(ascending=False)
    region_labels = [REGION_NAMES.get(str(value), str(value)) for value in region_rows.index]
    axes[1, 1].barh(region_labels[::-1], (region_rows.values / 1_000)[::-1], color="#6C5CE7")
    axes[1, 1].set_title("Retained sales by region")
    axes[1, 1].set_xlabel("Rows (thousands)")
    figure.suptitle("Phase 4 national data preflight", fontsize=17, fontweight="bold")
    figure.tight_layout()
    path = output_dir / "phase4_preflight_data_audit.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    outputs.append(path)

    clusters = sorted(profiles["market_cluster"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))
    color_map = dict(zip(clusters, colors))
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for cluster, group in profiles.groupby("market_cluster", observed=True):
        axes[0].scatter(
            group["median_comparable_ppm2"],
            group["transactions"],
            s=55,
            alpha=0.8,
            label=cluster,
            color=color_map[cluster],
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Median lagged comparable EUR/m²")
    axes[0].set_ylabel("Development transactions (log scale)")
    axes[0].set_title("Department market-structure clusters")
    axes[0].legend(ncol=2, fontsize=8)
    for cluster, group in profiles.groupby("market_cluster", observed=True):
        axes[1].scatter(
            group["centroid_longitude"],
            group["centroid_latitude"],
            s=np.clip(np.sqrt(group["transactions"]) * 2, 20, 180),
            alpha=0.8,
            label=cluster,
            color=color_map[cluster],
        )
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    axes[1].set_title("Geographic distribution of learned clusters")
    figure.suptitle(
        "Leakage-safe department clustering fitted on 2021–2023 only",
        fontsize=16,
        fontweight="bold",
    )
    figure.tight_layout()
    path = output_dir / "phase4_department_clusters.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    outputs.append(path)
    return outputs


def markdown_report(summary: dict[str, Any], profiles: pd.DataFrame) -> str:
    clustering = summary["clustering"]
    score_rows = "\n".join(
        f"| {clusters} | {score:.4f} |"
        for clusters, score in clustering["silhouette_scores"].items()
    )
    cluster_rows = "\n".join(
        f"| {cluster} | {len(group):,} | {group['transactions'].sum():,} | "
        f"{group['median_comparable_ppm2'].median():,.0f} | "
        f"{group['weighted_comparable_coverage'].median():.1%} |"
        for cluster, group in profiles.groupby("market_cluster", observed=True)
    )
    year_rows = "\n".join(
        f"| {year} | {rows:,} |" for year, rows in summary["rows_by_year"].items()
    )
    return f"""# Phase 4 national preflight and market clustering

**Execution status: empirical preflight complete; final model benchmark remains separate.**

This audit describes the model-ready national cohort before CatBoost training. It checks temporal and
territorial coverage, missingness, distribution shape, historical-comparable availability, and a
development-only clustering experiment. No 2024 or 2025 transaction value is used to fit the clusters.

## Dataset overview

| Year | Retained apartment sales |
|---:|---:|
{year_rows}

- Period: **{summary['date_min']} to {summary['date_max']}**
- Total rows: **{summary['rows']:,}**
- Represented departments: **{summary['departments']}**
- Represented regions: **{summary['regions']}**
- Median transaction value: **EUR {summary['median_transaction_value_eur']:,.0f}**
- Median price per m²: **EUR {summary['median_price_per_m2_eur']:,.0f}**
- Lagged weighted-comparable coverage: **{summary['weighted_comparable_coverage']:.2%}**

![National data preflight](../../visuals/phase4/phase4_preflight_data_audit.png)

## Preprocessing decision

CatBoost inputs are not standardized. Tree splits do not require a common numeric scale, categorical
territorial identifiers must remain strings, and the target is already fitted as `log1p(value)`. Numeric
standardization is used only inside K-means after median imputation of department-level profiles.
`surface_terrain` is retained in the audited Gold table but excluded from the fitted contract after the
preflight found **{summary['preprocessing_decisions']['surface_terrain_missingness']:.2%} missingness** for
apartment sales. Missing historical-comparable values remain meaningful and are accompanied by explicit
coverage and count features, so CatBoost handles them natively.

## Leakage-safe clustering

Departments are summarized using 2021–2023 transaction volume, apartment structure, and comparable-sale
features that already enforce the 90-day availability lag. The current transaction target is excluded.
The number of clusters is selected by silhouette score.

| Clusters | Silhouette score |
|---:|---:|
{score_rows}

Selected: **{clustering['selected_clusters']} clusters**.

| Cluster | Departments | Development rows | Median lagged EUR/m² | Median comparable coverage |
|---|---:|---:|---:|---:|
{cluster_rows}

![Department market clusters](../../visuals/phase4/phase4_department_clusters.png)

## Interpretation and next decision

{summary['interpretation']} The benchmark compares otherwise identical CatBoost candidates with and
without the cluster feature on 2024. Only that chronological ablation can justify retaining clustering
for the frozen 2025 test and production bundle.
"""


def run_preflight(
    gold_path: Path,
    output_dir: Path,
    visual_dir: Path,
    *,
    train_end_year: int = 2023,
    requested_clusters: int | None = None,
) -> dict[str, Any]:
    data = pd.read_parquet(gold_path)
    data["date_mutation"] = pd.to_datetime(data["date_mutation"])
    validate_phase4_table(data)
    development = data.loc[data["year"].le(train_end_year)].copy()
    profiles, clustering = fit_department_clusters(
        development, requested_clusters=requested_clusters
    )
    clustering["fit_period"] = f"through {train_end_year}"
    summary = audit_summary(data, profiles, clustering)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(profiles, output_dir / "department_market_clusters.csv")
    write_json_atomic(output_dir / "preflight_summary.json", summary)
    draw_preflight(data, profiles, visual_dir)
    report = markdown_report(summary, profiles)
    report_path = output_dir / "PHASE4_PREFLIGHT_REPORT.md"
    write_atomic(report_path, lambda temporary: temporary.write_text(report, encoding="utf-8"))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved {report_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold",
        type=Path,
        default=PROJECT_ROOT / "data/gold/phase4_france_sale_features.parquet",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports/phase4")
    parser.add_argument("--visual-dir", type=Path, default=PROJECT_ROOT / "visuals/phase4")
    parser.add_argument("--train-end-year", type=int, default=2023)
    parser.add_argument("--clusters", type=int)
    args = parser.parse_args()
    run_preflight(
        args.gold,
        args.output_dir,
        args.visual_dir,
        train_end_year=args.train_end_year,
        requested_clusters=args.clusters,
    )


if __name__ == "__main__":
    main()
