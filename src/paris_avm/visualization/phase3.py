"""Create publication-quality figures for the Phase 3 external-data paper.

All performance graphics use the frozen 2025 prediction export.  Coverage and
context graphics use the audited Phase 3 Gold table and feature-quality report.
The script does not fit or select a model.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np
import pandas as pd

from paris_avm.paths import PROJECT_ROOT


ROOT = PROJECT_ROOT
REPORT_DIR = ROOT / "reports" / "phase3"
OUTPUT_DIR = ROOT / "visuals" / "phase3"
GOLD_PATH = ROOT / "data" / "gold" / "phase3_sale_features.parquet"
MODEL_PATH = ROOT / "models" / "phase3" / "phase3_selected_model.joblib"

NAVY = "#102A43"
BLUE = "#1677B8"
CYAN = "#38A3A5"
GREEN = "#2A9D6F"
GOLD = "#E9A23B"
RED = "#D1495B"
PURPLE = "#7158A6"
INK = "#243B53"
MUTED = "#627D98"
GRID = "#D9E2EC"
LIGHT = "#F4F7FA"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK,
            "axes.edgecolor": GRID,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 220,
        }
    )


def finish(fig: plt.Figure, name: str) -> None:
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path.relative_to(ROOT)}")


def clean_axis(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.75)
        ax.set_axisbelow(True)


def euro_thousands(value: float, _: int | None = None) -> str:
    return f"EUR {value / 1_000:.0f}k"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    comparison = pd.read_csv(REPORT_DIR / "phase3_model_comparison.csv")
    predictions = pd.read_csv(REPORT_DIR / "phase3_test_predictions.csv")
    predictions["date_mutation"] = pd.to_datetime(predictions["date_mutation"])
    gold = pd.read_parquet(GOLD_PATH)
    gold["date_mutation"] = pd.to_datetime(gold["date_mutation"])
    quality = json.loads((REPORT_DIR / "feature_quality.json").read_text(encoding="utf-8"))
    results = json.loads((REPORT_DIR / "phase3_results.json").read_text(encoding="utf-8"))
    return comparison, predictions, gold, quality, results


def draw_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(13.8, 7.2))
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    fig.suptitle(
        "Phase 3 data architecture: authoritative sources to leakage-aware valuation",
        fontsize=18,
        fontweight="bold",
        color=NAVY,
        y=0.975,
    )
    ax.text(
        0.2,
        6.42,
        "Immutable snapshots, auditable identity resolution, point-in-time DPE, and chronological evaluation",
        color=MUTED,
        fontsize=11,
    )

    def box(x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.025,rounding_size=0.12",
            linewidth=1.6,
            edgecolor=color,
            facecolor="white",
        )
        patch.set_path_effects([patheffects.SimplePatchShadow(offset=(1.5, -1.5), alpha=0.12), patheffects.Normal()])
        ax.add_patch(patch)
        ax.add_patch(FancyBboxPatch((x, y + h - 0.36), w, 0.36, boxstyle="round,pad=0.02,rounding_size=0.1", facecolor=color, edgecolor=color))
        ax.text(x + 0.16, y + h - 0.19, title, color="white", fontweight="bold", fontsize=10.2, va="center")
        ax.text(x + 0.16, y + h - 0.58, body, color=INK, fontsize=8.8, va="top", linespacing=1.35)

    sources = [
        ("BAN", "Normalized address\nFANTOIR identity", BLUE),
        ("BDNB + DPE", "Building, age, state\nDated energy record", PURPLE),
        ("IDFM", "Metro, RER, train\nand tram access", CYAN),
        ("Paris Open Data", "Schools and public\ngreen spaces", GREEN),
        ("DRIEAT", "Road, RATP and SNCF\nstrategic noise", GOLD),
    ]
    for i, (title, body, color) in enumerate(sources):
        box(0.2 + i * 2.62, 5.05, 2.35, 1.0, title, body, color)

    box(0.65, 2.87, 2.85, 1.42, "BRONZE", "Versioned raw files\nURL + licence + timestamp\nsize + SHA-256", MUTED)
    box(5.25, 2.87, 2.85, 1.42, "SILVER", "BAN exact key\nBAN--BDNB reliability rank\nparcel fallback + match audit", BLUE)
    box(9.85, 2.87, 2.95, 1.42, "GOLD", "143,009 sale identities\n141 typed columns\npoint-in-time quality contract", GREEN)
    box(2.55, 0.67, 3.25, 1.38, "FEATURE CONTRACT", "DPE date <= sale date\ncomparables lagged >= 90 days\nstatic context explicitly flagged", PURPLE)
    box(7.6, 0.67, 3.25, 1.38, "CHRONOLOGICAL MODEL", "2021--23 development\n2024 selection\n2025 frozen test", RED)

    arrows = [
        ((6.7, 5.02), (2.15, 4.34)),
        ((3.53, 3.58), (5.22, 3.58)),
        ((8.13, 3.58), (9.82, 3.58)),
        ((11.3, 2.84), (9.55, 2.09)),
        ((5.83, 1.36), (7.57, 1.36)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, linewidth=1.8, color=MUTED))

    ax.text(6.7, 0.18, "One immutable sale row in  |  one enriched sale row out", ha="center", fontsize=11, color=NAVY, fontweight="bold")
    finish(fig, "phase3_data_pipeline.png")


def draw_results_dashboard(comparison: pd.DataFrame, quality: dict, results: dict) -> None:
    selected = comparison.loc[
        comparison["model_key"].isin(
            [
                "catboost_full_identity",
                "lightgbm_full_no_identity_mae",
                "lightgbm_building_mae",
                "lightgbm_dpe_safe_mae",
                "lightgbm_context_mae",
                "phase2_comparable_correction",
            ]
        )
    ].sort_values("mae_eur", ascending=True)
    labels = {
        "catboost_full_identity": "CatBoost: full identity + enrichment",
        "lightgbm_full_no_identity_mae": "LightGBM: full, no raw identity",
        "lightgbm_building_mae": "LightGBM: building only",
        "lightgbm_dpe_safe_mae": "LightGBM: dated DPE only",
        "lightgbm_context_mae": "LightGBM: neighborhood context",
        "phase2_comparable_correction": "Phase 2 reference",
    }
    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.12, 1], width_ratios=[1.55, 1], hspace=0.36, wspace=0.28)
    fig.suptitle("Phase 3 executive result", fontsize=19, fontweight="bold", color=NAVY, y=0.985)
    fig.text(0.5, 0.947, "Frozen 2025 test: 28,330 Paris apartment sales", ha="center", color=MUTED, fontsize=11)

    ax = fig.add_subplot(gs[0, :])
    y = np.arange(len(selected))
    colors = [GREEN if key == "catboost_full_identity" else (MUTED if key == "phase2_comparable_correction" else BLUE) for key in selected["model_key"]]
    xerr = np.vstack(
        [
            selected["mae_eur"] - selected["mae_ci95_lower_eur"],
            selected["mae_ci95_upper_eur"] - selected["mae_eur"],
        ]
    )
    ax.barh(y, selected["mae_eur"], color=colors, height=0.62, alpha=0.94)
    ax.errorbar(selected["mae_eur"], y, xerr=xerr, fmt="none", ecolor=NAVY, capsize=3, linewidth=1.2)
    ax.set_yticks(y, [labels[k] for k in selected["model_key"]])
    ax.invert_yaxis()
    ax.set_xlim(90_000, 99_200)
    ax.xaxis.set_major_formatter(FuncFormatter(euro_thousands))
    ax.set_title("Mean absolute error with 95% bootstrap intervals", loc="left")
    for yi, value in zip(y, selected["mae_eur"]):
        ax.text(value + 180, yi, f"EUR {value:,.0f}", va="center", fontsize=9.5, fontweight="bold", color=INK)
    clean_axis(ax, "x")

    ax = fig.add_subplot(gs[1, 0])
    ablations = selected.loc[selected["model_key"] != "phase2_comparable_correction"].sort_values("mae_reduction_vs_baseline_eur")
    y = np.arange(len(ablations))
    reductions = ablations["mae_reduction_vs_baseline_eur"].to_numpy()
    lo = ablations["reduction_ci95_lower_eur"].to_numpy()
    hi = ablations["reduction_ci95_upper_eur"].to_numpy()
    ax.barh(y, reductions, color=[GREEN if k == "catboost_full_identity" else BLUE for k in ablations["model_key"]], height=0.6)
    ax.errorbar(reductions, y, xerr=np.vstack([reductions - lo, hi - reductions]), fmt="none", ecolor=NAVY, capsize=3)
    ax.axvline(0, color=RED, linewidth=1.1)
    ax.set_yticks(y, [labels[k].replace("LightGBM: ", "").replace("CatBoost: ", "") for k in ablations["model_key"]])
    ax.set_xlabel("MAE reduction versus Phase 2 (EUR)")
    ax.set_title("Paired improvement on identical transactions", loc="left")
    clean_axis(ax, "x")

    ax = fig.add_subplot(gs[1, 1])
    coverage_labels = ["Exact BAN\naddress", "Any BDNB\nbuilding", "High-confidence\nBDNB", "DPE available\nby sale date"]
    coverage = np.array(
        [
            quality["match_quality"]["ban_exact_match_rate"],
            quality["match_quality"]["bdnb_building_match_rate"],
            quality["match_quality"]["bdnb_high_confidence_rate"],
            quality["dpe_available_at_sale_rate"],
        ]
    )
    bars = ax.bar(np.arange(4), coverage * 100, color=[BLUE, GREEN, CYAN, PURPLE], width=0.64)
    ax.set_ylim(0, 108)
    ax.set_xticks(np.arange(4), coverage_labels)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_ylabel("Coverage")
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_title("Entity-resolution and temporal coverage", loc="left")
    for bar, value in zip(bars, coverage):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{value:.2%}", ha="center", fontweight="bold", fontsize=9.5)
    clean_axis(ax)

    winner = results["winner"]
    fig.text(0.755, 0.005, f"Selected MAE reduction: EUR {winner['mae_reduction_vs_baseline_eur']:,.0f} ({winner['mae_reduction_vs_baseline_eur']/results['reference_2025_mae_eur']:.2%})  |  Future-DPE leakage: 0", ha="center", color=NAVY, fontsize=9.5, fontweight="bold")
    finish(fig, "phase3_results_dashboard.png")


def draw_diagnostics(predictions: pd.DataFrame) -> None:
    actual = predictions["valeur_fonciere"].to_numpy(dtype=float)
    phase3 = predictions["prediction_catboost_full_identity"].to_numpy(dtype=float)
    phase2 = predictions["prediction_phase2_comparable_correction"].to_numpy(dtype=float)
    abs3 = np.abs(actual - phase3)
    abs2 = np.abs(actual - phase2)

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0))
    fig.suptitle("Out-of-sample prediction diagnostics", fontsize=18, fontweight="bold", color=NAVY, y=0.99)

    ax = axes[0, 0]
    hb = ax.hexbin(actual, phase3, gridsize=62, xscale="log", yscale="log", bins="log", cmap="viridis", mincnt=1)
    limits = [50_000, 8_500_000]
    ax.plot(limits, limits, color=RED, linestyle="--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Observed transaction value (EUR, log scale)")
    ax.set_ylabel("Phase 3 prediction (EUR, log scale)")
    ax.set_title("Observed versus predicted", loc="left")
    ax.legend(frameon=False, loc="lower right")
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("log density")
    clean_axis(ax, None)

    ax = axes[0, 1]
    frame = predictions[["valeur_fonciere"]].copy()
    frame["price_decile"] = pd.qcut(frame["valeur_fonciere"], 10, labels=False, duplicates="drop")
    frame["phase2"] = abs2
    frame["phase3"] = abs3
    grouped = frame.groupby("price_decile", observed=True).agg(actual=("valeur_fonciere", "median"), phase2=("phase2", "mean"), phase3=("phase3", "mean"))
    x = np.arange(len(grouped))
    ax.plot(x, grouped["phase2"], marker="o", color=MUTED, linewidth=2, label="Phase 2")
    ax.plot(x, grouped["phase3"], marker="o", color=GREEN, linewidth=2.3, label="Phase 3")
    ax.set_xticks(x, [f"EUR {v/1_000:.0f}k" for v in grouped["actual"]], rotation=35, ha="right")
    ax.yaxis.set_major_formatter(FuncFormatter(euro_thousands))
    ax.set_ylabel("MAE")
    ax.set_title("Error across transaction-value deciles", loc="left")
    ax.legend(frameon=False)
    clean_axis(ax)

    ax = axes[1, 0]
    monthly = predictions.assign(
        phase2_error=abs2,
        phase3_error=abs3,
        month=predictions["date_mutation"].dt.month,
    ).groupby("month").agg(phase2=("phase2_error", "mean"), phase3=("phase3_error", "mean"))
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.plot(monthly.index, monthly["phase2"], marker="o", color=MUTED, linewidth=2, label="Phase 2")
    ax.plot(monthly.index, monthly["phase3"], marker="o", color=GREEN, linewidth=2.3, label="Phase 3")
    ax.set_xticks(range(1, 13), months)
    ax.yaxis.set_major_formatter(FuncFormatter(euro_thousands))
    ax.set_ylabel("Monthly MAE")
    ax.set_title("Performance stability during 2025", loc="left")
    ax.legend(frameon=False)
    clean_axis(ax)

    ax = axes[1, 1]
    signed_gain = abs2 - abs3
    clipped = np.clip(signed_gain, -300_000, 300_000)
    ax.hist(clipped, bins=65, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.25)
    ax.axvline(0, color=RED, linewidth=1.4)
    ax.axvline(np.mean(signed_gain), color=GREEN, linewidth=2, label=f"Mean gain = EUR {np.mean(signed_gain):,.0f}")
    ax.set_xlabel("Transaction-level absolute-error reduction (EUR, clipped)")
    ax.set_ylabel("Sales")
    ax.set_title("Where Phase 3 gains and loses", loc="left")
    ax.legend(frameon=False)
    clean_axis(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.965])
    finish(fig, "phase3_prediction_diagnostics.png")


def draw_spatial_performance(gold: pd.DataFrame, predictions: pd.DataFrame) -> None:
    test = gold.loc[gold["date_mutation"].dt.year.eq(2025), ["id_mutation", "longitude", "latitude", "code_postal"]].merge(
        predictions[["id_mutation", "valeur_fonciere", "prediction_phase2_comparable_correction", "prediction_catboost_full_identity"]],
        on="id_mutation",
        how="inner",
        validate="one_to_one",
    )
    test["phase2_error"] = np.abs(test["valeur_fonciere"] - test["prediction_phase2_comparable_correction"])
    test["phase3_error"] = np.abs(test["valeur_fonciere"] - test["prediction_catboost_full_identity"])
    test["gain"] = test["phase2_error"] - test["phase3_error"]

    fig = plt.figure(figsize=(14.2, 7.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1], wspace=0.42)
    fig.suptitle("Spatial distribution of Phase 3 performance", fontsize=18, fontweight="bold", color=NAVY, y=0.985)
    ax = fig.add_subplot(gs[0, 0])
    vmax = np.nanpercentile(np.abs(test["gain"]), 92)
    sc = ax.scatter(test["longitude"], test["latitude"], c=test["gain"], s=5, alpha=0.52, cmap="RdYlGn", vmin=-vmax, vmax=vmax, linewidths=0)
    ax.set_aspect(1 / np.cos(np.deg2rad(48.86)))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Transaction-level absolute-error change", loc="left")
    ax.text(0.02, 0.02, "Green: Phase 3 improves\nRed: Phase 3 worsens", transform=ax.transAxes, fontsize=9, color=INK, bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": GRID, "alpha": 0.92})
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.025)
    cb.set_label("Error reduction (EUR)", fontsize=9)
    clean_axis(ax, None)

    ax = fig.add_subplot(gs[0, 1])
    arr = test.assign(arrondissement=test["code_postal"].astype(str).str[-2:].astype(int)).groupby("arrondissement").agg(
        gain=("gain", "mean"),
        phase3_mae=("phase3_error", "mean"),
        n=("id_mutation", "size"),
    ).sort_values("gain")
    colors = np.where(arr["gain"] >= 0, GREEN, RED)
    y = np.arange(len(arr))
    ax.barh(y, arr["gain"], color=colors, height=0.7)
    ax.axvline(0, color=NAVY, linewidth=1)
    ax.set_yticks(y, [f"750{i:02d}" for i in arr.index])
    ax.set_xlabel("Mean MAE reduction versus Phase 2 (EUR)")
    ax.set_title("Improvement by arrondissement", loc="left")
    clean_axis(ax, "x")
    fig.text(0.5, 0.015, "The identity/enrichment gain is geographically broad but heterogeneous; green does not imply a causal amenity effect.", ha="center", color=MUTED, fontsize=9.5)
    finish(fig, "phase3_spatial_performance.png")


def draw_enrichment_profile(gold: pd.DataFrame) -> None:
    test = gold.loc[gold["date_mutation"].dt.year.eq(2025)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6))
    fig.suptitle("What the external enrichment adds", fontsize=18, fontweight="bold", color=NAVY, y=0.99)

    ax = axes[0, 0]
    dpe_order = list("ABCDEFG")
    counts = test["classe_bilan_dpe"].astype("string").value_counts().reindex(dpe_order).fillna(0)
    dpe_colors = ["#2E7D32", "#66A83D", "#B7C943", "#F0D84A", "#F2A93B", "#E66A35", "#C93D3D"]
    bars = ax.bar(dpe_order, counts / counts.sum() * 100, color=dpe_colors)
    ax.set_ylabel("Share of dated DPE matches")
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_title("Energy rating available by sale date", loc="left")
    for b, v in zip(bars, counts / counts.sum() * 100):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.0f}%", ha="center", fontsize=8.5)
    clean_axis(ax)

    ax = axes[0, 1]
    age = test["building_age_at_sale"].dropna().clip(0, 250)
    ax.hist(age, bins=np.arange(0, 261, 10), color=PURPLE, alpha=0.88, edgecolor="white")
    ax.axvline(age.median(), color=GOLD, linewidth=2, label=f"Median: {age.median():.0f} years")
    ax.set_xlabel("Building age at sale (years; clipped at 250)")
    ax.set_ylabel("Sales")
    ax.set_title("Building vintage reconstructed from BDNB", loc="left")
    ax.legend(frameon=False)
    clean_axis(ax)

    ax = axes[1, 0]
    distance_cols = {
        "Metro": "distance_metro_m",
        "School": "distance_school_m",
        "Park": "distance_green_space_centroid_m",
        "Shop/service": "distance_shop_service_m",
        "RER": "distance_rer_m",
    }
    values = [test[col].median() for col in distance_cols.values()]
    bars = ax.barh(list(distance_cols.keys()), values, color=[BLUE, CYAN, GREEN, GOLD, PURPLE])
    ax.invert_yaxis()
    ax.set_xlabel("Median straight-line distance (metres)")
    ax.set_title("Accessibility signals", loc="left")
    for b, v in zip(bars, values):
        ax.text(v + max(values) * 0.02, b.get_y() + b.get_height() / 2, f"{v:,.0f} m", va="center", fontsize=9, fontweight="bold")
    clean_axis(ax, "x")

    ax = axes[1, 1]
    noise = test["noise_transport_lden_max_db"].dropna()
    categories = pd.cut(noise, bins=[0, 54.999, 59.999, 64.999, 69.999, np.inf], labels=["No band / <55", "55--59", "60--64", "65--69", "70+"])
    shares = categories.value_counts(sort=False) / len(noise) * 100
    bars = ax.bar(shares.index.astype(str), shares, color=[GREEN, "#A8C66C", GOLD, "#E77C40", RED])
    ax.set_xlabel("Maximum mapped transport Lden band (dB)")
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_ylabel("Sales")
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_title("Strategic transport-noise context", loc="left")
    for b, v in zip(bars, shares):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.7, f"{v:.1f}%", ha="center", fontsize=8.5)
    clean_axis(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.965])
    finish(fig, "phase3_enrichment_profile.png")


def feature_family(feature: str) -> str:
    if feature in {"ban_address_id", "bdnb_building_id", "bdnb_match_method", "h3_r8", "h3_r9"}:
        return "Identity / micro-location"
    if feature.startswith(("local_", "weighted_", "similar_", "effective_", "nearest_", "median_comparable")):
        return "Historical comparables"
    if feature.startswith(("dpe_", "classe_", "conso_", "emission_", "type_installation", "type_energie", "annee_construction_dpe", "nombre_niveau", "surface_habitable")):
        return "DPE / energy"
    if feature.startswith(("building_", "bdnb_", "ban_", "usage_", "mat_")):
        return "Building / linkage"
    if feature.startswith(("distance_metro", "distance_rer", "distance_train", "distance_tram", "rail_", "distance_school", "school_", "distance_green", "green_", "distance_shop", "shop_", "noise_")):
        return "Neighborhood context"
    return "Core DVF / spatial"


def pretty_feature(name: str) -> str:
    replacements = {
        "bdnb_building_id": "BDNB building identity",
        "ban_address_id": "BAN address identity",
        "surface_reelle_bati": "Apartment surface",
        "local_365d_500m_median_ppm2": "Local median EUR/m2 (365d, 500m)",
        "local_730d_500m_median_ppm2": "Local median EUR/m2 (730d, 500m)",
        "weighted_comp_ppm2": "Weighted comparable EUR/m2",
        "similar_weighted_comp_ppm2": "Similar comparable EUR/m2",
        "longitude": "Longitude",
        "latitude": "Latitude",
        "building_age_at_sale": "Building age at sale",
        "building_year": "Construction year",
        "classe_bilan_dpe": "DPE energy class",
        "distance_metro_m": "Distance to Metro",
        "noise_transport_lden_max_db": "Maximum transport noise",
        "code_postal": "Arrondissement",
        "months_since_2021": "Market time trend",
    }
    return replacements.get(name, name.replace("_", " ").capitalize())


def draw_feature_importance() -> None:
    bundle = joblib.load(MODEL_PATH)
    importance = pd.DataFrame(
        {
            "feature": bundle["features"],
            "importance": bundle["model"].get_feature_importance(),
        }
    )
    importance["family"] = importance["feature"].map(feature_family)
    family = importance.groupby("family", as_index=False)["importance"].sum().sort_values("importance")
    top = importance.nlargest(18, "importance").sort_values("importance")
    palette = {
        "Identity / micro-location": GREEN,
        "Historical comparables": BLUE,
        "DPE / energy": PURPLE,
        "Building / linkage": GOLD,
        "Neighborhood context": CYAN,
        "Core DVF / spatial": MUTED,
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 8.2), gridspec_kw={"width_ratios": [0.72, 1.58], "wspace": 0.62})
    fig.suptitle("How the selected CatBoost model uses Phase 3 information", fontsize=18, fontweight="bold", color=NAVY, y=0.985)
    ax = axes[0]
    ax.barh(family["family"], family["importance"], color=[palette[x] for x in family["family"]])
    ax.set_xlabel("Total CatBoost importance")
    ax.set_title("Importance by feature family", loc="left")
    clean_axis(ax, "x")

    ax = axes[1]
    ax.barh([pretty_feature(x) for x in top["feature"]], top["importance"], color=[palette[x] for x in top["family"]])
    ax.tick_params(axis="y", labelsize=8.7)
    ax.set_xlabel("CatBoost prediction-value-change importance")
    ax.set_title("Top individual predictors", loc="left")
    clean_axis(ax, "x")
    handles = [plt.Line2D([0], [0], color=color, lw=7, label=family_name) for family_name, color in palette.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.text(0.5, 0.085, "Importance measures model usage, not causal impact or economic willingness to pay.", ha="center", color=MUTED, fontsize=9.5)
    fig.subplots_adjust(bottom=0.17, top=0.91)
    finish(fig, "phase3_feature_importance.png")


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=(
            "Generate the publication figures for the Phase 3 external-data "
            "paper from existing benchmark artifacts."
        )
    )


def main(argv: Sequence[str] | None = None) -> None:
    build_parser().parse_args(argv)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    comparison, predictions, gold, quality, results = load_inputs()
    draw_pipeline()
    draw_results_dashboard(comparison, quality, results)
    draw_diagnostics(predictions)
    draw_spatial_performance(gold, predictions)
    draw_enrichment_profile(gold)
    draw_feature_importance()


if __name__ == "__main__":
    main()
