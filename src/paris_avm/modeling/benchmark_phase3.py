"""Chronologically benchmark Phase 3 Paris valuation feature families.

Development: 2021--2023; model/feature selection: 2024; frozen test: 2025.
The script reports separate ablations for point-in-time-safe and retrospective
static enrichments so a backtest gain is never presented as a live historical
gain without qualification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from paris_avm.artifacts import write_csv_atomic, write_json_atomic
from paris_avm.modeling.benchmark_phase1 import bootstrap_intervals
from paris_avm.modeling.benchmark_phase2 import (
    BASE_CATEGORICAL,
    COMPARABLE_NUMERIC,
    PROPERTY_ENGINEERED_NUMERIC,
    SPATIAL_CATEGORICAL,
    SPATIAL_NUMERIC,
    prepare_categorical_data,
)
from paris_avm.modeling.train_phase1 import NUMERIC_FEATURES, regression_metrics
from paris_avm.paths import PROJECT_ROOT


IDENTITY_CATEGORICAL = [
    "ban_address_id",
    "bdnb_building_id",
    "bdnb_match_method",
]
BUILDING_NUMERIC = [
    "ban_match_distance_m",
    "ban_match_confidence",
    "bdnb_match_confidence",
    "bdnb_address_candidate_count",
    "bdnb_parcel_candidate_count",
    "building_level_count",
    "building_year",
    "building_dwelling_count",
    "building_height_m",
    "building_footprint_m2",
    "building_parking_lot_count",
    "building_total_lot_count",
    "building_dwelling_count_rnc",
    "building_age_at_sale",
    "building_year_missing",
]
BUILDING_CATEGORICAL = [
    "usage_niveau_1_txt",
    "mat_mur_txt",
    "mat_toit_txt",
    "building_state",
    "building_period_rnc",
]
DPE_NUMERIC = [
    "dpe_available_at_sale",
    "dpe_age_days",
    "dpe_surface_difference_ratio",
    "dpe_match_confidence",
    "annee_construction_dpe",
    "nombre_niveau_logement",
    "nombre_niveau_immeuble",
    "surface_habitable_immeuble",
    "surface_habitable_logement",
    "conso_5_usages_ep_m2",
    "emission_ges_5_usages_m2",
]
DPE_CATEGORICAL = [
    "classe_bilan_dpe",
    "classe_emission_ges",
    "type_installation_chauffage",
    "type_energie_chauffage",
]
CONTEXT_NUMERIC = [
    "distance_metro_m",
    "distance_rer_m",
    "distance_train_m",
    "distance_tram_m",
    "rail_station_mode_count_1000m",
    "rail_line_count_1000m",
    "distance_school_m",
    "school_count_1000m",
    "distance_green_space_centroid_m",
    "green_space_count_500m",
    "green_area_500m2",
    "distance_shop_service_m",
    "shop_service_count_500m",
    "shop_service_count_1000m",
    "noise_road_lden_db",
    "noise_rail_ratp_lden_db",
    "noise_rail_sncf_lden_db",
    "noise_transport_lden_max_db",
]

BASE_NUMERIC = NUMERIC_FEATURES + PROPERTY_ENGINEERED_NUMERIC + SPATIAL_NUMERIC + COMPARABLE_NUMERIC
BASE_CAT = BASE_CATEGORICAL + SPATIAL_CATEGORICAL
FEATURE_SETS = {
    "phase2": (BASE_NUMERIC, BASE_CAT),
    "dpe_safe": (BASE_NUMERIC + DPE_NUMERIC, BASE_CAT + DPE_CATEGORICAL),
    "building": (BASE_NUMERIC + BUILDING_NUMERIC, BASE_CAT + BUILDING_CATEGORICAL),
    "context": (BASE_NUMERIC + CONTEXT_NUMERIC, BASE_CAT),
    "full_no_identity": (
        BASE_NUMERIC + DPE_NUMERIC + BUILDING_NUMERIC + CONTEXT_NUMERIC,
        BASE_CAT + DPE_CATEGORICAL + BUILDING_CATEGORICAL,
    ),
    "full_identity": (
        BASE_NUMERIC + DPE_NUMERIC + BUILDING_NUMERIC + CONTEXT_NUMERIC,
        BASE_CAT + DPE_CATEGORICAL + BUILDING_CATEGORICAL + IDENTITY_CATEGORICAL,
    ),
}


def dump_joblib_atomic(artifact: Any, path: Path) -> None:
    """Serialize a model without replacing a valid artifact on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        joblib.dump(artifact, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_categories(data: pd.DataFrame) -> pd.DataFrame:
    output, _ = prepare_categorical_data(data)
    all_categories = set(
        BASE_CAT + DPE_CATEGORICAL + BUILDING_CATEGORICAL + IDENTITY_CATEGORICAL
    )
    for column in all_categories:
        output[column] = output[column].astype("string").fillna("__MISSING__")
        if column not in IDENTITY_CATEGORICAL:
            output[column] = output[column].astype("category")
    return output


def make_lightgbm(loss: str, configuration: dict[str, Any]) -> LGBMRegressor:
    objective = "huber" if loss == "huber" else "regression_l1"
    return LGBMRegressor(
        **configuration,
        n_estimators=650,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        objective=objective,
        n_jobs=-1,
        random_state=42,
        verbosity=-1,
    )


def fit_lightgbm(
    train: pd.DataFrame,
    target: pd.DataFrame,
    feature_set: str,
    loss: str,
    configuration: dict[str, Any],
) -> tuple[Any, np.ndarray, float]:
    numeric, categorical = FEATURE_SETS[feature_set]
    features = numeric + categorical
    start = perf_counter()
    model = make_lightgbm(loss, configuration)
    model.fit(
        train[features],
        np.log1p(train["valeur_fonciere"]),
        categorical_feature=categorical,
    )
    predicted = np.maximum(np.expm1(model.predict(target[features])), 0)
    return model, predicted, perf_counter() - start


def catboost_frame(data: pd.DataFrame, features: list[str], categorical: list[str]) -> pd.DataFrame:
    output = data[features].copy()
    for column in categorical:
        output[column] = output[column].astype("string").fillna("__MISSING__")
    for column in set(features) - set(categorical):
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(np.nan)
    return output


def fit_catboost(
    train: pd.DataFrame,
    target: pd.DataFrame,
    configuration: dict[str, Any],
) -> tuple[Any, np.ndarray, float]:
    numeric, categorical = FEATURE_SETS["full_identity"]
    features = numeric + categorical
    start = perf_counter()
    model = CatBoostRegressor(
        **configuration,
        iterations=650,
        learning_rate=0.05,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    train_x = catboost_frame(train, features, categorical)
    target_x = catboost_frame(target, features, categorical)
    model.fit(
        train_x,
        np.log1p(train["valeur_fonciere"]),
        cat_features=categorical,
    )
    predicted = np.maximum(np.expm1(model.predict(target_x)), 0)
    return model, predicted, perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=PROJECT_ROOT / "data/gold/phase3_sale_features.parquet"
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports/phase3")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models/phase3")
    parser.add_argument("--bootstrap-repetitions", type=int, default=500)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_categories(pd.read_parquet(args.features))
    development = data.loc[data["year"].le(2023)].copy()
    validation = data.loc[data["year"].eq(2024)].copy()
    final_train = data.loc[data["year"].le(2024)].copy()
    test = data.loc[data["year"].eq(2025)].copy()

    configurations = [
        {"num_leaves": 31, "min_child_samples": 40, "reg_lambda": 2.0},
        {"num_leaves": 63, "min_child_samples": 60, "reg_lambda": 3.0},
    ]
    validation_trials: list[dict[str, Any]] = []
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    validation_predictions: dict[str, np.ndarray] = {}

    for feature_set in ("phase2", "dpe_safe", "building", "context", "full_no_identity"):
        for loss in ("mae", "huber"):
            key = f"lightgbm_{feature_set}_{loss}"
            best: tuple[float, dict[str, Any], np.ndarray] | None = None
            for candidate, configuration in enumerate(configurations, start=1):
                _, prediction, seconds = fit_lightgbm(
                    development, validation, feature_set, loss, configuration
                )
                metrics = regression_metrics(validation["valeur_fonciere"], prediction)
                validation_trials.append(
                    {
                        "model_key": key,
                        "feature_set": feature_set,
                        "loss": loss,
                        "candidate": candidate,
                        "configuration": json.dumps(configuration, sort_keys=True),
                        "fit_predict_seconds": round(seconds, 3),
                        **metrics,
                    }
                )
                if best is None or metrics["mae_eur"] < best[0]:
                    best = (metrics["mae_eur"], configuration.copy(), prediction)
            assert best is not None
            selected[(feature_set, loss)] = best[1]
            validation_predictions[key] = best[2]
            print(f"{key}: validation MAE EUR {best[0]:,.0f}", flush=True)

    catboost_configs = [
        {"depth": 7, "l2_leaf_reg": 5.0, "loss_function": "MAE"},
        {"depth": 8, "l2_leaf_reg": 8.0, "loss_function": "Huber:delta=1.0"},
    ]
    best_catboost: tuple[float, dict[str, Any], np.ndarray] | None = None
    for candidate, configuration in enumerate(catboost_configs, start=1):
        _, prediction, seconds = fit_catboost(development, validation, configuration)
        metrics = regression_metrics(validation["valeur_fonciere"], prediction)
        validation_trials.append(
            {
                "model_key": "catboost_full_identity",
                "feature_set": "full_identity",
                "loss": configuration["loss_function"],
                "candidate": candidate,
                "configuration": json.dumps(configuration, sort_keys=True),
                "fit_predict_seconds": round(seconds, 3),
                **metrics,
            }
        )
        if best_catboost is None or metrics["mae_eur"] < best_catboost[0]:
            best_catboost = (metrics["mae_eur"], configuration.copy(), prediction)
    assert best_catboost is not None
    validation_predictions["catboost_full_identity"] = best_catboost[2]
    print(
        f"catboost_full_identity: validation MAE EUR {best_catboost[0]:,.0f}",
        flush=True,
    )

    # Refit every selected ablation on 2021--2024 and evaluate once on 2025.
    results: list[dict[str, Any]] = []
    test_predictions: dict[str, np.ndarray] = {}
    fitted: dict[str, tuple[Any, str, str, dict[str, Any]]] = {}
    for (feature_set, loss), configuration in selected.items():
        key = f"lightgbm_{feature_set}_{loss}"
        model, prediction, seconds = fit_lightgbm(
            final_train, test, feature_set, loss, configuration
        )
        test_predictions[key] = prediction
        fitted[key] = (model, "lightgbm", feature_set, configuration)
        results.append(
            {
                "model_key": key,
                "family": "LightGBM",
                "feature_set": feature_set,
                "loss": loss,
                "selected_configuration": json.dumps(configuration, sort_keys=True),
                "validation_mae_eur": regression_metrics(
                    validation["valeur_fonciere"], validation_predictions[key]
                )["mae_eur"],
                "final_fit_predict_seconds": round(seconds, 3),
                **regression_metrics(test["valeur_fonciere"], prediction),
            }
        )

    model, prediction, seconds = fit_catboost(final_train, test, best_catboost[1])
    test_predictions["catboost_full_identity"] = prediction
    fitted["catboost_full_identity"] = (
        model,
        "catboost",
        "full_identity",
        best_catboost[1],
    )
    results.append(
        {
            "model_key": "catboost_full_identity",
            "family": "CatBoost",
            "feature_set": "full_identity",
            "loss": best_catboost[1]["loss_function"],
            "selected_configuration": json.dumps(best_catboost[1], sort_keys=True),
            "validation_mae_eur": best_catboost[0],
            "final_fit_predict_seconds": round(seconds, 3),
            **regression_metrics(test["valeur_fonciere"], prediction),
        }
    )

    # Phase 2's selected transparent correction is the honest reference.
    phase2_test = pd.read_csv(
        PROJECT_ROOT / "reports/phase2/phase2_test_predictions.csv"
    )
    reference = test[["id_mutation"]].merge(
        phase2_test[["id_mutation", "prediction_phase2_comparable_correction"]],
        on="id_mutation",
        validate="one_to_one",
    )["prediction_phase2_comparable_correction"].to_numpy()
    test_predictions["phase2_comparable_correction"] = reference
    results.append(
        {
            "model_key": "phase2_comparable_correction",
            "family": "Reference",
            "feature_set": "phase2_selected",
            "loss": "blend",
            "selected_configuration": "frozen Phase 2",
            "validation_mae_eur": 95337.07,
            "final_fit_predict_seconds": 0.0,
            **regression_metrics(test["valeur_fonciere"], reference),
        }
    )

    actual = test["valeur_fonciere"].to_numpy()
    intervals = bootstrap_intervals(
        actual,
        test_predictions,
        baseline_key="phase2_comparable_correction",
        repetitions=args.bootstrap_repetitions,
    )
    for row in results:
        row.update(intervals[row["model_key"]])
    comparison = pd.DataFrame(results).sort_values("mae_eur").reset_index(drop=True)
    comparison.insert(0, "mae_rank", np.arange(1, len(comparison) + 1))
    write_csv_atomic(
        comparison, args.output_dir / "phase3_model_comparison.csv"
    )
    write_csv_atomic(
        pd.DataFrame(validation_trials),
        args.output_dir / "phase3_validation_search.csv",
    )

    predictions = test[
        ["id_mutation", "date_mutation", "code_postal", "surface_reelle_bati", "valeur_fonciere"]
    ].copy()
    for key, values in test_predictions.items():
        predictions[f"prediction_{key}"] = np.round(values, 2)
    write_csv_atomic(predictions, args.output_dir / "phase3_test_predictions.csv")

    winner_key = comparison.iloc[0]["model_key"]
    if winner_key in fitted:
        model, family, feature_set, configuration = fitted[winner_key]
        numeric, categorical = FEATURE_SETS[feature_set]
        artifact = {
            "model": model,
            "model_family": family,
            "feature_set": feature_set,
            "features": numeric + categorical,
            "categorical_features": categorical,
            "configuration": configuration,
            "training_period": "2021-2024",
            "test_period": "2025",
            "retrospective_static_context": feature_set
            in {"building", "context", "full_no_identity", "full_identity"},
        }
        dump_joblib_atomic(
            artifact, args.model_dir / "phase3_selected_model.joblib"
        )

    summary = {
        "schema_version": 1,
        "protocol": {
            "development": "2021-2023",
            "validation_selection": "2024",
            "final_test": "2025",
        },
        "winner": comparison.iloc[0].to_dict(),
        "reference_2025_mae_eur": 96428.47,
        "interpretation_rule": (
            "DPE-only features are point-in-time; building/context/identity features "
            "use current static snapshots and are reported as retrospective reconstruction."
        ),
    }
    write_json_atomic(
        args.output_dir / "phase3_results.json", summary, default=str
    )
    print(comparison[["mae_rank", "model_key", "mae_eur", "r2"]].to_string(index=False))


if __name__ == "__main__":
    main()
