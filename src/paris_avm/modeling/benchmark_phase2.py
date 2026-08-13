"""Benchmark leakage-safe Phase 2 spatial and comparable-sale features."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from paris_avm.modeling.benchmark_phase1 import bootstrap_intervals
from paris_avm.modeling.train_phase1 import NUMERIC_FEATURES, regression_metrics
from paris_avm.paths import PROJECT_ROOT


SPATIAL_NUMERIC = [
    "x_l93",
    "y_l93",
    "distance_paris_center_m",
    "bearing_sin",
    "bearing_cos",
]
PROPERTY_ENGINEERED_NUMERIC = [
    "log_surface",
    "rooms_per_10m2",
    "surface_terrain",
]
COMPARABLE_NUMERIC = [
    "local_365d_250m_count",
    "local_365d_250m_median_ppm2",
    "local_180d_500m_count",
    "local_180d_500m_median_ppm2",
    "local_365d_500m_count",
    "local_365d_500m_median_ppm2",
    "local_730d_500m_count",
    "local_730d_500m_median_ppm2",
    "local_180d_1000m_count",
    "local_180d_1000m_median_ppm2",
    "local_365d_1000m_count",
    "local_365d_1000m_median_ppm2",
    "local_730d_1000m_count",
    "local_730d_1000m_median_ppm2",
    "local_365d_500m_mad_ppm2",
    "weighted_comp_ppm2",
    "similar_weighted_comp_ppm2",
    "similar_comparable_count",
    "effective_comparable_count",
    "nearest_prior_sale_m",
    "median_comparable_age_days",
    "local_momentum_180v365_500m",
    "local_momentum_180v365_1000m",
]
BASE_CATEGORICAL = ["code_postal"]
# Street and H3 resolution 10 remain in the Gold table for auditable entity
# history, but are deliberately not fitted as raw categories: their thousands
# of levels are expensive and invite micro-location memorization.
SPATIAL_CATEGORICAL = ["h3_r8", "h3_r9"]

FEATURE_SETS = {
    "base": (NUMERIC_FEATURES, BASE_CATEGORICAL),
    "spatial": (
        NUMERIC_FEATURES + PROPERTY_ENGINEERED_NUMERIC + SPATIAL_NUMERIC,
        BASE_CATEGORICAL + SPATIAL_CATEGORICAL,
    ),
    "comparables": (
        NUMERIC_FEATURES + PROPERTY_ENGINEERED_NUMERIC + COMPARABLE_NUMERIC,
        BASE_CATEGORICAL,
    ),
    "full": (
        NUMERIC_FEATURES
        + PROPERTY_ENGINEERED_NUMERIC
        + SPATIAL_NUMERIC
        + COMPARABLE_NUMERIC,
        BASE_CATEGORICAL + SPATIAL_CATEGORICAL,
    ),
}

SEARCH_SPACES = {
    "lightgbm": [
        {"num_leaves": 31, "min_child_samples": 25, "reg_lambda": 1.0},
        {"num_leaves": 63, "min_child_samples": 40, "reg_lambda": 2.0},
        {"num_leaves": 31, "min_child_samples": 60, "reg_lambda": 3.0},
    ],
    "xgboost": [
        {"max_depth": 4, "min_child_weight": 5.0, "reg_lambda": 1.0},
        {"max_depth": 6, "min_child_weight": 5.0, "reg_lambda": 2.0},
        {"max_depth": 8, "min_child_weight": 10.0, "reg_lambda": 3.0},
    ],
}


def prepare_categorical_data(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    output = data.copy()
    category_levels: dict[str, list[str]] = {}
    for column in set(BASE_CATEGORICAL + SPATIAL_CATEGORICAL):
        values = output[column].astype("string").fillna("__MISSING__")
        levels = sorted(values.unique().tolist())
        output[column] = pd.Categorical(values, categories=levels)
        category_levels[column] = levels
    return output, category_levels


def make_model(model_family: str, configuration: dict[str, Any]) -> Any:
    if model_family == "lightgbm":
        return LGBMRegressor(
            **configuration,
            n_estimators=500,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            objective="regression",
            n_jobs=-1,
            random_state=42,
            verbosity=-1,
        )
    if model_family == "xgboost":
        return XGBRegressor(
            **configuration,
            n_estimators=500,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            objective="reg:squarederror",
            eval_metric="mae",
            tree_method="hist",
            enable_categorical=True,
            max_cat_to_onehot=32,
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        )
    raise KeyError(model_family)


def target_values(data: pd.DataFrame, target_type: str) -> np.ndarray:
    if target_type == "total":
        return np.log1p(data["valeur_fonciere"].to_numpy())
    if target_type == "ppm2":
        return np.log1p(data["price_per_m2"].to_numpy())
    if target_type == "residual":
        anchor = data["local_365d_500m_median_ppm2"].to_numpy()
        return np.log(data["price_per_m2"].to_numpy() / anchor)
    raise KeyError(target_type)


def inverse_prediction(
    raw_prediction: np.ndarray, data: pd.DataFrame, target_type: str
) -> np.ndarray:
    transformed = np.maximum(np.expm1(raw_prediction), 0)
    if target_type == "ppm2":
        return transformed * data["surface_reelle_bati"].to_numpy()
    if target_type == "residual":
        return (
            np.exp(raw_prediction)
            * data["local_365d_500m_median_ppm2"].to_numpy()
            * data["surface_reelle_bati"].to_numpy()
        )
    return transformed


def fit_and_predict(
    model_family: str,
    configuration: dict[str, Any],
    train: pd.DataFrame,
    target: pd.DataFrame,
    feature_set: str,
    target_type: str,
) -> tuple[Any, np.ndarray, float, float]:
    numeric, categorical = FEATURE_SETS[feature_set]
    features = numeric + categorical
    model = make_model(model_family, configuration)
    fit_start = perf_counter()
    training_target = target_values(train, target_type)
    eligible_training = np.isfinite(training_target)
    model.fit(
        train.loc[eligible_training, features],
        training_target[eligible_training],
        categorical_feature=categorical if model_family == "lightgbm" else None,
    ) if model_family == "lightgbm" else model.fit(
        train.loc[eligible_training, features], training_target[eligible_training]
    )
    fit_seconds = perf_counter() - fit_start
    predict_start = perf_counter()
    predicted = inverse_prediction(model.predict(target[features]), target, target_type)
    prediction_seconds = perf_counter() - predict_start
    return model, predicted, fit_seconds, prediction_seconds


def select_blend_weights(
    actual: np.ndarray, predictions: dict[str, np.ndarray], step: float = 0.1
) -> tuple[dict[str, float], np.ndarray]:
    keys = list(predictions)
    matrix = np.column_stack([predictions[key] for key in keys])
    units = int(round(1 / step))
    best_mae = float("inf")
    best_weights: np.ndarray | None = None
    for combination in itertools.product(range(units + 1), repeat=len(keys)):
        if sum(combination) != units:
            continue
        weights = np.asarray(combination, dtype=float) / units
        prediction = matrix @ weights
        mae = float(np.mean(np.abs(actual - prediction)))
        if mae < best_mae:
            best_mae = mae
            best_weights = weights
    if best_weights is None:
        raise RuntimeError("No blend weights found")
    return dict(zip(keys, best_weights.tolist())), matrix @ best_weights


def gain_importance(model: Any, features: list[str], model_family: str) -> pd.DataFrame:
    if model_family == "lightgbm":
        values = model.booster_.feature_importance(importance_type="gain")
        names = model.booster_.feature_name()
        return pd.DataFrame({"feature": names, "gain": values})
    scores = model.get_booster().get_score(importance_type="gain")
    return pd.DataFrame(
        {"feature": features, "gain": [float(scores.get(feature, 0.0)) for feature in features]}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=PROJECT_ROOT / "data/gold/phase2_sale_features.parquet",
    )
    parser.add_argument(
        "--phase1-comparison",
        type=Path,
        default=PROJECT_ROOT / "reports/benchmark/model_comparison.csv",
    )
    parser.add_argument(
        "--phase1-predictions",
        type=Path,
        default=PROJECT_ROOT / "reports/benchmark/test_predictions_all_models.csv",
    )
    parser.add_argument(
        "--phase1-validation-predictions",
        type=Path,
        default=PROJECT_ROOT / "reports/benchmark/validation_predictions_all_models.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports/phase2")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models/phase2")
    parser.add_argument("--bootstrap-repetitions", type=int, default=500)
    args = parser.parse_args()

    started = perf_counter()
    data = pd.read_parquet(args.features)
    data, category_levels = prepare_categorical_data(data)
    development = data.loc[data["year"].le(2023)].copy()
    validation = data.loc[data["year"].eq(2024)].copy()
    final_train = data.loc[data["year"].le(2024)].copy()
    test = data.loc[data["year"].eq(2025)].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    trials: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    validation_predictions: dict[str, np.ndarray] = {}

    for family in ("lightgbm", "xgboost"):
        for target_type in ("total", "ppm2", "residual"):
            experiment_key = f"phase2_{family}_{target_type}"
            best_mae = float("inf")
            best_configuration: dict[str, Any] | None = None
            best_prediction: np.ndarray | None = None
            print(f"\nSelecting {experiment_key}", flush=True)
            for candidate_number, configuration in enumerate(SEARCH_SPACES[family], start=1):
                _, predicted, fit_seconds, prediction_seconds = fit_and_predict(
                    family,
                    configuration,
                    development,
                    validation,
                    feature_set="full",
                    target_type=target_type,
                )
                metrics = regression_metrics(
                    validation["valeur_fonciere"].to_numpy(), predicted
                )
                trials.append(
                    {
                        "experiment_key": experiment_key,
                        "family": family,
                        "target_type": target_type,
                        "candidate": candidate_number,
                        "configuration": json.dumps(configuration, sort_keys=True),
                        "fit_seconds": round(fit_seconds, 3),
                        "prediction_seconds": round(prediction_seconds, 3),
                        **metrics,
                    }
                )
                print(
                    f"  candidate {candidate_number}: MAE=€{metrics['mae_eur']:,.0f}, "
                    f"MdAPE={metrics['median_absolute_percentage_error']:.2f}%",
                    flush=True,
                )
                if metrics["mae_eur"] < best_mae:
                    best_mae = metrics["mae_eur"]
                    best_configuration = configuration.copy()
                    best_prediction = predicted
            if best_configuration is None or best_prediction is None:
                raise RuntimeError(f"No candidate selected for {experiment_key}")
            selected[experiment_key] = best_configuration
            validation_predictions[experiment_key] = best_prediction

    # Select an ensemble on validation predictions only.
    blend_weights, validation_blend = select_blend_weights(
        validation["valeur_fonciere"].to_numpy(), validation_predictions
    )
    validation_predictions["phase2_blend"] = validation_blend

    test_predictions: dict[str, np.ndarray] = {}
    results: list[dict[str, Any]] = []
    fitted_models: dict[str, tuple[Any, str, str, list[str]]] = {}

    for experiment_key, configuration in selected.items():
        _, family, target_type = experiment_key.split("_", maxsplit=2)
        model, predicted, fit_seconds, prediction_seconds = fit_and_predict(
            family,
            configuration,
            final_train,
            test,
            feature_set="full",
            target_type=target_type,
        )
        numeric, categorical = FEATURE_SETS["full"]
        features = numeric + categorical
        model_path = args.model_dir / f"{experiment_key}.joblib"
        joblib.dump(
            {
                "model": model,
                "model_family": family,
                "target_type": target_type,
                "feature_set": "full",
                "features": features,
                "categorical_features": categorical,
                "category_levels": {key: category_levels[key] for key in categorical},
                "configuration": configuration,
                "training_period": "2021-2024",
                "comparable_availability_lag_days": 90,
            },
            model_path,
        )
        test_predictions[experiment_key] = predicted
        fitted_models[experiment_key] = (model, family, target_type, features)
        results.append(
            {
                "model_key": experiment_key,
                "model": experiment_key.replace("phase2_", "").replace("_", " ").title(),
                "feature_set": "full",
                "target_type": target_type,
                "selected_configuration": json.dumps(configuration, sort_keys=True),
                "validation_mae_eur": regression_metrics(
                    validation["valeur_fonciere"].to_numpy(),
                    validation_predictions[experiment_key],
                )["mae_eur"],
                "final_fit_seconds": round(fit_seconds, 3),
                "prediction_ms_per_1000": round(
                    prediction_seconds / len(test) * 1_000_000, 3
                ),
                "artifact_size_mb": round(model_path.stat().st_size / 1_048_576, 4),
                **regression_metrics(test["valeur_fonciere"].to_numpy(), predicted),
            }
        )

    # Spatial-only and comparable-only ablations use the selected LightGBM-total settings.
    lgbm_configuration = selected["phase2_lightgbm_total"]
    for feature_set in ("spatial", "comparables"):
        experiment_key = f"phase2_lightgbm_{feature_set}_ablation"
        validation_model, validation_prediction, _, _ = fit_and_predict(
            "lightgbm",
            lgbm_configuration,
            development,
            validation,
            feature_set=feature_set,
            target_type="total",
        )
        model, predicted, fit_seconds, prediction_seconds = fit_and_predict(
            "lightgbm",
            lgbm_configuration,
            final_train,
            test,
            feature_set=feature_set,
            target_type="total",
        )
        del validation_model
        validation_predictions[experiment_key] = validation_prediction
        test_predictions[experiment_key] = predicted
        results.append(
            {
                "model_key": experiment_key,
                "model": f"LightGBM {feature_set}-only ablation",
                "feature_set": feature_set,
                "target_type": "total",
                "selected_configuration": json.dumps(lgbm_configuration, sort_keys=True),
                "validation_mae_eur": regression_metrics(
                    validation["valeur_fonciere"].to_numpy(), validation_prediction
                )["mae_eur"],
                "final_fit_seconds": round(fit_seconds, 3),
                "prediction_ms_per_1000": round(
                    prediction_seconds / len(test) * 1_000_000, 3
                ),
                "artifact_size_mb": np.nan,
                **regression_metrics(test["valeur_fonciere"].to_numpy(), predicted),
            }
        )

    test_matrix = np.column_stack(
        [test_predictions[key] for key in blend_weights]
    )
    weight_vector = np.asarray([blend_weights[key] for key in blend_weights])
    test_blend = test_matrix @ weight_vector
    test_predictions["phase2_blend"] = test_blend
    results.append(
        {
            "model_key": "phase2_blend",
            "model": "Phase 2 validation-weighted blend",
            "feature_set": "full",
            "target_type": "blend",
            "selected_configuration": json.dumps(blend_weights, sort_keys=True),
            "validation_mae_eur": regression_metrics(
                validation["valeur_fonciere"].to_numpy(), validation_blend
            )["mae_eur"],
            "final_fit_seconds": round(
                sum(
                    row["final_fit_seconds"]
                    for row in results
                    if row["model_key"] in blend_weights
                ),
                3,
            ),
            "prediction_ms_per_1000": round(
                sum(
                    row["prediction_ms_per_1000"]
                    for row in results
                    if row["model_key"] in blend_weights
                ),
                3,
            ),
            "artifact_size_mb": round(
                sum(
                    row["artifact_size_mb"]
                    for row in results
                    if row["model_key"] in blend_weights
                ),
                4,
            ),
            **regression_metrics(test["valeur_fonciere"].to_numpy(), test_blend),
        }
    )

    # Bring forward the Phase 1 winner as the frozen reference.
    phase1_comparison = pd.read_csv(args.phase1_comparison)
    phase1_row = phase1_comparison.loc[
        phase1_comparison["model_key"].eq("lightgbm")
    ].iloc[0]
    phase1_predictions_frame = pd.read_csv(args.phase1_predictions)
    phase1_validation_frame = pd.read_csv(args.phase1_validation_predictions)
    if not phase1_predictions_frame["id_mutation"].astype(str).equals(
        test["id_mutation"].astype(str).reset_index(drop=True)
    ):
        raise RuntimeError("Phase 1 and Phase 2 test rows are not aligned")
    phase1_prediction = phase1_predictions_frame["prediction_lightgbm"].to_numpy()
    if not phase1_validation_frame["id_mutation"].astype(str).equals(
        validation["id_mutation"].astype(str).reset_index(drop=True)
    ):
        raise RuntimeError("Phase 1 and Phase 2 validation rows are not aligned")
    phase1_validation_prediction = phase1_validation_frame[
        "prediction_lightgbm"
    ].to_numpy()
    test_predictions["phase1_lightgbm"] = phase1_prediction
    results.append(
        {
            "model_key": "phase1_lightgbm",
            "model": "Phase 1 LightGBM reference",
            "feature_set": "base",
            "target_type": "total",
            "selected_configuration": phase1_row["selected_configuration"],
            "validation_mae_eur": phase1_row["validation_mae_eur"],
            "final_fit_seconds": phase1_row["final_fit_seconds"],
            "prediction_ms_per_1000": phase1_row["prediction_ms_per_1000"],
            "artifact_size_mb": phase1_row["artifact_size_mb"],
            **regression_metrics(test["valeur_fonciere"].to_numpy(), phase1_prediction),
        }
    )

    # A transparent AVM/comparable blend: choose both the historical anchor and
    # its weight exclusively on 2024, then apply that frozen choice to 2025.
    comparable_candidates = {
        "weighted_comp_ppm2": "weighted_comp_ppm2",
        "similar_weighted_comp_ppm2": "similar_weighted_comp_ppm2",
        "local_365d_500m_median_ppm2": "local_365d_500m_median_ppm2",
        "local_365d_1000m_median_ppm2": "local_365d_1000m_median_ppm2",
    }
    blend_trials = []
    best_correction: dict[str, Any] | None = None
    for label, column in comparable_candidates.items():
        validation_comparable = (
            validation[column] * validation["surface_reelle_bati"]
        ).fillna(pd.Series(phase1_validation_prediction, index=validation.index)).to_numpy()
        test_comparable = (
            test[column] * test["surface_reelle_bati"]
        ).fillna(pd.Series(phase1_prediction, index=test.index)).to_numpy()
        for base_weight in np.linspace(0, 1, 101):
            validation_corrected = (
                base_weight * phase1_validation_prediction
                + (1 - base_weight) * validation_comparable
            )
            validation_mae = float(
                np.mean(
                    np.abs(
                        validation["valeur_fonciere"].to_numpy()
                        - validation_corrected
                    )
                )
            )
            trial = {
                "comparable_feature": label,
                "phase1_weight": float(base_weight),
                "validation_mae_eur": validation_mae,
            }
            blend_trials.append(trial)
            if best_correction is None or validation_mae < best_correction["validation_mae_eur"]:
                best_correction = {
                    **trial,
                    "test_prediction": (
                        base_weight * phase1_prediction
                        + (1 - base_weight) * test_comparable
                    ),
                }
    if best_correction is None:
        raise RuntimeError("No comparable correction selected")
    correction_prediction = best_correction.pop("test_prediction")
    test_predictions["phase2_comparable_correction"] = correction_prediction
    results.append(
        {
            "model_key": "phase2_comparable_correction",
            "model": "Phase 1 + historical comparable correction",
            "feature_set": "transparent_blend",
            "target_type": "total",
            "selected_configuration": json.dumps(best_correction, sort_keys=True),
            "validation_mae_eur": round(best_correction["validation_mae_eur"], 2),
            "final_fit_seconds": phase1_row["final_fit_seconds"],
            "prediction_ms_per_1000": phase1_row["prediction_ms_per_1000"],
            "artifact_size_mb": phase1_row["artifact_size_mb"],
            **regression_metrics(
                test["valeur_fonciere"].to_numpy(), correction_prediction
            ),
        }
    )
    pd.DataFrame(blend_trials).to_csv(
        args.output_dir / "comparable_correction_search.csv", index=False
    )

    intervals = bootstrap_intervals(
        test["valeur_fonciere"].to_numpy(),
        test_predictions,
        baseline_key="phase1_lightgbm",
        repetitions=args.bootstrap_repetitions,
    )
    for row in results:
        row.update(intervals[row["model_key"]])
    comparison = pd.DataFrame(results).sort_values("mae_eur").reset_index(drop=True)
    comparison.insert(0, "mae_rank", np.arange(1, len(comparison) + 1))
    comparison.to_csv(args.output_dir / "phase2_model_comparison.csv", index=False)
    pd.DataFrame(trials).to_csv(args.output_dir / "phase2_validation_search.csv", index=False)

    output_predictions = test[
        ["id_mutation", "date_mutation", "code_postal", "surface_reelle_bati", "valeur_fonciere"]
    ].copy()
    for key, values in test_predictions.items():
        output_predictions[f"prediction_{key}"] = np.round(values, 2)
    output_predictions.to_csv(args.output_dir / "phase2_test_predictions.csv", index=False)

    importance_frames = []
    for key, (model, family, target_type, features) in fitted_models.items():
        frame = gain_importance(model, features, family)
        frame["model_key"] = key
        frame["target_type"] = target_type
        frame["normalized_gain"] = frame["gain"] / frame["gain"].sum()
        importance_frames.append(frame)
    pd.concat(importance_frames, ignore_index=True).to_csv(
        args.output_dir / "phase2_feature_importance.csv", index=False
    )

    report = {
        "schema_version": 1,
        "title": "Phase 2 leakage-safe spatial and comparable-sale benchmark",
        "gold_feature_table": str(args.features.resolve()),
        "protocol": {
            "development_train": "2021-2023",
            "validation_and_selection": "2024",
            "final_train": "2021-2024",
            "test": "2025",
            "comparable_availability_lag_days": 90,
            "selection_metric": "2024 validation MAE",
            "blend_weight_step": 0.1,
        },
        "cohort_rows": {
            "development": int(len(development)),
            "validation": int(len(validation)),
            "final_train": int(len(final_train)),
            "test": int(len(test)),
        },
        "feature_sets": FEATURE_SETS,
        "selected_configurations": selected,
        "blend_weights": blend_weights,
        "results": comparison.to_dict(orient="records"),
        "runtime_seconds": round(perf_counter() - started, 3),
        "interpretation_constraint": "2025 is computationally held out but had been inspected during Phase 1; improvements require future external confirmation.",
    }
    (args.output_dir / "phase2_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nPhase 2 ranking", flush=True)
    print(
        comparison[
            [
                "mae_rank",
                "model",
                "mae_eur",
                "median_absolute_percentage_error",
                "r2",
                "within_20_percent",
                "mae_reduction_vs_baseline_eur",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(f"\nBlend weights: {blend_weights}", flush=True)
    print(f"Saved Phase 2 results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
