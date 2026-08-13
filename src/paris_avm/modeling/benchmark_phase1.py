"""Benchmark six Paris valuation methods under one chronological protocol.

The benchmark selects each learned model using 2024 validation MAE, refits the
selected configuration on 2021--2024, and evaluates it on the common 2025 test
cohort. All learned models predict log transaction value from identical input
features. The arrondissement median-price-per-m2 method is the non-ML baseline.
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from paris_avm.paths import PROJECT_ROOT
from xgboost import XGBRegressor

from paris_avm.modeling.train_phase1 import (
    CATEGORICAL_FEATURES,
    DEFAULT_DATA_FILES,
    FEATURES,
    NUMERIC_FEATURES,
    load_and_clean_many,
    median_price_m2_baseline,
    regression_metrics,
)


MODEL_LABELS = {
    "median_price_m2": "Median price/m² baseline",
    "ridge": "Ridge regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "Histogram Gradient Boosting",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}

MODEL_FAMILIES = {
    "median_price_m2": "Market heuristic",
    "ridge": "Regularized linear model",
    "random_forest": "Bagged decision trees",
    "hist_gradient_boosting": "Gradient-boosted trees",
    "xgboost": "Gradient-boosted trees",
    "lightgbm": "Gradient-boosted trees",
}

SEARCH_SPACES: dict[str, list[dict[str, Any]]] = {
    "ridge": [
        {"alpha": 0.1},
        {"alpha": 1.0},
        {"alpha": 10.0},
        {"alpha": 100.0},
    ],
    "random_forest": [
        {"n_estimators": 240, "max_features": 0.7, "min_samples_leaf": 2},
        {"n_estimators": 240, "max_features": 1.0, "min_samples_leaf": 2},
        {"n_estimators": 240, "max_features": 0.7, "min_samples_leaf": 5},
    ],
    "hist_gradient_boosting": [
        {"max_leaf_nodes": 15, "min_samples_leaf": 25, "l2_regularization": 1.0},
        {"max_leaf_nodes": 31, "min_samples_leaf": 25, "l2_regularization": 1.0},
        {"max_leaf_nodes": 63, "min_samples_leaf": 40, "l2_regularization": 2.0},
    ],
    "xgboost": [
        {"max_depth": 4, "min_child_weight": 5.0},
        {"max_depth": 6, "min_child_weight": 5.0},
        {"max_depth": 8, "min_child_weight": 10.0},
    ],
    "lightgbm": [
        {"num_leaves": 15, "min_child_samples": 25},
        {"num_leaves": 31, "min_child_samples": 25},
        {"num_leaves": 63, "min_child_samples": 40},
    ],
}


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )


def make_model(model_key: str, configuration: dict[str, Any]) -> Any:
    if model_key == "ridge":
        estimator = Ridge(alpha=configuration["alpha"])
        scale_numeric = True
    elif model_key == "random_forest":
        estimator = RandomForestRegressor(
            **configuration,
            max_depth=None,
            bootstrap=True,
            n_jobs=-1,
            random_state=42,
        )
        scale_numeric = False
    elif model_key == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(
            **configuration,
            learning_rate=0.06,
            max_iter=450,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
        )
        scale_numeric = False
    elif model_key == "xgboost":
        estimator = XGBRegressor(
            **configuration,
            n_estimators=650,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            eval_metric="mae",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        )
        scale_numeric = False
    elif model_key == "lightgbm":
        estimator = LGBMRegressor(
            **configuration,
            n_estimators=650,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="regression",
            n_jobs=-1,
            random_state=42,
            verbosity=-1,
        )
        scale_numeric = False
    else:
        raise KeyError(f"Unknown model: {model_key}")

    pipeline = Pipeline(
        [("preprocess", make_preprocessor(scale_numeric)), ("regressor", estimator)]
    )
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def timed_predict(model: Any, features: pd.DataFrame) -> tuple[np.ndarray, float]:
    start = perf_counter()
    prediction = np.maximum(model.predict(features), 0)
    duration = perf_counter() - start
    return prediction, duration


def bootstrap_intervals(
    actual: np.ndarray,
    predictions: dict[str, np.ndarray],
    baseline_key: str,
    repetitions: int,
) -> dict[str, dict[str, float]]:
    """Return paired bootstrap intervals for MAE and reduction versus baseline."""
    keys = list(predictions)
    absolute_errors = np.column_stack(
        [np.abs(actual - predictions[key]) for key in keys]
    )
    baseline_index = keys.index(baseline_key)
    rng = np.random.default_rng(20250811)
    bootstrap_mae = np.empty((repetitions, len(keys)), dtype=float)
    batch_size = 25
    for start in range(0, repetitions, batch_size):
        stop = min(start + batch_size, repetitions)
        indices = rng.integers(0, len(actual), size=(stop - start, len(actual)))
        bootstrap_mae[start:stop] = absolute_errors[indices].mean(axis=1)

    output: dict[str, dict[str, float]] = {}
    baseline_samples = bootstrap_mae[:, baseline_index]
    for index, key in enumerate(keys):
        model_samples = bootstrap_mae[:, index]
        reduction = baseline_samples - model_samples
        output[key] = {
            "mae_ci95_lower_eur": round(float(np.quantile(model_samples, 0.025)), 2),
            "mae_ci95_upper_eur": round(float(np.quantile(model_samples, 0.975)), 2),
            "mae_reduction_vs_baseline_eur": round(
                float(absolute_errors[:, baseline_index].mean() - absolute_errors[:, index].mean()),
                2,
            ),
            "reduction_ci95_lower_eur": round(float(np.quantile(reduction, 0.025)), 2),
            "reduction_ci95_upper_eur": round(float(np.quantile(reduction, 0.975)), 2),
        }
    return output


def save_test_predictions(
    test: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    output = test[
        [
            "id_mutation",
            "date_mutation",
            "code_postal",
            "surface_reelle_bati",
            "nombre_pieces_principales",
            "latitude",
            "longitude",
            "valeur_fonciere",
        ]
    ].copy()
    for key, predicted in predictions.items():
        output[f"prediction_{key}"] = np.round(predicted, 2)
    output.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", type=Path, default=DEFAULT_DATA_FILES)
    parser.add_argument("--train-end-year", type=int, default=2023)
    parser.add_argument("--validation-year", type=int, default=2024)
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--bootstrap-repetitions", type=int, default=500)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "reports/benchmark"
    )
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models/benchmark")
    args = parser.parse_args()

    total_start = perf_counter()
    data, counts = load_and_clean_many(args.data)
    development_train = data.loc[data["year"].le(args.train_end_year)].copy()
    validation = data.loc[data["year"].eq(args.validation_year)].copy()
    test = data.loc[data["year"].eq(args.test_year)].copy()
    final_train = data.loc[data["year"].le(args.validation_year)].copy()
    if min(len(development_train), len(validation), len(test)) < 1_000:
        raise RuntimeError("At least one temporal cohort contains fewer than 1,000 rows.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    validation_trials: list[dict[str, Any]] = []
    selected_configurations: dict[str, dict[str, Any]] = {}
    tuning_seconds: dict[str, float] = {}

    for model_key, configurations in SEARCH_SPACES.items():
        model_tuning_start = perf_counter()
        best_mae = float("inf")
        best_configuration: dict[str, Any] | None = None
        print(f"\nSelecting {MODEL_LABELS[model_key]} ({len(configurations)} candidates)")
        for candidate_number, configuration in enumerate(configurations, start=1):
            candidate = make_model(model_key, configuration)
            fit_start = perf_counter()
            candidate.fit(
                development_train[FEATURES], development_train["valeur_fonciere"]
            )
            fit_seconds = perf_counter() - fit_start
            validation_prediction, prediction_seconds = timed_predict(
                candidate, validation[FEATURES]
            )
            metrics = regression_metrics(
                validation["valeur_fonciere"].to_numpy(), validation_prediction
            )
            trial = {
                "model_key": model_key,
                "model": MODEL_LABELS[model_key],
                "candidate": candidate_number,
                "configuration": json.dumps(configuration, sort_keys=True),
                "fit_seconds": round(fit_seconds, 3),
                "prediction_seconds": round(prediction_seconds, 3),
                **metrics,
            }
            validation_trials.append(trial)
            print(
                f"  {candidate_number}/{len(configurations)} "
                f"MAE=€{metrics['mae_eur']:,.0f}, "
                f"MdAPE={metrics['median_absolute_percentage_error']:.2f}%, "
                f"fit={fit_seconds:.1f}s"
            )
            if metrics["mae_eur"] < best_mae:
                best_mae = metrics["mae_eur"]
                best_configuration = configuration.copy()
        if best_configuration is None:
            raise RuntimeError(f"No configuration selected for {model_key}")
        selected_configurations[model_key] = best_configuration
        tuning_seconds[model_key] = perf_counter() - model_tuning_start

    validation_trials_frame = pd.DataFrame(validation_trials)
    validation_trials_frame.to_csv(
        args.output_dir / "validation_search_results.csv", index=False
    )

    test_actual = test["valeur_fonciere"].to_numpy()
    validation_actual = validation["valeur_fonciere"].to_numpy()
    test_predictions: dict[str, np.ndarray] = {}
    validation_predictions: dict[str, np.ndarray] = {}
    results: list[dict[str, Any]] = []

    # Transparent market baseline.
    baseline_fit_start = perf_counter()
    postcode_medians = final_train.groupby("code_postal")["price_per_m2"].median()
    global_median = float(final_train["price_per_m2"].median())
    baseline_fit_seconds = perf_counter() - baseline_fit_start
    baseline_prediction_start = perf_counter()
    test_baseline = median_price_m2_baseline(final_train, test)
    baseline_prediction_seconds = perf_counter() - baseline_prediction_start
    validation_baseline = median_price_m2_baseline(development_train, validation)
    baseline_artifact = {
        "model_key": "median_price_m2",
        "postcode_median_price_per_m2": postcode_medians.to_dict(),
        "global_median_price_per_m2": global_median,
        "training_end_year": args.validation_year,
    }
    baseline_path = args.model_dir / "median_price_m2.joblib"
    joblib.dump(baseline_artifact, baseline_path)
    test_predictions["median_price_m2"] = test_baseline
    validation_predictions["median_price_m2"] = validation_baseline
    results.append(
        {
            "model_key": "median_price_m2",
            "model": MODEL_LABELS["median_price_m2"],
            "family": MODEL_FAMILIES["median_price_m2"],
            "selected_configuration": "arrondissement median €/m²",
            "validation_mae_eur": regression_metrics(
                validation_actual, validation_baseline
            )["mae_eur"],
            "tuning_seconds": 0.0,
            "final_fit_seconds": round(baseline_fit_seconds, 3),
            "test_prediction_seconds": round(baseline_prediction_seconds, 3),
            "prediction_ms_per_1000": round(
                baseline_prediction_seconds / len(test) * 1_000_000, 3
            ),
            "artifact_size_mb": round(baseline_path.stat().st_size / 1_048_576, 4),
            **regression_metrics(test_actual, test_baseline),
        }
    )

    # Refit each selected learned model through 2024 and evaluate the same 2025 rows.
    for model_key, configuration in selected_configurations.items():
        print(f"\nFinal fit: {MODEL_LABELS[model_key]} with {configuration}")
        model = make_model(model_key, configuration)
        fit_start = perf_counter()
        model.fit(final_train[FEATURES], final_train["valeur_fonciere"])
        fit_seconds = perf_counter() - fit_start
        test_prediction, prediction_seconds = timed_predict(model, test[FEATURES])

        # Retain the genuine 2024 validation prediction from a model trained only
        # through 2023 for calibration and diagnostics.
        validation_model = make_model(model_key, configuration)
        validation_model.fit(
            development_train[FEATURES], development_train["valeur_fonciere"]
        )
        validation_prediction = np.maximum(
            validation_model.predict(validation[FEATURES]), 0
        )

        model_path = args.model_dir / f"{model_key}.joblib"
        joblib.dump(
            {
                "model": model,
                "features": FEATURES,
                "model_key": model_key,
                "configuration": configuration,
                "training_period": f"2021-{args.validation_year}",
            },
            model_path,
        )
        test_predictions[model_key] = test_prediction
        validation_predictions[model_key] = validation_prediction
        validation_metrics = regression_metrics(
            validation_actual, validation_prediction
        )
        results.append(
            {
                "model_key": model_key,
                "model": MODEL_LABELS[model_key],
                "family": MODEL_FAMILIES[model_key],
                "selected_configuration": json.dumps(configuration, sort_keys=True),
                "validation_mae_eur": validation_metrics["mae_eur"],
                "tuning_seconds": round(tuning_seconds[model_key], 3),
                "final_fit_seconds": round(fit_seconds, 3),
                "test_prediction_seconds": round(prediction_seconds, 3),
                "prediction_ms_per_1000": round(
                    prediction_seconds / len(test) * 1_000_000, 3
                ),
                "artifact_size_mb": round(model_path.stat().st_size / 1_048_576, 4),
                **regression_metrics(test_actual, test_prediction),
            }
        )

    intervals = bootstrap_intervals(
        test_actual,
        test_predictions,
        baseline_key="median_price_m2",
        repetitions=args.bootstrap_repetitions,
    )
    for result in results:
        result.update(intervals[result["model_key"]])

    results_frame = pd.DataFrame(results).sort_values("mae_eur").reset_index(drop=True)
    results_frame.insert(0, "mae_rank", np.arange(1, len(results_frame) + 1))
    results_frame.to_csv(args.output_dir / "model_comparison.csv", index=False)
    save_test_predictions(
        test,
        test_predictions,
        args.output_dir / "test_predictions_all_models.csv",
    )

    validation_output = validation[
        ["id_mutation", "date_mutation", "code_postal", "valeur_fonciere"]
    ].copy()
    for key, predicted in validation_predictions.items():
        validation_output[f"prediction_{key}"] = np.round(predicted, 2)
    validation_output.to_csv(
        args.output_dir / "validation_predictions_all_models.csv", index=False
    )

    winner_key = str(results_frame.iloc[0]["model_key"])
    report = {
        "schema_version": 1,
        "benchmark_title": "Paris apartment valuation model benchmark",
        "scope": "Ordinary Paris apartment sales containing exactly one apartment",
        "protocol": {
            "development_training": f"2021-{args.train_end_year}",
            "model_selection": str(args.validation_year),
            "final_training": f"2021-{args.validation_year}",
            "common_test": str(args.test_year),
            "selection_metric": "validation MAE in euros",
            "target_for_learned_models": "log1p transaction value",
            "random_seed": 42,
            "bootstrap_seed": 20250811,
            "bootstrap_repetitions": args.bootstrap_repetitions,
        },
        "counts": counts,
        "cohort_rows": {
            "development_train": int(len(development_train)),
            "validation": int(len(validation)),
            "final_train": int(len(final_train)),
            "test": int(len(test)),
        },
        "features": FEATURES,
        "search_spaces": SEARCH_SPACES,
        "results": results_frame.to_dict(orient="records"),
        "winner_by_test_mae": winner_key,
        "winner_label": MODEL_LABELS[winner_key],
        "software": {
            package: version(package)
            for package in [
                "python",
                "numpy",
                "pandas",
                "scikit-learn",
                "xgboost",
                "lightgbm",
                "joblib",
            ]
            if package != "python"
        },
        "total_runtime_seconds": round(perf_counter() - total_start, 3),
        "limitations": [
            "The benchmark uses a compact, predefined hyperparameter search rather than an exhaustive optimization.",
            "The 2025 cohort was not used for fitting or hyperparameter selection, but its results had been inspected during earlier baseline development; it is therefore held out computationally but not fully blind to the analyst.",
            "DVF omits condition, floor, energy rating, view, occupancy, and detailed building quality.",
            "Reported bootstrap intervals quantify sampling variability on this cohort, not uncertainty under future market regime changes.",
        ],
    }
    (args.output_dir / "benchmark_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nFinal 2025 ranking by MAE")
    print(
        results_frame[
            [
                "mae_rank",
                "model",
                "mae_eur",
                "median_absolute_percentage_error",
                "r2",
                "within_20_percent",
                "final_fit_seconds",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved benchmark outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
