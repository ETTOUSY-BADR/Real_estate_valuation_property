"""Train and evaluate the nationwide region-aware Phase 4 CatBoost model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from paris_avm.analysis.phase4_preflight import fit_department_clusters
from paris_avm.artifacts import write_csv_atomic, write_json_atomic, write_parquet_atomic
from paris_avm.features.phase4 import (
    MAX_TRANSACTION_VALUE_EUR,
    MIN_TRANSACTION_VALUE_EUR,
    PHASE4_CATEGORICAL_FEATURES,
    PHASE4_MODEL_FEATURES,
    PHASE4_NUMERIC_FEATURES,
    file_sha256,
    validate_phase4_table,
)
from paris_avm.geography import REGION_NAMES
from paris_avm.modeling.benchmark_phase3 import dump_joblib_atomic
from paris_avm.modeling.train_phase1 import regression_metrics
from paris_avm.paths import PROJECT_ROOT


MARKET_CLUSTER_FEATURE = "department_market_cluster"


def safe_inverse_log_prediction(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Invert log1p predictions within the registered national value bounds."""
    raw = np.asarray(values, dtype=float)
    lower = float(np.log1p(MIN_TRANSACTION_VALUE_EUR))
    upper = float(np.log1p(MAX_TRANSACTION_VALUE_EUR))
    invalid = (~np.isfinite(raw)) | (raw < lower) | (raw > upper)
    bounded = np.nan_to_num(raw, nan=lower, posinf=upper, neginf=lower)
    bounded = np.clip(bounded, lower, upper)
    return np.expm1(bounded), int(invalid.sum())


def prepare_frame(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    for column in PHASE4_CATEGORICAL_FEATURES:
        output[column] = output[column].astype("string").fillna("__MISSING__")
    for column in PHASE4_NUMERIC_FEATURES:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def make_model(configuration: dict[str, Any]) -> CatBoostRegressor:
    return CatBoostRegressor(
        **configuration,
        iterations=700,
        learning_rate=0.055,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def fit_predict(
    train: pd.DataFrame,
    target: pd.DataFrame,
    configuration: dict[str, Any],
    *,
    features: tuple[str, ...] = PHASE4_MODEL_FEATURES,
    categorical_features: tuple[str, ...] = PHASE4_CATEGORICAL_FEATURES,
) -> tuple[CatBoostRegressor, np.ndarray, float, int]:
    started = perf_counter()
    model = make_model(configuration)
    model.fit(
        train[list(features)],
        np.log1p(train["valeur_fonciere"]),
        cat_features=list(categorical_features),
    )
    prediction, clipped_predictions = safe_inverse_log_prediction(
        model.predict(target[list(features)])
    )
    return model, prediction, perf_counter() - started, clipped_predictions


def add_department_clusters(
    reference: pd.DataFrame, *frames: pd.DataFrame
) -> tuple[list[pd.DataFrame], dict[str, str], dict[str, Any]]:
    """Learn clusters from a reference period and apply them to supplied frames."""
    profiles, metadata = fit_department_clusters(reference)
    mapping = dict(
        zip(profiles["code_departement"].astype(str), profiles["market_cluster"])
    )
    transformed: list[pd.DataFrame] = []
    for frame in frames:
        output = frame.copy()
        output[MARKET_CLUSTER_FEATURE] = (
            output["code_departement"].astype(str).map(mapping).fillna("market_unknown")
        )
        transformed.append(output)
    return transformed, mapping, metadata


def hierarchical_baseline(reference: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    """Use only training-period median prices/m², falling back geographically."""
    estimate = pd.Series(np.nan, index=target.index, dtype=float)
    for column in ("code_postal", "code_commune", "code_departement", "code_region"):
        medians = reference.groupby(column, observed=True)["price_per_m2"].median()
        estimate = estimate.fillna(target[column].map(medians))
    estimate = estimate.fillna(float(reference["price_per_m2"].median()))
    return estimate.to_numpy() * target["surface_reelle_bati"].to_numpy(float)


def prediction_frame(
    data: pd.DataFrame, prediction: np.ndarray, baseline_prediction: np.ndarray
) -> pd.DataFrame:
    output = data[
        [
            "id_mutation",
            "date_mutation",
            "code_region",
            "code_departement",
            "code_commune",
            "code_postal",
            "surface_reelle_bati",
            "valeur_fonciere",
        ]
    ].copy()
    output["predicted_value"] = np.round(prediction, 2)
    output["baseline_predicted_value"] = np.round(baseline_prediction, 2)
    output["absolute_error"] = np.round(
        np.abs(output["valeur_fonciere"].to_numpy() - prediction), 2
    )
    output["baseline_absolute_error"] = np.round(
        np.abs(output["valeur_fonciere"].to_numpy() - baseline_prediction), 2
    )
    output["paired_improvement_eur"] = np.round(
        output["baseline_absolute_error"] - output["absolute_error"], 2
    )
    output["absolute_percentage_error"] = np.round(
        output["absolute_error"] / output["valeur_fonciere"] * 100, 2
    )
    return output


def paired_bootstrap_summary(
    actual: np.ndarray,
    prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    *,
    repetitions: int = 300,
    maximum_rows: int = 250_000,
) -> dict[str, float | int]:
    """Return bounded-memory paired MAE intervals on a deterministic sample."""
    if repetitions < 1 or maximum_rows < 1:
        raise ValueError("bootstrap repetitions and maximum_rows must be positive")
    model_error = np.abs(actual - prediction)
    baseline_error = np.abs(actual - baseline_prediction)
    rng = np.random.default_rng(20260826)
    if len(actual) > maximum_rows:
        sample = rng.choice(len(actual), size=maximum_rows, replace=False)
        model_error = model_error[sample]
        baseline_error = baseline_error[sample]
    model_samples = np.empty(repetitions, dtype=float)
    baseline_samples = np.empty(repetitions, dtype=float)
    batch_size = 10
    for start in range(0, repetitions, batch_size):
        stop = min(start + batch_size, repetitions)
        indices = rng.integers(0, len(model_error), size=(stop - start, len(model_error)))
        model_samples[start:stop] = model_error[indices].mean(axis=1)
        baseline_samples[start:stop] = baseline_error[indices].mean(axis=1)
    reduction = baseline_samples - model_samples
    return {
        "repetitions": repetitions,
        "sample_rows": int(len(model_error)),
        "model_mae_ci95_lower_eur": round(float(np.quantile(model_samples, 0.025)), 2),
        "model_mae_ci95_upper_eur": round(float(np.quantile(model_samples, 0.975)), 2),
        "baseline_mae_ci95_lower_eur": round(float(np.quantile(baseline_samples, 0.025)), 2),
        "baseline_mae_ci95_upper_eur": round(float(np.quantile(baseline_samples, 0.975)), 2),
        "reduction_ci95_lower_eur": round(float(np.quantile(reduction, 0.025)), 2),
        "reduction_ci95_upper_eur": round(float(np.quantile(reduction, 0.975)), 2),
    }


def segment_metrics(
    data: pd.DataFrame, prediction: np.ndarray, column: str, *, minimum_rows: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    predicted = pd.Series(prediction, index=data.index)
    for value, group in data.groupby(column, observed=True):
        if len(group) < minimum_rows:
            continue
        metrics = regression_metrics(
            group["valeur_fonciere"].to_numpy(), predicted.loc[group.index].to_numpy()
        )
        row: dict[str, object] = {column: value, "rows": int(len(group)), **metrics}
        if column == "code_region":
            row["region_name"] = REGION_NAMES.get(str(value), "Unknown")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mae_eur").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=PROJECT_ROOT / "data/gold/phase4_france_sale_features.parquet",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports/phase4")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models/phase4")
    parser.add_argument("--train-end-year", type=int, default=2023)
    parser.add_argument("--validation-year", type=int, default=2024)
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument(
        "--max-development-rows",
        type=int,
        help="Optional deterministic cap for rapid experiments; omit for the final run.",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=300)
    parser.add_argument("--bootstrap-max-rows", type=int, default=250_000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_frame(pd.read_parquet(args.features))
    validate_phase4_table(data)
    development = data.loc[data["year"].le(args.train_end_year)].copy()
    validation = data.loc[data["year"].eq(args.validation_year)].copy()
    final_train = data.loc[data["year"].le(args.validation_year)].copy()
    test = data.loc[data["year"].eq(args.test_year)].copy()
    if min(len(development), len(validation), len(test)) < 1_000:
        raise RuntimeError("Phase 4 requires at least 1,000 rows in each temporal split")
    if min(args.bootstrap_repetitions, args.bootstrap_max_rows) < 1:
        parser.error("bootstrap repetitions and maximum rows must be positive")
    if args.max_development_rows:
        if args.max_development_rows < 1_000:
            parser.error("--max-development-rows must be at least 1000")
        development = development.sample(
            min(args.max_development_rows, len(development)), random_state=42
        ).sort_values("date_mutation")

    cluster_frames, _, validation_cluster_metadata = add_department_clusters(
        development, development, validation
    )
    clustered_development, clustered_validation = cluster_frames
    configurations = (
        (False, {"depth": 8, "l2_leaf_reg": 8.0, "loss_function": "MAE"}),
        (False, {"depth": 9, "l2_leaf_reg": 10.0, "loss_function": "Huber:delta=1.0"}),
        (True, {"depth": 8, "l2_leaf_reg": 8.0, "loss_function": "MAE"}),
        (True, {"depth": 9, "l2_leaf_reg": 10.0, "loss_function": "Huber:delta=1.0"}),
    )
    validation_trials: list[dict[str, object]] = []
    winner: tuple[float, dict[str, Any], bool, np.ndarray] | None = None
    for candidate, (use_market_cluster, configuration) in enumerate(configurations, 1):
        candidate_train = clustered_development if use_market_cluster else development
        candidate_validation = clustered_validation if use_market_cluster else validation
        features = (
            PHASE4_MODEL_FEATURES + (MARKET_CLUSTER_FEATURE,)
            if use_market_cluster
            else PHASE4_MODEL_FEATURES
        )
        categoricals = (
            PHASE4_CATEGORICAL_FEATURES + (MARKET_CLUSTER_FEATURE,)
            if use_market_cluster
            else PHASE4_CATEGORICAL_FEATURES
        )
        _, prediction, seconds, clipped_predictions = fit_predict(
            candidate_train,
            candidate_validation,
            configuration,
            features=features,
            categorical_features=categoricals,
        )
        metrics = regression_metrics(validation["valeur_fonciere"].to_numpy(), prediction)
        validation_trials.append(
            {
                "candidate": candidate,
                "configuration": json.dumps(configuration, sort_keys=True),
                "market_cluster_feature": use_market_cluster,
                "market_clusters": (
                    validation_cluster_metadata["selected_clusters"]
                    if use_market_cluster
                    else 0
                ),
                "fit_predict_seconds": round(seconds, 3),
                "clipped_predictions": clipped_predictions,
                **metrics,
            }
        )
        if winner is None or metrics["mae_eur"] < winner[0]:
            winner = (metrics["mae_eur"], configuration.copy(), use_market_cluster, prediction)
        cluster_label = "with clusters" if use_market_cluster else "without clusters"
        print(
            f"Candidate {candidate} ({cluster_label}): validation MAE EUR {metrics['mae_eur']:,.0f}",
            flush=True,
        )
    assert winner is not None

    selected_features = PHASE4_MODEL_FEATURES
    selected_categoricals = PHASE4_CATEGORICAL_FEATURES
    evaluation_mapping: dict[str, str] = {}
    evaluation_cluster_metadata: dict[str, Any] | None = None
    evaluation_train = final_train
    evaluation_test = test
    if winner[2]:
        cluster_frames, evaluation_mapping, evaluation_cluster_metadata = add_department_clusters(
            final_train, final_train, test
        )
        evaluation_train, evaluation_test = cluster_frames
        selected_features = PHASE4_MODEL_FEATURES + (MARKET_CLUSTER_FEATURE,)
        selected_categoricals = PHASE4_CATEGORICAL_FEATURES + (MARKET_CLUSTER_FEATURE,)
    evaluation_model, test_prediction, test_seconds, test_clipped_predictions = fit_predict(
        evaluation_train,
        evaluation_test,
        winner[1],
        features=selected_features,
        categorical_features=selected_categoricals,
    )
    baseline_prediction = hierarchical_baseline(final_train, test)
    model_metrics = regression_metrics(test["valeur_fonciere"].to_numpy(), test_prediction)
    baseline_metrics = regression_metrics(test["valeur_fonciere"].to_numpy(), baseline_prediction)
    calibration_prediction = winner[3]
    calibration_ratios = validation["valeur_fonciere"].to_numpy() / np.maximum(
        calibration_prediction, 1
    )
    lower_ratio, upper_ratio = np.quantile(calibration_ratios, (0.05, 0.95))
    actual = test["valeur_fonciere"].to_numpy()
    coverage = np.mean(
        (actual >= test_prediction * lower_ratio) & (actual <= test_prediction * upper_ratio)
    )
    bootstrap = paired_bootstrap_summary(
        actual,
        test_prediction,
        baseline_prediction,
        repetitions=args.bootstrap_repetitions,
        maximum_rows=args.bootstrap_max_rows,
    )

    region_intervals: dict[str, dict[str, float | int]] = {}
    validation_with_prediction = validation.assign(_prediction=calibration_prediction)
    for region, group in validation_with_prediction.groupby("code_region", observed=True):
        if len(group) < 500:
            continue
        ratios = group["valeur_fonciere"] / np.maximum(group["_prediction"], 1)
        lower, upper = np.quantile(ratios, (0.05, 0.95))
        region_intervals[str(region)] = {
            "rows": int(len(group)),
            "lower_price_ratio": round(float(lower), 6),
            "upper_price_ratio": round(float(upper), 6),
        }

    feature_importance = pd.DataFrame(
        {
            "feature": list(selected_features),
            "importance": evaluation_model.get_feature_importance(),
        }
    ).sort_values("importance", ascending=False)
    region_performance = segment_metrics(test, test_prediction, "code_region", minimum_rows=100)
    department_performance = segment_metrics(
        test, test_prediction, "code_departement", minimum_rows=100
    )
    predictions = prediction_frame(test, test_prediction, baseline_prediction)
    benchmark_comparison = pd.DataFrame(
        [
            {
                "method": "Phase 4 CatBoost",
                **model_metrics,
                "mae_ci95_lower_eur": bootstrap["model_mae_ci95_lower_eur"],
                "mae_ci95_upper_eur": bootstrap["model_mae_ci95_upper_eur"],
                "mae_reduction_vs_baseline_eur": round(
                    baseline_metrics["mae_eur"] - model_metrics["mae_eur"], 2
                ),
                "reduction_ci95_lower_eur": bootstrap["reduction_ci95_lower_eur"],
                "reduction_ci95_upper_eur": bootstrap["reduction_ci95_upper_eur"],
            },
            {
                "method": "Hierarchical geographic median baseline",
                **baseline_metrics,
                "mae_ci95_lower_eur": bootstrap["baseline_mae_ci95_lower_eur"],
                "mae_ci95_upper_eur": bootstrap["baseline_mae_ci95_upper_eur"],
                "mae_reduction_vs_baseline_eur": 0.0,
                "reduction_ci95_lower_eur": 0.0,
                "reduction_ci95_upper_eur": 0.0,
            },
        ]
    )

    report: dict[str, object] = {
        "schema_version": 1,
        "phase": 4,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "One nationwide model for ordinary apartment sales available in Geo-DVF",
        "model_family": "CatBoostRegressor on log transaction value",
        "feature_table": str(args.features.resolve()),
        "feature_table_sha256": file_sha256(args.features),
        "protocol": {
            "development": f"through {args.train_end_year}",
            "validation": args.validation_year,
            "test_training": f"through {args.validation_year}",
            "untouched_test": args.test_year,
        },
        "rows": {
            "development": int(len(development)),
            "validation": int(len(validation)),
            "test_training": int(len(final_train)),
            "test": int(len(test)),
            "production": int(len(data)),
        },
        "selected_configuration": winner[1],
        "selected_market_cluster_feature": winner[2],
        "market_clustering": {
            "validation_fit": validation_cluster_metadata,
            "evaluation_fit": evaluation_cluster_metadata,
            "unseen_test_departments": sorted(
                set(test["code_departement"].astype(str)).difference(evaluation_mapping)
            ) if winner[2] else [],
        },
        "validation_metrics": regression_metrics(
            validation["valeur_fonciere"].to_numpy(), calibration_prediction
        ),
        "test_metrics": model_metrics,
        "baseline_test_metrics": baseline_metrics,
        "mae_reduction_vs_baseline_eur": round(
            baseline_metrics["mae_eur"] - model_metrics["mae_eur"], 2
        ),
        "paired_bootstrap": bootstrap,
        "test_fit_predict_seconds": round(test_seconds, 3),
        "test_clipped_predictions": test_clipped_predictions,
        "prediction_interval": {
            "nominal_coverage": 0.9,
            "observed_test_coverage": round(float(coverage), 4),
            "lower_price_ratio": round(float(lower_ratio), 6),
            "upper_price_ratio": round(float(upper_ratio), 6),
            "regional_calibration": region_intervals,
        },
        "features": list(selected_features),
        "limitations": [
            "DVF has no open data for departments 57, 67, 68 and 976.",
            "The model covers apartments, not houses, commercial property, or certified appraisals.",
            "DVF omits condition, floor, view, occupancy, and renovation quality.",
            "Sparse rural and overseas submarkets can have weak nearby-comparable coverage.",
        ],
    }

    write_csv_atomic(pd.DataFrame(validation_trials), args.output_dir / "validation_trials.csv")
    write_csv_atomic(benchmark_comparison, args.output_dir / "benchmark_comparison.csv")
    write_csv_atomic(feature_importance, args.output_dir / "feature_importance.csv")
    write_csv_atomic(region_performance, args.output_dir / "region_performance.csv")
    write_csv_atomic(department_performance, args.output_dir / "department_performance.csv")
    write_parquet_atomic(predictions, args.output_dir / "test_predictions.parquet")
    write_json_atomic(args.output_dir / "benchmark.json", report)

    production_data = data
    production_mapping: dict[str, str] = {}
    production_cluster_metadata: dict[str, Any] | None = None
    if winner[2]:
        cluster_frames, production_mapping, production_cluster_metadata = add_department_clusters(
            data, data
        )
        production_data = cluster_frames[0]
    production_model = make_model(winner[1])
    production_model.fit(
        production_data[list(selected_features)],
        np.log1p(production_data["valeur_fonciere"]),
        cat_features=list(selected_categoricals),
    )
    bundle = {
        "phase": 4,
        "model": production_model,
        "features": list(selected_features),
        "numeric_features": list(PHASE4_NUMERIC_FEATURES),
        "categorical_features": list(selected_categoricals),
        "department_market_cluster_mapping": production_mapping,
        "market_clustering": production_cluster_metadata,
        "training_date_min": data["date_mutation"].min().date().isoformat(),
        "training_date_max": data["date_mutation"].max().date().isoformat(),
        "scope": report["scope"],
        "prediction_interval": report["prediction_interval"],
        "feature_table_sha256": report["feature_table_sha256"],
    }
    model_path = args.model_dir / "france_catboost.joblib"
    dump_joblib_atomic(bundle, model_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved Phase 4 model: {model_path}")


if __name__ == "__main__":
    main()
