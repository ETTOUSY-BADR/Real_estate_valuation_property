"""Train a leakage-aware Paris apartment valuation model from 2021–2025 DVF.

DVF records a mutation-level sale value on one or more property rows. This
pipeline keeps transactions containing exactly one apartment, permits bundled
dependencies such as a cellar or parking space, and rejects transactions that
also contain a house or commercial property.

Temporal protocol:
    development train: 2021–2023
    validation/calibration: 2024
    final untouched test: 2025
    production artifact: refit on 2021–2025 after evaluation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from paris_avm.paths import PROJECT_ROOT, RAW_DVF_DIR


NUMERIC_FEATURES = [
    "surface_reelle_bati",
    "nombre_pieces_principales",
    "latitude",
    "longitude",
    "adresse_numero",
    "nombre_lots",
    "month",
    "day_of_year",
    "year",
    "months_since_2021",
]
CATEGORICAL_FEATURES = ["code_postal"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
DEFAULT_DATA_FILES = [RAW_DVF_DIR / f"75_{year}.csv" for year in range(2021, 2025)] + [
    RAW_DVF_DIR / "75.csv"
]


def load_and_clean(csv_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return one clean apartment observation per eligible transaction."""
    usecols = [
        "id_mutation",
        "date_mutation",
        "nature_mutation",
        "valeur_fonciere",
        "adresse_numero",
        "adresse_suffixe",
        "adresse_nom_voie",
        "adresse_code_voie",
        "code_postal",
        "code_commune",
        "id_parcelle",
        "code_type_local",
        "surface_reelle_bati",
        "nombre_pieces_principales",
        "nombre_lots",
        "surface_terrain",
        "longitude",
        "latitude",
    ]
    raw = pd.read_csv(
        csv_path,
        usecols=usecols,
        dtype={
            "id_mutation": "string",
            "adresse_suffixe": "string",
            "adresse_nom_voie": "string",
            "adresse_code_voie": "string",
            "code_postal": "string",
            "code_commune": "string",
            "id_parcelle": "string",
            "code_type_local": "string",
        },
        low_memory=False,
    )
    counts = {
        "raw_rows": int(len(raw)),
        "raw_mutations": int(raw["id_mutation"].nunique()),
    }

    raw["date_mutation"] = pd.to_datetime(raw["date_mutation"], errors="coerce")
    numeric_cols = [
        "valeur_fonciere",
        "adresse_numero",
        "surface_reelle_bati",
        "nombre_pieces_principales",
        "nombre_lots",
        "surface_terrain",
        "longitude",
        "latitude",
    ]
    for column in numeric_cols:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    # Stable DVF codes: 1=house, 2=apartment, 3=dependency, 4=commercial.
    sale_rows = raw.loc[raw["nature_mutation"].eq("Vente")].copy()
    transaction_summary = sale_rows.groupby("id_mutation")["code_type_local"].agg(
        apartment_count=lambda values: int(values.eq("2").sum()),
        incompatible_count=lambda values: int(values.isin(["1", "4"]).sum()),
    )
    eligible_ids = transaction_summary.index[
        transaction_summary["apartment_count"].eq(1)
        & transaction_summary["incompatible_count"].eq(0)
    ]
    apartments = sale_rows.loc[
        sale_rows["id_mutation"].isin(eligible_ids)
        & sale_rows["code_type_local"].eq("2")
    ].copy()
    counts["single_apartment_sale_rows"] = int(len(apartments))

    required = [
        "date_mutation",
        "valeur_fonciere",
        "surface_reelle_bati",
        "code_postal",
        "latitude",
        "longitude",
    ]
    apartments = apartments.dropna(subset=required)
    apartments["code_postal"] = (
        apartments["code_postal"].astype("Int64").astype("string")
    )
    apartments["price_per_m2"] = (
        apartments["valeur_fonciere"] / apartments["surface_reelle_bati"]
    )

    # Transparent fixed bounds keep evaluation cohorts comparable across years.
    apartments = apartments.loc[
        apartments["surface_reelle_bati"].between(9, 300)
        & apartments["valeur_fonciere"].between(50_000, 10_000_000)
        & apartments["price_per_m2"].between(2_000, 30_000)
        & apartments["code_postal"].str.fullmatch(r"750(?:0[1-9]|1[0-9]|20)", na=False)
        & apartments["latitude"].between(48.80, 48.92)
        & apartments["longitude"].between(2.20, 2.48)
    ].copy()
    apartments["month"] = apartments["date_mutation"].dt.month
    apartments["day_of_year"] = apartments["date_mutation"].dt.dayofyear
    apartments["year"] = apartments["date_mutation"].dt.year
    apartments["months_since_2021"] = (
        (apartments["year"] - 2021) * 12 + apartments["month"] - 1
    )
    apartments = apartments.sort_values(
        ["date_mutation", "id_mutation"]
    ).reset_index(drop=True)
    counts["clean_model_rows"] = int(len(apartments))
    counts["removed_during_quality_filters"] = (
        counts["single_apartment_sale_rows"] - counts["clean_model_rows"]
    )
    return apartments, counts


def load_and_clean_many(
    csv_paths: list[Path],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load annual files independently, validate IDs, then concatenate them."""
    frames: list[pd.DataFrame] = []
    per_file: dict[str, dict[str, int]] = {}
    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing annual data file: {csv_path}")
        frame, file_counts = load_and_clean(csv_path)
        frame["source_file"] = csv_path.name
        frames.append(frame)
        per_file[csv_path.name] = file_counts

    combined = pd.concat(frames, ignore_index=True)
    duplicated_ids = combined["id_mutation"].duplicated(keep=False)
    if duplicated_ids.any():
        examples = combined.loc[duplicated_ids, "id_mutation"].head(5).tolist()
        raise RuntimeError(f"Mutation IDs overlap across annual files: {examples}")
    combined = combined.sort_values(
        ["date_mutation", "id_mutation"]
    ).reset_index(drop=True)

    count_fields = [
        "raw_rows",
        "raw_mutations",
        "single_apartment_sale_rows",
        "clean_model_rows",
        "removed_during_quality_filters",
    ]
    counts: dict[str, object] = {
        field: int(sum(file_counts[field] for file_counts in per_file.values()))
        for field in count_fields
    }
    counts["per_file"] = per_file
    return combined, counts


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    absolute_percentage_error = np.abs((actual - predicted) / actual) * 100
    return {
        "mae_eur": round(float(mean_absolute_error(actual, predicted)), 2),
        "rmse_eur": round(float(mean_squared_error(actual, predicted) ** 0.5), 2),
        "r2": round(float(r2_score(actual, predicted)), 4),
        "mape_percent": round(float(np.mean(absolute_percentage_error)), 2),
        "median_absolute_percentage_error": round(
            float(np.median(absolute_percentage_error)), 2
        ),
        "within_10_percent": round(float(np.mean(absolute_percentage_error <= 10)), 4),
        "within_20_percent": round(float(np.mean(absolute_percentage_error <= 20)), 4),
    }


def make_model() -> TransformedTargetRegressor:
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    regressor = HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )
    pipeline = Pipeline([("preprocess", preprocessing), ("regressor", regressor)])
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def median_price_m2_baseline(
    reference: pd.DataFrame, target: pd.DataFrame
) -> np.ndarray:
    by_postcode = reference.groupby("code_postal")["price_per_m2"].median()
    global_median = float(reference["price_per_m2"].median())
    price_m2 = target["code_postal"].map(by_postcode).fillna(global_median)
    return price_m2.to_numpy() * target["surface_reelle_bati"].to_numpy()


def prediction_frame(data: pd.DataFrame, predicted: np.ndarray) -> pd.DataFrame:
    output = data[
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
    output["predicted_value"] = np.round(predicted, 2)
    output["predicted_price_per_m2"] = np.round(
        predicted / output["surface_reelle_bati"], 2
    )
    output["absolute_error"] = np.round(
        np.abs(output["valeur_fonciere"] - predicted), 2
    )
    output["absolute_percentage_error"] = np.round(
        output["absolute_error"] / output["valeur_fonciere"] * 100, 2
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", type=Path, default=DEFAULT_DATA_FILES)
    parser.add_argument("--train-end-year", type=int, default=2023)
    parser.add_argument("--validation-year", type=int, default=2024)
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument(
        "--model", type=Path, default=PROJECT_ROOT / "models/valuation_model.joblib"
    )
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports/metrics.json")
    parser.add_argument(
        "--validation-predictions",
        type=Path,
        default=PROJECT_ROOT / "reports/validation_predictions.csv",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_ROOT / "reports/test_predictions.csv",
    )
    args = parser.parse_args()

    data, counts = load_and_clean_many(args.data)
    train = data.loc[data["year"].le(args.train_end_year)].copy()
    validation = data.loc[data["year"].eq(args.validation_year)].copy()
    test = data.loc[data["year"].eq(args.test_year)].copy()
    if min(len(train), len(validation), len(test)) < 1_000:
        raise RuntimeError(
            "Temporal split produced fewer than 1,000 rows in train, validation, or test."
        )

    # Model selection view: past years predict the dedicated validation year.
    development_model = make_model()
    development_model.fit(train[FEATURES], train["valeur_fonciere"])
    validation_predicted = np.maximum(
        development_model.predict(validation[FEATURES]), 0
    )

    # Final test view: after validation decisions, refit through 2024 and touch
    # 2025 exactly once. This mirrors a realistic annual model update.
    test_training = pd.concat([train, validation], ignore_index=True)
    evaluation_model = make_model()
    evaluation_model.fit(test_training[FEATURES], test_training["valeur_fonciere"])
    test_predicted = np.maximum(evaluation_model.predict(test[FEATURES]), 0)

    validation_baseline = median_price_m2_baseline(train, validation)
    test_baseline = median_price_m2_baseline(test_training, test)

    # Calibrate uncertainty on 2024 errors and evaluate it only on 2025.
    calibration_ratios = validation["valeur_fonciere"].to_numpy() / np.maximum(
        validation_predicted, 1
    )
    interval_lower_ratio, interval_upper_ratio = np.quantile(
        calibration_ratios, [0.05, 0.95]
    )
    test_actual = test["valeur_fonciere"].to_numpy()
    interval_coverage = np.mean(
        (test_actual >= test_predicted * interval_lower_ratio)
        & (test_actual <= test_predicted * interval_upper_ratio)
    )

    importance_sample = test.sample(n=min(5_000, len(test)), random_state=42)
    importance_result = permutation_importance(
        evaluation_model,
        importance_sample[FEATURES],
        importance_sample["valeur_fonciere"],
        scoring="neg_mean_absolute_error",
        n_repeats=3,
        random_state=42,
        n_jobs=1,
    )
    feature_importance = {
        feature: round(float(increase), 2)
        for feature, increase in sorted(
            zip(FEATURES, importance_result.importances_mean),
            key=lambda item: item[1],
            reverse=True,
        )
    }

    report = {
        "schema_version": 2,
        "scope": "Ordinary Paris apartment sales with exactly one apartment per transaction",
        "data_files": [str(path.resolve()) for path in args.data],
        "counts": counts,
        "protocol": {
            "development_train": f"2021–{args.train_end_year}",
            "validation_and_interval_calibration": str(args.validation_year),
            "test_training": f"2021–{args.validation_year}",
            "untouched_test": str(args.test_year),
            "production_refit": "2021–2025",
        },
        "date_range": {
            "all_clean_start": data["date_mutation"].min().date().isoformat(),
            "all_clean_end": data["date_mutation"].max().date().isoformat(),
            "train_start": train["date_mutation"].min().date().isoformat(),
            "train_end": train["date_mutation"].max().date().isoformat(),
            "validation_start": validation["date_mutation"].min().date().isoformat(),
            "validation_end": validation["date_mutation"].max().date().isoformat(),
            "test_start": test["date_mutation"].min().date().isoformat(),
            "test_end": test["date_mutation"].max().date().isoformat(),
        },
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "production_fit_rows": int(len(data)),
        "validation_model_metrics": regression_metrics(
            validation["valeur_fonciere"].to_numpy(), validation_predicted
        ),
        "validation_baseline_metrics": regression_metrics(
            validation["valeur_fonciere"].to_numpy(), validation_baseline
        ),
        "model_metrics": regression_metrics(test_actual, test_predicted),
        "baseline_metrics": regression_metrics(test_actual, test_baseline),
        "feature_importance_mae_increase_eur": feature_importance,
        "prediction_interval": {
            "nominal_coverage": 0.90,
            "observed_test_coverage": round(float(interval_coverage), 4),
            # Retained for compatibility with existing visualizations.
            "observed_later_holdout_coverage": round(float(interval_coverage), 4),
            "lower_price_ratio": round(float(interval_lower_ratio), 6),
            "upper_price_ratio": round(float(interval_upper_ratio), 6),
            "calibration_rows": int(len(validation)),
            "evaluation_rows": int(len(test)),
            "method": "Rolling-origin empirical target/prediction ratios calibrated on 2024 and evaluated on 2025",
        },
        "features": FEATURES,
        "limitations": [
            "This is an automated valuation model, not a future-market forecast or certified appraisal.",
            "DVF prices may include bundled dependencies such as parking or a cellar.",
            "Renovation quality, floor level, energy rating, and building condition are unavailable.",
            "The fixed quality bounds restrict the model to ordinary Paris apartments.",
            "The empirical interval is evaluated under temporal shift and is not a formal conditional guarantee.",
        ],
    }

    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.validation_predictions.parent.mkdir(parents=True, exist_ok=True)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)

    # Evaluation is complete; the deployable artifact may now use every row.
    production_model = make_model()
    production_model.fit(data[FEATURES], data["valeur_fonciere"])
    bundle = {
        "model": production_model,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "training_date_min": report["date_range"]["all_clean_start"],
        "training_date_max": report["date_range"]["all_clean_end"],
        "scope": report["scope"],
        "prediction_interval": {
            "nominal_coverage": 0.90,
            "lower_price_ratio": float(interval_lower_ratio),
            "upper_price_ratio": float(interval_upper_ratio),
            "calibration_year": args.validation_year,
            "evaluation_year": args.test_year,
        },
    }
    joblib.dump(bundle, args.model)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    prediction_frame(validation, validation_predicted).to_csv(
        args.validation_predictions, index=False
    )
    prediction_frame(test, test_predicted).to_csv(args.predictions, index=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved production model: {args.model}")
    print(f"Saved report: {args.report}")
    print(f"Saved 2024 validation predictions: {args.validation_predictions}")
    print(f"Saved 2025 test predictions: {args.predictions}")


if __name__ == "__main__":
    main()
