import json
import os

import joblib
import pandas as pd
from helper_functions import (
    feature_engineer_interest_rates,
    get_features_and_labels,
    load_interest_rate_data,
    make_numeric_pipeline,
    select_numeric_features,
    split_train_test_chronological,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import HalvingRandomSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

TARGET_COL = "Federal Funds Target Rate"
LEAK_COLS = [
    "Federal Funds Upper Target",
    "Federal Funds Lower Target",
    "Effective Federal Funds Rate",
]


def is_verbose() -> bool:
    # Local/dev default: verbose. Vertex default: quiet.
    # Override with TRAIN_VERBOSE=1 or TRAIN_VERBOSE=0.
    env_override = os.environ.get("TRAIN_VERBOSE")
    if env_override is not None:
        return env_override.strip().lower() in {"1", "true", "yes", "on"}
    return os.environ.get("CLOUD_ML_PROJECT_ID") is None


def log(*args, **kwargs) -> None:
    if is_verbose():
        print(*args, **kwargs)


def build_model_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocessing", make_numeric_pipeline()),
            (
                "feature_select",
                SelectFromModel(
                    estimator=RandomForestRegressor(
                        n_estimators=200,
                        random_state=42,
                        n_jobs=-1,
                    ),
                    threshold="median",
                ),
            ),
            ("random_forest", RandomForestRegressor(random_state=42, n_jobs=-1)),
        ]
    )


def build_training_frame(df: pd.DataFrame):
    features, labels, label_mask = get_features_and_labels(df, TARGET_COL)
    features = features.drop(columns=[c for c in LEAK_COLS if c in features.columns])
    numeric_features = select_numeric_features(features)
    x = numeric_features.loc[label_mask]
    y = labels.loc[label_mask]
    return x, y


def main() -> None:
    data = load_interest_rate_data().copy()
    data = feature_engineer_interest_rates(data)

    train_set, test_set = split_train_test_chronological(data, test_size=0.2)
    if "date" in train_set.columns:
        train_set = train_set.drop(columns=["date"])
    if "date" in test_set.columns:
        test_set = test_set.drop(columns=["date"])

    x_train, y_train = build_training_frame(train_set)
    x_test, y_test = build_training_frame(test_set)

    param_distributions = {
        "feature_select__threshold": ["median", "1.25*median", "mean"],
        "random_forest__max_depth": [3, 5, 8, None],
        "random_forest__min_samples_split": [2, 5, 10, 20],
        "random_forest__min_samples_leaf": [1, 2, 4, 8],
        "random_forest__max_features": ["sqrt", 0.5, None],
        "random_forest__ccp_alpha": [0.0, 1e-4, 1e-3],
        "random_forest__bootstrap": [True],
    }

    tscv = TimeSeriesSplit(n_splits=5)
    halving_search = HalvingRandomSearchCV(
        estimator=build_model_pipeline(),
        param_distributions=param_distributions,
        cv=tscv,
        factor=3,
        resource="random_forest__n_estimators",
        min_resources=100,
        max_resources=1000,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=42,
        verbose=1 if is_verbose() else 0,
    )
    halving_search.fit(x_train, y_train)

    final_model = halving_search.best_estimator_
    log("Best parameters:", halving_search.best_params_)
    log("Best cross-validation score:", halving_search.best_score_)

    best_rf = final_model.named_steps["random_forest"]
    best_preprocessing = final_model.named_steps["preprocessing"]
    best_selector = final_model.named_steps["feature_select"]

    all_feature_names = best_preprocessing.get_feature_names_out(x_train.columns)
    selected_mask = best_selector.get_support()
    selected_feature_names = all_feature_names[selected_mask]

    feature_importances = best_rf.feature_importances_
    importance_pairs = sorted(
        zip(feature_importances, selected_feature_names),
        key=lambda x: x[0],
        reverse=True,
    )
    log("Selected features:", selected_feature_names.tolist())
    log("Top feature importances (selected features only):")
    log(importance_pairs)

    final_predictions = final_model.predict(x_test)
    final_rmse = root_mean_squared_error(y_test, final_predictions)
    log("final_test_rmse:", final_rmse)

    model_dir = os.environ.get("AIP_MODEL_DIR", ".")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model.joblib")
    feature_schema_path = os.path.join(model_dir, "feature_columns.json")

    joblib.dump(final_model, model_path)
    with open(feature_schema_path, "w", encoding="utf-8") as f:
        json.dump({"feature_columns": x_train.columns.tolist()}, f, indent=2)

    log("Saved model artifact:", model_path)
    log("Saved feature schema:", feature_schema_path)


if __name__ == "__main__":
    main()
