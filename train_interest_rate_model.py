import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helper_functions import (
    correlation_with_target,
    feature_engineer_interest_rates,
    get_features_and_labels,
    load_interest_rate_data,
    make_numeric_pipeline,
    plot_scatter_matrix,
    plot_target_vs_feature,
    quick_sanity_peeks,
    select_numeric_features,
    split_train_test_chronological,
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.feature_selection import SelectFromModel
import joblib


# Get data once
data = load_interest_rate_data().copy()

# Quick sanity peeks (uncomment as needed)
# quick_sanity_peeks(data)

# Build a proper datetime, fill targets, and keep optional engineered cols nearby.
data = feature_engineer_interest_rates(data)

train_set, test_set = split_train_test_chronological(data, test_size=0.2)
train_set = train_set.drop(columns=["date"])
test_set  = test_set.drop(columns=["date"])


# Optional: Cross validation (uncomment to use)
# tscv = TimeSeriesSplit(n_splits=5)
# for train_idx, test_idx in tscv.split(data):
#     train_fold = data.iloc[train_idx]
#     test_fold  = data.iloc[test_idx]
#     # fit/evaluate here

train_set_copy = train_set.copy()

# Correlation on numeric columns only (avoids non-numeric 'source_file')
# print(correlation_with_target(train_set_copy, "Federal Funds Target Rate"))

attributes = ["Federal Funds Target Rate", "Effective Federal Funds Rate",
              "Inflation Rate", "Unemployment Rate"]
# Optional: uncomment to visualize
# plot_scatter_matrix(train_set_copy, attributes)


# Optional: uncomment to visualize
# plot_target_vs_feature(train_set_copy,
#                        target_col="Federal Funds Target Rate",
#                        feature_col="Inflation Rate",
#                        alpha=0.1, grid=True)

features, labels, label_mask = get_features_and_labels(train_set, "Federal Funds Target Rate")
# Remove target-leaking proxy columns before modeling
leak_cols = [
    "Federal Funds Upper Target",
    "Federal Funds Lower Target",
    "Effective Federal Funds Rate",
]
features = features.drop(columns=[c for c in leak_cols if c in features.columns])
num_features = select_numeric_features(features)

# Quick diagnostic: check NaNs before/after preprocessing
diag_pipeline = make_numeric_pipeline()
X_train = num_features.loc[label_mask]
y_train = labels.loc[label_mask]
X_train_prepared = diag_pipeline.fit_transform(X_train)
print("Nans before preprocessing", X_train.isna().sum())
print("Post-pipeline NaNs per column:",
      pd.DataFrame(X_train_prepared, columns=num_features.columns).isna().sum())

# Linear regression pipeline fit on raw numeric features (pipeline handles prep)
lin_reg = make_pipeline(make_numeric_pipeline(), LinearRegression())
lin_reg.fit(num_features.loc[label_mask], y_train)
lin_predictions = lin_reg.predict(num_features.loc[label_mask])
print("lin_predictions: ", lin_predictions[:5])
print("actual: ", y_train.iloc[:5].values)

# Time-aware cross validation to reduce leakage
tscv = TimeSeriesSplit(n_splits=5)
lin_cv_scores = -cross_val_score(
    lin_reg,
    num_features.loc[label_mask],
    y_train,
    scoring="neg_root_mean_squared_error",
    cv=tscv,
)
print("lin_cv_rmse mean:", lin_cv_scores.mean(), "std:", lin_cv_scores.std())

# Evaluate on held-out chronological test set
test_features, test_labels, test_mask = get_features_and_labels(
    test_set, "Federal Funds Target Rate"
)
test_features = test_features.drop(columns=[c for c in leak_cols if c in test_features.columns])
test_num = select_numeric_features(test_features)
y_test = test_labels.loc[test_mask]
lin_test_pred = lin_reg.predict(test_num.loc[test_mask])
lin_test_rmse = root_mean_squared_error(y_test, lin_test_pred)
print("lin_test_rmse:", lin_test_rmse)

# Decision tree with the same preprocessing
tree_reg = make_pipeline(make_numeric_pipeline(), DecisionTreeRegressor(random_state=42))
tree_reg.fit(num_features.loc[label_mask], y_train)
tree_predictions = tree_reg.predict(num_features.loc[label_mask])
print("tree_predictions: ", tree_predictions[:5])
print("actual: ", y_train.iloc[:5].values)

tree_cv_scores = -cross_val_score(
    tree_reg,
    num_features.loc[label_mask],
    y_train,
    scoring="neg_root_mean_squared_error",
    cv=tscv,
)
print("tree_cv_rmse mean:", tree_cv_scores.mean(), "std:", tree_cv_scores.std())

tree_test_pred = tree_reg.predict(test_num.loc[test_mask])
tree_test_rmse = root_mean_squared_error(y_test, tree_test_pred)
print("tree_test_rmse:", tree_test_rmse)

#Random Forest regression model fit on raw numeric features (pipeline handles prep)
forest_reg = make_pipeline(make_numeric_pipeline(),
                           RandomForestRegressor(random_state=42))
forest_rmses = -cross_val_score(
    forest_reg,
    num_features.loc[label_mask],
    y_train,
    scoring="neg_root_mean_squared_error",
    cv=tscv,
)
print("forest_rmses: ", forest_rmses)
print("forest_rmse mean: ", forest_rmses.mean())
print("forest_rmse std: ", forest_rmses.std())
print(pd.Series(forest_rmses).describe())


#Fine-tuning the Random Forest model
preprocessing = make_numeric_pipeline()

full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("feature_select", SelectFromModel(
        estimator=RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
        threshold="median",
    )),
    ("random_forest", RandomForestRegressor(random_state=42, n_jobs=-1)),
])

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
    full_pipeline,
    param_distributions=param_distributions,
    cv=tscv,
    factor=3,
    resource="random_forest__n_estimators",
    min_resources=100,
    max_resources=1000,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    random_state=42,
    verbose=1,
)
halving_search.fit(num_features.loc[label_mask], y_train)

final_model = halving_search.best_estimator_  # full pipeline: preprocessing + selector + RF

print("Best parameters:", halving_search.best_params_)
print("Best cross-validation score:", halving_search.best_score_)

# Show selected features and their importances
best_pipeline = halving_search.best_estimator_
best_rf = best_pipeline.named_steps["random_forest"]
best_preprocessing = best_pipeline.named_steps["preprocessing"]
best_selector = best_pipeline.named_steps["feature_select"]

try:
    all_feature_names = best_preprocessing.get_feature_names_out(num_features.columns)
except Exception:
    all_feature_names = num_features.columns.to_numpy()

selected_mask = best_selector.get_support()
selected_feature_names = all_feature_names[selected_mask]

print("Selected features:")
print(selected_feature_names)

feature_importances = best_rf.feature_importances_
importance_pairs = sorted(
    zip(feature_importances, selected_feature_names),
    key=lambda x: x[0],
    reverse=True,
)

print("Top feature importances (selected features only):")
print(importance_pairs)


# Evaluate on test set using the same feature filtering as training
test_features_final, test_labels_final, test_mask_final = get_features_and_labels(
    test_set, "Federal Funds Target Rate"
)
test_features_final = test_features_final.drop(
    columns=[c for c in leak_cols if c in test_features_final.columns]
)
test_num_final = select_numeric_features(test_features_final)
y_test_final = test_labels_final.loc[test_mask_final]

final_predictions = final_model.predict(test_num_final.loc[test_mask_final])
final_rmse = root_mean_squared_error(y_test_final, final_predictions)
print("final_test_rmse:", final_rmse)

#Prepare for launch and save model 
joblib.dump(final_model, "fed_rate_model.pkl")
