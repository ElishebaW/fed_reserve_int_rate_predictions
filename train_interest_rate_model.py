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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

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
