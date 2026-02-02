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
num_features = select_numeric_features(features)

# Numeric preprocessing pipeline: median impute then standardize
num_pipeline = make_numeric_pipeline()

# Fit/transform on rows with non-null labels
X_train = num_features.loc[label_mask]
y_train = labels.loc[label_mask]
X_train_prepared = num_pipeline.fit_transform(X_train)
print("Post-pipeline NaNs per column:",
      pd.DataFrame(X_train_prepared, columns=num_features.columns).isna().sum())

model = TransformedTargetRegressor(LinearRegression(),
                                   transformer=StandardScaler())
model.fit(X_train_prepared, y_train)

# Predict on a small batch (first 5 rows) using the same pipeline
predictions = model.predict(num_pipeline.transform(num_features.iloc[:5]))
print("predictions: ", predictions)
