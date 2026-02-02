import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helper_functions import load_interest_rate_data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer

# Get data once
data = load_interest_rate_data().copy()

# Basic look if needed
# print(data.head())
# print(data.info())
# print(data.describe())

# Build a proper datetime from Year/Month/Day
data["Day"] = data["Day"].fillna(1).astype(int)
data["date"] = pd.to_datetime(dict(year=data["Year"], month=data["Month"], day=data["Day"]), errors="coerce")
data = data.sort_values("date")

# Feature engineering (similar style to rooms_per_house, etc.)
# Fill missing upper/lower targets with target rate, then effective rate as last resort.
upper = (
    data["Federal Funds Upper Target"]
    .fillna(data["Federal Funds Target Rate"])
    .fillna(data["Effective Federal Funds Rate"])
)
lower = (
    data["Federal Funds Lower Target"]
    .fillna(data["Federal Funds Target Rate"])
    .fillna(data["Effective Federal Funds Rate"])
)

data["rate_spread"] = (upper - lower).fillna(0)
data["implementation_gap"] = (data["Effective Federal Funds Rate"] - upper).fillna(0)
data["real_ffr"] = data["Effective Federal Funds Rate"] - data["Inflation Rate"]

train_set, test_set = train_test_split(data, test_size=0.2, shuffle=False)
train_set = train_set.drop(columns=["date"])
test_set  = test_set.drop(columns=["date"])


#Cross validation

# tscv = TimeSeriesSplit(n_splits=5)
# for train_idx, test_idx in tscv.split(data):
#     train_fold = data.iloc[train_idx]
#     test_fold  = data.iloc[test_idx]
#     # fit/evaluate here

train_set_copy = train_set.copy()

# Correlation on numeric columns only (avoid 'source_file' strings)
numeric_cols = train_set_copy.select_dtypes(include=[np.number])
corr_matrix = numeric_cols.corr()
print(corr_matrix["Federal Funds Target Rate"].sort_values(ascending=False))

# attributes = ["Federal Funds Target Rate", "Effective Federal Funds Rate", "Inflation Rate",
#               "Unemployment Rate"]
# scatter_matrix(train_set_copy[attributes], figsize=(12, 8))
# plt.show()


# train_set_copy.plot(kind="scatter", x="Federal Funds Target Rate", y="Inflation Rate",
#              alpha=0.1, grid=True)
# plt.show()

train_set_copy = train.drop(columns=["Federal Funds Target Rate"], axis=1)
train_set_copy_labels = train_set_copy["Federal Funds Target Rate"].copy()

imputer = SimpleImputer(strategy="median")
train_set_copy_numeric =  train_set_copy.select_dtypes(include=[np.number])
train_set_copy_numeric = imputer.fit(train_set_copy_numeric)




