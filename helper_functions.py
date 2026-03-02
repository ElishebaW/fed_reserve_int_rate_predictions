from pathlib import Path

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

load_dotenv()


def load_interest_rate_data() -> pd.DataFrame:
    """
    Download the Federal Reserve interest rates dataset via kagglehub
    and return a concatenated DataFrame of all CSV files in the package.

    If network resolution fails, fall back to local kagglehub cache.
    """
    dataset_dir = None
    try:
        print("Downloading dataset...")
        dataset_dir = Path(kagglehub.dataset_download("federalreserve/interest-rates"))
        print(f"Dataset directory: {dataset_dir}")
        csv_files = sorted(dataset_dir.glob("*.csv"))
        if csv_files:
            frames = []
            for csv_path in csv_files:
                df = pd.read_csv(csv_path)
                df["source_file"] = csv_path.name
                frames.append(df)
            return pd.concat(frames, ignore_index=True)
    except Exception as exc:
        print(f"Dataset download failed, trying local cache fallback: {exc}")

    cache_index = Path.home() / ".cache" / "kagglehub" / "datasets" / "federalreserve" / "interest-rates" / "versions" / "1" / "index.csv"
    if cache_index.exists():
        print(f"Using local cache file: {cache_index}")
        df = pd.read_csv(cache_index)
        df["source_file"] = cache_index.name
        return df

    raise FileNotFoundError(
        "Could not load interest-rate dataset from kagglehub or local cache. "
        f"Last dataset_dir={dataset_dir}"
    )


def shuffle_and_split_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def add_rate_category(df: pd.DataFrame, col: str = "Effective Federal Funds Rate") -> pd.DataFrame:
    """
    Add a categorical 'rate_cat' column for stratified sampling on a rate column.
    Bins cover the historical range (~0–20%) while keeping strata large enough.
    """
    bins = [-0.01, 1, 2, 3, 4, 5, 7.5, 10, 15, np.inf]
    labels = range(1, len(bins))
    df = df.copy()
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame")
    df["rate_cat"] = pd.cut(df[col], bins=bins, labels=labels)
    return df


# --- Convenience utilities for EDA/inspection (optional) ---

def quick_sanity_peeks(df: pd.DataFrame, n: int = 5) -> None:
    """Print quick structural views to sanity-check the raw data."""
    print(df.head(n))
    print(df.info())
    print(df.describe())


def correlation_with_target(df: pd.DataFrame, target_col: str) -> pd.Series:
    """Return correlations (numeric only) with the target, sorted descending."""
    numeric_cols = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_cols.corr()
    return corr_matrix[target_col].sort_values(ascending=False)


def plot_scatter_matrix(df: pd.DataFrame, attributes: list[str], figsize=(12, 8)) -> None:
    """Convenience wrapper for pandas scatter_matrix."""
    scatter_matrix(df[attributes], figsize=figsize)
    plt.show()


def plot_target_vs_feature(df: pd.DataFrame, target_col: str, feature_col: str, **kwargs) -> None:
    """Quick scatter plot of a single feature against the target."""
    df.plot(kind="scatter", x=target_col, y=feature_col, **kwargs)
    plt.show()


def feature_engineer_interest_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light feature engineering:
    - build a real date
    - fill upper/lower targets from target rate, then effective rate
    - optional engineered spreads (kept commented so you can enable easily)
    """
    data = df.copy()
    data["Day"] = data["Day"].fillna(1).astype(int)
    data["date"] = pd.to_datetime(
        dict(year=data["Year"], month=data["Month"], day=data["Day"]),
        errors="coerce",
    )
    data = data.sort_values("date")

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

    # Persist the filled versions so downstream steps see non-NaN values
    data["Federal Funds Upper Target"] = upper
    data["Federal Funds Lower Target"] = lower

    # data["rate_spread"] = (upper - lower).fillna(0)
    # data["implementation_gap"] = (data["Effective Federal Funds Rate"] - upper).fillna(0)
    # data["real_ffr"] = data["Effective Federal Funds Rate"] - data["Inflation Rate"]
    return data


def split_train_test_chronological(df: pd.DataFrame, test_size: float = 0.2):
    """Chronological split (no shuffle) to respect time ordering."""
    return train_test_split(df, test_size=test_size, shuffle=False)


def get_features_and_labels(df: pd.DataFrame, target_col: str):
    """
    Drop the target from features and return labels + non-null mask for training.
    """
    features = df.drop(columns=[target_col])
    labels = df[target_col].copy()
    label_mask = labels.notna()
    return features, labels, label_mask


def select_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep numeric columns only (avoids non-numeric columns like source_file)."""
    return df.select_dtypes(include=[np.number])


def make_numeric_pipeline():
    """Median impute, then standardize numeric predictors."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("standardize", StandardScaler()),
        ]
    )
