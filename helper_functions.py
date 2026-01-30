from pathlib import Path

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import kagglehub

load_dotenv()


def load_interest_rate_data() -> pd.DataFrame:
    """
    Download the Federal Reserve interest rates dataset via kagglehub
    and return a concatenated DataFrame of all CSV files in the package.
    """
    print("Downloading dataset...")
    dataset_dir = Path(kagglehub.dataset_download("federalreserve/interest-rates"))
    print(f"Dataset directory: {dataset_dir}")
    csv_files = sorted(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")

    frames = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        # Track origin file so it's easy to filter/inspect by source.
        df["source_file"] = csv_path.name
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


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
