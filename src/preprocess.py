"""
preprocess.py - Data loading and preprocessing for UAV anomaly detection.
"""

import pandas as pd


def load_and_preprocess(filepath: str):
    """
    Load the CSV dataset, drop missing values, drop 'timestamp' column
    (to prevent data leakage), and split into features (X) and target (y).

    Args:
        filepath: Path to the CSV file.

    Returns:
        X: Feature DataFrame (all columns except the last).
        y: Target Series (last column).
    """
    df = pd.read_csv(filepath)
    df = df.dropna()

    # Drop timestamp to avoid data leakage
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])
        print("[preprocess] Dropped 'timestamp' column.")

    # Last column is the target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    print(f"[preprocess] Loaded {len(df)} rows, {X.shape[1]} features.")
    print(f"[preprocess] Target distribution:\n{y.value_counts()}")

    return X, y
