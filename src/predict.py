"""
predict.py - Load saved model and predict on a single sample.
"""

import os
import sys

import joblib
import pandas as pd

# Ensure src/ is importable when running from project root
sys.path.insert(0, os.path.dirname(__file__))


def main():
    DATASET_PATH = os.path.join("archive", "dataset", "Fusion_Data.csv")
    MODEL_PATH = os.path.join("models", "model.pkl")

    # --- Load model ---
    if not os.path.exists(MODEL_PATH):
        print(f"[error] Model not found at '{MODEL_PATH}'. Run train.py first.")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    print(f"[predict] Model loaded from {MODEL_PATH}")

    # --- Load dataset and pick 1 sample ---
    df = pd.read_csv(DATASET_PATH)

    # Drop timestamp if present
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])

    # Separate features (all columns except last)
    X = df.iloc[:, :-1]
    y_true = df.iloc[:, -1]

    # Take 1 random sample
    sample = X.sample(n=1, random_state=42)
    true_label = y_true.iloc[sample.index[0]]

    # --- Predict ---
    prediction = model.predict(sample)

    print(f"\n{'=' * 45}")
    print(f"  PREDICTION RESULT")
    print(f"{'=' * 45}")
    print(f"  Sample index : {sample.index[0]}")
    print(f"  True label   : {true_label}")
    print(f"  Predicted    : {prediction[0]}")
    match = "YES - Correct" if prediction[0] == true_label else "NO - Wrong"
    print(f"  Match        : {match}")
    print(f"{'=' * 45}")


if __name__ == "__main__":
    main()
