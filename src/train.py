"""
train.py - Main training pipeline for UAV anomaly detection.
"""

import os
import sys
import time

import joblib
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Ensure src/ is importable when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from preprocess import load_and_preprocess
from model import get_model


def main():
    # --- Config ---
    DATASET_PATH = os.path.join("archive", "dataset", "Fusion_Data.csv")
    MODEL_DIR = "models"
    MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

    # --- Load & Preprocess ---
    print("=" * 55)
    print("  UAV Anomaly Detection - Training Pipeline")
    print("=" * 55)

    X, y = load_and_preprocess(DATASET_PATH)

    # --- Train/Test Split (80/20) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[split] Train: {len(X_train)} | Test: {len(X_test)}")

    # --- Train ---
    model = get_model()
    print("\n[train] Training RandomForestClassifier (n_estimators=50)...")
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"[train] Done in {elapsed:.2f}s")

    # --- Evaluate ---
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'=' * 55}")
    print(f"  RESULTS")
    print(f"{'=' * 55}")
    print(f"\n  Accuracy: {acc:.4f} ({acc * 100:.2f}%)")

    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred))

    print(f"  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # --- 5-Fold Cross-Validation ---
    print(f"\n{'=' * 55}")
    print(f"  5-FOLD CROSS-VALIDATION")
    print(f"{'=' * 55}")
    cv_model = get_model()
    cv_scores = cross_val_score(cv_model, X, y, cv=5, scoring="accuracy")
    print(f"\n  Fold scores: {np.round(cv_scores, 4)}")
    print(f"  Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # --- Save Model ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n[save] Model saved to {MODEL_PATH}")

    # --- Create output dirs ---
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print(f"\n{'=' * 55}")
    print("  Pipeline complete!")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
