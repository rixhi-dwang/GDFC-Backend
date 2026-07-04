"""
model.py - Model definition for UAV anomaly classification.
"""

from sklearn.ensemble import RandomForestClassifier


def get_model():
    """
    Returns a configured RandomForestClassifier.
    Uses n_estimators=50 for fast training, n_jobs=-1 for parallelism.
    """
    return RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        n_jobs=-1,
    )
