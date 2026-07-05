import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow all origins; tighten with origins=[...] in production

# --- 1. Robust Model Loading ---
# Resolve absolute path to models/model.pkl relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Model file not found exactly at: {MODEL_PATH}")


# --- 2 & 3. Extract Feature Names ---
# FIRST priority: extract from trained model
if hasattr(model, "feature_names_in_"):
    FEATURES = list(model.feature_names_in_)
else:
    raise RuntimeError(
        "The loaded model does not expose 'feature_names_in_'. "
        "Either retrain using a pandas DataFrame or manually define features."
    )


@app.route('/')
def home():
    return jsonify({
        "status": "Server is running",
        "expected_features_count": len(FEATURES),
        "expected_features": FEATURES
    })


@app.route('/features', methods=['GET'])
def features():
    """Return the ordered list of feature names the model expects."""
    return jsonify({"features": FEATURES})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data or not isinstance(data, dict):
            return jsonify({
                "error": "Request body must be a valid JSON object"
            }), 400

        # Ensure correct ordering
        ordered_row = {}
        missing_features = []

        for feature in FEATURES:
            if feature not in data:
                missing_features.append(feature)
            else:
                ordered_row[feature] = float(data[feature])

        if missing_features:
            return jsonify({
                "error": "Missing required features",
                "missing": missing_features,
                "schema": FEATURES
            }), 400

        # Create DataFrame
        df = pd.DataFrame([ordered_row], columns=FEATURES)

        # Predict
        prediction = model.predict(df)
        pred_value = int(prediction[0])

        return jsonify({
            "prediction": pred_value
        })

    except ValueError as e:
        return jsonify({
            "error": f"Invalid numeric value: {str(e)}"
        }), 400

    except Exception as e:
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)