from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

model = joblib.load("models/model.pkl")

@app.route("/")
def home():
    return "API is running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])

    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])

    prediction = model.predict(df)[0]

    return jsonify({
        "prediction": int(prediction),
        "message": "Anomaly ⚠️" if prediction == 1 else "Normal ✅"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

@app.route("/")
def home():
    return "UAV Backend is Running 🚀"