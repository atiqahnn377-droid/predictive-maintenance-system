# ===============================
# IMPORT LIBRARIES
# ===============================
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np


# ===============================
# INITIALIZE FLASK APP
# ===============================
app = Flask(__name__)


# ===============================
# LOAD TRAINED MODEL + SCALER
# ===============================
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")


# ===============================
# THRESHOLD SETTINGS
# ===============================
PREDICTION_THRESHOLD = 0.3
ALERT_THRESHOLD = 0.7


# ===============================
# HOME ROUTE
# ===============================
@app.route("/")
def home():
    return "Predictive Maintenance API is running."


# ===============================
# DASHBOARD ROUTE
# ===============================
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# ===============================
# PREDICT ROUTE
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # --------------------------------
        # RECEIVE JSON INPUT
        # --------------------------------
        data = request.get_json()

        # Input features from dashboard/API
        features = data["features"]

        # Convert to numpy array
        features_array = np.array(features).reshape(1, -1)

        # --------------------------------
        # APPLY SAME SCALER AS TRAINING
        # --------------------------------
        features_scaled = scaler.transform(features_array)

        # --------------------------------
        # PREDICT FAILURE PROBABILITY
        # --------------------------------
        failure_probability = model.predict_proba(features_scaled)[0][1]

        # --------------------------------
        # APPLY PREDICTION THRESHOLD
        # --------------------------------
        prediction = 1 if failure_probability > PREDICTION_THRESHOLD else 0

        # --------------------------------
        # ALERT LOGIC
        # --------------------------------
        alert = True if failure_probability > ALERT_THRESHOLD else False

        # --------------------------------
        # RETURN RESPONSE
        # --------------------------------
        return jsonify({
            "prediction": int(prediction),
            "failure_probability": round(float(failure_probability), 4),
            "alert": alert
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400



# ======================================================
# DATASET ANALYTICS MODULE (CSV PREVIEW/HISTORICAL DATA)
# ======================================================
@app.route("/dataset-preview")
def dataset_preview():
    import pandas as pd 

    # Load original CSV dataset
    df = pd.read_csv("data/predictive_maintenance_v3.csv")

    # Show first 50 rows only
    return df.head(50).to_json(orient="records")



# ================================
# DATASET SUMMARY MODULE
# ================================
@app.route("/dataset-summary")
def dataset_summary():
    import pandas as pd
    import os
    
    csv_path = os.path.join(os.getcwd(), "data", "predictive_maintenance_v3.csv")
    
    df = pd.read_csv(csv_path)
    
    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns)
    }



# ===============================
# MODEL COMPARISON API
# ===============================
@app.route("/model-comparison")
def model_comparison():
    return {
        "Random Forest Accuracy": "99.40%",
        "SVM Accuracy": "96.28%",
        "Logistic Regression Accuracy": "91.16%",
        "Best Model": "Random Forest"
    }



# ===============================
# RUN APPLICATION
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)