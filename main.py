from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model import train_model
from src.evaluate import evaluate_model

import joblib

# =============================
#  LOAD DATA
# =============================
df = load_data("data/predictive_maintenance_v3.csv")


# =============================
#  PREPROCESS DATA
# =============================
x_train, x_test, y_train, y_test, scaler = preprocess_data(df)


# =============================
# TRAIN + COMPARE MODELS
# =============================
model = train_model(x_train, y_train, x_test, y_test)


# =============================
# EVALUATE BEST MODEL
# =============================
evaluate_model(model, x_test, y_test)


# =============================
# SAVE BEST MODEL + SCALER
# =============================
print("Saving model...")
joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Done!")