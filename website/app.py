from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Food Sales Prediction API")

# Load your trained models (save them as .joblib after training)
rf = joblib.load("models/random_forest.joblib")
lgbm = joblib.load("models/lightgbm.joblib")
xgb = joblib.load("models/xgboost.joblib")

# IMPORTANT: Make sure your feature order here matches training exactly
FEATURE_NAMES = [
    "weekday",
    "is_school_holiday",
    "is_state_holiday",
    "is_special_day",
    "temp_mean",
    "temp_max",
    "temp_min",
    "sunshine",
    "precipitation",
    "store_id",
]

class PredictRequest(BaseModel):
    weekday: int                 # 0=Mon ... 6=Sun
    is_school_holiday: int       # 0/1
    is_state_holiday: int        # 0/1
    is_special_day: int          # 0/1
    temp_mean: float
    temp_max: float
    temp_min: float
    sunshine: float
    precipitation: float
    store_id: int

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    x = np.array([[getattr(req, f) for f in FEATURE_NAMES]], dtype=float)

    y_rf = float(rf.predict(x)[0])
    y_lgbm = float(lgbm.predict(x)[0])
    y_xgb = float(xgb.predict(x)[0])

    # Optional: a simple "recommended production" rule-of-thumb
    # (choose one model or average)
    y_avg = float((y_rf + y_lgbm + y_xgb) / 3.0)

    return {
        "random_forest": y_rf,
        "lightgbm": y_lgbm,
        "xgboost": y_xgb,
        "average": y_avg
    }