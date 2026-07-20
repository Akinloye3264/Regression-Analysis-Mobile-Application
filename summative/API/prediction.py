"""
IoT Edge-Device Energy Prediction API
Serves the trained Random Forest model (Task 1) over HTTP.

Endpoints
---------
GET  /            -> health check / basic info
POST /predict     -> predict energy consumption (mJ) from device state
POST /retrain     -> upload new telemetry (CSV) and retrain the model

Run locally:
    uv run uvicorn prediction:app --reload
Docs (Swagger UI):  http://127.0.0.1:8000/docs
"""

import io
import os
from enum import IntEnum

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load trained artifacts (saved by the notebook in Task 1)
HERE = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(HERE, "best_model.pkl"))
scaler = joblib.load(os.path.join(HERE, "scaler.pkl"))
feature_columns = joblib.load(os.path.join(HERE, "feature_columns.pkl"))
MODEL_PATH = os.path.join(HERE, "best_model.pkl")
SCALER_PATH = os.path.join(HERE, "scaler.pkl")

TARGET = "energy_consumed_mJ"


class DeviceAction(IntEnum):
    SLEEP_DEEP = 0
    SLEEP_LIGHT = 1
    ACTIVE_LOW = 2
    TX_LOW = 3
    TX_MED = 4
    TX_HIGH = 5


# FastAPI app
app = FastAPI(
    title="IoT Edge-Device Energy Prediction API",
    description="Predicts energy consumed (mJ) per cycle from an IoT node's operating state.",
    version="1.0.0",
)

# --------------------------------------------------------------------------- #
# CORS middleware — deliberately not a wildcard.
# Only the origins that call this API are allowed (Flutter dev server / localhost).
# Methods limited to GET/POST/OPTIONS (OPTIONS needed for browser preflight).
# Headers limited to Content-Type (JSON only). Credentials off — no auth cookies.
# --------------------------------------------------------------------------- #
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    allow_credentials=False,
)

# Pydantic input/output — enforced types and realistic ranges (from the data)
class PredictionInput(BaseModel):
    cpu_usage: float = Field(..., ge=0.0, le=1.0,
                             description="CPU utilisation fraction (0–1).")
    memory_usage: float = Field(..., ge=0.0, le=1.0,
                                description="Memory utilisation fraction (0–1).")
    signal_quality: float = Field(..., ge=0.0, le=1.0,
                                  description="Radio signal quality (0–1).")
    action: DeviceAction = Field(..., description="Power mode 0=SLEEP_DEEP … 5=TX_HIGH.")
    queue_size: int = Field(..., ge=0, le=100,
                            description="Number of packets waiting in the queue.")
    temperature_C: float = Field(..., ge=-20.0, le=60.0,
                                 description="Ambient temperature in °C.")

    class Config:
        json_schema_extra = {
            "example": {
                "cpu_usage": 0.65, "memory_usage": 0.40, "signal_quality": 0.75,
                "action": 5, "queue_size": 12, "temperature_C": 9.5,
            }
        }


class PredictionOutput(BaseModel):
    predicted_energy_mJ: float = Field(..., description="Predicted energy consumed (mJ).")
    action_name: str
    efficiency_note: str


def energy_note(mj: float) -> str:
    if mj <= 100:
        return "Low draw — sustainable on harvested energy."
    if mj <= 400:
        return "Moderate draw."
    if mj <= 800:
        return "High draw — monitor battery."
    return "Very high draw — may exceed harvest budget."


# Routes
@app.get("/")
def root():
    return {
        "message": "IoT Edge-Device Energy Prediction API",
        "docs": "/docs",
        "predict": "POST /predict",
        "retrain": "POST /retrain",
        "features": feature_columns,
        "actions": {a.value: a.name for a in DeviceAction},
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput):
    # Build the feature row in the exact training column order
    row = {
        "cpu_usage": payload.cpu_usage,
        "memory_usage": payload.memory_usage,
        "signal_quality": payload.signal_quality,
        "action": int(payload.action),
        "queue_size": payload.queue_size,
        "temperature_C": payload.temperature_C,
    }
    features = pd.DataFrame([[row[c] for c in feature_columns]], columns=feature_columns)
    features_scaled = scaler.transform(features)
    prediction = float(model.predict(features_scaled)[0])
    prediction = round(max(prediction, 0.0), 2)

    return PredictionOutput(
        predicted_energy_mJ=prediction,
        action_name=DeviceAction(int(payload.action)).name,
        efficiency_note=energy_note(prediction),
    )


@app.post("/retrain")
async def retrain(file: UploadFile = File(...)):
    """
    Retrain on newly uploaded telemetry.

    Expects a CSV containing at least the feature columns plus the target
    `energy_consumed_mJ`. A fresh Random Forest is fitted and the saved model on disk
    is replaced so subsequent /predict calls use the updated model.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=422, detail="Please upload a .csv file.")

    contents = await file.read()
    try:
        new_df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read CSV: {exc}")

    required = set(feature_columns) | {TARGET}
    missing = required - set(new_df.columns)
    if missing:
        raise HTTPException(status_code=422,
                            detail=f"CSV missing required columns: {sorted(missing)}")

    new_df = new_df.dropna(subset=list(required))
    if len(new_df) < 50:
        raise HTTPException(status_code=422,
                            detail="Need at least 50 valid rows to retrain.")

    X_new = new_df[feature_columns]
    y_new = new_df[TARGET]

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    new_scaler = StandardScaler()
    X_scaled = new_scaler.fit_transform(X_new)
    new_model = RandomForestRegressor(
        n_estimators=100, max_depth=20, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )
    new_model.fit(X_scaled, y_new)

    joblib.dump(new_model, MODEL_PATH, compress=3)
    joblib.dump(new_scaler, SCALER_PATH)

    global model, scaler
    model = new_model
    scaler = new_scaler

    return {"message": "Model retrained and saved successfully.",
            "rows_used": int(len(X_new))}