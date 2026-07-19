"""
Ghana PM2.5 Air Quality Prediction API
======================================
Serves the trained Random Forest model (Task 1) over HTTP.

Endpoints
---------
GET  /            -> health check / basic info
POST /predict     -> predict PM2.5 for a city + date
POST /retrain     -> upload new daily data (CSV) and retrain the model

Run locally:
    uv run uvicorn prediction:app --reload
Docs (Swagger UI):  http://127.0.0.1:8000/docs
"""

import io
import os
from datetime import date
from enum import Enum

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "best_model.pkl")
SCALER_PATH = os.path.join(HERE, "scaler.pkl")
COLUMNS_PATH = os.path.join(HERE, "feature_columns.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(COLUMNS_PATH)

# The 20 cities in the dataset. "Accra" was the drop_first baseline, so it has
# no dummy column, but it is still a valid input (all city dummies = 0).
CITIES = [
    "Accra", "Bolgatanga", "Cape Coast", "Dambai", "Damongo", "Goaso", "Ho",
    "Kintampo", "Koforidua", "Kumasi", "Nalerigu", "Navrongo", "Sefwi Wiawso",
    "Sekondi-Takoradi", "Somanya", "Sunyani", "Tamale", "Techiman", "Tema", "Wa",
]

CityEnum = Enum("CityEnum", {c.replace(" ", "_").replace("-", "_"): c for c in CITIES})

# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="Ghana PM2.5 Air Quality Prediction API",
    description="Predicts daily PM2.5 (µg/m³) for Ghanaian cities from date + city.",
    version="1.0.0",
)

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

# --------------------------------------------------------------------------- #
# Request / response models (Pydantic) — enforce types and realistic ranges
# --------------------------------------------------------------------------- #
class PredictionInput(BaseModel):
    city: CityEnum = Field(..., description="One of the 20 supported Ghanaian cities.")
    year: int = Field(..., ge=2005, le=2035, description="Calendar year (2005–2035).")
    month: int = Field(..., ge=1, le=12, description="Month of year (1–12).")
    day: int = Field(..., ge=1, le=31, description="Day of month (1–31).")

    class Config:
        json_schema_extra = {
            "example": {"city": "Tamale", "year": 2025, "month": 1, "day": 15}
        }


class PredictionOutput(BaseModel):
    city: str
    date: str
    predicted_pm25: float = Field(..., description="Predicted PM2.5 in µg/m³.")
    air_quality: str = Field(..., description="WHO-style qualitative band.")


def build_feature_row(city: str, d: date) -> pd.DataFrame:
    """Turn a city + date into the exact 23-column feature row the model expects."""
    harmattan = 1 if d.month in (11, 12, 1, 2, 3) else 0
    row = {
        "Year": d.year,
        "Month": d.month,
        "DayOfYear": d.timetuple().tm_yday,
        "Harmattan": harmattan,
    }
    # One-hot the city to match training (Accra baseline => all dummies 0)
    for col in feature_columns:
        if col.startswith("City_"):
            row[col] = 1 if col == f"City_{city}" else 0
    return pd.DataFrame([[row[c] for c in feature_columns]], columns=feature_columns)


def air_quality_band(pm25: float) -> str:
    """Rough WHO-aligned qualitative label for the predicted value."""
    if pm25 <= 15:
        return "Good"
    if pm25 <= 35:
        return "Moderate"
    if pm25 <= 55:
        return "Unhealthy for sensitive groups"
    if pm25 <= 110:
        return "Unhealthy"
    return "Very unhealthy"


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/")
def root():
    return {
        "message": "Ghana PM2.5 Air Quality Prediction API",
        "docs": "/docs",
        "predict": "POST /predict",
        "retrain": "POST /retrain",
        "cities": CITIES,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput):
    city = payload.city.value
    # Validate the calendar date (catches e.g. 31 February)
    try:
        d = date(payload.year, payload.month, payload.day)
    except ValueError:
        raise HTTPException(status_code=422,
                            detail="Invalid calendar date for the given year/month/day.")

    features = build_feature_row(city, d)
    features_scaled = scaler.transform(features)
    prediction = float(model.predict(features_scaled)[0])
    prediction = round(max(prediction, 0.0), 2)  # PM2.5 can't be negative

    return PredictionOutput(
        city=city,
        date=d.isoformat(),
        predicted_pm25=prediction,
        air_quality=air_quality_band(prediction),
    )


@app.post("/retrain")
async def retrain(file: UploadFile = File(...)):
    """
    Retrain the model on newly uploaded data.

    Expects a CSV with the same schema as the original dataset:
        Date,City,PM25
    The new rows are combined with retraining logic, a fresh Random Forest is
    fitted, and the saved model on disk is replaced so subsequent /predict calls
    use the updated model.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=422, detail="Please upload a .csv file.")

    contents = await file.read()
    try:
        new_df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read CSV: {exc}")

    required = {"Date", "City", "PM25"}
    if not required.issubset(new_df.columns):
        raise HTTPException(
            status_code=422,
            detail=f"CSV must contain columns {sorted(required)}.",
        )

    # --- Feature engineering (identical to Task 1) ---
    new_df["Date"] = pd.to_datetime(new_df["Date"], errors="coerce")
    new_df = new_df.dropna(subset=["Date", "PM25"])
    new_df["Year"] = new_df["Date"].dt.year
    new_df["Month"] = new_df["Date"].dt.month
    new_df["DayOfYear"] = new_df["Date"].dt.dayofyear
    new_df["Harmattan"] = new_df["Month"].isin([11, 12, 1, 2, 3]).astype(int)

    encoded = pd.get_dummies(new_df.drop(columns=["Date"]),
                             columns=["City"], drop_first=True)
    for col in encoded.select_dtypes("bool").columns:
        encoded[col] = encoded[col].astype(int)

    # Align to the training feature columns (add any missing city dummies as 0)
    y_new = encoded["PM25"]
    X_new = encoded.drop(columns=["PM25"])
    for col in feature_columns:
        if col not in X_new.columns:
            X_new[col] = 0
    X_new = X_new[feature_columns]

    if len(X_new) < 50:
        raise HTTPException(status_code=422,
                            detail="Need at least 50 valid rows to retrain.")

    # --- Refit scaler + model, then persist (hot-swap on disk) ---
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

    # Refresh the in-memory objects so predictions use the new model immediately
    global model, scaler
    model = new_model
    scaler = new_scaler

    return {
        "message": "Model retrained and saved successfully.",
        "rows_used": int(len(X_new)),
    }