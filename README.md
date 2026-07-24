

# IoT Edge-Device Energy Prediction

## Mission & Problem
Solar/wind-powered IoT sensor nodes in remote African deployments must budget energy
carefully to survive on harvested power, but measuring consumption directly needs extra
hardware. This project predicts a node's energy consumption per operating cycle from its
current state, so devices can forecast their own power use and adapt their behaviour to
stay alive on harvested energy.

## Dataset
**IoT RL: Solar/Wind and Adaptive Transmission Control** (Kaggle, author *wisam1985*).
20,000 timesteps of simulated solar/wind-powered IoT node telemetry across 30 columns
(battery, solar/wind input, CPU/memory load, signal quality, queue, environment, and
transmission actions). **Target:** `energy_consumed_mJ` (millijoules, continuous).
Source: https://www.kaggle.com/datasets/wisam1985/iot-rl-solarwind-and-adaptive-transmission-control

Six leakage-free predictors describe the device's operating state at decision time:
`cpu_usage`, `memory_usage`, `signal_quality`, `action` (0=SLEEP_DEEP … 5=TX_HIGH),
`queue_size`, `temperature_C`. Columns derived from the target (e.g. `net_energy_mJ`)
were dropped to avoid data leakage.

## Model performance
Three scikit-learn models were trained and compared on an 80/20 split:

| Model            | RMSE (mJ) | R²    |
|------------------|-----------|-------|
| Random Forest    | 30.98     | 0.993 |
| Decision Tree    | 32.40     | 0.992 |
| Linear (SGD)     | 66.16     | 0.966 |

Random Forest had the lowest loss and was saved as `best_model.pkl`.

## Live API (Swagger UI)
**Public endpoint:** https://ghana-pm25-api.onrender.com/docs

- `POST /predict` — returns predicted energy (mJ) for a device state.
- `POST /retrain` — upload new telemetry CSV to retrain and hot-swap the model.
- Input validation and realistic ranges are enforced with Pydantic.

Example request body:
```json
{ "cpu_usage": 0.65, "memory_usage": 0.40, "signal_quality": 0.75,
  "action": 5, "queue_size": 12, "temperature_C": 9.5 }
```

## Video demo
**YouTube (≤7 min):** <ADD_YOUTUBE_LINK_HERE>

## Repository structure
```
summative/
├── linear_regression/
│   ├── multivariate.ipynb        # data, EDA, 3 models, best-model saving
│   ├── FINAL_IoT_RL_dataset_2026.csv
│   └── best_model.pkl / scaler.pkl / feature_columns.pkl
├── API/
│   ├── prediction.py             # FastAPI app (predict + retrain, CORS, Pydantic)
│   ├── requirements.txt
│   └── best_model.pkl / scaler.pkl / feature_columns.pkl
└── FlutterApp/                    # main.dart lives in flutter_app/lib/
```

## Running the mobile app
1. Install Flutter (`flutter doctor` should pass for the Android toolchain).
2. From `summative/`, create the project if not present: `flutter create flutter_app`
3. Add the HTTP package: `cd flutter_app && flutter pub add http`
4. Replace `lib/main.dart` with the provided `main.dart`.
5. Connect an Android device or start an emulator: `flutter devices`
6. Run: `flutter run` (select the Android device).
7. Enter a device state, tap **Predict**, and the app displays the predicted energy in mJ.

> Note: the API is hosted on Render's free tier, which sleeps after inactivity. The first
> request may take up to a minute to wake the service.

## Running the API locally
```bash
cd summative/API
uv run uvicorn prediction:app --reload
# Swagger UI at https://ghana-pm25-api.onrender.com/docs
```