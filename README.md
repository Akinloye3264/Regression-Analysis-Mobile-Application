# IoT Edge-Device Energy Prediction


# Mission & Problem
Under rapid urbanization, cities increasingly rely on dense IoT sensor networks for
monitoring, but these renewable-powered nodes must survive on harvested solar/wind
energy without extra metering hardware. This project predicts a device's energy
consumption (mJ) from its operating state, so urban sensor deployments can budget power and scale sustainably.

# Public API (Swagger UI)
**Live endpoint:** https://ghana-pm25-api.onrender.com/docs

- `POST /predict` — returns predicted energy (mJ) for a device state.
- `POST /retrain` — upload new telemetry CSV to retrain and hot-swap the model.
- Inputs are validated with Pydantic (enforced datatypes and realistic ranges).

Example request body:
```json
{ "cpu_usage": 0.65, "memory_usage": 0.40, "signal_quality": 0.75,
  "action": 5, "queue_size": 12, "temperature_C": 9.5 }
```

# Video Demo
YouTube- https://youtu.be/gdBrs4-314A?si=Zq16gJw4AixXYGQ1

# Running the Mobile App
1. Install Flutter and confirm the Android toolchain: `flutter doctor`.
2. From `summative/`, create the project if needed: `flutter create flutter_app`
3. Add the HTTP package: `cd flutter_app && flutter pub add http`
4. Replace `lib/main.dart` with the provided `main.dart`.
5. Connect an Android device or start an emulator: `flutter devices`
6. Run it: `flutter run` (select the Android device - not Chrome/web).
7. Enter a device state, tap **Predict**, and the predicted energy appears on screen.

Apk url- https://drive.google.com/drive/folders/1NUxkKWFlMNXYb9-wsJnV5WKj5MrnHAyu?usp=sharing

> The API is on Render's free tier and sleeps when idle - the first request may take up
> to a minute to wake it.

# Dataset
IoT RL: Solar/Wind and Adaptive Transmission Control (Kaggle, wisam1985) - 20,000 rows,
30 columns of simulated node telemetry. Target: `energy_consumed_mJ`.
Source: https://www.kaggle.com/datasets/wisam1985/iot-rl-solarwind-and-adaptive-transmission-control

# Model Performance
| Model         | RMSE (mJ) | R2    |
|---------------|-----------|-------|
| Random Forest | 30.98     | 0.993 |
| Decision Tree | 32.40     | 0.992 |
| Linear (OLS)  | 65.99     | 0.966 |
| Linear (SGD)  | 66.16     | 0.966 |

Best model (lowest test RMSE) is saved and served by the API.

# Repository Structure
```
summative/
├── linear_regression/multivariate.ipynb   # data, EDA, 4 models, best-model saving
├── API/prediction.py                       # FastAPI (predict + retrain, CORS, Pydantic)
└── flutter_app/                             # main.dart -> flutter_app/lib/main.dart
```