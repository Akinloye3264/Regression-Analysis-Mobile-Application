
# Loads the best-performing model saved by the notebook and exposes a single
# `predict_energy(...)` function that returns the predicted energy consumption (mJ)
# for one device state. The API in Task 2 imports and uses this same function.

import os
import joblib
import pandas as pd

# Load the saved artifacts (produced by multivariate.ipynb)
HERE = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(HERE, "best_model.pkl"))
scaler = joblib.load(os.path.join(HERE, "scaler.pkl"))
feature_columns = joblib.load(os.path.join(HERE, "feature_columns.pkl"))


def predict_energy(cpu_usage, memory_usage, signal_quality,
                   action, queue_size, temperature_C):
    # Return predicted energy consumed (mJ) for one device state.
    # Args:
    #     cpu_usage (float): 0.0-1.0
    #     memory_usage (float): 0.0-1.0
    #     signal_quality (float): 0.0-1.0
    #     action (int): 0=SLEEP_DEEP ... 5=TX_HIGH
    #     queue_size (int): packets waiting
    #  temperature_C (float): ambient temperature
    
    row = {
        "cpu_usage": cpu_usage,
        "memory_usage": memory_usage,
        "signal_quality": signal_quality,
        "action": int(action),
        "queue_size": queue_size,
        "temperature_C": temperature_C,
    }
    # Order columns exactly as during training, scale, then predict
    features = pd.DataFrame([[row[c] for c in feature_columns]], columns=feature_columns)
    features_scaled = scaler.transform(features)
    prediction = float(model.predict(features_scaled)[0])
    return round(max(prediction, 0.0), 2)


if __name__ == "__main__":
    result = predict_energy(
        cpu_usage=0.65, memory_usage=0.40, signal_quality=0.75,
        action=5, queue_size=12, temperature_C=9.5,
    )
    print("Predicted energy consumed:", result, "mJ")