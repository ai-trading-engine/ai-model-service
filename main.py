from fastapi import FastAPI
import joblib
import numpy as np
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "ai_signal_model.pkl")

model = joblib.load(model_path)

@app.post("/predict")
def predict(data: dict):
    
    features = np.array([[
        data["rsi"],
        data["ema20"],
        data["ema50"],
        data["volume"]
    ]])
    
    probability = model.predict_proba(features)[0][1]
    
    return {
        "probability": float(round(probability, 4))
    }
