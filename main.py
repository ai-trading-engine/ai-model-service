from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import os
import uvicorn
from pydantic import BaseModel

# 1. Define the input schema for better validation
class SignalData(BaseModel):
    rsi: float
    ema20: float
    ema50: float
    volume: float

app = FastAPI()

# 2. Robust Path Handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "ai_signal_model.pkl")

# Load the model once
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

@app.get("/")
def home():
    return {"status": "AI Model Ready"}

@app.post("/predict")
def predict(data: SignalData): # Using Pydantic here
    try:
        # 3. Convert Pydantic model to numpy array
        features = np.array([[
            data.rsi,
            data.ema20,
            data.ema50,
            data.volume
        ]])

        # Get probability for the positive class
        probability = model.predict_proba(features)[0][1]

        return {
            "probability": float(round(probability, 4)),
            "signal": "BUY" if probability > 0.5 else "SELL"
        }

    except Exception as e:
        # Use HTTPException for proper API error responses
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # Note: "main:app" assumes your file is named main.py
    uvicorn.run(app, host="0.0.0.0", port=port)
