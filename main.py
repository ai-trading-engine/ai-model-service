from fastapi import FastAPI
import xgboost as xgb
import numpy as np
import os
import uvicorn

app = FastAPI()

model = xgb.XGBClassifier()
model.load_model("model.json")

@app.post("/predict")
def predict(data: dict):
    features = np.array([[
        data["rsi"],
        data["ema20"],
        data["ema50"],
        data["volume"]
    ]])

    prob = model.predict_proba(features)[0][1]

    if prob > 0.75:
        return {"signal": "TRADE", "probability": float(prob)}
    else:
        return {"signal": "NO TRADE", "probability": float(prob)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
