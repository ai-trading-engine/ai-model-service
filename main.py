from fastapi import FastAPI
import xgboost as xgb
import numpy as np

app = FastAPI()

model = xgb.Booster()
model.load_model("model.json")

@app.post("/predict")
def predict(data: dict):
    features = np.array([[data["rsi"], data["ema9"], data["ema21"], data["atr"]]])
    dmatrix = xgb.DMatrix(features)
    prob = model.predict(dmatrix)[0]

    return {"probability": float(prob)}

import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # default 8000 if not set
    uvicorn.run("main:app", host="0.0.0.0", port=port)
