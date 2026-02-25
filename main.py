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
