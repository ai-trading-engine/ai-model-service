from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("ai_signal_model.pkl")

if os.path.exists("model.json"):
    model.load_model("model.json")
else:
    print("WARNING: model.json not found")

@app.get("/")
def home():
    return {"status": "AI Model Ready"}

@app.post("/predict")
def predict(data: dict):
    try:
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

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
