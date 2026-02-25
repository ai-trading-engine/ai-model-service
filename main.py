from fastapi import FastAPI
import os
import uvicorn

app = FastAPI()

@app.get("/")
def home():
    return {"status": "AI service running"}

@app.post("/predict")
def predict(data: dict):
    # Temporary dummy probability
    return {"probability": 0.82}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
