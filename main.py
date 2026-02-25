from fastapi import FastAPI
import xgboost as xgb
import pandas as pd
import numpy as np
import requests
import ta
import os
import uvicorn

app = FastAPI()

# -----------------------
# Load Model
# -----------------------
model = xgb.XGBClassifier()
model.load_model("model.json")

# -----------------------
# Top 20 USDT-M Pairs
# -----------------------
TOP_PAIRS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT",
    "ADAUSDT","DOGEUSDT","AVAXUSDT","LINKUSDT","MATICUSDT",
    "DOTUSDT","TRXUSDT","LTCUSDT","BCHUSDT","APTUSDT",
    "ARBUSDT","OPUSDT","ATOMUSDT","FILUSDT","NEARUSDT"
]

# -----------------------
# Fetch Futures Data
# -----------------------
def fetch_data(symbol):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=1h&limit=200"
    data = requests.get(url).json()

    columns = ["time","open","high","low","close","volume",
               "close_time","qav","trades","taker_base","taker_quote","ignore"]

    df = pd.DataFrame(data, columns=columns)

    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["open"] = df["open"].astype(float)
    df["volume"] = df["volume"].astype(float)

    return df

# -----------------------
# Feature Engineering
# -----------------------
def add_features(df):
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["ema20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df = df.dropna()
    return df

# -----------------------
# Generate Signals
# -----------------------
def analyze_symbol(symbol):
    df = fetch_data(symbol)
    df = add_features(df)

    latest = df.iloc[-1]

    features = np.array([[
        latest["rsi"],
        latest["ema20"],
        latest["ema50"],
        latest["volume"]
    ]])

    probability = model.predict_proba(features)[0][1]

    if probability < 0.75:
        return None

    entry = latest["close"]
    stop = entry * 0.99
    take_profit = entry * 1.02

    return {
        "symbol": symbol,
        "probability": float(probability),
        "entry": round(entry, 4),
        "stop_loss": round(stop, 4),
        "take_profit": round(take_profit, 4)
    }

# -----------------------
# API Endpoints
# -----------------------
@app.get("/")
def home():
    return {"status": "AI Breakout Engine Running"}

@app.get("/scan")
def scan_market():
    signals = []

    for pair in TOP_PAIRS:
        result = analyze_symbol(pair)
        if result:
            signals.append(result)

    signals = sorted(signals, key=lambda x: x["probability"], reverse=True)

    return {
        "total_signals": len(signals),
        "signals": signals
    }

# -----------------------
# Run Server (Render Compatible)
# -----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
