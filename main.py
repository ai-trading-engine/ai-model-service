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
# Load Model Safely
# -----------------------
model = xgb.XGBClassifier()

if os.path.exists("model.json"):
    model.load_model("model.json")
else:
    print("WARNING: model.json not found")

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
# Fetch Futures Data Safely
# -----------------------
def fetch_data(symbol):
    try:
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=1h&limit=200"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)

        if response.status_code != 200:
            return None

        data = response.json()

        if not isinstance(data, list) or len(data) == 0:
            return None

        columns = [
            "time","open","high","low","close","volume",
            "close_time","qav","trades","taker_base","taker_quote","ignore"
        ]

        df = pd.DataFrame(data, columns=columns)

        df["close"] = df["close"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["open"] = df["open"].astype(float)
        df["volume"] = df["volume"].astype(float)

        return df

    except Exception as e:
        print(f"Fetch error for {symbol}: {e}")
        return None

# -----------------------
# Feature Engineering
# -----------------------
def add_features(df):
    try:
        df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        df["ema20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
        df["ema50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
        df = df.dropna()
        return df
    except:
        return None

# -----------------------
# Analyze One Symbol
# -----------------------
def analyze_symbol(symbol):
    try:
        df = fetch_data(symbol)

        if df is None or df.empty:
            return None

        df = add_features(df)

        if df is None or df.empty or len(df) < 1:
            return None

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
            "probability": float(round(probability, 4)),
            "entry": round(entry, 4),
            "stop_loss": round(stop, 4),
            "take_profit": round(take_profit, 4)
        }

    except Exception as e:
        print(f"Analysis error for {symbol}: {e}")
        return None

# -----------------------
# API Routes
# -----------------------
@app.get("/")
def home():
    return {"status": "AI Breakout Engine Running"}

@app.get("/scan")
def scan_market():
    results = []

    for pair in TOP_PAIRS:
        try:
            print(f"Checking {pair}")

            df = fetch_data(pair)

            if df is None:
                print(f"{pair}: fetch_data returned None")
                continue

            print(f"{pair}: fetched {len(df)} rows")

            df = add_features(df)

            if df is None or df.empty:
                print(f"{pair}: features empty")
                continue

            latest = df.iloc[-1]

            features = np.array([[
                latest["rsi"],
                latest["ema20"],
                latest["ema50"],
                latest["volume"]
            ]])

            probability = model.predict_proba(features)[0][1]

            print(f"{pair}: probability {probability}")

            results.append({
                "symbol": pair,
                "probability": float(round(probability, 4))
            })

        except Exception as e:
            print(f"{pair} ERROR: {e}")
            continue

    results = sorted(results, key=lambda x: x["probability"], reverse=True)

    return results
# -----------------------
# Run Server (Render Compatible)
# -----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
