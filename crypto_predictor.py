# /root/backend/crypto_predictor.py

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb

app = Flask(__name__)

# ۱. دریافت داده‌های تاریخی از CoinGecko
def get_crypto_data(crypto_id, days):
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
    params = {"vs_currency": "usd", "days": days}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return None
    data = response.json()
    prices = [item[1] for item in data.get("prices", [])]
    df = pd.DataFrame(prices, columns=["price"])
    df["price_change"] = df["price"].pct_change()
    df["SMA"] = df["price"].rolling(window=14).mean()
    df.dropna(inplace=True)
    return df

# ۲. مدل رگرسیون خطی
def train_regression_model(data):
    features = data[["price", "price_change", "SMA"]]
    target = data["price"]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# ۳. مدل LSTM
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
    return model

def train_lstm_model(data):
    prices = data["price"].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices.reshape(-1, 1))
    n_steps = 30  # استفاده از 30 داده گذشته به عنوان ورودی
    X, y = [], []
    for i in range(n_steps, len(scaled_prices)):
        X.append(scaled_prices[i-n_steps:i, 0])
        y.append(scaled_prices[i, 0])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = create_lstm_model((X.shape[1], 1))
    model.fit(X, y, epochs=20, verbose=0)
    return model, n_steps, scaler

# ۴. مدل XGBoost
def predict_xgboost(data):
    X = np.arange(len(data)).reshape(-1, 1)
    y = data["price"].values
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model.fit(X, y)
    prediction = model.predict(np.array([[len(data)]]))[0]
    return prediction

# ۵. مدل ARIMA
def predict_arima(data):
    model = ARIMA(data["price"], order=(5, 1, 0))
    model_fit = model.fit()
    prediction = model_fit.forecast()[0]
    return prediction

# ۶. API پیش‌بینی
@app.route("/predict", methods=["GET"])
def predict_price():
    crypto_id = request.args.get("crypto_id", default="bitcoin")
    
    try:
        days = int(request.args.get("days", default=30))
    except ValueError:
        return jsonify({"error": "Invalid value for 'days'. Must be an integer."}), 400

    algorithm = request.args.get("algorithm", default="regression")

    # دریافت داده‌های ارز دیجیتال
    data = get_crypto_data(crypto_id, days)
    if data is None or data.empty:
        return jsonify({"error": "Unable to retrieve data"}), 400

    # پیش‌بینی قیمت بر اساس الگوریتم انتخابی
    if algorithm == "regression":
        model = train_regression_model(data)
        latest_data = data[["price", "price_change", "SMA"]].iloc[-1:].values
        predicted_price = model.predict(latest_data)[0]

    elif algorithm == "lstm":
        lstm_model, n_steps, scaler = train_lstm_model(data)
        prices = data["price"].values.reshape(-1, 1)  # تبدیل قیمت‌ها به [rows, 1]
        scaled_prices = scaler.transform(prices)

        last_sequence = scaled_prices[-n_steps:].reshape(1, n_steps, 1)

        print("X_test shape:", last_sequence.shape)  # دیباگ ورودی مدل

        predicted_scaled = lstm_model.predict(last_sequence)[0][0]
        predicted_price = scaler.inverse_transform([[predicted_scaled]])[0][0]

    elif algorithm == "xgboost":
        predicted_price = predict_xgboost(data)

    elif algorithm == "arima":
        predicted_price = predict_arima(data)

    else:
        return jsonify({"error": "Invalid algorithm. Choose 'regression', 'lstm', 'xgboost', or 'arima'."}), 400

    # تجزیه و تحلیل داده‌های فعلی
    analysis = {
        "latest_price": float(data["price"].iloc[-1]),
        "price_change": float(data["price_change"].iloc[-1]),
        "SMA": float(data["SMA"].iloc[-1])
    }

    return jsonify({
        "predicted_price": float(predicted_price),
        "algorithm": algorithm,
        "analysis": analysis
    })

# ۷. اجرای سرور Flask
if __name__ == "__main__":
    # اطمینان از عدم استفاده از reloader برای جلوگیری از اجرای چندباره
    app.run(host="0.0.0.0", port=8080, debug=True, use_reloader=False)

