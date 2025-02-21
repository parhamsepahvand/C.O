from django.db import models
# mymodels.py
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb

def get_crypto_data(crypto, days):
    """
    دریافت داده‌های تاریخی ارز دیجیتال از CoinGecko
    """
    url = f"https://api.coingecko.com/api/v3/coins/{crypto}/market_chart?vs_currency=usd&days={days}"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json()
    prices = [item[1] for item in data.get("prices", [])]
    if not prices:
        return None
    # ایجاد DataFrame نمونه
    dates = pd.date_range(end=datetime.now(), periods=len(prices))
    df = pd.DataFrame({"price": prices}, index=dates)
    df["price_change"] = df["price"].pct_change()
    df["SMA"] = df["price"].rolling(window=3, min_periods=1).mean()
    df.dropna(inplace=True)
    return df

def train_regression_model(data):
    features = data[["price", "price_change", "SMA"]]
    target = data["price"]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_lstm_model(data):
    prices = data["price"].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices.reshape(-1, 1))
    n_steps = 3  # استفاده از 3 داده گذشته (برای نمونه)
    X, y = [], []
    for i in range(n_steps, len(scaled_prices)):
        X.append(scaled_prices[i - n_steps:i, 0])
        y.append(scaled_prices[i, 0])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
    model.fit(X, y, epochs=5, verbose=0)
    return model, n_steps, scaler

def predict_xgboost(data):
    X = np.arange(len(data)).reshape(-1, 1)
    y = data["price"].values
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=10)
    model.fit(X, y)
    prediction = model.predict(np.array([[len(data)]]))[0]
    return prediction

def predict_arima(data):
    model = ARIMA(data["price"], order=(1, 1, 0))
    model_fit = model.fit()
    prediction = model_fit.forecast()[0]
    return prediction
