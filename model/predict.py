import os
import numpy as np
import yfinance as yf
import joblib
import tensorflow as tf

SEQ_LEN = 60
MODEL_DIR = os.path.dirname(__file__)


def predict_next(ticker: str, days: int = 30) -> dict:
    model_path = os.path.join(MODEL_DIR, f'{ticker}_lstm.h5')
    scaler_path = os.path.join(MODEL_DIR, f'{ticker}_scaler.pkl')

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    df = yf.download(ticker, period='1y', progress=False)
    prices = df['Close'].dropna().values.reshape(-1, 1)
    scaled = scaler.transform(prices)

    # Build actual vs predicted for last portion
    X, y_actual = [], []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i - SEQ_LEN:i])
        y_actual.append(prices[i][0])

    X = np.array(X)
    y_pred_scaled = model.predict(X, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten().tolist()

    # Future predictions
    last_seq = scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
    future = []
    for _ in range(days):
        pred = model.predict(last_seq, verbose=0)[0][0]
        future.append(float(scaler.inverse_transform([[pred]])[0][0]))
        last_seq = np.append(last_seq[:, 1:, :], [[[pred]]], axis=1)

    import pandas as pd
    dates = df.index[-len(y_pred):].strftime('%Y-%m-%d').tolist()

    return {
        'ticker': ticker,
        'dates': dates,
        'actual': [round(v, 2) for v in y_actual[-len(y_pred):]],
        'predicted': [round(v, 2) for v in y_pred],
        'future': [round(v, 2) for v in future]
    }
