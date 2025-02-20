"""
Train LSTM stock price predictor.
Usage: python model/train.py --ticker AAPL --epochs 50
"""
import argparse
import os
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from sklearn.preprocessing import MinMaxScaler
from lstm_model import build_lstm

SEQ_LEN = 60
MODEL_DIR = os.path.dirname(__file__)


def fetch_data(ticker: str, period: str = '5y') -> pd.Series:
    df = yf.download(ticker, period=period, progress=False)
    return df['Close'].dropna()


def make_sequences(data: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def train(ticker: str, epochs: int):
    print(f"Fetching data for {ticker}...")
    prices = fetch_data(ticker).values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    X, y = make_sequences(scaled, SEQ_LEN)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_lstm(seq_len=SEQ_LEN)
    model.summary()

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=32,
        callbacks=[
            __import__('tensorflow').keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            __import__('tensorflow').keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
        ]
    )

    model_path = os.path.join(MODEL_DIR, f'{ticker}_lstm.h5')
    scaler_path = os.path.join(MODEL_DIR, f'{ticker}_scaler.pkl')
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved model → {model_path}")
    print(f"Saved scaler → {scaler_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', default='AAPL')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    train(args.ticker, args.epochs)
