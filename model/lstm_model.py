import tensorflow as tf
from tensorflow.keras import layers, models


def build_lstm(seq_len: int = 60, n_features: int = 1) -> tf.keras.Model:
    """
    Stacked LSTM for univariate time-series prediction.
    Input shape: (batch, seq_len, n_features)
    Output: single value (next closing price, normalized)
    """
    model = models.Sequential([
        layers.Input(shape=(seq_len, n_features)),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# feat-3layer-lstm
