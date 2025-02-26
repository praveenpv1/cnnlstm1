import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Enable mixed precision for faster training on H100
mixed_precision.set_global_policy('mixed_float16')

# Check GPU availability
if tf.config.list_physical_devices('GPU'):
    print("✅ GPU detected! Training will be accelerated.")
else:
    print("❌ No GPU found! Training will be slow.")

# Load OHLC data from CSV
CSV_FILE_PATH = "ohlc_data.csv"

def load_data():
    print("Loading OHLC data...")
    df = pd.read_csv(CSV_FILE_PATH)
    df = df[['Open', 'High', 'Low', 'Close']]  # Ensure only OHLC columns are used
    print(f"Data loaded: {df.shape[0]} rows")
    return df

# Feature extraction
def compute_features(df, max_window_size=10):
    print("Extracting features...")
    body_sizes, upper_wicks, lower_wicks, volatilities, directions = [], [], [], [], []
    for i in range(len(df)):
        window = df.iloc[max(0, i - max_window_size):i + 1]
        body_size = abs(window["Close"].iloc[-1] - window["Open"].iloc[0])
        upper_wick = window["High"].max() - max(window["Open"].iloc[0], window["Close"].iloc[-1])
        lower_wick = min(window["Open"].iloc[0], window["Close"].iloc[-1]) - window["Low"].min()
        volatility = (window["High"].max() - window["Low"].min()) / window["Open"].iloc[0]
        direction = int(window["Close"].iloc[-1] > window["Open"].iloc[0])

        body_sizes.append(body_size)
        upper_wicks.append(upper_wick)
        lower_wicks.append(lower_wick)
        volatilities.append(volatility)
        directions.append(direction)

    features = pd.DataFrame({
        "body_size": body_sizes,
        "upper_wick": upper_wicks,
        "lower_wick": lower_wicks,
        "volatility": volatilities,
        "direction": directions
    })
    print("Feature extraction completed.")
    return features

# Dynamic window size function
def dynamic_window_size(df, min_window=3, max_window=10):
    print("Computing dynamic window size based on clustering and volatility...")
    features = compute_features(df)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(features_scaled)
    df = df.copy()
    df.loc[:, "pattern_cluster"] = clusters
    df = df.loc[df["pattern_cluster"] >= 0].copy()
    label_encoder = LabelEncoder()
    df.loc[:, "pattern_cluster"] = label_encoder.fit_transform(df["pattern_cluster"])
    dynamic_windows = []
    for i in range(len(df)):
        cluster_id = df["pattern_cluster"].iloc[i]
        window_size = min(max_window, max(min_window, int(1 + np.abs(cluster_id) * 0.5)))
        dynamic_windows.append(window_size)
    print("Window size determination complete.")
    return dynamic_windows, df

# NaN detection callback
class NaNStopping(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('loss') is None or np.isnan(logs['loss']):
            print(f"❌ NaN detected at epoch {epoch}, stopping training.")
            self.model.stop_training = True

# Model training function
def train_model():
    df = load_data()
    dynamic_windows, df = dynamic_window_size(df)
    print("Preparing training and validation datasets...")
    max_window_size = max(dynamic_windows)
    X, y = [], []
    for i in range(len(df)):
        window_size = dynamic_windows[i]
        if i >= window_size:
            window = df[['Open', 'High', 'Low', 'Close']].values[i - window_size:i]
            if len(window) < max_window_size:
                pad = np.zeros((max_window_size - len(window), 4))
                window = np.vstack((pad, window))
            elif len(window) > max_window_size:
                window = window[-max_window_size:]
            X.append(window)
            y.append(df["pattern_cluster"].iloc[i])
    X = np.array(X)
    y = np.array(y)
    print(f"Training data prepared: X shape {X.shape}, y shape {y.shape}")
    unique_labels = np.unique(y)
    print(f"Unique labels: {unique_labels}")
    y = to_categorical(y)
    print("Building model...")
    optimizer = Adam(clipnorm=1.0)
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(max_window_size, 4)),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(128, return_sequences=False),
        Dense(128, activation='relu'),
        Dense(len(set(np.argmax(y, axis=1))), activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print("Starting model training...")
    nan_callback = NaNStopping()
    history = model.fit(X, y, epochs=25, batch_size=32, validation_split=0.2, callbacks=[nan_callback])
    print("Model training completed successfully.")
    model.save('candlestickfivemin_model_dynamic.h5')
    print("Model saved as 'candlestickfivemin_model_dynamic.h5'.")
    return history.history

if __name__ == '__main__':
    history = train_model()
    print(f"📊 Final Training Loss: {history['loss'][-1]}")
    print(f"📊 Final Validation Loss: {history['val_loss'][-1]}")
    print("🎉 Model training and saving complete!")
