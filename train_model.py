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

# Enable mixed precision for faster training
mixed_precision.set_global_policy('mixed_float16')

# Check GPU availability
if tf.config.list_physical_devices('GPU'):
    print("✅ GPU detected! Training will be accelerated.")
else:
    print("❌ No GPU found! Training will be slow.")

CSV_FILE_PATH = "ohlc_data.csv"

def load_data():
    print("Loading OHLC data...")
    df = pd.read_csv(CSV_FILE_PATH)
    df = df[['Open', 'High', 'Low', 'Close']].dropna()  # Drop NaN values
    print(f"Data loaded: {df.shape[0]} rows")
    return df

# Feature extraction with NaN checks
def compute_features(df, max_window_size=10):
    print("Extracting features...")
    body_sizes, upper_wicks, lower_wicks, volatilities, directions = [], [], [], [], []
    for i in range(len(df)):
        window = df.iloc[max(0, i - max_window_size):i + 1]
        body_size = abs(window["Close"].iloc[-1] - window["Open"].iloc[0])
        upper_wick = window["High"].max() - max(window["Open"].iloc[0], window["Close"].iloc[-1])
        lower_wick = min(window["Open"].iloc[0], window["Close"].iloc[-1]) - window["Low"].min()
        volatility = (window["High"].max() - window["Low"].min()) / max(1e-6, window["Open"].iloc[0])  # Avoid div by zero
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
    }).replace([np.inf, -np.inf], np.nan).dropna()  # Replace infinite values and drop NaNs

    print("Feature extraction completed.")
    return features

# Dynamic window size function
def dynamic_window_size(df, min_window=3, max_window=10):
    print("Computing dynamic window size...")
    features = compute_features(df)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(features_scaled)
    
    if np.all(clusters == -1):
        raise ValueError("All data points were classified as noise by DBSCAN. Adjust parameters.")
    
    df = df.copy()
    df["pattern_cluster"] = clusters
    df = df[df["pattern_cluster"] >= 0]  # Remove noise points
    
    label_encoder = LabelEncoder()
    df["pattern_cluster"] = label_encoder.fit_transform(df["pattern_cluster"])
    
    dynamic_windows = [min(max_window, max(min_window, 3 + abs(c) // 2)) for c in df["pattern_cluster"]]
    print("Window size determination complete.")
    return dynamic_windows, df

class NaNStopping(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('loss') is None or np.isnan(logs['loss']):
            print(f"❌ NaN detected at epoch {epoch}, stopping training.")
            self.model.stop_training = True

# Model training function
def train_model():
    df = load_data()
    dynamic_windows, df = dynamic_window_size(df)
    print("Preparing training datasets...")
    max_window_size = max(dynamic_windows)
    X, y = [], []
    
    for i in range(len(df)):
        window_size = dynamic_windows[i]
        if i >= window_size:
            window = df[['Open', 'High', 'Low', 'Close']].values[i - window_size:i]
            if len(window) < max_window_size:
                pad = np.zeros((max_window_size - len(window), 4))
                window = np.vstack((pad, window))
            X.append(window)
            y.append(df["pattern_cluster"].iloc[i])
    
    X = np.array(X, dtype=np.float32)  # Ensure float32 precision
    y = np.array(y, dtype=np.int32)
    print(f"Training data prepared: X shape {X.shape}, y shape {y.shape}")
    
    y = to_categorical(y, num_classes=len(set(y)))  # Ensure correct label encoding
    
    print("Building model...")
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=(max_window_size, 4)),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(128, return_sequences=False),
        Dense(128, activation='relu'),
        Dense(y.shape[1], activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=1e-3, clipnorm=1.0), loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Starting model training...")
    nan_callback = NaNStopping()
    history = model.fit(X, y, epochs=25, batch_size=32, validation_split=0.2, callbacks=[nan_callback])
    
    print("Model training completed successfully.")
    model.save('candlestick_model_fixed.h5')
    print("Model saved as 'candlestick_model_fixed.h5'.")
    return history.history

if __name__ == '__main__':
    history = train_model()
    print(f"📊 Final Training Loss: {history['loss'][-1]}")
    print(f"📊 Final Validation Loss: {history['val_loss'][-1]}")
    print("🎉 Training complete!")
