import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization, Input
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

# Load data from CSV or existing npy files
def load_data():
    if os.path.exists("X_train.npy") and os.path.exists("y_train.npy"):
        print("Loading preprocessed data from .npy files...")
        X_train = np.load("X_train.npy")
        y_train = np.load("y_train.npy")
    else:
        print("Loading OHLC data from CSV...")
        df = pd.read_csv(CSV_FILE_PATH)
        df = df[['Open', 'High', 'Low', 'Close']].dropna()
        print(f"Data loaded: {df.shape[0]} rows")
        
        X_train, y_train = preprocess_data(df)
        np.save("X_train.npy", X_train)
        np.save("y_train.npy", y_train)
    
    return X_train, y_train

# Feature extraction

def compute_features(df, max_window_size=10):
    print("Extracting features...")
    features = pd.DataFrame({
        "body_size": abs(df["Close"] - df["Open"]),
        "upper_wick": df["High"] - df[["Open", "Close"]].max(axis=1),
        "lower_wick": df[["Open", "Close"]].min(axis=1) - df["Low"],
        "volatility": (df["High"] - df["Low"]) / (df["Open"] + 1e-6),
        "direction": (df["Close"] > df["Open"]).astype(int)
    })
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    print("Feature extraction completed.")
    return features

# Compute dynamic window sizes
def dynamic_window_size(df, min_window=3, max_window=10):
    print("Computing dynamic window size...")
    features = compute_features(df)
    scaler = StandardScaler()
    features_scaled = np.nan_to_num(scaler.fit_transform(features))
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(features_scaled)
    
    if np.all(clusters == -1):
        raise ValueError("❌ All data points were classified as noise by DBSCAN. Adjust clustering parameters.")
    
    df = df.copy()
    df["pattern_cluster"] = LabelEncoder().fit_transform(np.clip(clusters, 0, None))
    dynamic_windows = [min(max_window, max(min_window, 3 + abs(c) // 2)) for c in df["pattern_cluster"]]
    print("Window size determination complete.")
    return dynamic_windows, df

# Prepare training data
def preprocess_data(df):
    dynamic_windows, df = dynamic_window_size(df)
    X, y = [], []
    
    max_window_size = max(dynamic_windows)
    for i in range(len(df)):
        window_size = dynamic_windows[i]
        if i >= window_size:
            window = df[['Open', 'High', 'Low', 'Close']].values[i - window_size:i]
            window = np.pad(window, ((max_window_size - len(window), 0), (0, 0)), mode='constant')
            X.append(window)
            y.append(df["pattern_cluster"].iloc[i])
    
    X = np.array(X, dtype=np.float32)
    y = to_categorical(np.array(y, dtype=np.int32))
    
    print(f"Training data prepared: X shape {X.shape}, y shape {y.shape}")
    return X, y

# Model training function
def train_model():
    X_train, y_train = load_data()
    print("Building model...")
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        Conv1D(64, 3, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(128, return_sequences=False),
        Dense(128, activation='relu'),
        Dense(y_train.shape[1], activation='softmax')
    ])

    optimizer = Adam(learning_rate=1e-5, clipnorm=1.0)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    print("Starting model training...")
    history = model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.2,
                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
    
    print("Model training completed successfully.")
    model.save('candlestick_model_fixed.keras')
    print("Model saved as 'candlestick_model_fixed.keras'.")
    
    return history.history

if __name__ == '__main__':
    history = train_model()
    print(f"📊 Final Training Loss: {history['loss'][-1]}")
    print(f"📊 Final Validation Loss: {history['val_loss'][-1]}")
    print("🎉 Training complete!")
