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

def load_data():
    print("Loading OHLC data...")
    df = pd.read_csv(CSV_FILE_PATH)
    df = df[['Open', 'High', 'Low', 'Close']].dropna()  # Drop NaN values
    print(f"Data loaded: {df.shape[0]} rows")
    return df

# Feature extraction with stability fixes
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
    })
    
    features = features.replace([np.inf, -np.inf], np.nan).dropna()  # Replace infinite values and drop NaNs
    print("Feature extraction completed.")
    return features

# Dynamic window size function
def dynamic_window_size(df, min_window=3, max_window=10):
    print("Computing dynamic window size...")
    features = compute_features(df)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    # Ensure no NaN or Inf
    features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(features_scaled)
    
    if np.all(clusters == -1):
        raise ValueError("❌ All data points were classified as noise by DBSCAN. Adjust clustering parameters.")

    df = df.copy()
    df["pattern_cluster"] = clusters
    df = df[df["pattern_cluster"] >= 0]  # Remove noise points

    label_encoder = LabelEncoder()
    df["pattern_cluster"] = label_encoder.fit_transform(df["pattern_cluster"])
    
    dynamic_windows = [min(max_window, max(min_window, 3 + abs(c) // 2)) for c in df["pattern_cluster"]]
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
    
    # X = np.array(X, dtype=np.float32)  # Ensure stable precision
    # y = np.array(y, dtype=np.int32)

    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
    y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
    
    print(f"Training data prepared: X shape {X.shape}, y shape {y.shape}")

    unique_labels = np.unique(y)
    print(f"Unique labels: {unique_labels}")

    y = to_categorical(y, num_classes=len(unique_labels))  # Ensure correct one-hot encoding

    print("Building model...")
    model = Sequential([
        Input(shape=(max_window_size, 4)),  # Explicit input layer
        Conv1D(64, 3, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(128, return_sequences=False),
        Dense(128, activation='relu'),
        Dense(y.shape[1], activation='softmax')
    ])

    optimizer = Adam(learning_rate=1e-4, clipvalue=1.0)  # Lower LR & gradient clipping
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    def custom_loss(y_true, y_pred):
        loss = loss_fn(y_true, y_pred)
        loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)  # Replace NaNs with 0
        return loss
    model.compile(
        optimizer=Adam(learning_rate=1e-4, clipvalue=1.0),
        loss=custom_loss,  # Use NaN-safe loss
        metrics=['accuracy']
    )

    print("Starting model training...")
    nan_callback = NaNStopping()
    print(f"NaNs in X: {np.isnan(X).sum()}, Infs in X: {np.isinf(X).sum()}")
    print(f"NaNs in y: {np.isnan(y).sum()}, Infs in y: {np.isinf(y).sum()}")
    print(f"X min/max: {X.min()}/{X.max()}, y min/max: {y.min()}/{y.max()}")
    history = model.fit(X, y, epochs=25, batch_size=32, validation_split=0.2, callbacks=[nan_callback])

    print("Model training completed successfully.")
    model.save('candlestick_model_fixed.keras')  # Save in the recommended format
    print("Model saved as 'candlestick_model_fixed.keras'.")
    
    return history.history

if __name__ == '__main__':
    history = train_model()
    print(f"📊 Final Training Loss: {history['loss'][-1]}")
    print(f"📊 Final Validation Loss: {history['val_loss'][-1]}")
    print("🎉 Training complete!")
