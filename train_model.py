import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense
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
    # Initialize lists to store features for each window
    body_sizes, upper_wicks, lower_wicks, volatilities, directions = [], [], [], [], []
    for i in range(len(df)):
        window = df.iloc[max(0, i-max_window_size):i+1]  # Use a window of up to `max_window_size` candles for flexibility
        # Features for the window
        body_size = abs(window["Close"].iloc[-1] - window["Open"].iloc[0])
        upper_wick = window["High"].max() - max(window["Open"].iloc[0], window["Close"].iloc[-1])
        lower_wick = min(window["Open"].iloc[0], window["Close"].iloc[-1]) - window["Low"].min()
        volatility = (window["High"].max() - window["Low"].min()) / window["Open"].iloc[0]
        direction = (window["Close"].iloc[-1] > window["Open"].iloc[0]).astype(int)

        # Store features
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
    # Compute features
    features = compute_features(df)
    
    # Scale features for DBSCAN
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply DBSCAN to detect regions of interest
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(features_scaled)
    df["pattern_cluster"] = clusters
    df = df[df["pattern_cluster"] >= 0] 
    # Reindex labels to ensure they are contiguous
    label_encoder = LabelEncoder()
    df["pattern_cluster"] = label_encoder.fit_transform(df["pattern_cluster"])
    
    # Dynamically adjust window size based on the cluster size or volatility
    dynamic_windows = []
    
    for i in range(len(df)):
        # Determine window size dynamically based on the cluster size or volatility of the pattern
        cluster_id = df["pattern_cluster"].iloc[i]
        
        # For large patterns (more volatile), use a larger window
        window_size = min(max_window, max(min_window, int(1 + np.abs(cluster_id) * 0.5)))
        
        dynamic_windows.append(window_size)
    
    print("Window size determination complete.")
    return dynamic_windows, df

# Model training function
def train_model():
    df = load_data()
    dynamic_windows, df = dynamic_window_size(df)
    print("Preparing training and validation datasets...")
    X = []
    y = df["pattern_cluster"].values
    # Create input sequences for varying window sizes
    for i in range(len(df)):
        window_size = dynamic_windows[i]
        # Ensure we have enough data for the current window size
        if i >= window_size:
            X.append(df[['Open', 'High', 'Low', 'Close']].values[i - window_size:i])

    X = np.array(X)
    y = np.array(y[window_size - 1:])  # Align labels to match the rolling window size
    print(f"Training data prepared: X shape {X.shape}, y shape {y.shape}")

    print("Building model...")
    # Build CNN + LSTM model
    model = Sequential([
        Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(None, 4)),  # Input shape varies dynamically
        LSTM(64, return_sequences=False),
        Dense(64, activation='relu'),
        Dense(len(set(y)), activation='softmax')  # Dynamic classification
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Starting model training...")
    history = model.fit(X, y, epochs=15, batch_size=16, validation_split=0.2)
    print("Model training completed successfully.")
    # Save the trained model (Optional)
    model.save('candlestickfivemin_model_dynamic.h5')
    print("Model saved as 'candlestickfivemin_model_dynamic.h5'.")
    return history.history

if __name__ == '__main__':
    history = train_model()
    print(f"📊 Final Training Loss: {history['loss'][-1]}")
    print(f"📊 Final Validation Loss: {history['val_loss'][-1]}")
    print("🎉 Model training and saving complete!")