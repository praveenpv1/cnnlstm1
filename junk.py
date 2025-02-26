# Dynamic window size function old function
def dynamic_window_size(df, min_window=3, max_window=10):
    print("Computing dynamic window size based on clustering and volatility...")
    features = compute_features(df)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(features_scaled)
    df = df.iloc[:len(clusters)]
    df["pattern_cluster"] = clusters
    df = df[df["pattern_cluster"] >= 0]
    label_encoder = LabelEncoder()
    df["pattern_cluster"] = label_encoder.fit_transform(df["pattern_cluster"])

    dynamic_windows = [
        min(max_window, max(min_window, int(1 + np.abs(cluster_id) * 0.5)))
        for cluster_id in df["pattern_cluster"].values
    ]

    print("Window size determination complete.")
    return dynamic_windows, df


# Model training function
def train_model():
    df = load_data()
    dynamic_windows, df = dynamic_window_size(df)
    print("Preparing training and validation datasets...")
    max_window_size = max(dynamic_windows)
    X = []
    y = df["pattern_cluster"].values
    for i, window_size in enumerate(dynamic_windows):
        if i >= window_size:
            window = df[['Open', 'High', 'Low', 'Close']].values[i - window_size:i]
            pad = np.zeros((max_window_size - len(window), 4))
            window = np.vstack((pad, window)) if len(window) < max_window_size else window[-max_window_size:]
            X.append(window)
    X = np.array(X)
    y = np.array(y[max_window_size - 1:])

    print(f"Training data prepared: X shape {X.shape}, y shape {y.shape}")

    print("Building model...")
    model = Sequential([
        Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(max_window_size, 4)),
        LSTM(64, return_sequences=False),
        Dense(64, activation='relu'),
        Dense(len(np.unique(y)), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Starting model training...")
    history = model.fit(X, y, epochs=15, batch_size=16, validation_split=0.2)
    print("Model training completed successfully.")
    model.save('candlestickfivemin_model_dynamic.h5')
    print("Model saved as 'candlestickfivemin_model_dynamic.h5'.")
    return history.history