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
