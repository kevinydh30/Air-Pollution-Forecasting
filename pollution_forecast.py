import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def generate_synthetic_data(n_samples=5000):
    """Generates synthetic multivariate time series data."""
    time = np.arange(n_samples)
    
    # Pollution: Trend + Seasonality + Noise
    pollution = 50 + 0.01 * time + 20 * np.sin(time / 50) + np.random.normal(0, 5, n_samples)
    
    # Temperature: Seasonality + Noise
    temp = 20 + 10 * np.sin(time / 100) + np.random.normal(0, 2, n_samples)
    
    # Humidity: Inverse to Temp + Noise
    humidity = 60 - 5 * np.sin(time / 100) + np.random.normal(0, 3, n_samples)
    
    # Wind Speed: Random Walk
    wind_speed = 5 + np.cumsum(np.random.normal(0, 0.1, n_samples))
    wind_speed = np.abs(wind_speed)
    
    data = pd.DataFrame({
        'pollution': pollution,
        'temp': temp,
        'humidity': humidity,
        'wind_speed': wind_speed
    })
    
    return data

def create_sequences(data, seq_length, predict_steps=1):
    """Converts data into sequences for LSTM training."""
    xs, ys = [], []
    for i in range(len(data) - seq_length - predict_steps + 1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, 0]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def compute_metrics(y_true, y_pred):
    """Computes MAE, RMSE, MSE, and R² Score."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R²": r2}

def train_model(X_train, y_train, X_test, y_test, units=50, dropout=0.2, epochs=20, batch_size=32, early_stopping=False, patience=5):
    """Builds and trains the LSTM model with optional Early Stopping."""
    model = Sequential([
        LSTM(units=units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(dropout),
        LSTM(units=units, return_sequences=False),
        Dropout(dropout),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    callbacks = []
    if early_stopping:
        callbacks.append(EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ))
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    return model, history

if __name__ == "__main__":
    print("Generating synthetic data...")
    df = generate_synthetic_data()
    print(f"Data shape: {df.shape}")
    
    # Plot raw data
    plt.figure(figsize=(12, 6))
    plt.plot(df['pollution'][:500], label='Pollution')
    plt.title('Synthetic Air Pollution Data (First 500 hours)')
    plt.legend()
    plt.savefig('pollution_data_sample.png')
    print("Saved pollution_data_sample.png")

    # scaling
    print("Scaling data...")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Parameters
    SEQ_LENGTH = 24  # Use past 24 hours to predict next
    TRAIN_SPLIT = 0.8
    
    # Create sequences
    X, y = create_sequences(scaled_data, SEQ_LENGTH)
    
    # Split into train/test
    train_size = int(len(X) * TRAIN_SPLIT)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    model, history = train_model(X_train, y_train, X_test, y_test)
    
    # Visualize Loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss (Train vs Validation)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('loss_plot.png')
    print("Saved loss_plot.png")
    
    # Evaluate
    loss = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")

    print("Done.")
