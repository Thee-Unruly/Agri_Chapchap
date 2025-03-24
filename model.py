import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("ml_ready_data.csv")

# Convert date column to datetime
data['date'] = pd.to_datetime(data['date'])

# Sort by date
data = data.sort_values(by='date')

# Define features and target
features = ['precipitation_sum', 'precipitation_hours', 'wind_direction_10m_dominant', 
            'et0_fao_evapotranspiration', 'sunshine_duration', 'shortwave_radiation_sum']
target = 'rain_sum'

# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features + [target]])

# Convert data to sequences for LSTM
def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :-1])  # Features
        y.append(data[i+seq_length, -1])  # Target
    return np.array(X), np.array(y)

# Define sequence length
seq_length = 10
X, y = create_sequences(scaled_data, seq_length)

# Train-Test Split (80% train, 20% test)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM Model
model = keras.Sequential([
    keras.layers.LSTM(50, return_sequences=True, input_shape=(seq_length, len(features))),
    keras.layers.LSTM(50, return_sequences=False),
    keras.layers.Dense(25, activation='relu'),
    keras.layers.Dense(1)
])

# Compile Model
model.compile(optimizer='adam', loss='mse')

# Train Model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Make Predictions
y_pred = model.predict(X_test)

# Convert back to original scale
y_pred_rescaled = scaler.inverse_transform(np.hstack((X_test[:, -1, :], y_pred.reshape(-1, 1))))[:, -1]
y_test_rescaled = scaler.inverse_transform(np.hstack((X_test[:, -1, :], y_test.reshape(-1, 1))))[:, -1]

# Evaluate Model
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
r2 = r2_score(y_test_rescaled, y_pred_rescaled)

print(f"✅ LSTM Model Performance:")
print(f"📉 MAE: {mae:.3f}")
print(f"📊 RMSE: {rmse:.3f}")
print(f"📈 R² Score: {r2:.3f}")

# Plot Predictions vs Actual
plt.figure(figsize=(10,5))
plt.plot(y_test_rescaled, label="Actual Rainfall", color='blue')
plt.plot(y_pred_rescaled, label="Predicted Rainfall", color='red', linestyle='dashed')
plt.xlabel("Time")
plt.ylabel("Rainfall")
plt.legend()
plt.title("LSTM Rainfall Prediction")
plt.show()
