import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Step 1: Fetch weather data from Open-Meteo
API_URL = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": -1.286389,  # Nairobi, Kenya (change to your location)
    "longitude": 36.817223,
    "daily": ["precipitation_sum"],
    "temperature_unit": "celsius",
    "wind_speed_unit": "kmh",
    "precipitation_unit": "mm",
    "timezone": "Africa/Nairobi",
    "past_days": 60  # Get past 60 days of rainfall data
}
response = requests.get(API_URL, params=params)
data = response.json()

# Step 2: Convert to DataFrame
df = pd.DataFrame({
    "date": pd.to_datetime(data["daily"]["time"]),
    "rainfall": data["daily"]["precipitation_sum"]
})

# Step 3: Prepare data for ML model
df["day"] = df["date"].dt.day
X = df[["day"]]  # Simple feature: Day of the month
y = df["rainfall"]

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f} mm")

# Step 8: Visualize Rainfall Trend
plt.figure(figsize=(10, 5))
plt.scatter(df["day"], df["rainfall"], label="Actual Data", alpha=0.6)
plt.plot(X_test, y_pred, color='red', label="Predicted Trend")
plt.xlabel("Day of the Month")
plt.ylabel("Rainfall (mm)")
plt.title("Rainfall Prediction using Linear Regression")
plt.legend()
plt.show()
