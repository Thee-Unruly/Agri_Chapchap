import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
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

# Step 8: Streamlit App
st.title("🌦️ Rainfall Prediction Dashboard")
st.write("This application provides insights into rainfall trends and predictions.")

# Show Rainfall Data Table
st.subheader("📋 Rainfall Data (Past 60 Days)")
st.dataframe(df)

# Show Model Accuracy
st.subheader("📊 Model Performance")
st.write(f"Mean Absolute Error: {mae:.2f} mm")

# Show Visuals with Explanation
st.subheader("📈 Rainfall Trend Analysis")
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(df["day"], df["rainfall"], label="Actual Data", alpha=0.6)
ax.plot(X_test, y_pred, color='red', label="Predicted Trend")
ax.set_xlabel("Day of the Month")
ax.set_ylabel("Rainfall (mm)")
ax.set_title("Rainfall Prediction using Linear Regression")
ax.legend()
st.pyplot(fig)

# Step 9: Recommendations based on Rainfall Prediction
st.subheader("🌱 Recommendations for Farmers")
if mae > 5:
    st.warning("⚠️ Prediction accuracy is low. Consider using additional features like humidity and temperature.")
if df["rainfall"].mean() > 10:
    st.success("🌧️ Expect heavy rainfall. Farmers should prepare for irrigation drainage and soil erosion control.")
elif df["rainfall"].mean() < 2:
    st.info("☀️ Dry conditions expected. Consider irrigation or drought-resistant crops.")
else:
    st.info("🌤️ Moderate rainfall expected. Plan farming activities accordingly.")
