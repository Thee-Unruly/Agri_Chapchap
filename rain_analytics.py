import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.patheffects as path_effects

# Load the data
daily_dataframe = pd.read_csv("daily_weather_data.csv")

# Convert date column to datetime format
daily_dataframe["date"] = pd.to_datetime(daily_dataframe["date"])

# Reverse the data (most recent dates first)
daily_dataframe = daily_dataframe.sort_values("date", ascending=False)

# Streamlit app
st.subheader("🌦️ Weather Data Visualization")

# Dropdown to select the variable to visualize
variable_options = [col for col in daily_dataframe.columns[1:] if col != "weather_code"]
selected_variable = st.selectbox("📊 Select a variable to visualize", ["Select a variable"] + variable_options)

# Function to plot a smooth and visually interesting curve
def plot_variable(variable):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#f4f4f8")  # Light background

    # Get x and y values
    x = np.arange(len(daily_dataframe))  # Use index as x-axis for interpolation
    y = daily_dataframe[variable].values

    # Smooth curve using spline interpolation
    if len(x) > 3:
        x_smooth = np.linspace(x.min(), x.max(), 300)
        spline = make_interp_spline(x, y, k=3)
        y_smooth = spline(x_smooth)

        ax.scatter(daily_dataframe["date"], y, color="#ff6361", alpha=0.7, s=50, label="Raw Data")  # Stylish scatter
        ax.plot(pd.to_datetime(np.interp(x_smooth, x, daily_dataframe["date"].astype(int))),
                y_smooth, linestyle="-", linewidth=3, color="#003f5c", alpha=0.9, label="Smoothed Curve", 
                path_effects=[path_effects.withStroke(linewidth=5, foreground='white')])  # Shadow effect
    else:
        ax.plot(daily_dataframe["date"], y, marker="o", linestyle="-", color="#58508d", label="Raw Data")

    ax.set_title(f"📈 {variable} Over Time", fontsize=16, fontweight="bold", color="#333333")
    ax.set_xlabel("Date", fontsize=12, fontweight="bold", color="#555555")
    ax.set_ylabel(variable, fontsize=12, fontweight="bold", color="#555555")
    ax.legend(frameon=False, loc="best")

    # Remove gridlines for a cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#bbbbbb")
    ax.spines["bottom"].set_color("#bbbbbb")

    st.pyplot(fig)

# Show the plot **only if the user selects a variable**
if selected_variable != "Select a variable":
    plot_variable(selected_variable)

    # Display the reversed data after the visualization
    st.subheader(f"📋 Data for {selected_variable} (Most Recent First)")
    st.dataframe(daily_dataframe[["date", selected_variable]])
