import pandas as pd

# Load the data
daily_dataframe = pd.read_csv("daily_weather_data.csv")

# Convert date column to datetime format
daily_dataframe["date"] = pd.to_datetime(daily_dataframe["date"])

# Drop non-numeric columns
numeric_df = daily_dataframe.select_dtypes(include=["number"])

# Ensure "rain_sum" exists in the dataset
if "rain_sum" in numeric_df.columns:
    # Compute correlations
    correlations = numeric_df.corr()["rain_sum"].drop("rain_sum").sort_values(ascending=False)

    # Select top correlated features (positive or negative)
    top_features = correlations[abs(correlations) > 0.6].index.tolist()

    # Create new dataset with selected features
    ml_dataset = numeric_df[["rain_sum"] + top_features]

    # Save for ML modeling
    ml_dataset.to_csv("ml_ready_data.csv", index=False)

    print(f"✅ Selected Features for ML: {top_features}")
else:
    print("⚠️ 'rain_sum' column not found in the dataset. Please check your data.")
