import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": -1.2833,
	"longitude": 36.8167,
	"start_date": "2025-01-01",
	"end_date": "2025-03-22",
	"daily": ["weather_code", "temperature_2m_max", "daylight_duration", "apparent_temperature_min", "apparent_temperature_max", "wind_direction_10m_dominant", "precipitation_hours", "temperature_2m_mean", "sunrise", "precipitation_sum", "et0_fao_evapotranspiration", "sunset", "rain_sum", "wind_gusts_10m_max", "snowfall_sum", "apparent_temperature_mean", "temperature_2m_min", "sunshine_duration", "shortwave_radiation_sum", "wind_speed_10m_max"],
	"hourly": "temperature_2m",
	"timezone": "auto"
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation {response.Elevation()} m asl")
print(f"Timezone {response.Timezone()}{response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

# Process daily data. The order of variables needs to be the same as requested.
daily = response.Daily()
daily_weather_code = daily.Variables(0).ValuesAsNumpy()
daily_temperature_2m_max = daily.Variables(1).ValuesAsNumpy()
daily_daylight_duration = daily.Variables(2).ValuesAsNumpy()
daily_apparent_temperature_min = daily.Variables(3).ValuesAsNumpy()
daily_apparent_temperature_max = daily.Variables(4).ValuesAsNumpy()
daily_wind_direction_10m_dominant = daily.Variables(5).ValuesAsNumpy()
daily_precipitation_hours = daily.Variables(6).ValuesAsNumpy()
daily_temperature_2m_mean = daily.Variables(7).ValuesAsNumpy()
daily_sunrise = daily.Variables(8).ValuesAsNumpy()
daily_precipitation_sum = daily.Variables(9).ValuesAsNumpy()
daily_et0_fao_evapotranspiration = daily.Variables(10).ValuesAsNumpy()
daily_sunset = daily.Variables(11).ValuesAsNumpy()
daily_rain_sum = daily.Variables(12).ValuesAsNumpy()
daily_wind_gusts_10m_max = daily.Variables(13).ValuesAsNumpy()
daily_snowfall_sum = daily.Variables(14).ValuesAsNumpy()
daily_apparent_temperature_mean = daily.Variables(15).ValuesAsNumpy()
daily_temperature_2m_min = daily.Variables(16).ValuesAsNumpy()
daily_sunshine_duration = daily.Variables(17).ValuesAsNumpy()
daily_shortwave_radiation_sum = daily.Variables(18).ValuesAsNumpy()
daily_wind_speed_10m_max = daily.Variables(19).ValuesAsNumpy()

daily_data = {"date": pd.date_range(
	start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
	end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = daily.Interval()),
	inclusive = "left"
)}

daily_data["weather_code"] = daily_weather_code
daily_data["temperature_2m_max"] = daily_temperature_2m_max
daily_data["daylight_duration"] = daily_daylight_duration
daily_data["apparent_temperature_min"] = daily_apparent_temperature_min
daily_data["apparent_temperature_max"] = daily_apparent_temperature_max
daily_data["wind_direction_10m_dominant"] = daily_wind_direction_10m_dominant
daily_data["precipitation_hours"] = daily_precipitation_hours
daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
daily_data["sunrise"] = daily_sunrise
daily_data["precipitation_sum"] = daily_precipitation_sum
daily_data["et0_fao_evapotranspiration"] = daily_et0_fao_evapotranspiration
daily_data["sunset"] = daily_sunset
daily_data["rain_sum"] = daily_rain_sum
daily_data["wind_gusts_10m_max"] = daily_wind_gusts_10m_max
daily_data["snowfall_sum"] = daily_snowfall_sum
daily_data["apparent_temperature_mean"] = daily_apparent_temperature_mean
daily_data["temperature_2m_min"] = daily_temperature_2m_min
daily_data["sunshine_duration"] = daily_sunshine_duration
daily_data["shortwave_radiation_sum"] = daily_shortwave_radiation_sum
daily_data["wind_speed_10m_max"] = daily_wind_speed_10m_max

daily_dataframe = pd.DataFrame(data = daily_data)
print(daily_dataframe)

# Save the daily weather data to a CSV file
daily_dataframe.to_csv("daily_weather_data.csv", index=False)
print("Daily weather data saved to 'daily_weather_data.csv'")
