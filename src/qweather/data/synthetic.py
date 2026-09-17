import numpy as np
import pandas as pd
from qweather.data.base import validate_layer1_contract

def generate_synthetic_weather(
    start_date: str = "2023-01-01 00:00:00",
    num_hours: int = 720,  # Default: 30 days of hourly data
    seed: int = 42,
) -> pd.DataFrame:
    """Generates a deterministic, contract-compliant synthetic weather DataFrame
    for offline testing, development, and CI pipelines.
    """
    np.random.seed(seed)
    date_index = pd.date_range(
        start=start_date, periods=num_hours, freq="h"
    )

    t = np.arange(num_hours)

    # 1. Temperature: Seasonal baseline + diurnal cycle + Gaussian noise
    diurnal_temp = 5.0 * np.sin(2 * np.pi * (t - 14) / 24.0)
    seasonal_temp = 15.0 + 10.0 * np.sin(
        2 * np.pi * t / (365.25 * 24)
    )  # Annual macro trend
    noise_temp = np.random.normal(0, 1.0, size=num_hours)
    temp_2m = seasonal_temp + diurnal_temp + noise_temp

    # 2. Relative Humidity: Strongly inverse to temperature with noise
    relative_humidity_2m = np.clip(
        80.0 - 2.0 * diurnal_temp + np.random.normal(0, 5.0, size=num_hours),
        10.0,
        100.0
    )

    # 3. Pressure: Stable base with slow random walk
    pressure_noise = np.cumsum(np.random.normal(0, 0.2, size=num_hours))
    pressure_msl = 1013.25 + pressure_noise

    # 4. Wind Speed: Positive bounded distribution
    wind_speed_10m = np.abs(np.random.normal(5.0, 2.5, size=num_hours))

    # 5. Shortwave Radiation: Sine wave during daylight hours (6 AM to 6 PM), zero at night
    hour_of_date = date_index.hour.values
    solar_raw = 800.0 * np.sin(np.pi * (hour_of_date - 6) / 12.0)
    shortwave_radiation = np.where(
        (hour_of_date >= 6) & (hour_of_date <= 18), solar_raw, 0.0
    )
    cloud_mask = np.random.uniform(0.7, 1.0, size=num_hours)
    shortwave_radiation = np.where(
        shortwave_radiation > 0, shortwave_radiation * cloud_mask, 0.0
    )

    df = pd.DataFrame(
        {
            "temp_2m": temp_2m.astype(np.float32),
            "relative_humidity_2m": relative_humidity_2m.astype(np.float32),
            "pressure_msl": pressure_msl.astype(np.float32),
            "wind_speed_10m": wind_speed_10m.astype(np.float32),
            "shortwave_radiation": shortwave_radiation.astype(np.float32),
        },
        index=date_index,
    )

    return validate_layer1_contract(df)