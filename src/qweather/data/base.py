import pandas as pd

EXPECTED_COLUMNS = [
    "temp_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "wind_speed_10m",
    "shortwave_radiation",
]

def validate_layer1_contract(df: pd.DataFrame) -> pd.DataFrame:
    """Validates that a pandas DataFrame satisfies the Layer 1 data contract.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Layer 1 data contract violation: Index must be a DatetimeIndex.")

    if not df.index.is_monotonic_increasing:
        raise ValueError("Layer 1 data contract violation: Index timestamps must be monotonically increasing.")

    # Explicit spacing validation via timestamp differences (safely handles single-row case where len <= 1)
    if len(df) > 1:
        diffs = df.index.to_series().diff().dropna()
        if not (diffs == pd.Timedelta(hours=1)).all():
            raise ValueError(
                "Layer 1 data contract violation: Timestamps must have exactly 1-hour spacing between consecutive records."
            )

    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Layer 1 data contract violation: Missing expected columns: {missing_cols}"
        )

    if df[EXPECTED_COLUMNS].isna().any().any():
        raise ValueError(
            "Layer 1 data contract violation: Dataset contains NaN values."
        )

    if (df[EXPECTED_COLUMNS].isin([float("inf"), float("-inf")])).any().any():
        raise ValueError(
            "Layer 1 data contract violation: Dataset contains infinite values."
        )

    return df[EXPECTED_COLUMNS]