import numpy as np
import pandas as pd
import pytest
import torch

from qweather.data import generate_synthetic_weather
from qweather.features.preprocessing import (
    WeatherDataset,
    create_split_windows,
    fit_scaler,
    split_chronological,
    transform_dataframe,
)


def test_arbitrary_dataset_lengths_and_splits():
  """Tests chronological splitting across non-standard dataset sizes."""
  df = generate_synthetic_weather(num_hours=500, seed=42)
  train, val, test = split_chronological(df, train_ratio=0.7, val_ratio=0.15)
  assert len(train) == 350
  assert len(val) == 75
  assert len(test) == 75
  assert train.index[-1] < val.index[0]
  assert val.index[-1] < test.index[0]


def test_scaler_fitting_leakage_prevention():
  """Tests that MinMaxScaler statistics are derived ONLY from training data."""
  df = generate_synthetic_weather(num_hours=1000, seed=42)
  train, val, test = split_chronological(df)
  scaler = fit_scaler(train)

  for i, col in enumerate(train.columns):
    assert scaler.data_min_[i] == train[col].min()
    assert scaler.data_max_[i] == train[col].max()


def test_window_alignment_and_shapes():
  """Tests exact 168 -> 1 window alignment and shape structures."""
  df = generate_synthetic_weather(num_hours=400, seed=42)
  train, val, test = split_chronological(df)
  scaler = fit_scaler(train)

  train_scaled = transform_dataframe(train, scaler)
  val_scaled = transform_dataframe(val, scaler)
  test_scaled = transform_dataframe(test, scaler)
  full_scaled = np.concatenate([train_scaled, val_scaled, test_scaled], axis=0)

  # Train valid targets start at index 168
  X, y = create_split_windows(full_scaled, start_idx=168, end_idx=len(train), lookback=168)
  assert X.shape == (len(train) - 168, 168, 5)
  assert y.shape == (len(train) - 168,)

  # Verify exact 1-step-ahead target mapping (column 0 is temp_2m)
  assert np.isclose(y[0], full_scaled[168, 0])
  assert np.isclose(y[1], full_scaled[169, 0])


def test_split_boundaries_and_context_crossing():
  """Tests that validation/test targets stay inside their split while crossing context boundaries."""
  df = generate_synthetic_weather(num_hours=500, seed=42)
  train, val, test = split_chronological(df)
  scaler = fit_scaler(train)

  train_scaled = transform_dataframe(train, scaler)
  val_scaled = transform_dataframe(val, scaler)
  test_scaled = transform_dataframe(test, scaler)
  full_scaled = np.concatenate([train_scaled, val_scaled, test_scaled], axis=0)

  val_start = len(train)
  val_end = len(train) + len(val)

  # Validation window extraction starting at val_start
  X_val, y_val = create_split_windows(full_scaled, start_idx=val_start, end_idx=val_end, lookback=168)

  # The first validation target y_val[0] corresponds to index val_start (first hour of validation)
  assert np.isclose(y_val[0], full_scaled[val_start, 0])

  # Its input window X_val[0] should span from val_start - 168 to val_start - 1
  # This crosses the train/val boundary, utilizing the final 168 hours of training data!
  expected_history = full_scaled[val_start - 168 : val_start]
  np.testing.assert_allclose(X_val[0], expected_history)


def test_pytorch_dataset_tensors():
  """Tests PyTorch Dataset dtype and tensor dimensions."""
  df = generate_synthetic_weather(num_hours=300, seed=42)
  train, val, test = split_chronological(df)
  scaler = fit_scaler(train)
  full_scaled = np.concatenate([transform_dataframe(t, scaler) for t in [train, val, test]], axis=0)

  X, y = create_split_windows(full_scaled, start_idx=168, end_idx=250, lookback=168)
  dataset = WeatherDataset(X, y)
  sample_x, sample_y = dataset[0]

  assert isinstance(sample_x, torch.Tensor)
  assert isinstance(sample_y, torch.Tensor)
  assert sample_x.dtype == torch.float32
  assert sample_y.dtype == torch.float32
  assert sample_x.shape == (168, 5)
  assert sample_y.shape == ()


def test_edge_case_insufficient_global_history():
  """Tests that datasets shorter than lookback + 1 raise an explicit ValueError."""
  df = generate_synthetic_weather(num_hours=100, seed=42)
  train, val, test = split_chronological(df)
  scaler = fit_scaler(train)
  full_scaled = np.concatenate([transform_dataframe(t, scaler) for t in [train, val, test]], axis=0)

  with pytest.raises(
      ValueError,
      match=r"Global dataset length .* is smaller than lookback \+ 1"
  ):
    create_split_windows(full_scaled, start_idx=50, end_idx=80, lookback=168)


def test_edge_case_insufficient_split_history():
  """Tests that a start_idx < lookback raises an explicit ValueError."""
  df = generate_synthetic_weather(num_hours=300, seed=42)
  train, val, test = split_chronological(df)
  scaler = fit_scaler(train)
  full_scaled = np.concatenate([transform_dataframe(t, scaler) for t in [train, val, test]], axis=0)

  with pytest.raises(ValueError, match="cannot be less than lookback"):
    create_split_windows(full_scaled, start_idx=50, end_idx=200, lookback=168)


def test_invalid_split_ranges():
  """Tests invalid start/end index parameters."""
  df = generate_synthetic_weather(num_hours=300, seed=42)
  train, val, test = split_chronological(df)
  scaler = fit_scaler(train)
  full_scaled = np.concatenate([transform_dataframe(t, scaler) for t in [train, val, test]], axis=0)

  with pytest.raises(ValueError, match="cannot be greater than end_idx"):
    create_split_windows(full_scaled, start_idx=200, end_idx=180, lookback=168)