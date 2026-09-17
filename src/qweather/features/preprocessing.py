import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset


def split_chronological(
    df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Splits a DataFrame chronologically into train, validation, and test subsets

  based on configurable ratios, without random shuffling.
  """
  if not (0.0 < train_ratio < 1.0) or not (0.0 < val_ratio < 1.0):
    raise ValueError("Split ratios must be strictly between 0.0 and 1.0.")
  if train_ratio + val_ratio >= 1.0:
    raise ValueError("Sum of train_ratio and val_ratio must be less than 1.0.")

  n = len(df)
  train_end = int(n * train_ratio)
  val_end = int(n * (train_ratio + val_ratio))

  train_df = df.iloc[:train_end]
  val_df = df.iloc[train_end:val_end]
  test_df = df.iloc[val_end:]

  return train_df, val_df, test_df


def fit_scaler(train_df: pd.DataFrame) -> MinMaxScaler:
  """Initializes and fits MinMaxScaler(feature_range=(0, 1))

  EXCLUSIVELY on the training dataset to prevent data leakage.
  """
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaler.fit(train_df.values)
  return scaler


def transform_dataframe(df: pd.DataFrame, scaler: MinMaxScaler) -> np.ndarray:
  """Transforms a DataFrame using a pre-fitted scaler, returning a float32 numpy array."""
  return scaler.transform(df.values).astype(np.float32)


def create_split_windows(
    full_scaled_data: np.ndarray,
    start_idx: int,
    end_idx: int,
    lookback: int = 168,
    target_col_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
  """Extracts sliding windows X of shape (samples, lookback, features) and

  targets y of shape (samples,) for a specific target index range [start_idx, end_idx)
  within the continuous full scaled dataset.

  Enforces strict boundary and historical context validation.
  """
  if len(full_scaled_data) < lookback + 1:
    raise ValueError(
        f"Global dataset length ({len(full_scaled_data)}) is smaller than"
        f" lookback + 1 ({lookback + 1})."
    )
  if not (0 <= target_col_idx < full_scaled_data.shape[1]):
    raise ValueError(
        f"target_col_idx ({target_col_idx}) is out of bounds for "
        f"{full_scaled_data.shape[1]} features."
    )
  if start_idx < lookback:
    raise ValueError(
        f"start_idx ({start_idx}) cannot be less than lookback ({lookback})."
        " Targets require sufficient historical context preceding them."
    )
  if start_idx > end_idx:
    raise ValueError(
        f"start_idx ({start_idx}) cannot be greater than end_idx ({end_idx})."
    )
  if end_idx > len(full_scaled_data):
    raise ValueError(
        f"end_idx ({end_idx}) exceeds full_scaled_data length"
        f" ({len(full_scaled_data)})."
    )

  num_samples = end_idx - start_idx
  if num_samples <= 0:
    feature_dim = full_scaled_data.shape[1]
    return np.empty((0, lookback, feature_dim), dtype=np.float32), np.empty(
        (0,), dtype=np.float32
    )

  X_list = []
  y_list = []

  for t in range(start_idx, end_idx):
    X_win = full_scaled_data[t - lookback : t]
    y_val = full_scaled_data[t, target_col_idx]
    X_list.append(X_win)
    y_list.append(y_val)

  return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


class WeatherDataset(Dataset):
  """PyTorch Dataset wrapper casting X and y arrays into torch.float32 tensors."""

  def __init__(self, X: np.ndarray, y: np.ndarray):
    if len(X) != len(y):
      raise ValueError(
          f"Length mismatch between X ({len(X)}) and y ({len(y)})."
      )
    self.X = torch.tensor(X, dtype=torch.float32)
    self.y = torch.tensor(y, dtype=torch.float32)

  def __len__(self) -> int:
    return len(self.X)

  def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    return self.X[idx], self.y[idx]