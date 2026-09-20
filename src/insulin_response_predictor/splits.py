"""Time-aware train/test splits that never train on the future."""

from __future__ import annotations

import pandas as pd


def chronological_split(
    frame: pd.DataFrame,
    *,
    timestamp_column: str = "bolus_timestamp",
    train_fraction: float = 0.70,
    minimum_train_rows: int = 20,
    minimum_test_rows: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if timestamp_column not in frame.columns:
        raise KeyError(f"Missing timestamp column {timestamp_column!r}")
    ordered = frame.copy()
    ordered[timestamp_column] = pd.to_datetime(ordered[timestamp_column], utc=True)
    ordered = ordered.sort_values(timestamp_column, kind="stable").reset_index(drop=True)
    if len(ordered) < minimum_train_rows + minimum_test_rows:
        raise ValueError(
            f"Need at least {minimum_train_rows + minimum_test_rows} rows; found {len(ordered)}"
        )

    candidate = max(minimum_train_rows, int(len(ordered) * train_fraction))
    candidate = min(candidate, len(ordered) - minimum_test_rows)
    cutoff = ordered.iloc[candidate - 1][timestamp_column]
    train = ordered.loc[ordered[timestamp_column] <= cutoff].copy()
    test = ordered.loc[ordered[timestamp_column] > cutoff].copy()
    if len(test) < minimum_test_rows:
        raise ValueError("Timestamp ties leave too few test rows")
    if train[timestamp_column].max() >= test[timestamp_column].min():
        raise AssertionError("Chronological split overlaps")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def rolling_origin_splits(
    frame: pd.DataFrame,
    *,
    timestamp_column: str = "bolus_timestamp",
    initial_train_rows: int,
    test_rows: int,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    if initial_train_rows <= 0 or test_rows <= 0:
        raise ValueError("initial_train_rows and test_rows must be positive")
    ordered = frame.copy()
    ordered[timestamp_column] = pd.to_datetime(ordered[timestamp_column], utc=True)
    ordered = ordered.sort_values(timestamp_column, kind="stable").reset_index(drop=True)
    splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    train_end = initial_train_rows
    while train_end + test_rows <= len(ordered):
        train = ordered.iloc[:train_end].copy()
        test = ordered.iloc[train_end : train_end + test_rows].copy()
        if train[timestamp_column].max() >= test[timestamp_column].min():
            raise ValueError("Equal timestamps cross a rolling-origin boundary")
        splits.append((train.reset_index(drop=True), test.reset_index(drop=True)))
        train_end += test_rows
    return splits
