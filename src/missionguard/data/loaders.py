# src/missionguard/data/loaders.py
"""Data loaders for MissionGuard datasets."""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

from .schema import (
    SegmentsSchema,
    DatasetSchema,
    validate_segments_df,
    validate_dataset_df,
)


def load_segments(
    path: str,
    parse_timestamps: bool = True,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Load segments.csv (raw telemetry data).
    
    Args:
        path: Path to segments.csv file
        parse_timestamps: Whether to parse timestamp column to datetime
        validate: Whether to run schema validation
        
    Returns:
        DataFrame with raw telemetry data
        
    Raises:
        ValueError: If validation fails and validate=True
    """
    df = pd.read_csv(path)
    
    if parse_timestamps and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    
    if validate:
        result = validate_segments_df(df, SegmentsSchema())
        if not result["valid"]:
            raise ValueError(f"Schema validation failed: {result['errors']}")
        if result["warnings"]:
            print(f"Warnings: {result['warnings']}")
    
    return df


def load_dataset(
    path: str,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Load dataset.csv (segment-level features).
    
    Args:
        path: Path to dataset.csv file
        validate: Whether to run schema validation
        
    Returns:
        DataFrame with segment-level features
        
    Raises:
        ValueError: If validation fails and validate=True
    """
    df = pd.read_csv(path)
    
    if validate:
        result = validate_dataset_df(df, DatasetSchema())
        if not result["valid"]:
            raise ValueError(f"Schema validation failed: {result['errors']}")
        if result["warnings"]:
            print(f"Warnings: {result['warnings']}")
    
    return df


def get_train_test_split(
    segments_df: pd.DataFrame,
    dataset_df: Optional[pd.DataFrame] = None,
    split_column: str = "train",
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Split data into train/test using the provided split column.
    
    Args:
        segments_df: Raw telemetry DataFrame
        dataset_df: Optional segment features DataFrame
        split_column: Column name containing train/test flag (1=train, 0=test)
        
    Returns:
        Tuple of (train_segments, test_segments, train_dataset, test_dataset)
    """
    train_segments = segments_df[segments_df[split_column] == 1].copy()
    test_segments = segments_df[segments_df[split_column] == 0].copy()
    
    train_dataset = None
    test_dataset = None
    
    if dataset_df is not None:
        train_dataset = dataset_df[dataset_df[split_column] == 1].copy()
        test_dataset = dataset_df[dataset_df[split_column] == 0].copy()
    
    return train_segments, test_segments, train_dataset, test_dataset


def load_opssat_ad(
    data_dir: str,
    parse_timestamps: bool = True,
    validate: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load OPSSAT-AD dataset (both segments and dataset files).
    
    Args:
        data_dir: Directory containing segments.csv and dataset.csv
        parse_timestamps: Whether to parse timestamps
        validate: Whether to run schema validation
        
    Returns:
        Tuple of (segments_df, dataset_df)
    """
    data_path = Path(data_dir)
    
    segments_df = load_segments(
        data_path / "segments.csv",
        parse_timestamps=parse_timestamps,
        validate=validate,
    )
    
    dataset_df = load_dataset(
        data_path / "dataset.csv",
        validate=validate,
    )
    
    return segments_df, dataset_df


def get_temporal_train_test_split(
    segments_df: pd.DataFrame,
    test_ratio: float = 0.25,
    timestamp_column: str = "timestamp",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a temporal train/test split (no future data in train).
    
    Args:
        segments_df: Raw telemetry DataFrame with timestamps
        test_ratio: Fraction of data to use for test (most recent)
        timestamp_column: Name of timestamp column
        
    Returns:
        Tuple of (train_df, test_df)
    """
    if timestamp_column not in segments_df.columns:
        raise ValueError(f"Timestamp column '{timestamp_column}' not found")
    
    # Sort by timestamp
    df_sorted = segments_df.sort_values(timestamp_column).reset_index(drop=True)
    
    # Split point
    split_idx = int(len(df_sorted) * (1 - test_ratio))
    
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    
    return train_df, test_df