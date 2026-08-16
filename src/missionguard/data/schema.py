# src/missionguard/data/schema.py
"""Schema definitions and validation for MissionGuard datasets."""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
import numpy as np


@dataclass
class SegmentsSchema:
    """Expected schema for segments.csv (raw telemetry)."""
    required_columns: List[str] = None
    dtypes: dict = None
    
    def __post_init__(self):
        if self.required_columns is None:
            self.required_columns = [
                "channel", "timestamp", "value", "label", 
                "sampling", "anomaly", "segment", "train"
            ]
        if self.dtypes is None:
            self.dtypes = {
                "channel": "object",
                "timestamp": "object",  # parsed separately
                "value": "float64",
                "label": "object",
                "sampling": "int64",
                "anomaly": "int64",
                "segment": "int64",
                "train": "int64",
            }


@dataclass
class DatasetSchema:
    """Expected schema for dataset.csv (segment-level features)."""
    required_columns: List[str] = None
    dtypes: dict = None
    feature_columns: List[str] = None
    
    def __post_init__(self):
        if self.required_columns is None:
            self.required_columns = [
                "segment", "anomaly", "train", "channel", "sampling",
                "duration", "len", "mean", "var", "std", "kurtosis", "skew",
                "n_peaks", "smooth10_n_peaks", "smooth20_n_peaks",
                "diff_peaks", "diff2_peaks", "diff_var", "diff2_var",
                "gaps_squared", "len_weighted", "var_div_duration", "var_div_len"
            ]
        if self.dtypes is None:
            self.dtypes = {
                "segment": "int64",
                "anomaly": "int64",
                "train": "int64",
                "channel": "object",
                "sampling": "int64",
                "duration": "int64",
                "len": "int64",
                "mean": "float64",
                "var": "float64",
                "std": "float64",
                "kurtosis": "float64",
                "skew": "float64",
                "n_peaks": "int64",
                "smooth10_n_peaks": "int64",
                "smooth20_n_peaks": "int64",
                "diff_peaks": "int64",
                "diff2_peaks": "int64",
                "diff_var": "float64",
                "diff2_var": "float64",
                "gaps_squared": "float64",
                "len_weighted": "float64",
                "var_div_duration": "float64",
                "var_div_len": "float64",
            }
        if self.feature_columns is None:
            self.feature_columns = [
                "duration", "len", "mean", "var", "std", "kurtosis", "skew",
                "n_peaks", "smooth10_n_peaks", "smooth20_n_peaks",
                "diff_peaks", "diff2_peaks", "diff_var", "diff2_var",
                "gaps_squared", "len_weighted", "var_div_duration", "var_div_len"
            ]


def validate_segments_df(df: pd.DataFrame, schema: Optional[SegmentsSchema] = None) -> dict:
    """
    Validate segments DataFrame against expected schema.
    
    Returns:
        dict with validation results: {'valid': bool, 'errors': list, 'warnings': list}
    """
    if schema is None:
        schema = SegmentsSchema()
    
    errors = []
    warnings = []
    
    # Check required columns
    missing_cols = set(schema.required_columns) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    
    extra_cols = set(df.columns) - set(schema.required_columns)
    if extra_cols:
        warnings.append(f"Extra columns (ignored): {extra_cols}")
    
    # Check dtypes for present columns
    for col, expected_dtype in schema.dtypes.items():
        if col in df.columns:
            actual_dtype = str(df[col].dtype)
            if expected_dtype not in actual_dtype and not (expected_dtype == "object" and actual_dtype == "string"):
                warnings.append(f"Column '{col}': expected {expected_dtype}, got {actual_dtype}")
    
    # Check for missing values (only for columns that exist)
    present_required = [c for c in schema.required_columns if c in df.columns]
    if present_required:
        null_counts = df[present_required].isnull().sum()
        if null_counts.any():
            errors.append(f"Missing values found: {null_counts[null_counts > 0].to_dict()}")
    
    # Check anomaly values are binary
    if "anomaly" in df.columns:
        invalid_anomaly = df[~df["anomaly"].isin([0, 1])]
        if len(invalid_anomaly) > 0:
            errors.append(f"Invalid anomaly values (must be 0 or 1): {len(invalid_anomaly)} rows")
    
    # Check train values are binary
    if "train" in df.columns:
        invalid_train = df[~df["train"].isin([0, 1])]
        if len(invalid_train) > 0:
            errors.append(f"Invalid train values (must be 0 or 1): {len(invalid_train)} rows")
    
    # Check segment IDs are positive
    if "segment" in df.columns:
        if (df["segment"] <= 0).any():
            errors.append("Segment IDs must be positive integers")
    
    # Check sampling rates
    if "sampling" in df.columns:
        unique_sampling = df["sampling"].unique()
        if not set(unique_sampling).issubset({1, 5}):
            warnings.append(f"Unexpected sampling rates: {unique_sampling}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }


def validate_dataset_df(df: pd.DataFrame, schema: Optional[DatasetSchema] = None) -> dict:
    """
    Validate dataset DataFrame against expected schema.
    
    Returns:
        dict with validation results: {'valid': bool, 'errors': list, 'warnings': list}
    """
    if schema is None:
        schema = DatasetSchema()
    
    errors = []
    warnings = []
    
    # Check required columns
    missing_cols = set(schema.required_columns) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    
    # Check dtypes
    for col, expected_dtype in schema.dtypes.items():
        if col in df.columns:
            actual_dtype = str(df[col].dtype)
            if expected_dtype not in actual_dtype and not (expected_dtype == "object" and actual_dtype == "string"):
                warnings.append(f"Column '{col}': expected {expected_dtype}, got {actual_dtype}")
    
    # Check for missing values (only for columns that exist)
    present_required = [c for c in schema.required_columns if c in df.columns]
    if present_required:
        null_counts = df[present_required].isnull().sum()
        if null_counts.any():
            errors.append(f"Missing values found: {null_counts[null_counts > 0].to_dict()}")
    
    # Check anomaly values are binary
    if "anomaly" in df.columns:
        invalid_anomaly = df[~df["anomaly"].isin([0, 1])]
        if len(invalid_anomaly) > 0:
            errors.append(f"Invalid anomaly values (must be 0 or 1): {len(invalid_anomaly)} rows")
    
    # Check train values are binary
    if "train" in df.columns:
        invalid_train = df[~df["train"].isin([0, 1])]
        if len(invalid_train) > 0:
            errors.append(f"Invalid train values (must be 0 or 1): {len(invalid_train)} rows")
    
    # Check segment IDs are positive and unique
    if "segment" in df.columns:
        if (df["segment"] <= 0).any():
            errors.append("Segment IDs must be positive integers")
        if df["segment"].duplicated().any():
            errors.append("Duplicate segment IDs found")
    
    # Check feature columns for inf/nan
    for col in schema.feature_columns:
        if col in df.columns:
            if np.isinf(df[col]).any():
                warnings.append(f"Feature '{col}' contains infinite values")
            if df[col].isnull().any():
                warnings.append(f"Feature '{col}' contains NaN values")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }