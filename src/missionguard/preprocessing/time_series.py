# src/missionguard/preprocessing/time_series.py
"""Time series preprocessing utilities for raw telemetry."""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict
from pathlib import Path


def sort_by_segment_time(
    segments_df: pd.DataFrame,
    segment_column: str = "segment",
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """
    Sort raw telemetry by segment then timestamp.
    
    Essential because OPSSAT-AD segments are concatenated, not time-ordered globally.
    
    Args:
        segments_df: Raw telemetry DataFrame
        segment_column: Segment ID column
        timestamp_column: Timestamp column
        
    Returns:
        Sorted DataFrame
    """
    return segments_df.sort_values([segment_column, timestamp_column]).reset_index(drop=True)


def extract_segment_windows(
    segments_df: pd.DataFrame,
    segment_id: int,
    segment_column: str = "segment",
    timestamp_column: str = "timestamp",
    value_column: str = "value",
) -> pd.DataFrame:
    """
    Extract a single segment's time series window.
    
    Args:
        segments_df: Raw telemetry DataFrame
        segment_id: Segment ID to extract
        segment_column: Segment ID column
        timestamp_column: Timestamp column
        value_column: Value column
        
    Returns:
        DataFrame with single segment's data, sorted by time
    """
    mask = segments_df[segment_column] == segment_id
    segment_data = segments_df[mask].copy()
    segment_data = segment_data.sort_values(timestamp_column).reset_index(drop=True)
    return segment_data


def compute_rolling_features(
    series: pd.Series,
    windows: List[int] = [5, 10, 30, 60],
    features: List[str] = ["mean", "std", "min", "max", "skew", "kurt"],
) -> pd.DataFrame:
    """
    Compute rolling window features for a time series.
    
    Args:
        series: Time series values (index should be timestamps or integers)
        windows: List of window sizes
        features: List of features to compute
        
    Returns:
        DataFrame with rolling features (same index as input)
    """
    result = pd.DataFrame(index=series.index)
    
    for window in windows:
        rolling = series.rolling(window=window, min_periods=1)
        
        if "mean" in features:
            result[f"roll_{window}_mean"] = rolling.mean()
        if "std" in features:
            result[f"roll_{window}_std"] = rolling.std()
        if "min" in features:
            result[f"roll_{window}_min"] = rolling.min()
        if "max" in features:
            result[f"roll_{window}_max"] = rolling.max()
        if "skew" in features:
            result[f"roll_{window}_skew"] = rolling.skew()
        if "kurt" in features:
            result[f"roll_{window}_kurt"] = rolling.kurt()
    
    return result


def compute_differencing_features(
    series: pd.Series,
    lags: List[int] = [1, 2, 5, 10],
) -> pd.DataFrame:
    """
    Compute differencing features (first and second order differences).
    
    Args:
        series: Time series values
        lags: List of lag values for differencing
        
    Returns:
        DataFrame with difference features
    """
    result = pd.DataFrame(index=series.index)
    
    for lag in lags:
        result[f"diff_{lag}"] = series.diff(lag)
        result[f"diff2_{lag}"] = series.diff(lag).diff(lag)
    
    return result


def resample_to_regular(
    segments_df: pd.DataFrame,
    segment_column: str = "segment",
    timestamp_column: str = "timestamp",
    value_column: str = "value",
    freq: str = "1S",
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Resample each segment to regular frequency.
    
    Useful for channels with irregular sampling or gaps.
    
    Args:
        segments_df: Raw telemetry DataFrame
        segment_column: Segment ID column
        timestamp_column: Timestamp column
        value_column: Value column
        freq: Resampling frequency (pandas offset string)
        agg: Aggregation method for resampling
        
    Returns:
        DataFrame with regularly sampled segments
    """
    # Set timestamp as index for resampling
    df = segments_df.copy()
    df = df.set_index(timestamp_column)
    
    result_segments = []
    
    for seg_id, seg_data in df.groupby(segment_column):
        # Resample this segment
        resampled = seg_data[value_column].resample(freq).agg(agg).to_frame()
        resampled[segment_column] = seg_id
        
        # Forward fill other columns
        for col in df.columns:
            if col not in [value_column, segment_column]:
                resampled[col] = seg_data[col].resample(freq).ffill()
        
        result_segments.append(resampled)
    
    result = pd.concat(result_segments).reset_index()
    return result


def detect_gaps(
    segments_df: pd.DataFrame,
    segment_column: str = "segment",
    timestamp_column: str = "timestamp",
    max_gap_seconds: float = 60.0,
) -> pd.DataFrame:
    """
    Detect time gaps within segments.
    
    Args:
        segments_df: Raw telemetry DataFrame
        segment_column: Segment ID column
        timestamp_column: Timestamp column
        max_gap_seconds: Maximum expected gap (seconds)
        
    Returns:
        DataFrame with gap information per segment
    """
    gaps_info = []
    
    for seg_id, seg_data in segments_df.groupby(segment_column):
        seg_data = seg_data.sort_values(timestamp_column)
        time_diffs = seg_data[timestamp_column].diff().dt.total_seconds()
        
        large_gaps = time_diffs[time_diffs > max_gap_seconds]
        
        gaps_info.append({
            "segment": seg_id,
            "num_gaps": len(large_gaps),
            "max_gap_seconds": time_diffs.max() if len(time_diffs) > 1 else 0,
            "mean_gap_seconds": time_diffs.mean() if len(time_diffs) > 1 else 0,
        })
    
    return pd.DataFrame(gaps_info)


def align_channels_temporally(
    segments_df: pd.DataFrame,
    segment_column: str = "segment",
    channel_column: str = "channel",
    timestamp_column: str = "timestamp",
    value_column: str = "value",
    freq: str = "1S",
) -> pd.DataFrame:
    """
    Attempt to align multiple channels to common time grid.
    
    NOTE: Only works if channels have overlapping time ranges.
    For OPSSAT-AD, channels are NOT time-aligned (different segments = different channels/times).
    This function is provided for ESA-ADB where channels may be aligned.
    
    Args:
        segments_df: Raw telemetry DataFrame
        segment_column: Segment ID column
        channel_column: Channel ID column
        timestamp_column: Timestamp column
        value_column: Value column
        freq: Resampling frequency
        
    Returns:
        Wide-format DataFrame with channels as columns, common time index
    """
    # Check if channels have overlapping time ranges
    time_ranges = segments_df.groupby(channel_column)[timestamp_column].agg(["min", "max"])
    
    global_start = time_ranges["min"].max()
    global_end = time_ranges["max"].min()
    
    if global_start >= global_end:
        raise ValueError(
            f"Channels do not have overlapping time ranges. "
            f"Overlap: {global_start} to {global_end}"
        )
    
    # Create common time grid
    common_time = pd.date_range(start=global_start, end=global_end, freq=freq, tz="UTC")
    
    # Resample each channel to common grid
    aligned_data = {"timestamp": common_time}
    
    for channel in segments_df[channel_column].unique():
        ch_data = segments_df[segments_df[channel_column] == channel].copy()
        ch_data = ch_data.set_index(timestamp_column)
        ch_resampled = ch_data[value_column].resample(freq).mean()
        ch_aligned = ch_resampled.reindex(common_time).interpolate(method="time")
        aligned_data[channel] = ch_aligned.values
    
    return pd.DataFrame(aligned_data)