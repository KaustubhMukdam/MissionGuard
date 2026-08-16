# src/missionguard/utils/config.py
"""Configuration and paths for MissionGuard."""

from pathlib import Path
from typing import Dict, Any


# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Artifacts
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"


def get_data_dir() -> Path:
    """Get project data directory."""
    return DATA_DIR


def get_raw_data_path(dataset: str = "opssat-ad") -> Path:
    """Get raw data path for a dataset."""
    return RAW_DATA_DIR / dataset


def get_processed_data_path(dataset: str = "opssat-ad", filename: str = None) -> Path:
    """Get processed data path."""
    path = PROCESSED_DATA_DIR / dataset
    if filename:
        return path / filename
    return path


# OPSSAT-AD dataset configuration
OPSSAT_AD_CONFIG: Dict[str, Any] = {
    "name": "opssat-ad",
    "description": "ESA OPS-SAT telemetry anomaly benchmark",
    "source": "https://doi.org/10.5281/zenodo.12588359",
    "files": {
        "segments": "segments.csv",
        "dataset": "dataset.csv",
    },
    "segments_schema": {
        "columns": [
            "channel", "timestamp", "value", "label",
            "sampling", "anomaly", "segment", "train"
        ],
        "dtypes": {
            "channel": "object",
            "timestamp": "datetime64[ns, UTC]",
            "value": "float64",
            "label": "object",
            "sampling": "int64",
            "anomaly": "int64",
            "segment": "int64",
            "train": "int64",
        },
    },
    "dataset_schema": {
        "columns": [
            "segment", "anomaly", "train", "channel", "sampling",
            "duration", "len", "mean", "var", "std", "kurtosis", "skew",
            "n_peaks", "smooth10_n_peaks", "smooth20_n_peaks",
            "diff_peaks", "diff2_peaks", "diff_var", "diff2_var",
            "gaps_squared", "len_weighted", "var_div_duration", "var_div_len"
        ],
        "feature_columns": [
            "duration", "len", "mean", "var", "std", "kurtosis", "skew",
            "n_peaks", "smooth10_n_peaks", "smooth20_n_peaks",
            "diff_peaks", "diff2_peaks", "diff_var", "diff2_var",
            "gaps_squared", "len_weighted", "var_div_duration", "var_div_len"
        ],
        "target": "anomaly",
        "split_column": "train",
    },
    "statistics": {
        "n_segments": 2123,
        "n_anomaly_segments": 434,
        "anomaly_rate": 0.204,
        "n_channels": 9,
        "channels": [
            "CADC0872", "CADC0892", "CADC0874", "CADC0884",
            "CADC0873", "CADC0886", "CADC0888", "CADC0894", "CADC0890"
        ],
        "timestamp_range": {
            "start": "2022-01-04T20:00:50Z",
            "end": "2022-06-02T15:10:42Z",
        },
        "sampling_rates": [1, 5],
        "anomaly_types": ["anomaly", "a2", "a3", "a4"],
    },
    "limitations": {
        "cross_channel_aggregation": False,
        "channel_metadata": False,
        "time_aligned_channels": False,
        "temporal_split_provided": False,  # split is by segment, not time
    },
}

# ESA-ADB dataset configuration (for future use)
ESA_ADB_CONFIG: Dict[str, Any] = {
    "name": "esa-adb",
    "description": "ESA Anomaly Detection Benchmark - Mission 1",
    "source": "https://github.com/kplabs-pl/ESA-ADB",
    "zenodo": "https://doi.org/10.5281/zenodo.12528696",
    "subsets": {
        "3_months": {
            "train_rows": 262081,
            "test_rows": 7364161,
            "dimensions": 87,
            "contamination": 0.049,
        },
        "10_months": {
            "train_rows": 878401,
            "test_rows": 7364161,
            "dimensions": 87,
            "contamination": 0.022,
        },
        "21_months": {
            "train_rows": 1840321,
            "test_rows": 7364161,
            "dimensions": 87,
            "contamination": 0.027,
        },
        "42_months": {
            "train_rows": 3677761,
            "test_rows": 7364161,
            "dimensions": 87,
            "contamination": 0.018,
        },
        "84_months": {
            "train_rows": 7364161,
            "test_rows": 7364161,
            "dimensions": 87,
            "contamination": 0.019,
        },
    },
    "recommended_subset": "3_months",
    "limitations": {
        "full_download_size_gb": 3.7,
        "preprocessing_time_hours": "several on standard PC",
    },
}


# Model configuration
MODEL_CONFIG: Dict[str, Any] = {
    "random_state": 42,
    "test_size": 0.25,
    "cv_folds": 5,
    "scaler_type": "robust",  # "standard" or "robust"
    "baseline_threshold_percentile": 95,
}

# Evaluation configuration
EVAL_CONFIG: Dict[str, Any] = {
    "metrics": ["precision", "recall", "f1", "roc_auc", "pr_auc"],
    "operational_metrics": ["false_alarms_per_hour", "detection_delay_seconds"],
    "threshold_selection": "validation_f1",  # select threshold on validation, report on test
}