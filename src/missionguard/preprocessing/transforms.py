# src/missionguard/preprocessing/transforms.py
"""Feature scaling and transformation utilities."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import List, Optional, Tuple
import joblib
from pathlib import Path


class StandardScalerWrapper:
    """Wrapper for StandardScaler with feature names and persistence."""
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.scaler = StandardScaler()
        self.fitted = False
    
    def fit(self, X: pd.DataFrame) -> "StandardScalerWrapper":
        """Fit scaler on training data only."""
        X_features = X[self.feature_names]
        self.scaler.fit(X_features)
        self.fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scaler."""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")
        X_features = X[self.feature_names]
        X_scaled = self.scaler.transform(X_features)
        return pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform scaled features."""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")
        X_orig = self.scaler.inverse_transform(X[self.feature_names])
        return pd.DataFrame(X_orig, columns=self.feature_names, index=X.index)
    
    def save(self, path: str) -> None:
        """Save fitted scaler to disk."""
        if not self.fitted:
            raise ValueError("Cannot save unfitted scaler")
        joblib.dump({
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "fitted": self.fitted,
        }, path)
    
    @classmethod
    def load(cls, path: str) -> "StandardScalerWrapper":
        """Load fitted scaler from disk."""
        data = joblib.load(path)
        wrapper = cls(data["feature_names"])
        wrapper.scaler = data["scaler"]
        wrapper.fitted = data["fitted"]
        return wrapper


class RobustScalerWrapper:
    """Wrapper for RobustScaler (more robust to outliers)."""
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.scaler = RobustScaler()
        self.fitted = False
    
    def fit(self, X: pd.DataFrame) -> "RobustScalerWrapper":
        X_features = X[self.feature_names]
        self.scaler.fit(X_features)
        self.fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")
        X_features = X[self.feature_names]
        X_scaled = self.scaler.transform(X_features)
        return pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
    
    def save(self, path: str) -> None:
        if not self.fitted:
            raise ValueError("Cannot save unfitted scaler")
        joblib.dump({
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "fitted": self.fitted,
        }, path)
    
    @classmethod
    def load(cls, path: str) -> "RobustScalerWrapper":
        data = joblib.load(path)
        wrapper = cls(data["feature_names"])
        wrapper.scaler = data["scaler"]
        wrapper.fitted = data["fitted"]
        return wrapper


def fit_scaler(
    train_df: pd.DataFrame,
    feature_names: List[str],
    scaler_type: str = "robust",
) -> StandardScalerWrapper | RobustScalerWrapper:
    """
    Fit a scaler on training data only (no leakage).
    
    Args:
        train_df: Training DataFrame
        feature_names: List of feature column names
        scaler_type: "standard" or "robust"
        
    Returns:
        Fitted scaler wrapper
    """
    if scaler_type == "standard":
        scaler = StandardScalerWrapper(feature_names)
    elif scaler_type == "robust":
        scaler = RobustScalerWrapper(feature_names)
    else:
        raise ValueError(f"Unknown scaler_type: {scaler_type}")
    
    scaler.fit(train_df)
    return scaler


def transform_features(
    df: pd.DataFrame,
    scaler: StandardScalerWrapper | RobustScalerWrapper,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Transform features using fitted scaler.
    
    Args:
        df: DataFrame to transform
        scaler: Fitted scaler wrapper
        feature_names: List of feature column names
        
    Returns:
        DataFrame with scaled features (other columns preserved)
    """
    result = df.copy()
    scaled = scaler.transform(df)
    result[feature_names] = scaled[feature_names]
    return result


def prepare_features_target(
    dataset_df: pd.DataFrame,
    feature_names: List[str],
    target_column: str = "anomaly",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract feature matrix and target vector from dataset.
    
    Args:
        dataset_df: Segment-level features DataFrame
        feature_names: List of feature column names
        target_column: Target column name
        
    Returns:
        Tuple of (X, y)
    """
    X = dataset_df[feature_names].copy()
    y = dataset_df[target_column].copy()
    return X, y


def get_feature_names(dataset_df: pd.DataFrame, exclude: List[str] = None) -> List[str]:
    """
    Get feature column names from dataset, excluding metadata columns.
    
    Args:
        dataset_df: Segment-level features DataFrame
        exclude: Additional columns to exclude
        
    Returns:
        List of feature column names
    """
    default_exclude = ["segment", "anomaly", "train", "channel", "sampling"]
    if exclude:
        default_exclude.extend(exclude)
    
    feature_names = [c for c in dataset_df.columns if c not in default_exclude]
    return feature_names