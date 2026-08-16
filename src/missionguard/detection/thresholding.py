# src/missionguard/detection/thresholding.py
"""Threshold selection and evaluation for anomaly detection."""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)


@dataclass
class ThresholdConfig:
    """Configuration for threshold selection."""
    method: str = "percentile"  # "percentile", "f1_optimal", "precision_at_recall", "fixed"
    value: float = 95.0  # Percentile or target recall/precision
    metric: str = "f1"  # For f1_optimal
    fixed_threshold: Optional[float] = None  # For fixed method


def select_threshold(
    scores: np.ndarray,
    labels: Optional[np.ndarray] = None,
    config: Optional[ThresholdConfig] = None,
) -> float:
    """
    Select anomaly threshold based on configuration.
    
    Args:
        scores: Anomaly scores (validation or test)
        labels: True labels (required for supervised methods)
        config: ThresholdConfig object
        
    Returns:
        Selected threshold
    """
    if config is None:
        config = ThresholdConfig()
    
    if config.method == "percentile":
        return float(np.percentile(scores, config.value))
    
    elif config.method == "fixed":
        if config.fixed_threshold is None:
            raise ValueError("fixed_threshold required for 'fixed' method")
        return config.fixed_threshold
    
    elif config.method == "f1_optimal":
        if labels is None:
            raise ValueError("Labels required for f1_optimal threshold selection")
        return _select_threshold_f1_optimal(scores, labels)
    
    elif config.method == "precision_at_recall":
        if labels is None:
            raise ValueError("Labels required for precision_at_recall")
        return _select_threshold_precision_at_recall(scores, labels, config.value)
    
    elif config.method == "recall_at_precision":
        if labels is None:
            raise ValueError("Labels required for recall_at_precision")
        return _select_threshold_recall_at_precision(scores, labels, config.value)
    
    else:
        raise ValueError(f"Unknown threshold method: {config.method}")


def _select_threshold_f1_optimal(
    scores: np.ndarray, 
    labels: np.ndarray,
) -> float:
    """Select threshold that maximizes F1 score."""
    # Use precision_recall_curve to get thresholds
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    
    # F1 for each threshold (ignore last point where recall=1, precision=undefined)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx])


def _select_threshold_precision_at_recall(
    scores: np.ndarray, 
    labels: np.ndarray,
    target_recall: float,
) -> float:
    """Select highest threshold achieving at least target recall."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    
    # Find thresholds where recall >= target
    valid = recall[:-1] >= target_recall
    if not valid.any():
        # Fallback: return threshold at max recall
        return float(thresholds[np.argmax(recall[:-1])])
    
    # Among valid, choose highest threshold (max precision)
    valid_thresholds = thresholds[:-1][valid]
    return float(valid_thresholds.max())


def _select_threshold_recall_at_precision(
    scores: np.ndarray, 
    labels: np.ndarray,
    target_precision: float,
) -> float:
    """Select highest threshold achieving at least target precision."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    
    valid = precision[:-1] >= target_precision
    if not valid.any():
        return float(thresholds[np.argmax(precision[:-1])])
    
    valid_thresholds = thresholds[:-1][valid]
    return float(valid_thresholds.max())


def evaluate_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """
    Evaluate a threshold on labeled data.
    
    Returns:
        Dict with precision, recall, f1, accuracy, specificity, etc.
    """
    preds = (scores >= threshold).astype(int)
    
    tp = np.sum((preds == 1) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # ROC-AUC and PR-AUC (threshold-independent)
    roc_auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0.5
    pr_auc = average_precision_score(labels, scores) if len(np.unique(labels)) > 1 else 0.0
    
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "specificity": float(specificity),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    }


def evaluate_thresholds_sweep(
    scores: np.ndarray,
    labels: np.ndarray,
    n_thresholds: int = 100,
) -> pd.DataFrame:
    """
    Evaluate multiple thresholds and return DataFrame.
    
    Useful for plotting precision-recall curves with threshold annotations.
    """
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    results = []
    
    for thresh in thresholds:
        eval_result = evaluate_threshold(scores, labels, thresh)
        results.append(eval_result)
    
    return pd.DataFrame(results)


def find_optimal_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, Dict[str, float]]:
    """
    Find optimal threshold for a given metric.
    
    Returns:
        Tuple of (optimal_threshold, evaluation_dict)
    """
    if metric == "f1":
        thresh = _select_threshold_f1_optimal(scores, labels)
    elif metric == "precision":
        # Not directly optimizable - use precision_at_recall with high recall target
        thresh = _select_threshold_precision_at_recall(scores, labels, 0.5)
    elif metric == "recall":
        thresh = _select_threshold_recall_at_precision(scores, labels, 0.5)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    eval_result = evaluate_threshold(scores, labels, thresh)
    return thresh, eval_result


def get_false_alarm_rate(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    time_per_sample: float = 1.0,
) -> Dict[str, float]:
    """
    Compute false alarm metrics.
    
    Args:
        scores: Anomaly scores
        labels: True labels
        threshold: Anomaly threshold
        time_per_sample: Time per sample in seconds
        
    Returns:
        Dict with false alarm metrics
    """
    preds = (scores >= threshold).astype(int)
    
    # False positives per hour
    fp = np.sum((preds == 1) & (labels == 0))
    total_time_hours = len(scores) * time_per_sample / 3600.0
    false_alarms_per_hour = fp / total_time_hours if total_time_hours > 0 else 0.0
    
    # False alarm rate (FP / (FP + TN))
    tn = np.sum((preds == 0) & (labels == 0))
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # Mean time between false alarms (hours)
    mtbfa = total_time_hours / fp if fp > 0 else float("inf")
    
    return {
        "false_positives": int(fp),
        "false_alarms_per_hour": float(false_alarms_per_hour),
        "false_alarm_rate": float(far),
        "mtbfa_hours": float(mtbfa),
    }


def get_detection_delay(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    time_per_sample: float = 1.0,
) -> Dict[str, float]:
    """
    Compute detection delay for true positive events.
    
    Args:
        scores: Anomaly scores
        labels: True labels (1=anomaly, 0=normal)
        threshold: Anomaly threshold
        time_per_sample: Time per sample in seconds
        
    Returns:
        Dict with delay statistics
    """
    preds = (scores >= threshold).astype(int)
    
    # Find anomaly events in labels
    delays = []
    in_anomaly = False
    anomaly_start = 0
    
    for i, (label, pred) in enumerate(zip(labels, preds)):
        if label == 1 and not in_anomaly:
            in_anomaly = True
            anomaly_start = i
        elif label == 0 and in_anomaly:
            in_anomaly = False
            anomaly_end = i - 1
            
            # Check if detected within this anomaly window
            window_preds = preds[anomaly_start:anomaly_end+1]
            if window_preds.sum() > 0:
                # Find first detection
                first_detect = np.where(window_preds == 1)[0][0]
                delay_samples = first_detect
                delay_seconds = delay_samples * time_per_sample
                delays.append(delay_seconds)
    
    # Handle anomaly at end
    if in_anomaly:
        anomaly_end = len(labels) - 1
        window_preds = preds[anomaly_start:anomaly_end+1]
        if window_preds.sum() > 0:
            first_detect = np.where(window_preds == 1)[0][0]
            delay_seconds = first_detect * time_per_sample
            delays.append(delay_seconds)
    
    if not delays:
        return {
            "detected_events": 0,
            "mean_delay_seconds": 0.0,
            "median_delay_seconds": 0.0,
            "max_delay_seconds": 0.0,
        }
    
    return {
        "detected_events": len(delays),
        "mean_delay_seconds": float(np.mean(delays)),
        "median_delay_seconds": float(np.median(delays)),
        "max_delay_seconds": float(np.max(delays)),
        "min_delay_seconds": float(np.min(delays)),
    }