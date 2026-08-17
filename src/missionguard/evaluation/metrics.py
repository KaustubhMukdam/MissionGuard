# src/missionguard/evaluation/metrics.py
"""Evaluation metrics for anomaly detection."""

from dataclasses import dataclass
from typing import Optional, Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    confusion_matrix,
)


@dataclass
class MetricsResult:
    """Container for evaluation metrics."""
    # Point-level metrics
    precision: float
    recall: float
    f1: float
    accuracy: float
    specificity: float
    roc_auc: float
    pr_auc: float
    
    # Confusion matrix
    tp: int
    tn: int
    fp: int
    fn: int
    
    # Operational metrics
    false_alarms_per_hour: Optional[float] = None
    false_alarm_rate: Optional[float] = None
    mtbfa_hours: Optional[float] = None
    mean_detection_delay_seconds: Optional[float] = None
    median_detection_delay_seconds: Optional[float] = None
    max_detection_delay_seconds: Optional[float] = None
    detected_events: Optional[int] = None
    
    # Threshold info
    threshold: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "specificity": self.specificity,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "false_alarms_per_hour": self.false_alarms_per_hour,
            "false_alarm_rate": self.false_alarm_rate,
            "mtbfa_hours": self.mtbfa_hours,
            "mean_detection_delay_seconds": self.mean_detection_delay_seconds,
            "median_detection_delay_seconds": self.median_detection_delay_seconds,
            "max_detection_delay_seconds": self.max_detection_delay_seconds,
            "detected_events": self.detected_events,
            "threshold": self.threshold,
        }
    
    def __str__(self) -> str:
        lines = [
            f"Precision:     {self.precision:.4f}",
            f"Recall:        {self.recall:.4f}",
            f"F1:            {self.f1:.4f}",
            f"Accuracy:      {self.accuracy:.4f}",
            f"Specificity:   {self.specificity:.4f}",
            f"ROC-AUC:       {self.roc_auc:.4f}",
            f"PR-AUC:        {self.pr_auc:.4f}",
            f"TP: {self.tp}, TN: {self.tn}, FP: {self.fp}, FN: {self.fn}",
        ]
        if self.threshold is not None:
            lines.append(f"Threshold:     {self.threshold:.4f}")
        if self.false_alarms_per_hour is not None:
            lines.append(f"False alarms/hr: {self.false_alarms_per_hour:.2f}")
        if self.mean_detection_delay_seconds is not None:
            lines.append(f"Mean delay:      {self.mean_detection_delay_seconds:.1f}s")
        return "\n".join(lines)


def compute_all_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: Optional[float] = None,
    time_per_sample: float = 1.0,
) -> MetricsResult:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True binary labels (0=normal, 1=anomaly)
        y_scores: Anomaly scores (higher = more anomalous)
        threshold: Anomaly threshold (if None, uses F1-optimal)
        time_per_sample: Time per sample in seconds (for operational metrics)
        
    Returns:
        MetricsResult with all metrics
    """
    # Handle NaN in scores - drop corresponding samples
    valid_mask = ~np.isnan(y_scores)
    if not valid_mask.all():
        y_true = y_true[valid_mask]
        y_scores = y_scores[valid_mask]
        if len(y_true) == 0:
            raise ValueError("All scores are NaN")
    
    # Select threshold if not provided
    if threshold is None:
        threshold = _select_f1_optimal_threshold(y_scores, y_true)
    
    # Binary predictions
    y_pred = (y_scores >= threshold).astype(int)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Point-level metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Threshold-independent metrics
    roc_auc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = average_precision_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0
    
    # Operational metrics
    false_alarms_per_hour = None
    false_alarm_rate = None
    mtbfa_hours = None
    mean_detection_delay = None
    median_detection_delay = None
    max_detection_delay = None
    detected_events = None
    
    if time_per_sample > 0:
        # False alarms per hour
        total_time_hours = len(y_true) * time_per_sample / 3600.0
        if total_time_hours > 0:
            false_alarms_per_hour = fp / total_time_hours
        
        # False alarm rate
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # MTBFA
        mtbfa_hours = total_time_hours / fp if fp > 0 else float("inf")
        
        # Detection delay
        delays = _compute_detection_delays(y_true, y_pred, time_per_sample)
        if delays:
            detected_events = len(delays)
            mean_detection_delay = float(np.mean(delays))
            median_detection_delay = float(np.median(delays))
            max_detection_delay = float(np.max(delays))
    
    return MetricsResult(
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        specificity=specificity,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        tp=tp, tn=tn, fp=fp, fn=fn,
        false_alarms_per_hour=false_alarms_per_hour,
        false_alarm_rate=false_alarm_rate,
        mtbfa_hours=mtbfa_hours,
        mean_detection_delay_seconds=mean_detection_delay,
        median_detection_delay_seconds=median_detection_delay,
        max_detection_delay_seconds=max_detection_delay,
        detected_events=detected_events,
        threshold=threshold,
    )


def _select_f1_optimal_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Select threshold maximizing F1."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    return float(thresholds[np.argmax(f1_scores)])


def _compute_detection_delays(
    labels: np.ndarray,
    preds: np.ndarray,
    time_per_sample: float,
) -> List[float]:
    """Compute detection delay for each true anomaly event."""
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
            
            window_preds = preds[anomaly_start:anomaly_end+1]
            if window_preds.sum() > 0:
                first_detect = np.where(window_preds == 1)[0][0]
                delays.append(first_detect * time_per_sample)
    
    if in_anomaly:
        anomaly_end = len(labels) - 1
        window_preds = preds[anomaly_start:anomaly_end+1]
        if window_preds.sum() > 0:
            first_detect = np.where(window_preds == 1)[0][0]
            delays.append(first_detect * time_per_sample)
    
    return delays


def compute_metrics_at_thresholds(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """Compute metrics for multiple thresholds."""
    results = []
    for thresh in thresholds:
        m = compute_all_metrics(y_true, y_scores, threshold=thresh)
        results.append(m.to_dict())
    return pd.DataFrame(results)


def compare_models(
    y_true: np.ndarray,
    model_scores: Dict[str, np.ndarray],
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compare multiple models on the same test set.
    
    Args:
        y_true: True labels
        model_scores: Dict of {model_name: anomaly_scores}
        threshold: Fixed threshold (if None, uses F1-optimal per model)
        
    Returns:
        DataFrame with metrics for each model
    """
    results = []
    for name, scores in model_scores.items():
        thresh = threshold
        if thresh is None:
            thresh = _select_f1_optimal_threshold(scores, y_true)
        
        m = compute_all_metrics(y_true, scores, threshold=thresh)
        d = m.to_dict()
        d["model"] = name
        results.append(d)
    
    return pd.DataFrame(results)


def bootstrap_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    threshold: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute bootstrap confidence intervals for metrics.
    
    Returns:
        Dict of {metric: {"mean": ..., "ci_lower": ..., "ci_upper": ...}}
    """
    n = len(y_true)
    metrics_list = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        m = compute_all_metrics(
            y_true[indices], y_scores[indices], threshold=threshold
        )
        metrics_list.append(m.to_dict())
    
    df = pd.DataFrame(metrics_list)
    alpha = (1 - confidence) / 2
    
    result = {}
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            result[col] = {
                "mean": float(df[col].mean()),
                "ci_lower": float(df[col].quantile(alpha)),
                "ci_upper": float(df[col].quantile(1 - alpha)),
            }
    
    return result