# src/missionguard/evaluation/experiment.py
"""Experiment runner for reproducible ML experiments."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

from ..models import BaseAnomalyDetector
from ..data.loaders import get_train_test_split
from ..preprocessing import fit_scaler, transform_features, get_feature_names, prepare_features_target
from .metrics import compute_all_metrics, MetricsResult


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    experiment_id: str
    model_name: str
    model_params: Dict[str, Any]
    dataset: str  # "opssat-ad" or "esa-adb"
    split_strategy: str = "segment"  # "segment" or "temporal"
    test_ratio: float = 0.25
    scaler_type: str = "robust"
    threshold_method: str = "f1_optimal"
    threshold_percentile: float = 95.0
    random_state: int = 42
    notes: str = ""


@dataclass
class ExperimentResult:
    """Results from an experiment run."""
    experiment_id: str
    config: ExperimentConfig
    metrics: MetricsResult
    train_time_seconds: float
    inference_time_seconds: float
    model_path: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            "experiment_id": self.experiment_id,
            "config": self.config.__dict__,
            "metrics": self.metrics.to_dict(),
            "train_time_seconds": self.train_time_seconds,
            "inference_time_seconds": self.inference_time_seconds,
            "model_path": self.model_path,
            "timestamp": self.timestamp,
        }
    
    def save(self, path: Path) -> None:
        """Save experiment result to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class ExperimentRunner:
    """Runs reproducible ML experiments."""
    
    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        random_state: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.random_state = random_state
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run(
        self,
        config: ExperimentConfig,
        model_class,
        save_model: bool = True,
    ) -> ExperimentResult:
        """
        Run a complete experiment.
        
        Args:
            config: Experiment configuration
            model_class: Anomaly detector class (e.g., IsolationForestDetector)
            save_model: Whether to save fitted model
            
        Returns:
            ExperimentResult
        """
        import time
        
        print(f"Running experiment: {config.experiment_id}")
        
        # Load data
        if config.dataset == "opssat-ad":
            from ..data.loaders import load_opssat_ad
            segments, dataset = load_opssat_ad(self.data_dir / "opssat-ad")
        else:
            raise ValueError(f"Unknown dataset: {config.dataset}")
        
        # Split data
        if config.split_strategy == "segment":
            train_seg, test_seg, train_ds, test_ds = get_train_test_split(segments, dataset)
        elif config.split_strategy == "temporal":
            from ..data.loaders import get_temporal_train_test_split
            train_seg, test_seg = get_temporal_train_test_split(segments, config.test_ratio)
            train_ds = dataset[dataset["segment"].isin(train_seg["segment"])].copy()
            test_ds = dataset[dataset["segment"].isin(test_seg["segment"])].copy()
        else:
            raise ValueError(f"Unknown split strategy: {config.split_strategy}")
        
        # Prepare features
        feature_names = get_feature_names(train_ds)
        
        # Scale features
        scaler = fit_scaler(train_ds, feature_names, scaler_type=config.scaler_type)
        train_scaled = transform_features(train_ds, scaler, feature_names)
        test_scaled = transform_features(test_ds, scaler, feature_names)
        
        # Extract X, y
        X_train, y_train = prepare_features_target(train_scaled, feature_names)
        X_test, y_test = prepare_features_target(test_scaled, feature_names)
        
        # Create and train model
        model = model_class(**config.model_params)
        
        train_start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - train_start
        
        # Score test set
        infer_start = time.time()
        test_scores = model.score(X_test)
        inference_time = time.time() - infer_start
        
        # Select threshold on validation (use portion of train as val)
        val_size = int(0.2 * len(X_train))
        X_val = X_train.iloc[-val_size:]
        y_val = y_train.iloc[-val_size:]
        X_train_main = X_train.iloc[:-val_size]
        y_train_main = y_train.iloc[:-val_size]
        
        val_scores = model.score(X_val)
        
        if config.threshold_method == "f1_optimal":
            threshold = model.tune_threshold(X_val, y_val, metric="f1")
        elif config.threshold_method == "percentile":
            threshold = float(np.percentile(val_scores, config.threshold_percentile))
            model.set_threshold(threshold)
        else:
            raise ValueError(f"Unknown threshold method: {config.threshold_method}")
        
        # Evaluate on test
        metrics = compute_all_metrics(
            y_test.values, test_scores, threshold=model.threshold,
            time_per_sample=1.0  # OPSSAT-AD is 1Hz per segment point
        )
        
        # Save model
        model_path = None
        if save_model:
            model_path = self.output_dir / f"{config.experiment_id}_model.joblib"
            model.save(model_path)
        
        # Save experiment result
        result = ExperimentResult(
            experiment_id=config.experiment_id,
            config=config,
            metrics=metrics,
            train_time_seconds=train_time,
            inference_time_seconds=inference_time,
            model_path=str(model_path) if model_path else None,
        )
        
        result_path = self.output_dir / f"{config.experiment_id}_result.json"
        result.save(result_path)
        
        print(f"  F1: {metrics.f1:.4f}, Precision: {metrics.precision:.4f}, Recall: {metrics.recall:.4f}")
        print(f"  Train time: {train_time:.1f}s, Inference: {inference_time:.3f}s")
        
        return result
    
    def run_baseline_comparison(
        self,
        baseline_configs: List[ExperimentConfig],
        model_class,
    ) -> List[ExperimentResult]:
        """Run multiple baseline experiments."""
        results = []
        for config in baseline_configs:
            try:
                result = self.run(config, model_class)
                results.append(result)
            except Exception as e:
                print(f"Experiment {config.experiment_id} failed: {e}")
        return results


def create_baseline_configs() -> List[ExperimentConfig]:
    """Create standard baseline experiment configurations."""
    configs = []
    
    # Statistical baseline - MAD
    configs.append(ExperimentConfig(
        experiment_id="EXP-001-statistical-mad",
        model_name="StatisticalBaseline",
        model_params={"method": "mad", "aggregation": "max"},
        dataset="opssat-ad",
        split_strategy="segment",
        threshold_method="percentile",
        threshold_percentile=95.0,
        notes="Global MAD baseline on segment features",
    ))
    
    # Statistical baseline - Z-score
    configs.append(ExperimentConfig(
        experiment_id="EXP-002-statistical-zscore",
        model_name="StatisticalBaseline",
        model_params={"method": "zscore", "aggregation": "max"},
        dataset="opssat-ad",
        split_strategy="segment",
        threshold_method="percentile",
        threshold_percentile=95.0,
        notes="Global Z-score baseline on segment features",
    ))
    
    # Rolling MAD (needs raw time series - placeholder for now)
    # configs.append(ExperimentConfig(...))
    
    # Isolation Forest
    configs.append(ExperimentConfig(
        experiment_id="EXP-003-isolation-forest",
        model_name="IsolationForestDetector",
        model_params={
            "n_estimators": 100,
            "contamination": "auto",
            "score_normalization": "minmax",
        },
        dataset="opssat-ad",
        split_strategy="segment",
        threshold_method="f1_optimal",
        notes="Isolation Forest on 18 segment features",
    ))
    
    return configs