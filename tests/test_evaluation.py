# tests/test_evaluation.py
"""Tests for evaluation module."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from missionguard.evaluation.metrics import (
    compute_all_metrics,
    compute_metrics_at_thresholds,
    compare_models,
    bootstrap_metrics,
    MetricsResult,
)
from missionguard.evaluation.experiment import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentRunner,
    create_baseline_configs,
)


class TestMetrics:
    """Tests for metrics computation."""
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions and labels."""
        np.random.seed(42)
        n = 1000
        # Create imbalanced labels (10% anomalies)
        labels = np.zeros(n)
        labels[:100] = 1
        np.random.shuffle(labels)
        
        # Create scores: higher for anomalies
        scores = np.random.beta(1, 5, n)  # Most scores low
        scores[labels == 1] = np.random.beta(5, 1, 100)  # High scores for anomalies
        
        return labels, scores
    
    def test_compute_all_metrics(self, sample_predictions):
        """Test computing all metrics."""
        labels, scores = sample_predictions
        
        metrics = compute_all_metrics(labels, scores, time_per_sample=1.0)
        
        assert isinstance(metrics, MetricsResult)
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1 <= 1
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.specificity <= 1
        assert 0.5 <= metrics.roc_auc <= 1
        assert 0 <= metrics.pr_auc <= 1
        assert metrics.tp >= 0
        assert metrics.fp >= 0
        assert metrics.tn >= 0
        assert metrics.fn >= 0
        assert metrics.threshold is not None
    
    def test_metrics_with_fixed_threshold(self, sample_predictions):
        """Test metrics with fixed threshold."""
        labels, scores = sample_predictions
        
        metrics = compute_all_metrics(labels, scores, threshold=0.5)
        
        assert metrics.threshold == 0.5
    
    def test_operational_metrics(self, sample_predictions):
        """Test operational metrics (false alarms, detection delay)."""
        labels, scores = sample_predictions
        
        metrics = compute_all_metrics(labels, scores, time_per_sample=1.0)
        
        assert metrics.false_alarms_per_hour is not None
        assert metrics.false_alarm_rate is not None
        assert metrics.mtbfa_hours is not None
        # Detection delay only if there are detected events
        if metrics.detected_events and metrics.detected_events > 0:
            assert metrics.mean_detection_delay_seconds is not None
    
    def test_metrics_result_to_dict(self, sample_predictions):
        """Test MetricsResult serialization."""
        labels, scores = sample_predictions
        metrics = compute_all_metrics(labels, scores)
        
        d = metrics.to_dict()
        assert "precision" in d
        assert "recall" in d
        assert "f1" in d
        assert "threshold" in d
    
    def test_metrics_result_str(self, sample_predictions):
        """Test MetricsResult string representation."""
        labels, scores = sample_predictions
        metrics = compute_all_metrics(labels, scores)
        
        s = str(metrics)
        assert "Precision" in s
        assert "Recall" in s
        assert "F1" in s
    
    def test_compute_metrics_at_thresholds(self, sample_predictions):
        """Test computing metrics at multiple thresholds."""
        labels, scores = sample_predictions
        
        thresholds = np.linspace(0, 1, 11)
        df = compute_metrics_at_thresholds(labels, scores, thresholds)
        
        assert len(df) == 11
        assert "threshold" in df.columns
        assert "f1" in df.columns
    
    def test_compare_models(self, sample_predictions):
        """Test comparing multiple models."""
        labels, scores = sample_predictions
        
        # Create slightly different scores for two models
        model_scores = {
            "model_a": scores,
            "model_b": scores * 0.9 + np.random.rand(len(scores)) * 0.1,
        }
        
        df = compare_models(labels, model_scores)
        
        assert len(df) == 2
        assert set(df["model"]) == {"model_a", "model_b"}
    
    def test_bootstrap_metrics(self, sample_predictions):
        """Test bootstrap confidence intervals."""
        labels, scores = sample_predictions
        
        # Use small n_bootstrap for speed
        result = bootstrap_metrics(labels, scores, n_bootstrap=50, confidence=0.9)
        
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        for metric_name, vals in result.items():
            # Skip infinite values (e.g., mtbfa_hours when fp=0)
            if np.isinf(vals["mean"]) or np.isinf(vals["ci_lower"]) or np.isinf(vals["ci_upper"]):
                continue
            assert "mean" in vals
            assert "ci_lower" in vals
            assert "ci_upper" in vals
            assert vals["ci_lower"] <= vals["mean"] <= vals["ci_upper"]


class TestExperimentConfig:
    """Tests for ExperimentConfig."""
    
    def test_create_config(self):
        """Test creating experiment config."""
        config = ExperimentConfig(
            experiment_id="TEST-001",
            model_name="IsolationForestDetector",
            model_params={"n_estimators": 100},
            dataset="opssat-ad",
        )
        
        assert config.experiment_id == "TEST-001"
        assert config.model_name == "IsolationForestDetector"
        assert config.dataset == "opssat-ad"
        assert config.split_strategy == "segment"
    
    def test_config_defaults(self):
        """Test default values."""
        config = ExperimentConfig(
            experiment_id="TEST-001",
            model_name="Test",
            model_params={},
            dataset="opssat-ad",
        )
        
        assert config.split_strategy == "segment"
        assert config.test_ratio == 0.25
        assert config.scaler_type == "robust"
        assert config.threshold_method == "f1_optimal"
        assert config.random_state == 42


class TestExperimentResult:
    """Tests for ExperimentResult."""
    
    def test_create_result(self):
        """Test creating experiment result."""
        from missionguard.evaluation.metrics import MetricsResult
        
        config = ExperimentConfig(
            experiment_id="TEST-001",
            model_name="Test",
            model_params={},
            dataset="opssat-ad",
        )
        
        metrics = MetricsResult(
            precision=0.9, recall=0.8, f1=0.85, accuracy=0.95,
            specificity=0.97, roc_auc=0.95, pr_auc=0.9,
            tp=80, tn=900, fp=30, fn=20, threshold=0.5,
        )
        
        result = ExperimentResult(
            experiment_id="TEST-001",
            config=config,
            metrics=metrics,
            train_time_seconds=10.5,
            inference_time_seconds=0.1,
        )
        
        assert result.experiment_id == "TEST-001"
        assert result.train_time_seconds == 10.5
    
    def test_result_serialization(self, tmp_path):
        """Test saving experiment result."""
        from missionguard.evaluation.metrics import MetricsResult
        
        config = ExperimentConfig(
            experiment_id="TEST-001",
            model_name="Test",
            model_params={},
            dataset="opssat-ad",
        )
        
        metrics = MetricsResult(
            precision=0.9, recall=0.8, f1=0.85, accuracy=0.95,
            specificity=0.97, roc_auc=0.95, pr_auc=0.9,
            tp=80, tn=900, fp=30, fn=20, threshold=0.5,
        )
        
        result = ExperimentResult(
            experiment_id="TEST-001",
            config=config,
            metrics=metrics,
            train_time_seconds=10.5,
            inference_time_seconds=0.1,
        )
        
        save_path = tmp_path / "result.json"
        result.save(save_path)
        
        assert save_path.exists()
        
        # Load and verify
        import json
        with open(save_path) as f:
            loaded = json.load(f)
        
        assert loaded["experiment_id"] == "TEST-001"
        assert loaded["metrics"]["precision"] == 0.9


class TestCreateBaselineConfigs:
    """Tests for baseline config factory."""
    
    def test_create_baseline_configs(self):
        """Test creating baseline configs."""
        configs = create_baseline_configs()
        
        assert len(configs) >= 3
        exp_ids = [c.experiment_id for c in configs]
        assert "EXP-001-statistical-mad" in exp_ids
        assert "EXP-002-statistical-zscore" in exp_ids
        assert "EXP-003-isolation-forest" in exp_ids


class TestExperimentRunner:
    """Tests for ExperimentRunner (integration tests)."""
    
    @pytest.fixture
    def runner(self, tmp_path):
        """Create runner with temp directories."""
        return ExperimentRunner(
            data_dir=Path(__file__).parent.parent / "data",
            output_dir=tmp_path / "experiments",
        )
    
    def test_create_runner(self, runner):
        """Test creating runner."""
        assert runner.data_dir.exists()
        assert runner.output_dir.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])