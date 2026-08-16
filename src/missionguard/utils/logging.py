# src/missionguard/utils/logging.py
"""Logging utilities for MissionGuard."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
) -> None:
    """
    Configure root logger.
    
    Args:
        level: Logging level
        log_file: Optional file path for file logging
        format_string: Custom format string
    """
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get logger instance for a module."""
    return logging.getLogger(name)


class ExperimentLogger:
    """Logger for ML experiments with structured output."""
    
    def __init__(self, experiment_id: str, log_dir: Path = None):
        self.experiment_id = experiment_id
        self.log_dir = log_dir or Path("logs/experiments")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(f"experiment.{experiment_id}")
        self.logger.setLevel(logging.INFO)
        
        # File handler for experiment
        fh = logging.FileHandler(self.log_dir / f"{experiment_id}.log")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        self.logger.addHandler(fh)
        self.logger.propagate = False
    
    def log_params(self, params: dict) -> None:
        """Log experiment parameters."""
        self.logger.info(f"PARAMS: {params}")
    
    def log_metrics(self, metrics: dict, split: str = "val") -> None:
        """Log evaluation metrics."""
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        self.logger.info(f"METRICS [{split}]: {metrics_str}")
    
    def log_artifact(self, path: Path, description: str = "") -> None:
        """Log artifact path."""
        self.logger.info(f"ARTIFACT: {path} | {description}")
    
    def log_note(self, note: str) -> None:
        """Log arbitrary note."""
        self.logger.info(f"NOTE: {note}")