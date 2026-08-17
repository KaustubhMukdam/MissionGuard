# Phase 3b — Isolation Forest Production Baseline + Extended Experiments
# Run with: python notebooks/04_iforest_production.py

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from missionguard.data import load_opssat_ad, get_train_test_split
from missionguard.preprocessing import fit_scaler, transform_features, get_feature_names, prepare_features_target
from missionguard.models import IsolationForestDetector
from missionguard.evaluation import compute_all_metrics, compare_models, bootstrap_metrics
from missionguard.detection import scores_to_events, merge_events

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = Path("D:/Hackathon/ai-builder-ibm/MissionGuard/data/raw/opssat-ad")
OUTPUT_DIR = Path("D:/Hackathon/ai-builder-ibm/MissionGuard/artifacts/phase3b")
MODELS_DIR = Path("D:/Hackathon/ai-builder-ibm/MissionGuard/models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("PHASE 3b: Isolation Forest Production Baseline + Extended Experiments")
print("=" * 70)

# ============================================================
# 1. Load and prepare data (same pipeline as Phase 2)
# ============================================================
print("\n1. Loading OPSSAT-AD segment features...")
segments, dataset = load_opssat_ad(DATA_DIR)
train_seg, test_seg, train_ds, test_ds = get_train_test_split(segments, dataset)

feature_names = get_feature_names(train_ds)
scaler = fit_scaler(train_ds, feature_names, scaler_type="robust")
train_scaled = transform_features(train_ds, scaler, feature_names)
test_scaled = transform_features(test_ds, scaler, feature_names)

X_train, y_train = prepare_features_target(train_scaled, feature_names)
X_test, y_test = prepare_features_target(test_scaled, feature_names)

print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
print(f"   Features: {len(feature_names)}")
print(f"   Train anomaly rate: {y_train.mean():.3f}")
print(f"   Test anomaly rate: {y_test.mean():.3f}")

# ============================================================
# 2. Production Baseline Configuration
# ============================================================
print("\n2. Production Baseline Configuration...")

PROD_CONFIG = {
    "model_name": "IsolationForestDetector",
    "version": "1.0.0",
    "trained_on": "OPSSAT-AD segment features (dataset.csv)",
    "feature_names": feature_names,
    "scaler_type": "robust",
    "scaler_params": {
        "center_": scaler.scaler.center_.tolist() if hasattr(scaler.scaler, 'center_') else None,
        "scale_": scaler.scaler.scale_.tolist() if hasattr(scaler.scaler, 'scale_') else None,
    },
    "model_params": {
        "n_estimators": 200,
        "max_samples": "auto",
        "contamination": 0.1,  # Fixed contamination for reproducibility
        "max_features": 1.0,
        "bootstrap": False,
        "n_jobs": -1,
        "random_state": 42,
    },
    "score_normalization": "minmax",
    "threshold_method": "f1_optimal",
    "threshold_value": None,  # Set after training
    "training_date": datetime.now().isoformat(),
    "data_split": {
        "train_segments": int(len(train_ds)),
        "test_segments": int(len(test_ds)),
        "split_strategy": "segment_based",
    },
}

print(f"   Model params: {PROD_CONFIG['model_params']}")

# ============================================================
# 3. Train Production Model
# ============================================================
print("\n3. Training Production Isolation Forest...")

prod_model = IsolationForestDetector(**PROD_CONFIG["model_params"])
prod_model.fit(X_train)

# Score validation set (20% of train)
val_size = int(0.2 * len(X_train))
X_val = X_train.iloc[-val_size:]
y_val = y_train.iloc[-val_size:]

val_scores = prod_model.score(X_val)
prod_model.tune_threshold(X_val, y_val, metric="f1")

PROD_CONFIG["threshold_value"] = float(prod_model.threshold)
print(f"   Optimal threshold (F1): {prod_model.threshold:.6f}")

# Score test set
test_scores = prod_model.score(X_test)
test_preds = prod_model.predict(X_test)

# Evaluate
prod_metrics = compute_all_metrics(y_test.values, test_scores, threshold=prod_model.threshold)
print(f"\n   PRODUCTION BASELINE RESULTS:")
print(f"   F1: {prod_metrics.f1:.4f}")
print(f"   Precision: {prod_metrics.precision:.4f}")
print(f"   Recall: {prod_metrics.recall:.4f}")
print(f"   ROC-AUC: {prod_metrics.roc_auc:.4f}")
print(f"   PR-AUC: {prod_metrics.pr_auc:.4f}")
print(f"   False alarms/hr: {prod_metrics.false_alarms_per_hour:.1f}")
print(f"   Mean detection delay: {prod_metrics.mean_detection_delay_seconds:.1f}s")

# ============================================================
# 4. Save Production Artifacts
# ============================================================
print("\n4. Saving Production Artifacts...")

# Save model
model_path = MODELS_DIR / "isolation_forest_prod_v1.joblib"
prod_model.save(model_path)
print(f"   Model saved: {model_path}")

# Save scaler
scaler_path = MODELS_DIR / "robust_scaler_prod_v1.joblib"
scaler.save(scaler_path)
print(f"   Scaler saved: {scaler_path}")

# Save config
config_path = MODELS_DIR / "prod_config_v1.json"
with open(config_path, "w") as f:
    json.dump(PROD_CONFIG, f, indent=2, default=str)
print(f"   Config saved: {config_path}")

# Save metrics
metrics_path = OUTPUT_DIR / "prod_baseline_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(prod_metrics.to_dict(), f, indent=2, default=str)
print(f"   Metrics saved: {metrics_path}")

# ============================================================
# 5. Extended Experiments
# ============================================================
print("\n" + "=" * 70)
print("5. EXTENDED EXPERIMENTS")
print("=" * 70)

experiment_results = {}

# ---- Experiment 5.1: Contamination sweep ----
print("\n5.1 Contamination Parameter Sweep...")
contamination_values = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, "auto"]
contam_results = []

for contam in contamination_values:
    print(f"   Testing contamination={contam}...")
    model = IsolationForestDetector(
        n_estimators=200,
        contamination=contam,
        score_normalization="minmax",
        random_state=42,
    )
    model.fit(X_train)
    model.tune_threshold(X_val, y_val, metric="f1")
    scores = model.score(X_test)
    metrics = compute_all_metrics(y_test.values, scores, threshold=model.threshold)
    
    contam_results.append({
        "contamination": contam,
        "threshold": float(model.threshold),
        "f1": metrics.f1,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "roc_auc": metrics.roc_auc,
        "pr_auc": metrics.pr_auc,
        "false_alarms_per_hour": metrics.false_alarms_per_hour,
    })

contam_df = pd.DataFrame(contam_results)
contam_path = OUTPUT_DIR / "experiment_contamination_sweep.csv"
contam_df.to_csv(contam_path, index=False)
print(f"   Results saved: {contam_path}")
experiment_results["contamination_sweep"] = contam_results

# ---- Experiment 5.2: N_estimators sweep ----
print("\n5.2 N_Estimators Sweep...")
n_est_values = [50, 100, 200, 300, 500]
n_est_results = []

for n_est in n_est_values:
    print(f"   Testing n_estimators={n_est}...")
    model = IsolationForestDetector(
        n_estimators=n_est,
        contamination=0.1,
        score_normalization="minmax",
        random_state=42,
    )
    model.fit(X_train)
    model.tune_threshold(X_val, y_val, metric="f1")
    scores = model.score(X_test)
    metrics = compute_all_metrics(y_test.values, scores, threshold=model.threshold)
    
    n_est_results.append({
        "n_estimators": n_est,
        "threshold": float(model.threshold),
        "f1": metrics.f1,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "roc_auc": metrics.roc_auc,
        "pr_auc": metrics.pr_auc,
        "train_time_sec": None,  # Could measure
    })

n_est_df = pd.DataFrame(n_est_results)
n_est_path = OUTPUT_DIR / "experiment_n_estimators_sweep.csv"
n_est_df.to_csv(n_est_path, index=False)
experiment_results["n_estimators_sweep"] = n_est_results

# ---- Experiment 5.3: Feature Subset Analysis ----
print("\n5.3 Feature Subset Analysis...")
feature_groups = {
    "statistical": ["mean", "var", "std", "kurtosis", "skew"],
    "peak_based": ["n_peaks", "smooth10_n_peaks", "smooth20_n_peaks"],
    "diff_based": ["diff_peaks", "diff2_peaks", "diff_var", "diff2_var"],
    "duration_based": ["duration", "len", "len_weighted", "gaps_squared", "var_div_duration", "var_div_len"],
    "all": feature_names,
}

feature_results = []
for group_name, group_features in feature_groups.items():
    print(f"   Testing feature group: {group_name} ({len(group_features)} features)...")
    
    X_train_sub = X_train[group_features]
    X_val_sub = X_val[group_features]
    X_test_sub = X_test[group_features]
    
    # Scale subset
    scaler_sub = fit_scaler(X_train_sub, group_features, scaler_type="robust")
    X_train_scaled = transform_features(X_train_sub, scaler_sub, group_features)
    X_val_scaled = transform_features(X_val_sub, scaler_sub, group_features)
    X_test_scaled = transform_features(X_test_sub, scaler_sub, group_features)
    
    model = IsolationForestDetector(
        n_estimators=200,
        contamination=0.1,
        score_normalization="minmax",
        random_state=42,
    )
    model.fit(X_train_scaled)
    model.tune_threshold(X_val_scaled, y_val, metric="f1")
    scores = model.score(X_test_scaled)
    metrics = compute_all_metrics(y_test.values, scores, threshold=model.threshold)
    
    feature_results.append({
        "feature_group": group_name,
        "n_features": len(group_features),
        "threshold": float(model.threshold),
        "f1": metrics.f1,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "roc_auc": metrics.roc_auc,
        "pr_auc": metrics.pr_auc,
    })

feature_df = pd.DataFrame(feature_results)
feature_path = OUTPUT_DIR / "experiment_feature_groups.csv"
feature_df.to_csv(feature_path, index=False)
experiment_results["feature_groups"] = feature_results

# ---- Experiment 5.4: Score Normalization Comparison ----
print("\n5.4 Score Normalization Comparison...")
norm_results = []
for norm in ["minmax", "percentile", "none"]:
    print(f"   Testing normalization={norm}...")
    model = IsolationForestDetector(
        n_estimators=200,
        contamination=0.1,
        score_normalization=norm,
        random_state=42,
    )
    model.fit(X_train)
    model.tune_threshold(X_val, y_val, metric="f1")
    scores = model.score(X_test)
    metrics = compute_all_metrics(y_test.values, scores, threshold=model.threshold)
    
    norm_results.append({
        "normalization": norm,
        "threshold": float(model.threshold),
        "f1": metrics.f1,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "roc_auc": metrics.roc_auc,
        "pr_auc": metrics.pr_auc,
    })

norm_df = pd.DataFrame(norm_results)
norm_path = OUTPUT_DIR / "experiment_normalization.csv"
norm_df.to_csv(norm_path, index=False)
experiment_results["normalization"] = norm_results

# ---- Experiment 5.5: Bootstrap Confidence Intervals ----
print("\n5.5 Bootstrap Confidence Intervals (100 samples)...")
bootstrap_results = bootstrap_metrics(
    y_test.values, test_scores, 
    n_bootstrap=100, 
    confidence=0.95,
    threshold=prod_model.threshold
)
bootstrap_path = OUTPUT_DIR / "experiment_bootstrap.json"
with open(bootstrap_path, "w") as f:
    json.dump(bootstrap_results, f, indent=2, default=str)
experiment_results["bootstrap"] = bootstrap_results

# ============================================================
# 6. Summary & Save All Experiment Results
# ============================================================
print("\n" + "=" * 70)
print("6. EXPERIMENT SUMMARY")
print("=" * 70)

# Print key findings
print("\n--- Contamination Sweep ---")
for r in contam_results:
    print(f"  contam={r['contamination']}: F1={r['f1']:.4f}, P={r['precision']:.4f}, R={r['recall']:.4f}, thresh={r['threshold']:.4f}")

print("\n--- N_Estimators ---")
for r in n_est_results:
    print(f"  n_est={r['n_estimators']}: F1={r['f1']:.4f}, P={r['precision']:.4f}, R={r['recall']:.4f}")

print("\n--- Feature Groups ---")
for r in feature_results:
    print(f"  {r['feature_group']} ({r['n_features']} feat): F1={r['f1']:.4f}, P={r['precision']:.4f}, R={r['recall']:.4f}")

print("\n--- Normalization ---")
for r in norm_results:
    print(f"  norm={r['normalization']}: F1={r['f1']:.4f}, P={r['precision']:.4f}, R={r['recall']:.4f}")

print("\n--- Bootstrap CI (F1) ---")
if "f1" in bootstrap_results:
    b = bootstrap_results["f1"]
    print(f"  Mean: {b['mean']:.4f}, 95% CI: [{b['ci_lower']:.4f}, {b['ci_upper']:.4f}]")

# Save all experiment results
all_results_path = OUTPUT_DIR / "all_experiment_results.json"
with open(all_results_path, "w") as f:
    json.dump({
        "production_baseline": PROD_CONFIG,
        "production_metrics": prod_metrics.to_dict(),
        "experiments": experiment_results,
        "timestamp": datetime.now().isoformat(),
    }, f, indent=2, default=str)
print(f"\nAll results saved: {all_results_path}")

print("\n" + "=" * 70)
print("PHASE 3b COMPLETE")
print("=" * 70)
print(f"\nProduction model ready at: {model_path}")
print(f"Config: {config_path}")
print(f"All experiment data: {OUTPUT_DIR}")