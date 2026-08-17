# Phase 3a — Rolling Baselines on OPSSAT-AD Raw Telemetry
# Run with: python notebooks/03_rolling_baselines.py

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from missionguard.data import load_opssat_ad, get_train_test_split
from missionguard.preprocessing.time_series import (
    sort_by_segment_time,
    extract_segment_windows,
    compute_rolling_features,
    compute_differencing_features,
    detect_gaps,
)
from missionguard.models import RollingMADBaseline, RollingZScoreBaseline
from missionguard.evaluation import compute_all_metrics
from missionguard.detection import scores_to_events, merge_events

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = Path("D:/Hackathon/ai-builder-ibm/MissionGuard/data/raw/opssat-ad")
OUTPUT_DIR = Path("D:/Hackathon/ai-builder-ibm/MissionGuard/artifacts/phase3a")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PHASE 3a: Rolling Baselines on Raw Telemetry")
print("=" * 60)

# 1. Load and sort raw telemetry
print("\n1. Loading raw telemetry (segments.csv)...")
segments_df = pd.read_csv(DATA_DIR / "segments.csv")
segments_df["timestamp"] = pd.to_datetime(segments_df["timestamp"], utc=True)
print(f"   Raw rows: {len(segments_df):,}")
print(f"   Channels: {segments_df['channel'].nunique()}")
print(f"   Segments: {segments_df['segment'].nunique()}")

# Sort by segment then timestamp (critical for OPSSAT-AD)
segments_sorted = sort_by_segment_time(segments_df)
print(f"   Sorted by segment+timestamp")

# 2. Extract segment windows for a few representative segments
print("\n2. Extracting segment windows...")
segment_ids = segments_sorted["segment"].unique()

# Pick a few anomaly and normal segments for analysis
anomaly_segments = segments_sorted[segments_sorted["anomaly"] == 1]["segment"].unique()
normal_segments = segments_sorted[segments_sorted["anomaly"] == 0]["segment"].unique()

print(f"   Anomaly segments: {len(anomaly_segments)}")
print(f"   Normal segments: {len(normal_segments)}")

# Analyze first few segments of each type
for seg_id in list(anomaly_segments[:3]) + list(normal_segments[:3]):
    seg_data = extract_segment_windows(segments_sorted, seg_id)
    print(f"   Segment {seg_id}: {len(seg_data)} points, "
          f"channel={seg_data['channel'].iloc[0]}, "
          f"anomaly={seg_data['anomaly'].iloc[0]}, "
          f"sampling={seg_data['sampling'].iloc[0]}Hz, "
          f"duration={(seg_data['timestamp'].iloc[-1] - seg_data['timestamp'].iloc[0]).total_seconds():.0f}s")

# 3. Detect gaps in segments
print("\n3. Detecting sampling gaps...")
gaps_df = detect_gaps(segments_sorted, max_gap_seconds=5.0)
print(f"   Segments with gaps >5s: {(gaps_df['num_gaps'] > 0).sum()}")
print(f"   Max gap across all: {gaps_df['max_gap_seconds'].max():.1f}s")

# 4. Prepare rolling features for ALL segments
print("\n4. Computing rolling features per segment...")
window_sizes = [10, 30, 60]  # 10s, 30s, 60s windows
rolling_features_list = []

for seg_id in segment_ids:
    seg_data = extract_segment_windows(segments_sorted, seg_id)
    seg_data = seg_data.reset_index(drop=True)
    
    # Compute rolling features on value column
    rolling_df = compute_rolling_features(
        seg_data["value"], 
        windows=window_sizes,
        features=["mean", "std", "min", "max", "skew", "kurt"]
    )
    
    # Add metadata
    rolling_df["segment"] = seg_id
    rolling_df["channel"] = seg_data["channel"].iloc[0]
    rolling_df["anomaly"] = seg_data["anomaly"].iloc[0]
    rolling_df["sampling"] = seg_data["sampling"].iloc[0]
    rolling_df["timestamp"] = seg_data["timestamp"]
    
    rolling_features_list.append(rolling_df)

rolling_features = pd.concat(rolling_features_list, ignore_index=True)
print(f"   Rolling features shape: {rolling_features.shape}")
print(f"   Columns: {list(rolling_features.columns)}")

# Save rolling features
rolling_path = OUTPUT_DIR / "rolling_features.csv"
rolling_features.to_csv(rolling_path, index=False)
print(f"   Saved to {rolling_path}")

# 5. Train Rolling MAD and Z-Score baselines per channel
print("\n5. Training rolling baselines per channel...")

# Feature columns (rolling features)
feature_cols = [c for c in rolling_features.columns 
                if c not in ["segment", "channel", "anomaly", "sampling", "timestamp"]]

# Use provided train/test split
train_segments = segments_sorted[segments_sorted["train"] == 1]["segment"].unique()
test_segments = segments_sorted[segments_sorted["train"] == 0]["segment"].unique()

train_rolling = rolling_features[rolling_features["segment"].isin(train_segments)]
test_rolling = rolling_features[rolling_features["segment"].isin(test_segments)]

print(f"   Train segments: {len(train_segments)}, rows: {len(train_rolling):,}")
print(f"   Test segments: {len(test_segments)}, rows: {len(test_rolling):,}")

# Drop rows with NaN (from rolling window warmup)
train_rolling_clean = train_rolling.dropna(subset=feature_cols).reset_index(drop=True)
test_rolling_clean = test_rolling.dropna(subset=feature_cols).reset_index(drop=True)

print(f"   After dropping NaN - Train: {len(train_rolling_clean):,}, Test: {len(test_rolling_clean):,}")

X_train = train_rolling_clean[feature_cols]
y_train = train_rolling_clean["anomaly"]
X_test = test_rolling_clean[feature_cols]
y_test = test_rolling_clean["anomaly"]

print(f"   Feature count: {len(feature_cols)}")
print(f"   Train anomaly rate: {y_train.mean():.3f}")
print(f"   Test anomaly rate: {y_test.mean():.3f}")

# 6. Rolling MAD Baseline
print("\n6. Rolling MAD Baseline...")
for window in window_sizes:
    window_features = [c for c in feature_cols if f"roll_{window}_" in c]
    print(f"   Window {window}s: {len(window_features)} features")
    
    mad = RollingMADBaseline(window=window, min_periods=5, aggregation="max")
    mad.fit(X_train[window_features])
    
    # Score validation set
    val_size = int(0.2 * len(X_train))
    val_scores = mad.score(X_train[window_features].iloc[-val_size:])
    val_scores_valid = val_scores[~np.isnan(val_scores)]
    if len(val_scores_valid) == 0:
        print(f"     MAD w={window}: No valid validation scores")
        continue
    mad.set_threshold_from_scores(val_scores_valid, method="percentile", value=95.0)
    
    test_scores = mad.score(X_test[window_features])
    test_scores_valid = test_scores[~np.isnan(test_scores)]
    y_test_valid = y_test[~np.isnan(test_scores)]
    metrics = compute_all_metrics(y_test_valid.values, test_scores_valid, threshold=mad.threshold)
    
    print(f"     MAD w={window}: F1={metrics.f1:.4f}, P={metrics.precision:.4f}, R={metrics.recall:.4f}, "
          f"thresh={mad.threshold:.4f}")

# 7. Rolling Z-Score Baseline
print("\n7. Rolling Z-Score Baseline...")
for window in window_sizes:
    window_features = [c for c in feature_cols if f"roll_{window}_" in c]
    print(f"   Window {window}s: {len(window_features)} features")
    
    zscore = RollingZScoreBaseline(window=window, min_periods=5, aggregation="max")
    zscore.fit(X_train[window_features])
    
    val_size = int(0.2 * len(X_train))
    val_scores = zscore.score(X_train[window_features].iloc[-val_size:])
    val_scores_valid = val_scores[~np.isnan(val_scores)]
    if len(val_scores_valid) == 0:
        print(f"     Z-score w={window}: No valid validation scores")
        continue
    zscore.set_threshold_from_scores(val_scores_valid, method="percentile", value=95.0)
    
    test_scores = zscore.score(X_test[window_features])
    test_scores_valid = test_scores[~np.isnan(test_scores)]
    y_test_valid = y_test[~np.isnan(test_scores)]
    metrics = compute_all_metrics(y_test_valid.values, test_scores_valid, threshold=zscore.threshold)
    
    print(f"     Z-score w={window}: F1={metrics.f1:.4f}, P={metrics.precision:.4f}, R={metrics.recall:.4f}, "
          f"thresh={zscore.threshold:.4f}")

# 8. Event detection on test set
print("\n8. Event detection on test segments...")
# Use best window (60s) for event detection
best_window = 60
window_features = [c for c in feature_cols if f"roll_{best_window}_" in c]

mad_best = RollingMADBaseline(window=best_window, min_periods=5, aggregation="max")
mad_best.fit(X_train[window_features])
val_scores = mad_best.score(X_train[window_features].iloc[-val_size:])
val_scores_valid = val_scores[~np.isnan(val_scores)]
mad_best.set_threshold_from_scores(val_scores_valid, method="percentile", value=95.0)

test_scores = mad_best.score(X_test[window_features])
test_scores_valid = test_scores[~np.isnan(test_scores)]
test_timestamps = test_rolling_clean["timestamp"]
test_segments_arr = test_rolling_clean["segment"].values

# Convert scores to events per segment
events_by_segment = {}
for seg_id in test_segments[:10]:  # First 10 test segments
    seg_mask = test_rolling_clean["segment"] == seg_id
    if not seg_mask.any():
        continue
    
    seg_scores = test_scores[seg_mask]
    seg_times = test_timestamps[seg_mask]
    seg_channel = test_rolling_clean.loc[seg_mask, "channel"].iloc[0]
    
    events = scores_to_events(
        seg_scores, 
        seg_times, 
        seg_channel, 
        threshold=mad_best.threshold,
        min_duration=5
    )
    if events:
        events_by_segment[seg_id] = events

print(f"   Segments with events: {len(events_by_segment)}")
for seg_id, events in list(events_by_segment.items())[:5]:
    for e in events[:2]:
        print(f"     Seg {seg_id} ({e.channel}): {e.start_time} - {e.end_time}, "
              f"max_score={e.max_score:.3f}, dur={e.duration_seconds:.0f}s")

# 9. Merge events across segments (temporal grouping)
print("\n9. Temporal event grouping (incident aggregation)...")
all_events = []
for seg_id, events in events_by_segment.items():
    all_events.extend(events)

# Sort by time
all_events_sorted = sorted(all_events, key=lambda e: e.start_time)
merged_events = merge_events(all_events_sorted, max_gap_seconds=300.0)  # 5 min gap

print(f"   Raw events: {len(all_events)}")
print(f"   Merged incidents: {len(merged_events)}")
for i, inc in enumerate(merged_events[:5]):
    print(f"     INC-{i}: {inc.start_time} - {inc.end_time}, "
          f"channels={inc.channel}, max_score={inc.max_score:.3f}, "
          f"dur={inc.duration_seconds:.0f}s")

print("\n" + "=" * 60)
print("PHASE 3a COMPLETE")
print("=" * 60)