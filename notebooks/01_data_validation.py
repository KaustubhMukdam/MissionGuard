# Phase 0 — OPSSAT-AD Dataset Validation
# Run with: python notebooks/01_data_validation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_DIR = Path("D:/Hackathon/ai-builder-ibm/MissionGuard/data/raw/opssat-ad")
print(f"Data directory: {DATA_DIR.absolute()}")
print(f"Exists: {DATA_DIR.exists()}")

for f in DATA_DIR.iterdir():
    print(f"  {f.name}: {f.stat().st_size / 1024:.1f} KB")

# Load segments.csv (raw telemetry)
segments = pd.read_csv(DATA_DIR / "segments.csv")
print(f"\nShape: {segments.shape}")
print(f"Columns: {list(segments.columns)}")
print(f"\nDtypes:\n{segments.dtypes}")
print(f"\nFirst 5 rows:\n{segments.head()}")
print(f"\nLast 5 rows:\n{segments.tail()}")

# Basic statistics
print(f"\nUnique channels: {segments['channel'].unique()}")
print(f"Unique labels: {segments['label'].unique()}")
print(f"Anomaly distribution:\n{segments['anomaly'].value_counts()}")
print(f"Train distribution:\n{segments['train'].value_counts()}")
print(f"Segment IDs: {segments['segment'].nunique()} unique, range: {segments['segment'].min()}-{segments['segment'].max()}")

# Timestamp range
segments['timestamp'] = pd.to_datetime(segments['timestamp'])
print(f"\nTimestamp range: {segments['timestamp'].min()} to {segments['timestamp'].max()}")
print(f"Duration: {segments['timestamp'].max() - segments['timestamp'].min()}")

# Check for missing values
print(f"\nMissing values:\n{segments.isnull().sum()}")

# Check for duplicates
print(f"\nDuplicate rows: {segments.duplicated().sum()}")

# Check timestamp ordering
is_sorted = segments['timestamp'].is_monotonic_increasing
print(f"\nTimestamps monotonically increasing: {is_sorted}")

# Time gaps
time_diffs = segments['timestamp'].diff().dt.total_seconds()
print(f"\nTime diff stats (seconds):\n{time_diffs.describe()}")
print(f"\nGaps > 1s: {(time_diffs > 1).sum()}")

# Anomaly analysis
anomaly_segments = segments[segments['anomaly'] == 1]
nominal_segments = segments[segments['anomaly'] == 0]

print(f"\nAnomaly rows: {len(anomaly_segments)} ({len(anomaly_segments)/len(segments)*100:.2f}%)")
print(f"Nominal rows: {len(nominal_segments)} ({len(nominal_segments)/len(segments)*100:.2f}%)")

# Anomaly by segment
anom_by_segment = anomaly_segments.groupby('segment').size()
print(f"\nAnomaly segment lengths:\n{anom_by_segment.describe()}")

# Anomaly by train/test
print(f"\nAnomaly in train: {(anomaly_segments['train']==1).sum()}")
print(f"Anomaly in test: {(anomaly_segments['train']==0).sum()}")

# Load dataset.csv (segment-level features)
dataset = pd.read_csv(DATA_DIR / "dataset.csv")
print(f"\nShape: {dataset.shape}")
print(f"Columns: {list(dataset.columns)}")
print(f"\nFirst 5 rows:\n{dataset.head()}")

print(f"\nAnomaly distribution:\n{dataset['anomaly'].value_counts()}")
print(f"Train distribution:\n{dataset['train'].value_counts()}")

# Feature columns
feature_cols = [c for c in dataset.columns if c not in ['segment', 'anomaly', 'train', 'channel', 'sampling']]
print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")

# Summary statistics for documentation
print("\n" + "=" * 60)
print("OPSSAT-AD DATASET SUMMARY")
print("=" * 60)
print(f"Raw telemetry rows: {len(segments):,}")
print(f"Unique channels: {segments['channel'].nunique()}")
print(f"Channels: {list(segments['channel'].unique())}")
print(f"Timestamp range: {segments['timestamp'].min()} to {segments['timestamp'].max()}")
print(f"Sampling rate: {segments['sampling'].unique()} Hz")
print(f"\nAnomaly rows: {len(anomaly_segments):,} ({len(anomaly_segments)/len(segments)*100:.2f}%)")
print(f"Anomaly segments: {anomaly_segments['segment'].nunique()}")
print(f"Train rows: {(segments['train']==1).sum():,}")
print(f"Test rows: {(segments['train']==0).sum():,}")
print(f"\nSegment-level dataset: {len(dataset)} segments")
print(f"Segment features: {len(feature_cols)}")
print(f"Features: {feature_cols}")
print(f"\nAnomaly representation: Segment-based (each segment has anomaly label)")
print(f"Labels: Point-level in segments.csv, segment-level in dataset.csv")
print(f"\nCross-channel aggregation: NOT SUPPORTED (single channel: CADC0872)")
print(f"Channel metadata: NOT AVAILABLE (no subsystem/group mapping)")

# Plot 1: Raw telemetry time series (first 5000 points)
plt.figure(figsize=(14, 5))
sample = segments.head(5000)
colors = sample['anomaly'].map({0: 'blue', 1: 'red'})
plt.scatter(sample['timestamp'], sample['value'], c=colors, s=1, alpha=0.6)
plt.xlabel('Timestamp')
plt.ylabel('Value')
plt.title('OPSSAT-AD Raw Telemetry (first 5000 points) - Red = Anomaly')
plt.tight_layout()
plt.savefig('D:/Hackathon/ai-builder-ibm/MissionGuard/artifacts/plot1_raw_telemetry.png', dpi=150)
plt.close()

# Plot 2: Telemetry value distribution by anomaly label
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(nominal_segments['value'], bins=100, alpha=0.7, label='Nominal', density=True)
plt.hist(anomaly_segments['value'], bins=100, alpha=0.7, label='Anomaly', density=True)
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Value Distribution by Label')
plt.legend()
plt.yscale('log')

plt.subplot(1, 2, 2)
plt.boxplot([nominal_segments['value'], anomaly_segments['value']], tick_labels=['Nominal', 'Anomaly'])
plt.ylabel('Value')
plt.title('Value Boxplot by Label')
plt.yscale('symlog')

plt.tight_layout()
plt.savefig('D:/Hackathon/ai-builder-ibm/MissionGuard/artifacts/plot2_value_distribution.png', dpi=150)
plt.close()

# Plot 3: Segment-level feature distributions
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, col in enumerate(feature_cols[:9]):
    ax = axes[i]
    anom_data = dataset[dataset['anomaly']==1][col]
    nom_data = dataset[dataset['anomaly']==0][col]
    ax.hist(nom_data, bins=50, alpha=0.5, label='Nominal', density=True)
    ax.hist(anom_data, bins=50, alpha=0.5, label='Anomaly', density=True)
    ax.set_title(col)
    ax.legend(fontsize=8)
    ax.set_yscale('log')

plt.tight_layout()
plt.savefig('D:/Hackathon/ai-builder-ibm/MissionGuard/artifacts/plot3_feature_distributions.png', dpi=150)
plt.close()

# Plot 4: Anomaly segments over time
plt.figure(figsize=(14, 5))

# Get anomaly segment start/end times
anom_seg_info = anomaly_segments.groupby('segment').agg(
    start=('timestamp', 'min'),
    end=('timestamp', 'max'),
    duration=('timestamp', lambda x: (x.max() - x.min()).total_seconds())
).reset_index()

for _, row in anom_seg_info.iterrows():
    plt.axvspan(row['start'], row['end'], alpha=0.3, color='red')

# Plot nominal background
plt.scatter(segments['timestamp'], segments['value'], c='blue', s=0.5, alpha=0.3, label='Nominal')
plt.scatter(anomaly_segments['timestamp'], anomaly_segments['value'], c='red', s=1, alpha=0.8, label='Anomaly')

plt.xlabel('Timestamp')
plt.ylabel('Value')
plt.title('Telemetry with Anomaly Segments Highlighted')
plt.legend()
plt.tight_layout()
plt.savefig('D:/Hackathon/ai-builder-ibm/MissionGuard/artifacts/plot4_anomaly_segments.png', dpi=150)
plt.close()

# Plot 5: Correlation matrix of segment features
plt.figure(figsize=(12, 10))
corr = dataset[feature_cols].corr()
sns.heatmap(corr, cmap='RdBu_r', center=0, annot=False, fmt='.2f')
plt.title('Segment Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('D:/Hackathon/ai-builder-ibm/MissionGuard/artifacts/plot5_correlation_matrix.png', dpi=150)
plt.close()

print("\nAll plots saved to ../artifacts/")