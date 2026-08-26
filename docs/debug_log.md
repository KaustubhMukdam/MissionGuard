# Debug Log — MissionGuard

> Record non-trivial bugs here. Do not record every typo.

---

## Template

**Date:** YYYY-MM-DD
**Project:** MissionGuard
**Bug:** [short title]

**Error message:**
```text
[paste exact error]
```

**Root cause:** [precise cause]

**Fix:**
```python
# Before
...

# After
...
```

**Time lost:** [minutes]

**How I found it:** [debugging process/tool]

**Pattern to remember:** [generalizable lesson]

---

## Initial project note

No implementation bugs logged yet.

The first technical debugging session should start only after the ESA-ADB data-validation milestone is complete.

---

**Date:** 2026-08-25
**Project:** MissionGuard
**Bug:** `KeyError: 'channel'` in data bridge event extraction

**Error message:**
```text
E   KeyError: 'channel'
src\missionguard\detection\events.py:276: in get_events_per_channel
    for channel in segments_df[channel_col].unique():
```

**Root cause:** In `app/data_bridge.py`, the per-segment windows groupby aggregated `channel=("channel","first")` from segments.csv, then merged with test_ds (from dataset.csv) which *also* has a `channel` column. Pandas merge suffixes overlapping non-key columns, so the merged frame had `channel_x`/`channel_y` and no plain `channel`.

**Fix:**
```python
# Before
windows = segments.groupby("segment").agg(
    timestamp=("timestamp", "min"), end=("timestamp", "max"), channel=("channel", "first"),
)

# After — dataset.csv already carries `channel`; don't duplicate it in the merge
windows = segments.groupby("segment").agg(
    timestamp=("timestamp", "min"), end=("timestamp", "max"),
)
```

**Time lost:** 10 min (one red test run)

**How I found it:** pytest traceback pointed at the merge output missing the column; checked dataset.csv header to confirm it already has `channel`.

**Pattern to remember:** When merging two frames that share metadata column names, aggregate only what one side uniquely contributes, or rename explicitly before merging.

---

**Date:** 2026-08-25
**Project:** MissionGuard
**Bug:** sklearn raises on inf features during scaling — finite-mask ran too late

**Error message:**
```text
ValueError: Input X contains infinity or a value too large for dtype('float64').
  (RobustScaler.transform → check_array → _assert_all_finite)
```

**Root cause:** The bridge dropped non-finite rows AFTER calling `transform_features()`. But `RobustScaler.transform` validates finiteness and raises before returning. Also: `validate_dataset_df` treats inf as a *warning*, not an error — so inf rows pass loading and reach the scaler.

**Fix:**
```python
# Before — mask applied post-transform, never reached
transformed = transform_features(scored, scaler, feature_names)
finite_mask = np.isfinite(transformed[feature_names]...).all(axis=1)

# After — mask raw features BEFORE any sklearn call
finite_mask = np.isfinite(scored[feature_names].to_numpy(dtype=float)).all(axis=1)
scored = scored[finite_mask].reset_index(drop=True)
transformed = transform_features(scored, scaler, feature_names)
```

**Time lost:** 5 min

**How I found it:** Red test `test_nonfinite_features_dropped_not_fatal` failed with the sklearn ValueError instead of passing gracefully.

**Pattern to remember:** Validate/drop bad rows at the earliest boundary of the pipeline, not after intermediate transforms that themselves validate. Schema validators that only *warn* about inf/NaN are not protection — assume such rows reach your model code.
