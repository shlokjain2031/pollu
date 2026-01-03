# Temporal Split Specification for PM2.5 ML Model

This document defines the **immutable** train/validation/test split strategy for the PM2.5 prediction model. The splits are timestamp-based with mandatory gaps to prevent feature leakage.

## Quick Reference

| Split      | Start Date | End Date   | Duration | Purpose                           |
|------------|------------|------------|----------|-----------------------------------|
| Train      | 2019-01-01 | 2022-12-29 | 1459 days (4 years) | Model training          |
| **Gap 1**  | 2022-12-30 | 2022-12-31 | 2 days   | Leakage control (lag24) |
| Validation | 2023-01-02 | 2023-12-30 | 363 days (1 year)   | Hyperparameter tuning / early stopping |
| **Gap 2**  | 2023-12-31 | 2024-01-02 | 3 days   | Leakage control (lag24) |
| Test       | 2024-01-03 | 2024-12-31 | 364 days (1 year)   | Final evaluation only   |

## Excluded Data

- **2017-2018**: Sparse and inconsistent sensor coverage. Do not use for training.
- **2025**: Partial year (April-September only). Reserved for post-hoc evaluation after model freeze. Not in train/val/test.

## Rationale

### Why these specific dates?

1. **Train (2019-2022)**: 4 full years of stable, gradually increasing data density. Captures seasonal patterns across multiple years.

2. **Validation (2023)**: 1 complete dense year for hyperparameter selection and early stopping. Close enough to test to be representative, but separated by a gap.

3. **Test (2024)**: Latest complete year with highest data density. Closest proxy to deployment conditions.

4. **Gaps (48-72 hours)**: Due to `lag24` features (24-hour lagged pollution values), we need at least 48 hours between splits to prevent data leakage. The gaps ensure that:
   - No training sample's features are computed from validation/test timestamps
   - No validation sample's features are computed from test timestamps

### Why timestamp-based, not file-based?

File boundaries don't respect temporal ordering and can cause leakage if files contain mixed dates. Timestamp-based splitting ensures every sample is assigned to exactly one split based on its temporal position.

## Invariants (Enforced by Code)

These rules are **hard-coded** and will cause assertion failures if violated:

1. ✓ Splits are defined only by timestamps, never by file boundaries
2. ✓ A hard 48h+ gap is mandatory between train→val and val→test
3. ✓ No timestamp may appear in more than one split
4. ✓ Assertions hard-fail if boundaries or overlaps are violated
5. ✓ No hyperparameter tuning or early stopping may touch the test split
6. ✓ Seasonal coverage guarantee:
   - Train: all seasons × 4 years
   - Validation: all seasons × 1 year
   - Test: all seasons × 1 year

## Usage

### Materializing Splits

To physically create train/val/test parquet files from assembled training data:

```bash
# Create splits from training data in artifacts/v1/training/
python -m nalanda.cli create-splits

# Specify custom input/output directories
python -m nalanda.cli create-splits \
  --input-dir artifacts/v1/training \
  --output-dir artifacts

# This creates:
# - artifacts/train/train.parquet
# - artifacts/val/validation.parquet  
# - artifacts/test/test.parquet
# - artifacts/splits_metadata.json (checksums, row counts, date ranges)
```

The materialization process:
1. Loads all date-partitioned parquet files from input directory
2. Validates all timestamps respect split boundaries
3. Filters data by split using timestamp ranges
4. Saves to separate immutable parquet files
5. Records SHA256 checksums, row counts, and date ranges in metadata

### Python API

```python
from nalanda.splits import get_split, validate_splits, SPLIT_RANGES

# Check which split a timestamp belongs to
split = get_split(pd.Timestamp("2023-06-15"))  # Returns 'validation'

# Get split boundaries
train_start, train_end = SPLIT_RANGES["train"]

# Validate a DataFrame
df = pd.read_parquet("data.parquet")
results = validate_splits(df, timestamp_col='timestamp', strict=True)

# Materialize splits programmatically
from nalanda.training import materialize_splits
splits_metadata = materialize_splits(
    input_dir=Path("artifacts/v1/training"),
    output_base_dir=Path("artifacts"),
)
```

### CLI Commands

```bash
# Materialize splits (create physical train/val/test files)
python -m nalanda.cli create-splits

# Print split configuration
python -m nalanda.cli validate-splits --summary

# Validate a parquet file
python -m nalanda.cli validate-splits --file data/train.parquet

# Check train/val/test files for leakage
python -m nalanda.cli validate-splits --splits --output-dir artifacts/
```

### Filtering DataFrames by Split

```python
from nalanda.splits import get_split_mask

# Load your data
df = pd.read_parquet("full_dataset.parquet")

# Split into train/val/test
train_df = df[get_split_mask(df, "train", "timestamp")]
val_df = df[get_split_mask(df, "validation", "timestamp")]
test_df = df[get_split_mask(df, "test", "timestamp")]

# Verify no leakage (raises AssertionError if violated)
from nalanda.splits import assert_no_leakage
assert_no_leakage(train_df, val_df, test_df, timestamp_col="timestamp")
```

## Testing

Unit tests enforce all invariants:

```bash
pytest tests/test_splits.py -v
```

Tests verify:
- No overlap between splits
- 48h+ gaps enforced
- Correct timestamp assignment
- Seasonal coverage (all 12 months in each split)
- Boundary conditions (edge timestamps)
- Leakage detection

## Modification Policy

**This split is final unless the underlying data distribution changes materially.**

If you need to modify the split:

1. Document the reason (data quality issue, distribution shift, etc.)
2. Update `nalanda/splits.py` with new boundaries
3. Update this document
4. Re-run all tests: `pytest tests/test_splits.py`
5. Re-train all models from scratch (old checkpoints are invalid)
6. Update any published results/papers

Do **not** modify splits to improve test metrics. That defeats the purpose of a held-out test set.

## References

- Implementation: [`nalanda/splits.py`](nalanda/splits.py)
- Tests: [`tests/test_splits.py`](tests/test_splits.py)
- CLI: [`nalanda/cli.py`](nalanda/cli.py) (command: `validate-splits`)

---

**Version**: 1.0  
**Last Updated**: 2026-01-03  
**Status**: Immutable (unless data distribution changes)
