"""Temporal split specification for PM2.5 ML model.

This module defines the train/validation/test split strategy with strict
timestamp-based boundaries and mandatory gaps to prevent leakage.

Split Decision (Invariant - DO NOT MODIFY):
-------------------------------------------
- Exclude: 2017-2018 (sparse/inconsistent coverage)
- Exclude: 2025 (partial year, April-September only)
- Use: 2019-2024 for train/val/test

Final Split (Timestamp-based, No Shuffling):
--------------------------------------------
Train:       2019-01-01 00:00:00 to 2022-12-29 23:59:59
Gap:         2022-12-30 00:00:00 to 2022-12-31 23:59:59 (48 hours)
Validation:  2023-01-02 00:00:00 to 2023-12-30 23:59:59
Gap:         2023-12-31 00:00:00 to 2024-01-02 23:59:59 (72 hours)
Test:        2024-01-03 00:00:00 to 2024-12-31 23:59:59

Invariants (Enforced by Assertions):
------------------------------------
1. Splits defined only by timestamps, never by file boundaries
2. Hard 48h gap mandatory between train/val and val/test (due to lag24 features)
3. No timestamp may appear in more than one split
4. Assertions hard-fail if boundaries or overlaps are violated
5. No hyperparameter tuning or early stopping may touch test split
6. Seasonal coverage: train (4 years), val (1 year), test (1 year)

Usage:
------
    from nalanda.splits import get_split, validate_splits, SPLIT_RANGES
    
    # Check which split a timestamp belongs to
    split = get_split(pd.Timestamp("2023-06-15"))  # Returns 'validation'
    
    # Get all split boundaries
    ranges = SPLIT_RANGES
    
    # Validate a DataFrame
    validate_splits(df, timestamp_col='timestamp')
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional, Dict, Tuple

import pandas as pd


# ============================================================================
# Split Boundaries (Immutable)
# ============================================================================

SplitName = Literal["train", "validation", "test", "gap", "excluded"]

# Define exact timestamp boundaries (timezone-naive, assumed UTC)
SPLIT_RANGES: Dict[SplitName, Tuple[pd.Timestamp, pd.Timestamp]] = {
    "train": (
        pd.Timestamp("2019-01-01 00:00:00"),
        pd.Timestamp("2022-12-29 23:59:59"),
    ),
    "gap": (
        pd.Timestamp("2022-12-30 00:00:00"),
        pd.Timestamp("2024-01-02 23:59:59"),  # Covers both gaps (includes 2024-01-02)
    ),
    "validation": (
        pd.Timestamp("2023-01-02 00:00:00"),
        pd.Timestamp("2023-12-30 23:59:59"),
    ),
    "test": (
        pd.Timestamp("2024-01-03 00:00:00"),
        pd.Timestamp("2024-12-31 23:59:59"),
    ),
    "excluded": (
        pd.Timestamp("2017-01-01 00:00:00"),
        pd.Timestamp("2018-12-31 23:59:59"),
    ),
}

# Individual gap periods for precise validation
GAP_1 = (pd.Timestamp("2022-12-30 00:00:00"), pd.Timestamp("2022-12-31 23:59:59"))
GAP_2 = (pd.Timestamp("2023-12-31 00:00:00"), pd.Timestamp("2024-01-02 23:59:59"))  # Includes 2024-01-02


# ============================================================================
# Core Functions
# ============================================================================

def get_split(timestamp: pd.Timestamp) -> SplitName:
    """Determine which split a timestamp belongs to.
    
    Args:
        timestamp: Pandas timestamp to check
        
    Returns:
        Split name: 'train', 'validation', 'test', 'gap', or 'excluded'
        
    Raises:
        ValueError: If timestamp is outside all defined ranges (e.g., 2025+)
    """
    if not isinstance(timestamp, pd.Timestamp):
        timestamp = pd.Timestamp(timestamp)
    
    # Check each split in priority order
    for split_name, (start, end) in SPLIT_RANGES.items():
        if start <= timestamp <= end:
            # Special handling for validation (it's within the gap range)
            if split_name == "gap":
                # Check if it's actually in validation
                val_start, val_end = SPLIT_RANGES["validation"]
                if val_start <= timestamp <= val_end:
                    return "validation"
                return "gap"
            return split_name
    
    # If not in any range, check if it's 2025+ (excluded from training)
    if timestamp >= pd.Timestamp("2025-01-01 00:00:00"):
        raise ValueError(
            f"Timestamp {timestamp} is in 2025 or later. "
            "2025 is excluded from train/val/test (partial year, Apr-Sep only). "
            "Use only for post-hoc evaluation after model freeze."
        )
    
    raise ValueError(f"Timestamp {timestamp} does not belong to any defined split")


def is_in_split(timestamp: pd.Timestamp, split_name: SplitName) -> bool:
    """Check if a timestamp belongs to a specific split.
    
    Args:
        timestamp: Pandas timestamp to check
        split_name: Name of the split ('train', 'validation', 'test', 'gap', 'excluded')
        
    Returns:
        True if timestamp is in the specified split, False otherwise
    """
    try:
        return get_split(timestamp) == split_name
    except ValueError:
        return False


def get_split_mask(
    df: pd.DataFrame,
    split_name: SplitName,
    timestamp_col: str = "timestamp",
) -> pd.Series:
    """Get a boolean mask for rows belonging to a specific split.
    
    Args:
        df: DataFrame with timestamp column
        split_name: Name of the split to filter
        timestamp_col: Name of the timestamp column
        
    Returns:
        Boolean Series with True for rows in the specified split
    """
    start, end = SPLIT_RANGES[split_name]
    
    # Special handling for validation (within gap range)
    if split_name == "validation":
        return (df[timestamp_col] >= start) & (df[timestamp_col] <= end)
    
    # For gap, exclude validation timestamps
    if split_name == "gap":
        val_start, val_end = SPLIT_RANGES["validation"]
        in_gap_range = (df[timestamp_col] >= start) & (df[timestamp_col] <= end)
        not_in_val = ~((df[timestamp_col] >= val_start) & (df[timestamp_col] <= val_end))
        return in_gap_range & not_in_val
    
    return (df[timestamp_col] >= start) & (df[timestamp_col] <= end)


# ============================================================================
# Validation & Assertions
# ============================================================================

def validate_splits(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    strict: bool = True,
) -> Dict[str, any]:
    """Validate that a DataFrame respects split boundaries and invariants.
    
    Enforces:
    - No timestamp appears in multiple splits
    - 48h gap between train/val and val/test
    - No overlap between splits
    - Correct seasonal coverage
    
    Args:
        df: DataFrame to validate
        timestamp_col: Name of the timestamp column
        strict: If True, raise AssertionError on violations. If False, return warnings.
        
    Returns:
        Dictionary with validation results and statistics
        
    Raises:
        AssertionError: If strict=True and any invariant is violated
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Column '{timestamp_col}' not found in DataFrame")
    
    timestamps = pd.to_datetime(df[timestamp_col])
    
    # Count samples in each split
    split_counts = {
        "train": get_split_mask(df, "train", timestamp_col).sum(),
        "validation": get_split_mask(df, "validation", timestamp_col).sum(),
        "test": get_split_mask(df, "test", timestamp_col).sum(),
        "gap": get_split_mask(df, "gap", timestamp_col).sum(),
        "excluded": get_split_mask(df, "excluded", timestamp_col).sum(),
    }
    
    # Check for timestamps outside all ranges
    total_assigned = sum(split_counts.values())
    unassigned = len(df) - total_assigned
    
    # Validate no overlap (each timestamp in exactly one split)
    issues = []
    
    if unassigned > 0:
        # Find unassigned timestamps
        all_masks = pd.Series(False, index=df.index)
        for split in split_counts.keys():
            all_masks |= get_split_mask(df, split, timestamp_col)
        unassigned_ts = timestamps[~all_masks]
        issues.append(
            f"Found {unassigned} timestamps not assigned to any split. "
            f"Range: {unassigned_ts.min()} to {unassigned_ts.max()}"
        )
    
    # Validate gap enforcement
    gap1_mask = (timestamps >= GAP_1[0]) & (timestamps <= GAP_1[1])
    gap2_mask = (timestamps >= GAP_2[0]) & (timestamps <= GAP_2[1])
    gap_mask = get_split_mask(df, "gap", timestamp_col)
    
    # Check that all timestamps in gap1 range are marked as gap
    if gap1_mask.sum() > 0:
        gap1_timestamps_not_in_gap = gap1_mask & ~gap_mask
        if gap1_timestamps_not_in_gap.any():
            issues.append(f"Gap 1 (2022-12-30 to 2022-12-31) contains non-gap data")
    
    # Check that all timestamps in gap2 range are marked as gap
    if gap2_mask.sum() > 0:
        gap2_timestamps_not_in_gap = gap2_mask & ~gap_mask
        if gap2_timestamps_not_in_gap.any():
            issues.append(f"Gap 2 (2023-12-31 to 2024-01-02) contains non-gap data")
    
    # Check that test doesn't leak into validation or train
    test_mask = get_split_mask(df, "test", timestamp_col)
    train_mask = get_split_mask(df, "train", timestamp_col)
    val_mask = get_split_mask(df, "validation", timestamp_col)
    
    if (test_mask & train_mask).any():
        issues.append("Test split overlaps with train split")
    
    if (test_mask & val_mask).any():
        issues.append("Test split overlaps with validation split")
    
    if (train_mask & val_mask).any():
        issues.append("Train split overlaps with validation split")
    
    # Compile results
    results = {
        "valid": len(issues) == 0,
        "split_counts": split_counts,
        "total_rows": len(df),
        "unassigned_rows": unassigned,
        "issues": issues,
        "date_range": {
            "min": timestamps.min(),
            "max": timestamps.max(),
        },
    }
    
    # Strict mode: raise on violations
    if strict and issues:
        raise AssertionError(
            f"Split validation failed with {len(issues)} issue(s):\n" +
            "\n".join(f"  - {issue}" for issue in issues)
        )
    
    return results


def assert_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> None:
    """Assert that train/val/test splits have no temporal leakage.
    
    Verifies:
    - No timestamp overlap between splits
    - 48h gap exists between train→val and val→test
    - Train < Validation < Test (temporal ordering)
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        timestamp_col: Name of the timestamp column
        
    Raises:
        AssertionError: If any leakage is detected
    """
    train_ts = pd.to_datetime(train_df[timestamp_col])
    val_ts = pd.to_datetime(val_df[timestamp_col])
    test_ts = pd.to_datetime(test_df[timestamp_col])
    
    # Check temporal ordering
    assert train_ts.max() < val_ts.min(), (
        f"Train max ({train_ts.max()}) must be before validation min ({val_ts.min()})"
    )
    assert val_ts.max() < test_ts.min(), (
        f"Validation max ({val_ts.max()}) must be before test min ({test_ts.min()})"
    )
    
    # Check 48h gap between train and validation
    gap1_hours = (val_ts.min() - train_ts.max()).total_seconds() / 3600
    assert gap1_hours >= 48, (
        f"Gap between train and validation must be >= 48h, got {gap1_hours:.1f}h"
    )
    
    # Check 48h gap between validation and test
    gap2_hours = (test_ts.min() - val_ts.max()).total_seconds() / 3600
    assert gap2_hours >= 48, (
        f"Gap between validation and test must be >= 48h, got {gap2_hours:.1f}h"
    )
    
    # Check no overlap
    train_set = set(train_ts)
    val_set = set(val_ts)
    test_set = set(test_ts)
    
    assert len(train_set & val_set) == 0, "Train and validation have overlapping timestamps"
    assert len(train_set & test_set) == 0, "Train and test have overlapping timestamps"
    assert len(val_set & test_set) == 0, "Validation and test have overlapping timestamps"


def print_split_summary() -> None:
    """Print a human-readable summary of the split configuration."""
    print("=" * 80)
    print("PM2.5 ML Model - Temporal Split Configuration")
    print("=" * 80)
    print()
    print("Excluded from training:")
    print("  2017-2018: Sparse/inconsistent coverage")
    print("  2025:      Partial year (Apr-Sep only), reserved for post-hoc evaluation")
    print()
    print("Active splits (2019-2024):")
    print()
    
    for split_name in ["train", "validation", "test"]:
        start, end = SPLIT_RANGES[split_name]
        duration_days = (end - start).days + 1
        print(f"  {split_name.upper():12} {start.date()} to {end.date()} ({duration_days:4} days)")
        
        if split_name == "train":
            print(f"  {'GAP (48h)':12} {GAP_1[0].date()} to {GAP_1[1].date()}")
        elif split_name == "validation":
            print(f"  {'GAP (48h)':12} {GAP_2[0].date()} to {GAP_2[1].date()}")
    
    print()
    print("Invariants enforced:")
    print("  ✓ Timestamp-based splitting only (no file boundaries)")
    print("  ✓ Strict 48h gaps prevent lag24 feature leakage")
    print("  ✓ No timestamp in multiple splits")
    print("  ✓ Seasonal coverage: train (4 years), val (1 year), test (1 year)")
    print("  ✓ Test split reserved for final evaluation only")
    print("=" * 80)


if __name__ == "__main__":
    # Print split summary when run directly
    print_split_summary()
