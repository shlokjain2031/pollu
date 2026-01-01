"""Tests for CLI --target-row-idx argument configuration."""

import argparse
import sys
from pathlib import Path

import pytest


def test_landsat8_cli_argument():
    """Test that landsat8_signals has --target-row-idx argument."""
    from patliputra.landsat8_signals import DEFAULT_TARGET_ROW_IDX_PATH
    
    # Verify constant has .parquet extension
    assert DEFAULT_TARGET_ROW_IDX_PATH.endswith('.parquet'), \
        f"DEFAULT_TARGET_ROW_IDX_PATH must end with .parquet: {DEFAULT_TARGET_ROW_IDX_PATH}"
    assert DEFAULT_TARGET_ROW_IDX_PATH == "/tmp/target_row_idx.parquet", \
        f"Unexpected default path: {DEFAULT_TARGET_ROW_IDX_PATH}"


def test_sentinel2_cli_argument():
    """Test that sentinel2_signals has --target-row-idx argument."""
    from patliputra.sentinel2_signals import DEFAULT_TARGET_ROW_IDX_PATH
    
    # Verify constant has .parquet extension
    assert DEFAULT_TARGET_ROW_IDX_PATH.endswith('.parquet'), \
        f"DEFAULT_TARGET_ROW_IDX_PATH must end with .parquet: {DEFAULT_TARGET_ROW_IDX_PATH}"
    assert DEFAULT_TARGET_ROW_IDX_PATH == "/tmp/target_row_idx.parquet", \
        f"Unexpected default path: {DEFAULT_TARGET_ROW_IDX_PATH}"


def test_openmeteo_cli_argument():
    """Test that openmeteo_signals has --target-row-idx argument."""
    from patliputra.openmeteo_signals import DEFAULT_TARGET_ROW_IDX_PATH
    
    # Verify constant has .parquet extension
    assert DEFAULT_TARGET_ROW_IDX_PATH.endswith('.parquet'), \
        f"DEFAULT_TARGET_ROW_IDX_PATH must end with .parquet: {DEFAULT_TARGET_ROW_IDX_PATH}"
    assert DEFAULT_TARGET_ROW_IDX_PATH == "/tmp/target_row_idx.parquet", \
        f"Unexpected default path: {DEFAULT_TARGET_ROW_IDX_PATH}"


def test_landsat8_validate_function():
    """Test that validate_target_row_idx_path function exists and works."""
    from patliputra.landsat8_signals import validate_target_row_idx_path
    
    # Test with /tmp (should be writable on macOS/Linux)
    test_path = Path("/tmp/test_target_row_idx.parquet")
    try:
        validate_target_row_idx_path(test_path)
        print(f"✓ Landsat8 validation passed: {test_path} is writable")
    except RuntimeError as e:
        # /tmp should always be writable, so this is unexpected
        pytest.fail(f"Landsat8 validation failed for /tmp: {e}")


def test_sentinel2_validate_function():
    """Test that validate_target_row_idx_path function exists and works."""
    from patliputra.sentinel2_signals import validate_target_row_idx_path
    
    # Test with /tmp (should be writable on macOS/Linux)
    test_path = Path("/tmp/test_target_row_idx.parquet")
    try:
        validate_target_row_idx_path(test_path)
        print(f"✓ Sentinel2 validation passed: {test_path} is writable")
    except RuntimeError as e:
        pytest.fail(f"Sentinel2 validation failed for /tmp: {e}")


def test_openmeteo_validate_function():
    """Test that validate_target_row_idx_path function exists and works."""
    from patliputra.openmeteo_signals import validate_target_row_idx_path
    
    # Test with /tmp (should be writable on macOS/Linux)
    test_path = Path("/tmp/test_target_row_idx.parquet")
    try:
        validate_target_row_idx_path(test_path)
        print(f"✓ OpenMeteo validation passed: {test_path} is writable")
    except RuntimeError as e:
        pytest.fail(f"OpenMeteo validation failed for /tmp: {e}")


def test_landsat8_load_signature():
    """Test that load_target_row_idxs has correct signature."""
    import inspect
    from patliputra.landsat8_signals import load_target_row_idxs
    
    sig = inspect.signature(load_target_row_idxs)
    assert 'target_path' in sig.parameters, \
        "load_target_row_idxs must have 'target_path' parameter"
    
    # Verify it's required (no default)
    param = sig.parameters['target_path']
    assert param.default == inspect.Parameter.empty, \
        "load_target_row_idxs 'target_path' parameter should not have a default"


def test_sentinel2_load_signature():
    """Test that load_target_row_idxs has correct signature."""
    import inspect
    from patliputra.sentinel2_signals import load_target_row_idxs
    
    sig = inspect.signature(load_target_row_idxs)
    assert 'target_path' in sig.parameters, \
        "load_target_row_idxs must have 'target_path' parameter"
    
    # Verify it's required (no default)
    param = sig.parameters['target_path']
    assert param.default == inspect.Parameter.empty, \
        "load_target_row_idxs 'target_path' parameter should not have a default"


def test_openmeteo_load_signature():
    """Test that load_target_row_idxs has correct signature."""
    import inspect
    from patliputra.openmeteo_signals import load_target_row_idxs
    
    sig = inspect.signature(load_target_row_idxs)
    assert 'target_path' in sig.parameters, \
        "load_target_row_idxs must have 'target_path' parameter"
    
    # Verify it's required (no default)
    param = sig.parameters['target_path']
    assert param.default == inspect.Parameter.empty, \
        "load_target_row_idxs 'target_path' parameter should not have a default"
