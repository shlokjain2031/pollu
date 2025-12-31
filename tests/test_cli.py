"""Tests for nalanda/cli.py (CLI and validation)."""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import pytest

from nalanda.cli import (
    get_date_range,
    get_available_target_dates,
    get_available_training_dates,
    assemble_single,
    assemble_batch,
    validate_training_data,
    validate_batch,
    generate_report,
    AssemblyResult,
    ValidationResult,
    DEFAULT_TARGET_DIR,
    DEFAULT_OUTPUT_DIR,
)


# ============================================================================
# Date utilities tests
# ============================================================================

class TestGetDateRange:
    """Tests for date range generation."""
    
    def test_single_day_range(self):
        """Single day returns list with one date."""
        result = get_date_range('2021-06-12', '2021-06-12')
        assert result == ['2021-06-12']
    
    def test_multi_day_range(self):
        """Multi-day range returns correct dates."""
        result = get_date_range('2021-06-10', '2021-06-12')
        assert result == ['2021-06-10', '2021-06-11', '2021-06-12']
    
    def test_week_range_count(self):
        """Week range returns 7 dates."""
        result = get_date_range('2021-06-01', '2021-06-07')
        assert len(result) == 7
    
    def test_dates_are_sorted(self):
        """Returned dates are in chronological order."""
        result = get_date_range('2021-06-01', '2021-06-05')
        assert result == sorted(result)


class TestGetAvailableDates:
    """Tests for available date discovery."""
    
    def test_target_dates_nonempty(self):
        """At least one target date exists."""
        dates = get_available_target_dates()
        assert len(dates) > 0
    
    def test_target_dates_format(self):
        """Target dates are in YYYY-MM-DD format."""
        dates = get_available_target_dates()
        for d in dates:
            assert len(d) == 10
            assert d[4] == '-' and d[7] == '-'
    
    def test_target_dates_sorted(self):
        """Target dates are sorted."""
        dates = get_available_target_dates()
        assert dates == sorted(dates)
    
    def test_training_dates_format(self):
        """Training dates are in YYYY-MM-DD format."""
        dates = get_available_training_dates()
        for d in dates:
            assert len(d) == 10
            assert d[4] == '-' and d[7] == '-'


# ============================================================================
# Assembly tests
# ============================================================================

class TestAssemblySingle:
    """Tests for single-date assembly."""
    
    def test_assemble_returns_result(self):
        """Assembly returns AssemblyResult."""
        result = assemble_single('2021-06-12', dry_run=True)
        assert isinstance(result, AssemblyResult)
    
    def test_assemble_success_has_rows(self):
        """Successful assembly has row count."""
        result = assemble_single('2021-06-12', dry_run=True)
        assert result.success
        assert result.rows > 0
    
    def test_assemble_success_has_columns(self):
        """Successful assembly has column count."""
        result = assemble_single('2021-06-12', dry_run=True)
        assert result.columns > 50  # ~64 expected
    
    def test_assemble_dry_run_no_output(self):
        """Dry run doesn't set output_path."""
        result = assemble_single('2021-06-12', dry_run=True)
        assert result.output_path is None
    
    def test_assemble_invalid_date_fails(self):
        """Invalid date returns failure result."""
        result = assemble_single('1999-01-01', dry_run=True)
        assert not result.success
        assert result.error is not None
    
    def test_assemble_result_to_dict(self):
        """AssemblyResult converts to dict."""
        result = assemble_single('2021-06-12', dry_run=True)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert 'date' in d
        assert 'success' in d


# ============================================================================
# Validation tests
# ============================================================================

class TestValidation:
    """Tests for training data validation."""
    
    def test_validate_existing_date(self):
        """Validating existing training data succeeds."""
        result = validate_training_data('2021-06-12')
        assert isinstance(result, ValidationResult)
        assert result.valid
    
    def test_validate_has_row_count(self):
        """Validation result has row count."""
        result = validate_training_data('2021-06-12')
        assert result.rows > 70000  # ~76k expected
    
    def test_validate_has_column_count(self):
        """Validation result has column count."""
        result = validate_training_data('2021-06-12')
        assert result.columns > 50
    
    def test_validate_has_null_pct(self):
        """Validation result has null percentages."""
        result = validate_training_data('2021-06-12')
        assert isinstance(result.null_pct, dict)
        assert len(result.null_pct) > 0
    
    def test_validate_missing_date_invalid(self):
        """Missing date validation returns invalid."""
        result = validate_training_data('1999-01-01')
        assert not result.valid
        assert len(result.errors) > 0
    
    def test_validate_result_to_dict(self):
        """ValidationResult converts to dict."""
        result = validate_training_data('2021-06-12')
        d = result.to_dict()
        assert isinstance(d, dict)
        assert 'date' in d
        assert 'valid' in d


class TestValidationBatch:
    """Tests for batch validation."""
    
    def test_validate_batch_returns_list(self):
        """Batch validation returns list."""
        results = validate_batch(['2021-06-12'], verbose=False)
        assert isinstance(results, list)
        assert len(results) == 1
    
    def test_validate_batch_all_results(self):
        """Batch validation returns result per date."""
        dates = ['2021-06-12']
        results = validate_batch(dates, verbose=False)
        assert len(results) == len(dates)


class TestAssemblyBatch:
    """Tests for batch assembly (sequential and parallel)."""
    
    def test_assemble_batch_sequential(self):
        """Batch assembly works sequentially."""
        dates = ['2021-06-12']
        results = assemble_batch(dates, dry_run=True, verbose=False, parallel=False)
        assert len(results) == 1
        assert results[0].success
    
    def test_assemble_batch_parallel(self):
        """Batch assembly works in parallel."""
        dates = ['2021-06-12']
        results = assemble_batch(dates, dry_run=True, verbose=False, parallel=True, max_workers=2)
        assert len(results) == 1
        assert results[0].success
    
    def test_parallel_maintains_order(self):
        """Parallel processing returns results sorted by date."""
        dates = ['2021-06-12', '2021-06-11', '2021-06-13']
        results = assemble_batch(dates, dry_run=True, verbose=False, parallel=True)
        result_dates = [r.date for r in results]
        assert result_dates == sorted(result_dates)
    
    def test_parallel_multiple_dates(self):
        """Parallel processing handles multiple dates."""
        dates = ['2021-06-11', '2021-06-12']
        results = assemble_batch(dates, dry_run=True, verbose=False, parallel=True)
        assert len(results) == 2
        assert all(r.success for r in results)
        results = validate_batch(dates, verbose=False)
        assert len(results) == len(dates)


# ============================================================================
# Report tests
# ============================================================================

class TestReport:
    """Tests for pipeline report generation."""
    
    def test_report_returns_dict(self):
        """Report returns dictionary."""
        report = generate_report()
        assert isinstance(report, dict)
    
    def test_report_has_summary(self):
        """Report has summary section."""
        report = generate_report()
        assert 'summary' in report
        assert 'target_dates_available' in report['summary']
    
    def test_report_has_coverage(self):
        """Report has date coverage section."""
        report = generate_report()
        assert 'date_coverage' in report
    
    def test_report_has_validation(self):
        """Report has validation results."""
        report = generate_report()
        assert 'validation_results' in report
        assert isinstance(report['validation_results'], list)
    
    def test_report_has_timestamp(self):
        """Report has generation timestamp."""
        report = generate_report()
        assert 'generated_at' in report
    
    def test_report_is_json_serializable(self):
        """Report can be serialized to JSON."""
        report = generate_report()
        json_str = json.dumps(report)
        assert len(json_str) > 0


# ============================================================================
# Data classes tests
# ============================================================================

class TestDataClasses:
    """Tests for result data classes."""
    
    def test_assembly_result_defaults(self):
        """AssemblyResult has sensible defaults."""
        result = AssemblyResult(date='2021-06-12', success=True)
        assert result.rows == 0
        assert result.error is None
    
    def test_validation_result_defaults(self):
        """ValidationResult has sensible defaults."""
        result = ValidationResult(date='2021-06-12', valid=True)
        assert result.warnings == []
        assert result.errors == []
        assert result.null_pct == {}
    
    def test_validation_result_post_init(self):
        """ValidationResult __post_init__ handles None."""
        result = ValidationResult(
            date='2021-06-12', 
            valid=True,
            null_pct=None,
            warnings=None,
            errors=None,
        )
        assert result.null_pct == {}
        assert result.warnings == []


# ============================================================================
# Integration tests
# ============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_workflow_dry_run(self):
        """Full assemble+validate workflow works in dry-run."""
        # Assemble
        assemble_result = assemble_single('2021-06-12', dry_run=True)
        assert assemble_result.success
        
        # Validate existing data
        validate_result = validate_training_data('2021-06-12')
        assert validate_result.valid
    
    def test_report_consistency(self):
        """Report numbers are internally consistent."""
        report = generate_report()
        
        # Validation results count matches summary
        assert len(report['validation_results']) == report['summary']['training_dates_assembled']
        
        # Valid + invalid equals total
        s = report['summary']
        assert s['valid_training_files'] + s['invalid_training_files'] == s['training_dates_assembled']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
