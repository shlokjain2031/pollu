"""Tests for utility functions."""
import pytest
from datetime import datetime, timezone, timedelta

from takshashika.core.utils import (
    round_to_hour, validate_grid_id_format
)


class TestRoundToHour:
    """Test timestamp rounding semantics."""
    
    def test_round_exact_hour(self):
        """Test rounding at exact hour boundary."""
        dt = datetime(2026, 1, 19, 16, 0, 0, tzinfo=timezone.utc)
        result = round_to_hour(dt)
        assert result == datetime(2026, 1, 19, 16, 0, 0, tzinfo=timezone.utc)
    
    def test_round_partial_hour(self):
        """Test rounding with minutes and seconds."""
        dt = datetime(2026, 1, 19, 16, 47, 30, tzinfo=timezone.utc)
        result = round_to_hour(dt)
        assert result == datetime(2026, 1, 19, 16, 0, 0, tzinfo=timezone.utc)
    
    def test_round_edge_of_hour(self):
        """Test rounding at 59 minutes 59 seconds."""
        dt = datetime(2026, 1, 19, 16, 59, 59, tzinfo=timezone.utc)
        result = round_to_hour(dt)
        assert result == datetime(2026, 1, 19, 16, 0, 0, tzinfo=timezone.utc)
    
    def test_round_midnight(self):
        """Test rounding at midnight."""
        dt = datetime(2026, 1, 19, 23, 59, 59, tzinfo=timezone.utc)
        result = round_to_hour(dt)
        assert result == datetime(2026, 1, 19, 23, 0, 0, tzinfo=timezone.utc)
    
    def test_round_naive_datetime(self):
        """Test rounding naive datetime (assumes UTC)."""
        dt = datetime(2026, 1, 19, 16, 30, 0)  # No timezone
        result = round_to_hour(dt)
        assert result == datetime(2026, 1, 19, 16, 0, 0, tzinfo=timezone.utc)
        assert result.tzinfo == timezone.utc
    
    def test_round_with_microseconds(self):
        """Test rounding preserves microsecond stripping."""
        dt = datetime(2026, 1, 19, 16, 30, 45, 123456, tzinfo=timezone.utc)
        result = round_to_hour(dt)
        assert result == datetime(2026, 1, 19, 16, 0, 0, tzinfo=timezone.utc)
        assert result.microsecond == 0


class TestValidateGridIdFormat:
    """Test grid_id format validation."""
    
    def test_valid_grid_ids(self):
        """Test valid grid_id formats."""
        assert validate_grid_id_format("g00000001") is True
        assert validate_grid_id_format("g00509208") is True
        assert validate_grid_id_format("g12345678") is True
        assert validate_grid_id_format("g00000000") is True
    
    def test_invalid_grid_ids(self):
        """Test invalid grid_id formats."""
        assert validate_grid_id_format("invalid") is False
        assert validate_grid_id_format("g123") is False  # Too short
        assert validate_grid_id_format("g0000000123") is False  # Too long
        assert validate_grid_id_format("G00000001") is False  # Uppercase
        assert validate_grid_id_format("00000001") is False  # No 'g' prefix
        assert validate_grid_id_format("g0000000a") is False  # Non-numeric
    
    def test_invalid_types(self):
        """Test validation with invalid types."""
        assert validate_grid_id_format(123) is False
        assert validate_grid_id_format(None) is False
        assert validate_grid_id_format([]) is False
