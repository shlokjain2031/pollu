"""Utility functions for Takshashila inference system.

Timestamp normalization, logging setup, and helper functions.
"""
from datetime import datetime, timezone
import logging
from .config import LOG_LEVEL, LOG_FORMAT, LOG_DATEFMT


def round_to_hour(dt: datetime) -> datetime:
    """Floor timestamp to nearest hour boundary (UTC).
    
    Implements time semantics per SYSTEM_CONTRACTS.md §2.
    
    Args:
        dt: Input datetime (any timezone)
    
    Returns:
        Datetime floored to hour, in UTC
    
    Examples:
        >>> round_to_hour(datetime(2026, 1, 19, 16, 47, 30))
        datetime(2026, 1, 19, 16, 0, 0, tzinfo=timezone.utc)
        
        >>> round_to_hour(datetime(2026, 1, 19, 16, 0, 0))
        datetime(2026, 1, 19, 16, 0, 0, tzinfo=timezone.utc)
    """
    # Ensure UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    
    # Floor to hour
    return dt.replace(minute=0, second=0, microsecond=0)


def setup_logging(level: str = LOG_LEVEL) -> None:
    """Configure Python stdlib logging with structured output.
    
    Uses simple format for Phase 1; can be extended to structlog later.
    
    Args:
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=numeric_level,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        force=True  # Reconfigure if already set
    )
    
    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("duckdb").setLevel(logging.WARNING)


def validate_grid_id_format(grid_id: str) -> bool:
    """Validate grid_id matches expected format: g########
    
    Args:
        grid_id: Grid identifier (e.g., "g00000001")
    
    Returns:
        True if valid format, False otherwise
    
    Examples:
        >>> validate_grid_id_format("g00000001")
        True
        >>> validate_grid_id_format("g00509208")
        True
        >>> validate_grid_id_format("invalid")
        False
        >>> validate_grid_id_format("g123")
        False
    """
    if not isinstance(grid_id, str):
        return False
    
    if len(grid_id) != 9:
        return False
    
    if not grid_id.startswith("g"):
        return False
    
    try:
        int(grid_id[1:])
        return True
    except ValueError:
        return False
