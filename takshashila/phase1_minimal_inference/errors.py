"""Error definitions for Takshashila inference system.

All errors are explicit and carry structured metadata for observability.
Implements fail-loud semantics per SYSTEM_CONTRACTS.md §5.
"""


class InferenceError(Exception):
    """Base class for all inference errors."""
    pass


class FeatureAssemblyError(InferenceError):
    """Raised when feature assembly fails.
    
    Triggers:
    - Required feature is missing from DuckDB
    - Feature value is NULL when non-NULL expected
    - Feature table row does not exist for (grid_id, timestamp_hour)
    
    Attributes:
        grid_id: Grid cell identifier
        timestamp_hour: Prediction timestamp (hour-floored)
        missing_feature: Name of the feature that failed
    """
    
    def __init__(self, grid_id: str, timestamp_hour, missing_feature: str):
        self.grid_id = grid_id
        self.timestamp_hour = timestamp_hour
        self.missing_feature = missing_feature
        super().__init__(
            f"Feature '{missing_feature}' missing for grid_id={grid_id} "
            f"at timestamp={timestamp_hour}"
        )


class ModelInferenceError(InferenceError):
    """Raised when model prediction fails.
    
    Triggers:
    - Model.predict() raises exception
    - Model returns invalid output (NaN, negative, etc.)
    - Model input validation fails
    
    Attributes:
        grid_id: Grid cell identifier
        timestamp_hour: Prediction timestamp
        reason: Human-readable failure reason
    """
    
    def __init__(self, grid_id: str, timestamp_hour, reason: str):
        self.grid_id = grid_id
        self.timestamp_hour = timestamp_hour
        self.reason = reason
        super().__init__(
            f"Model inference failed for grid_id={grid_id} at timestamp={timestamp_hour}: "
            f"{reason}"
        )


class PersistenceError(InferenceError):
    """Raised on DuckDB read/write failure.
    
    Triggers:
    - DuckDB connection failure
    - Write constraint violation
    - Read-back validation failure
    - Schema mismatch
    
    Attributes:
        operation: "read" or "write"
        grid_id: Grid cell identifier
        timestamp_hour: Prediction timestamp
        reason: Human-readable failure reason
    """
    
    def __init__(self, operation: str, grid_id: str, timestamp_hour, reason: str):
        self.operation = operation
        self.grid_id = grid_id
        self.timestamp_hour = timestamp_hour
        self.reason = reason
        super().__init__(
            f"Persistence {operation} failed for grid_id={grid_id} "
            f"at timestamp={timestamp_hour}: {reason}"
        )


class TimeoutError(InferenceError):
    """Raised when computation exceeds timeout.
    
    Triggers:
    - Feature assembly exceeds budget
    - Model inference exceeds budget
    - DuckDB write exceeds budget
    
    Note: Phase 1 does not enforce timeouts yet; this is defined for forward compatibility.
    
    Attributes:
        grid_id: Grid cell identifier
        timestamp_hour: Prediction timestamp
        timeout_sec: Timeout threshold that was exceeded
    """
    
    def __init__(self, grid_id: str, timestamp_hour, timeout_sec: float):
        self.grid_id = grid_id
        self.timestamp_hour = timestamp_hour
        self.timeout_sec = timeout_sec
        super().__init__(
            f"Timeout ({timeout_sec}s) exceeded for grid_id={grid_id} "
            f"at timestamp={timestamp_hour}"
        )


class InvalidRequestError(InferenceError):
    """Raised for invalid grid_id, timestamp, or request parameters.
    
    Triggers:
    - grid_id not in grids table
    - Timestamp format invalid
    - Required parameter missing
    
    This is NOT an internal error; it indicates bad client input.
    
    Attributes:
        reason: Human-readable explanation of invalidity
    """
    
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Invalid request: {reason}")
