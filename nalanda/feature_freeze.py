"""Frozen feature schema: persistence, validation, and enforcement.

This module implements the model-feature contract. The contract states:

    The model assumes that:
    1. Input is a vector of length N
    2. Element i corresponds to feature_list[i]
    3. Values have the same meaning and scale as during training

If any of these are violated, model output is UNDEFINED.

The frozen feature list is not configuration, metadata, or documentation.
It is the INTERFACE DEFINITION between data, model, inference, and evaluation.

LightGBM implements f: ℝⁿ → ℝ
This module defines what ℝⁿ actually means.

Usage:
    # Freeze (once per model version)
    from nalanda.feature_freeze import freeze_feature_schema
    schema = freeze_feature_schema()
    schema.save("artifacts/v1/models/model_v1/feature_schema.json")
    
    # Load and validate (at training/inference time)
    from nalanda.feature_freeze import FrozenFeatureSchema
    schema = FrozenFeatureSchema.load("artifacts/v1/models/model_v1/feature_schema.json")
    schema.validate(df)  # raises if mismatch
    X = df[schema.feature_list]  # safe to use
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd


@dataclass
class FrozenFeatureSchema:
    """Immutable feature schema with validation and persistence.
    
    This is the contract between data and model. Once frozen, it must
    not change for a given model version.
    
    Attributes:
        feature_list: Ordered list of feature names. Index i = feature_i in LightGBM.
        schema_hash: SHA256 of the feature list (for integrity verification).
        created_at: ISO timestamp when schema was frozen.
        metadata: Optional dict for version info, notes, etc.
    """
    
    feature_list: List[str]
    schema_hash: str
    created_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate schema integrity after creation."""
        computed_hash = self._compute_hash(self.feature_list)
        if self.schema_hash != computed_hash:
            raise ValueError(
                f"Schema hash mismatch. Expected {self.schema_hash}, "
                f"computed {computed_hash}. Schema may be corrupted."
            )
    
    @staticmethod
    def _compute_hash(feature_list: List[str]) -> str:
        """Compute SHA256 hash of feature list.
        
        Hash is computed over the JSON-serialized list to ensure
        deterministic hashing across Python versions.
        """
        content = json.dumps(feature_list, sort_keys=False)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    @property
    def n_features(self) -> int:
        """Number of features in schema."""
        return len(self.feature_list)
    
    @property
    def feature_set(self) -> Set[str]:
        """Feature names as a set (for O(1) membership checks)."""
        return set(self.feature_list)
    
    def validate(
        self,
        df: pd.DataFrame,
        strict: bool = True,
    ) -> None:
        """Validate DataFrame columns against frozen schema.
        
        Args:
            df: DataFrame to validate.
            strict: If True, extra columns cause an error. If False, only
                    missing columns cause an error.
        
        Raises:
            ValueError: If validation fails.
        """
        df_cols = set(df.columns)
        schema_cols = self.feature_set
        
        missing = schema_cols - df_cols
        extra = df_cols - schema_cols
        
        errors = []
        
        if missing:
            errors.append(f"Missing features: {sorted(missing)}")
        
        if strict and extra:
            # Only flag extra columns that aren't explicitly excluded
            from nalanda.features import EXCLUDED_COLUMNS
            unexpected = extra - set(EXCLUDED_COLUMNS)
            if unexpected:
                errors.append(f"Unexpected features: {sorted(unexpected)}")
        
        if errors:
            raise ValueError(
                f"Schema validation failed:\n" + "\n".join(errors)
            )
    
    def reorder(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reorder DataFrame columns to match frozen schema order.
        
        Args:
            df: DataFrame with (at least) all schema columns.
        
        Returns:
            DataFrame with columns in exact schema order.
        
        Raises:
            ValueError: If required columns are missing.
        """
        missing = self.feature_set - set(df.columns)
        if missing:
            raise ValueError(f"Cannot reorder: missing columns {sorted(missing)}")
        
        return df[self.feature_list]
    
    def extract_X(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract feature matrix in exact schema order.
        
        This is the canonical way to go from raw DataFrame to model input.
        
        Args:
            df: DataFrame with features.
        
        Returns:
            DataFrame with only schema features in exact order.
        """
        self.validate(df, strict=False)
        return self.reorder(df)
    
    def save(self, path: str | Path) -> None:
        """Persist schema to JSON file.
        
        Args:
            path: Output path for schema JSON.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "feature_list": self.feature_list,
            "schema_hash": self.schema_hash,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "n_features": self.n_features,
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str | Path) -> "FrozenFeatureSchema":
        """Load schema from JSON file.
        
        Args:
            path: Path to schema JSON.
        
        Returns:
            FrozenFeatureSchema instance.
        
        Raises:
            FileNotFoundError: If schema file doesn't exist.
            ValueError: If schema hash doesn't match (corrupted file).
        """
        path = Path(path)
        
        with open(path) as f:
            data = json.load(f)
        
        return cls(
            feature_list=data["feature_list"],
            schema_hash=data["schema_hash"],
            created_at=data["created_at"],
            metadata=data.get("metadata", {}),
        )
    
    def __eq__(self, other: object) -> bool:
        """Two schemas are equal if their hashes match."""
        if not isinstance(other, FrozenFeatureSchema):
            return False
        return self.schema_hash == other.schema_hash
    
    def diff(self, other: "FrozenFeatureSchema") -> Dict[str, Any]:
        """Compute difference between two schemas.
        
        Args:
            other: Another schema to compare against.
        
        Returns:
            Dict with 'added', 'removed', 'reordered' keys.
        """
        self_set = self.feature_set
        other_set = other.feature_set
        
        added = other_set - self_set
        removed = self_set - other_set
        
        # Check for reordering (same features, different order)
        common = self_set & other_set
        self_order = [f for f in self.feature_list if f in common]
        other_order = [f for f in other.feature_list if f in common]
        reordered = self_order != other_order
        
        return {
            "added": sorted(added),
            "removed": sorted(removed),
            "reordered": reordered,
            "hash_changed": self.schema_hash != other.schema_hash,
        }


def freeze_feature_schema(
    metadata: Optional[Dict[str, Any]] = None,
    include_landsat: bool = True,
    include_sentinel: bool = True,
) -> FrozenFeatureSchema:
    """Create a frozen feature schema from the canonical definition.
    
    This should be called ONCE per model version, and the result
    persisted alongside the model.
    
    Args:
        metadata: Optional metadata to include (version, notes, etc.).
        include_landsat: Include Landsat features.
        include_sentinel: Include Sentinel features.
    
    Returns:
        FrozenFeatureSchema instance.
    """
    from nalanda.features import get_feature_list
    
    feature_list = get_feature_list(
        include_landsat=include_landsat,
        include_sentinel=include_sentinel,
    )
    
    schema_hash = FrozenFeatureSchema._compute_hash(feature_list)
    created_at = datetime.utcnow().isoformat() + "Z"
    
    return FrozenFeatureSchema(
        feature_list=feature_list,
        schema_hash=schema_hash,
        created_at=created_at,
        metadata=metadata or {},
    )


def freeze_from_dataframe(
    df: pd.DataFrame,
    metadata: Optional[Dict[str, Any]] = None,
) -> FrozenFeatureSchema:
    """Create a frozen schema from an existing DataFrame's columns.
    
    Use this when you want to freeze the schema based on what columns
    actually exist in a representative training file.
    
    WARNING: This trusts the DataFrame to have the correct columns.
    Prefer freeze_feature_schema() which uses the canonical definition.
    
    Args:
        df: Representative DataFrame.
        metadata: Optional metadata.
    
    Returns:
        FrozenFeatureSchema instance.
    """
    from nalanda.features import EXCLUDED_COLUMNS
    
    # Filter out excluded columns
    feature_list = [c for c in df.columns if c not in EXCLUDED_COLUMNS]
    
    schema_hash = FrozenFeatureSchema._compute_hash(feature_list)
    created_at = datetime.utcnow().isoformat() + "Z"
    
    meta = metadata or {}
    meta["frozen_from"] = "dataframe"
    meta["warning"] = "Schema derived from data, not definition. May include unexpected columns."
    
    return FrozenFeatureSchema(
        feature_list=feature_list,
        schema_hash=schema_hash,
        created_at=created_at,
        metadata=meta,
    )


def validate_splits_schema(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema: Optional[FrozenFeatureSchema] = None,
) -> None:
    """Validate that train/val/test splits have consistent schemas.
    
    This is a critical check: if schemas differ, LightGBM will still run,
    but the numbers will be meaningless.
    
    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        test_df: Test DataFrame.
        schema: Optional frozen schema to validate against. If None,
                validates that all three have the same columns.
    
    Raises:
        ValueError: If schemas are inconsistent.
    """
    train_cols = set(train_df.columns)
    val_cols = set(val_df.columns)
    test_cols = set(test_df.columns)
    
    errors = []
    
    # Check pairwise consistency
    if train_cols != val_cols:
        in_train = train_cols - val_cols
        in_val = val_cols - train_cols
        if in_train:
            errors.append(f"In train but not val: {sorted(in_train)}")
        if in_val:
            errors.append(f"In val but not train: {sorted(in_val)}")
    
    if train_cols != test_cols:
        in_train = train_cols - test_cols
        in_test = test_cols - train_cols
        if in_train:
            errors.append(f"In train but not test: {sorted(in_train)}")
        if in_test:
            errors.append(f"In test but not train: {sorted(in_test)}")
    
    # Check against frozen schema if provided
    if schema is not None:
        try:
            schema.validate(train_df, strict=False)
        except ValueError as e:
            errors.append(f"Train vs schema: {e}")
        
        try:
            schema.validate(val_df, strict=False)
        except ValueError as e:
            errors.append(f"Val vs schema: {e}")
        
        try:
            schema.validate(test_df, strict=False)
        except ValueError as e:
            errors.append(f"Test vs schema: {e}")
    
    if errors:
        raise ValueError(
            "Split schema validation failed:\n" + "\n".join(errors)
        )
