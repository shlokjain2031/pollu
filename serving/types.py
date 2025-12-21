from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Literal, Dict, Any


class BadRequest(ValueError):
    pass


@dataclass(frozen=True)
class CanonicalIdentity:
    artifact_type: str
    spatial_id: str
    timestamp: str
    version: str

    @classmethod
    def from_mapping(cls, mapping: Dict[str, Any]) -> "CanonicalIdentity":
        required = ["artifact_type", "spatial_id", "timestamp", "version"]
        missing = [k for k in required if not mapping.get(k)]
        if missing:
            raise BadRequest(f"Missing required identity fields: {missing}")
        # simple sanitization: ensure strings
        return cls(
            artifact_type=str(mapping["artifact_type"]),
            spatial_id=str(mapping["spatial_id"]),
            timestamp=str(mapping["timestamp"]),
            version=str(mapping["version"]),
        )


@dataclass
class ManifestRow:
    manifest_id: str
    state: Literal["ok", "producing", "error"]
    artifact_path: Optional[str]
    request_id: Optional[str]
    created_at: str
    updated_at: str
    error_text: Optional[str] = None


@dataclass
class IntentRow:
    intent_id: str
    artifact_identity: CanonicalIdentity
    status: Literal["pending", "queued", "claimed", "failed"]
    created_at: str
    expires_at: str


@dataclass
class Decision:
    http_status: int
    headers: Dict[str, str]
    body: Dict[str, Any]
