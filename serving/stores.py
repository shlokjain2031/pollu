from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict

from .types import CanonicalIdentity, ManifestRow, IntentRow


class ManifestStore:
    """Interface for manifest lookups (read-only for serving)."""

    def get(self, identity: CanonicalIdentity) -> Optional[ManifestRow]:
        raise NotImplementedError()


class InMemoryManifestStore(ManifestStore):
    """Simple in-memory manifest for testing.

    Keys are the stringified identity (artifact_type::spatial_id::timestamp::version).
    """

    def __init__(self):
        self._store: Dict[str, ManifestRow] = {}

    @staticmethod
    def _key(identity: CanonicalIdentity) -> str:
        return f"{identity.artifact_type}::{identity.spatial_id}::{identity.timestamp}::{identity.version}"

    def seed(self, row: ManifestRow):
        self._store[row.manifest_id] = row

    def get(self, identity: CanonicalIdentity) -> Optional[ManifestRow]:
        return self._store.get(self._key(identity))


class IntentStore:
    """Interface for short-lived intent rows used by serving for deduplication."""

    def get_or_create(self, identity: CanonicalIdentity) -> IntentRow:
        raise NotImplementedError()

    def get(self, intent_id: str):
        raise NotImplementedError()


class InMemoryIntentStore(IntentStore):
    def __init__(self, ttl_seconds: int = 300):
        self._by_identity: Dict[str, IntentRow] = {}
        self._by_id: Dict[str, IntentRow] = {}
        self._ttl = ttl_seconds

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _key(self, identity: CanonicalIdentity) -> str:
        return f"{identity.artifact_type}::{identity.spatial_id}::{identity.timestamp}::{identity.version}"

    def get(self, intent_id: str):
        return self._by_id.get(intent_id)

    def get_or_create(self, identity: CanonicalIdentity) -> IntentRow:
        k = self._key(identity)
        existing = self._by_identity.get(k)
        now = datetime.now(timezone.utc)
        if existing:
            # expire check
            expires = datetime.fromisoformat(existing.expires_at)
            if expires > now:
                return existing
            # expired -> remove
            del self._by_identity[k]
            del self._by_id[existing.intent_id]

        intent_id = uuid.uuid4().hex
        created = now.isoformat()
        expires = (now + timedelta(seconds=self._ttl)).isoformat()
        row = IntentRow(
            intent_id=intent_id,
            artifact_identity=identity,
            status="pending",
            created_at=created,
            expires_at=expires,
        )
        self._by_identity[k] = row
        self._by_id[intent_id] = row
        return row
