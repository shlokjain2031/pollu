"""Serving decision layer for Pollu (read-only, deterministic).

This package implements:
- CanonicalIdentity parsing/validation
- ManifestStore interface and in-memory mock
- IntentStore interface and in-memory mock (deduplication)
- Decision engine `decide_request`
- A minimal Flask HTTP adapter exposing `/artifact` and `/artifact/status`

This is intentionally small and safe: serving never mutates manifest lifecycle
and does not enqueue or run producers.
"""

from .app import create_app

__all__ = ["create_app"]
