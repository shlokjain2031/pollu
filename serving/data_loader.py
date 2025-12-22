from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from .types import ManifestRow


class ArtifactLoader(ABC):
    """Abstract boundary responsible for loading artifact bytes given a manifest row.

    Implementations MUST assume the manifest_row.state == "ok" and should
    immediately raise a ValueError (or subclass) if that's not true.
    """

    @abstractmethod
    def load(self, manifest_row: ManifestRow) -> bytes:
        raise NotImplementedError()


class LocalPlaceholderLoader(ArtifactLoader):
    """Trivial loader used as a placeholder for Stage 2.

    This loader ignores artifact_path and returns static bytes. It enforces
    the invariant that the provided manifest row must be in the "ok" state.
    """

    def load(self, manifest_row: ManifestRow) -> bytes:
        if manifest_row.state != "ok":
            raise ValueError("ArtifactLoader.load() requires manifest_row.state == 'ok'")
        return b"artifact data placeholder"


class LocalFileLoader(ArtifactLoader):
    """Loader that reads raw bytes from the filesystem using the manifest's artifact_path.

    This implementation is deliberately minimal and defensive:
    - It requires `manifest_row.state == 'ok'`.
    - It opens the file in binary read-only mode and returns the full contents.
    - Any I/O errors (missing file, permissions, etc.) are propagated as exceptions
      so the HTTP adapter can translate them to 500 responses.
    """

    def load(self, manifest_row: ManifestRow) -> bytes:
        if manifest_row.state != "ok":
            raise ValueError("ArtifactLoader.load() requires manifest_row.state == 'ok'")

        path = manifest_row.artifact_path
        if not path:
            raise ValueError("ManifestRow.artifact_path is required for LocalFileLoader")

        # Open and read the file in binary mode. Let any IOError propagate.
        with open(path, "rb") as fh:
            return fh.read()
