"""Fake consumer that reads artifacts via the manifest.

Usage:
  python -c "from scripts.fake_consumer import consume; print(consume({'row_idx':1,'timestamp':'2025-12-20','artifact_type':'grid_30m','version':'vtest'}))"
"""

from __future__ import annotations

from typing import Mapping, Any
from pathlib import Path

import pandas as pd

from manifest.db import Manifest, ArtifactKey
from scripts.util import load_artifact


def consume(key_mapping: Mapping[str, Any], manifest_path: str | None = None) -> Any:
    m = Manifest(db_path=manifest_path)
    row = m.get(key_mapping)
    if row is None:
        raise FileNotFoundError("Artifact missing in manifest")
    if row.get("state") != "ok":
        raise RuntimeError(f"Artifact not ready: state={row.get('state')}")
    artifact_path = Path(row["artifact_path"]) if row.get("artifact_path") else None
    if not artifact_path or not artifact_path.exists():
        raise FileNotFoundError("Artifact file missing on disk")
    # Use util.load_artifact to support multiple formats
    return load_artifact(artifact_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 5:
        print("Usage: python scripts/fake_consumer.py <row_idx> <timestamp> <artifact_type> <version>")
        raise SystemExit(2)
    row_idx, timestamp, artifact_type, version = sys.argv[1:5]
    key = {"row_idx": int(row_idx), "timestamp": timestamp, "artifact_type": artifact_type, "version": version}
    v = consume(key)
    if isinstance(v, pd.DataFrame):
        print(v.to_dict(orient="records"))
    else:
        print(v)
