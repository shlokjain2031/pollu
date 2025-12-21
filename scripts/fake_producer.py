"""Fake producer for testing the manifest/serving behavior.

Usage:
  python -c "from scripts.fake_producer import produce; produce({'row_idx':1,'timestamp':'2025-12-20','artifact_type':'grid_30m','version':'vtest'})"

This script will:
 - call manifest.mark_producing(...)
 - write a tiny parquet artifact (atomic tmp -> replace)
 - call manifest.mark_ok(...)
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from typing import Mapping

import pandas as pd

from manifest.db import Manifest, ArtifactKey


def _make_dummy_df() -> pd.DataFrame:
    return pd.DataFrame({"value": [42], "note": ["fake"]})


def produce(key_mapping: Mapping[str, object], request_id: str | None = None, manifest_path: str | None = None) -> dict:
    m = Manifest(db_path=manifest_path)
    rid = request_id or f"fake-{uuid.uuid4().hex}"
    # mark producing (producers must do this)
    m.mark_producing(key_mapping, rid)
    key = ArtifactKey.from_mapping(key_mapping)
    target = m.artifact_path_for_key(key)
    tmp = Path(str(target) + ".tmp")
    try:
        df = _make_dummy_df()
        tmp.parent.mkdir(parents=True, exist_ok=True)
        # write parquet to tmp
        df.to_parquet(tmp)
        # atomic replace
        os.replace(str(tmp), str(target))
        m.mark_ok(key_mapping, target)
        return {"status": "ok", "artifact_path": str(target), "request_id": rid}
    except Exception as e:
        try:
            m.mark_error(key_mapping, str(e))
        except Exception:
            pass
        raise


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 5:
        print("Usage: python scripts/fake_producer.py <row_idx> <timestamp> <artifact_type> <version>")
        raise SystemExit(2)
    row_idx, timestamp, artifact_type, version = sys.argv[1:5]
    key = {"row_idx": int(row_idx), "timestamp": timestamp, "artifact_type": artifact_type, "version": version}
    print(produce(key))
