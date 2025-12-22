#!/usr/bin/env python3
"""Small utility to demonstrate that `LocalFileLoader` reads a file.

Usage:
  python scripts/show_artifact_read.py /path/to/artifact.bin

This script constructs a minimal `ManifestRow` with `state == 'ok'` and
invokes `LocalFileLoader.load()` to read the bytes. It prints the byte
count and a short preview of the data.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone

# Ensure repository root is importable so we can import serving.*
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from serving.data_loader import LocalFileLoader
from serving.types import ManifestRow


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/show_artifact_read.py /path/to/artifact.bin")
        return 2

    p = Path(sys.argv[1])
    if not p.exists():
        print(f"File not found: {p}")
        return 1

    loader = LocalFileLoader()
    mr = ManifestRow(
        manifest_id="manual-check",
        state="ok",
        artifact_path=str(p),
        request_id=None,
        created_at=now_iso(),
        updated_at=now_iso(),
    )

    try:
        data = loader.load(mr)
    except Exception as exc:
        print("Loader error:", exc)
        return 3

    print(f"Read {len(data)} bytes from: {p}")
    preview = data[:200]
    print("--- Preview (first 200 bytes) ---")
    print(preview)
    try:
        text = preview.decode("utf-8")
        print("--- Preview as text ---")
        print(text)
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
