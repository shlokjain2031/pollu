#!/usr/bin/env python3
"""Check Earth Engine initialization using local `.env` settings.

This script will:
- load environment variables from the repository root `.env` file (if present)
- attempt EE initialization using the following strategies (in order):
  1. Service account credentials (if GOOGLE_APPLICATION_CREDENTIALS and EE_ACCOUNT_EMAIL are set)
  2. Explicit project id (GOOGLE_CLOUD_PROJECT)
  3. Default initialization (ee.Initialize())

Exit status: 0 on success, 1 on failure.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def load_dotenv_simple(path: Path) -> None:
    """Load simple KEY=VALUE pairs from a .env file into os.environ if not already set.

    This is intentionally minimal to avoid a runtime dependency.
    """
    if not path.exists():
        print(f"No .env file at {path}; skipping load")
        return

    print(f"Loading env from: {path}")
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        # don't overwrite existing environment values
        os.environ.setdefault(k, v)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / ".env"
    load_dotenv_simple(env_path)

    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    key_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    sa_email = os.getenv("EE_ACCOUNT_EMAIL")

    print("Current relevant env:")
    print("  GOOGLE_CLOUD_PROJECT:", project or "(not set)")
    print("  GOOGLE_APPLICATION_CREDENTIALS:", key_file or "(not set)")
    print("  EE_ACCOUNT_EMAIL:", sa_email or "(not set)")

    try:
        import ee
    except Exception as e:  # pragma: no cover - runtime check
        print(
            "Failed to import Earth Engine (ee). Is earthengine-api installed?",
            file=sys.stderr,
        )
        print(str(e), file=sys.stderr)
        return 1

    # Try service account flow if both key and account set
    try:
        if key_file and sa_email:
            key_path = Path(key_file)
            if not key_path.exists():
                print(f"Service account key file not found: {key_path}")
            else:
                print("Attempting service-account initialization...")
                creds = ee.ServiceAccountCredentials(sa_email, str(key_path))
                ee.Initialize(credentials=creds, project=project)
                print("EE initialized using service account credentials")
                print("ee.__version__:", getattr(ee, "__version__", "unknown"))
                return 0

        # Next try explicit project id
        if project:
            print(f"Attempting initialization with project='{project}'...")
            ee.Initialize(project=project)
            print("EE initialized with project")
            print("ee.__version__:", getattr(ee, "__version__", "unknown"))
            return 0

        # Fallback: default initialize
        print("Attempting default ee.Initialize()...")
        ee.Initialize()
        print("EE initialized with default credentials")
        print("ee.__version__:", getattr(ee, "__version__", "unknown"))
        return 0

    except Exception as e:
        print("Earth Engine initialization failed:", file=sys.stderr)
        print(str(e), file=sys.stderr)
        print(
            "See https://developers.google.com/earth-engine/guides/python_install for auth instructions."
        )
        return 1


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
