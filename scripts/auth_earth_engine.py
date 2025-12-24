#!/usr/bin/env python3
"""Authenticate Google Earth Engine.

Usage:
  Interactive (default):
    python scripts/auth_earth_engine.py

  Service account (non-interactive):
    python scripts/auth_earth_engine.py --service-account-key /path/to/key.json --service-account-email service-account@PROJECT.iam.gserviceaccount.com

Notes:
- Installs are not performed by this script. If `import ee` fails, run `pip install earthengine-api` in your desired venv.
- If you encounter a ModuleNotFoundError for "StringIO" from the earthengine package, that indicates an incompatible package install; prefer running this script in a supported Python (3.11/3.12) or follow the troubleshooting guidance in docs.
"""
from __future__ import annotations

import argparse
import os
import sys


def main():
    p = argparse.ArgumentParser(description="Authenticate Google Earth Engine (interactive or service-account)")
    p.add_argument("--service-account-key", help="Path to service account JSON key (non-interactive)")
    p.add_argument("--service-account-email", help="Service account email (required when using --service-account-key)")
    args = p.parse_args()

    try:
        import ee
    except Exception as e:
        print("ERROR: Failed to import earthengine-api (ee).\n")
        print("Please install it in your active environment:\n  pip install earthengine-api\n")
        print("If `pip install` fails, ensure you're using a supported Python version (3.11/3.12 recommended).")
        print(f"\nFull import error: {e}")
        sys.exit(2)

    # If service account flow requested
    if args.service_account_key:
        if not args.service_account_email:
            print("ERROR: --service-account-email is required when using --service-account-key")
            sys.exit(2)
        key_path = os.path.expanduser(args.service_account_key)
        if not os.path.exists(key_path):
            print(f"ERROR: service account key file not found: {key_path}")
            sys.exit(2)
        print("Using Service Account authentication (non-interactive)...")
        try:
            # Prefer environment variable for some EE libraries; set it but still pass credentials
            os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", key_path)
            # Create credentials and initialize
            creds = ee.ServiceAccountCredentials(args.service_account_email, key_path)
            ee.Initialize(creds)
            print("Earth Engine initialized with service account.")
            return
        except Exception as e:
            print("ERROR: Service account authentication failed:")
            print(e)
            print("Ensure the service account has Earth Engine access and the JSON key is valid.")
            sys.exit(3)

    # Interactive flow
    try:
        print("Starting interactive authentication flow. A browser window will open (or follow the URL printed).")
        ee.Authenticate()
        ee.Initialize()
        print("Earth Engine authenticated and initialized (interactive).")
    except Exception as e:
        # Common issue: old earthengine-api code importing Python2-only StringIO
        if isinstance(e, ModuleNotFoundError) and getattr(e, 'name', '') == 'StringIO':
            print("\nERROR: earthengine-api import failed with missing 'StringIO' module.\n")
            print("This usually indicates an incompatible install of earthengine-api (Python2-era import).\n")
            print("Workarounds:\n 1) Use a supported Python version (3.11/3.12) and reinstall:\n    python3.12 -m venv .venv_ee\n    source .venv_ee/bin/activate\n    pip install --upgrade pip\n    pip install earthengine-api\n    python scripts/auth_earth_engine.py\n\n 2) (Stopgap) Create a small compatibility shim named 'StringIO.py' in your venv's site-packages mapping to 'io.StringIO' — not recommended for production.\n")
            sys.exit(4)
        else:
            print("Interactive authentication failed with exception:")
            print(e)
            sys.exit(5)


if __name__ == '__main__':
    main()
