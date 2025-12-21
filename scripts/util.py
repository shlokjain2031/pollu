

from __future__ import annotations

from pathlib import Path
from typing import Any
from datetime import datetime, date
import json
import pandas as pd


def _safe(s: Any) -> str:
	"""Return a filesystem-safe string for path components."""
	if isinstance(s, (datetime, date)):
		s = s.isoformat()
	out = str(s)
	# replace problematic chars
	for ch in ("/", "\\", " "):
		out = out.replace(ch, "_")
	return out

def build_path(artifact_type: str, spatial_id: Any, timestamp: Any, version: Any) -> Path:
	"""Deterministically build an artifact Path.

	Path structure:
		artifacts/<artifact_type>/<timestamp>/<version>/<spatial_id>.parquet

	All components are stringified and sanitized (slashes and spaces -> underscore).

	This function does NOT create the file; it only returns the deterministic path.
	"""
	base = Path("artifacts")
	at = _safe(artifact_type)
	ts = _safe(timestamp)
	ver = _safe(version)
	sid = _safe(spatial_id)
	return base / at / ts / ver / f"{sid}.parquet"


def load_artifact(path: Union[str, Path]) -> Any:
	"""Load an artifact from `path` and return a Python object.

	- Parquet/CSV -> pandas.DataFrame
	- JSON -> dict/list

	Raises FileNotFoundError if the path does not exist, or ValueError for unsupported formats.
	"""
	p = Path(path)
	if not p.exists():
		raise FileNotFoundError(f"Artifact not found: {p}")
	suf = p.suffix.lower()
	if suf in (".parquet",):
		# Pandas uses pyarrow/fastparquet under the hood if available
		return pd.read_parquet(p)
	if suf in (".csv", ".txt"):
		return pd.read_csv(p)
	if suf in (".json",):
		with p.open("r", encoding="utf8") as fh:
			return json.load(fh)
	# fallback: try parquet first, then JSON
	try:
		return pd.read_parquet(p)
	except Exception:
		with p.open("r", encoding="utf8") as fh:
			return json.load(fh)

