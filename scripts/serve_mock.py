"""Minimal serving mock that returns HTTP responses based on manifest state.

Endpoints:
- GET /artifact?row_idx=&timestamp=&artifact_type=&version=

Responses:
- 200 OK: artifact content (JSON)
- 202 Accepted: producing (returns manifest row)
- 404 Not Found: missing
- 503 Service Unavailable: error state

This app uses Flask; install `flask` in your environment to run it.
"""

from __future__ import annotations

from flask import Flask, request, jsonify, make_response
from manifest.db import Manifest
from scripts.util import load_artifact
from pathlib import Path

app = Flask(__name__)


def _parse_key(req):
    try:
        row_idx = int(req.args.get("row_idx"))
        timestamp = req.args.get("timestamp")
        artifact_type = req.args.get("artifact_type")
        version = req.args.get("version")
    except Exception:
        return None
    if not all((timestamp, artifact_type, version)):
        return None
    return {"row_idx": row_idx, "timestamp": timestamp, "artifact_type": artifact_type, "version": version}


@app.route("/artifact", methods=["GET"])
def get_artifact():
    key = _parse_key(request)
    if key is None:
        return make_response(jsonify({"error": "missing or invalid identity parameters"}), 400)
    m = Manifest()
    row = m.get(key)
    if row is None:
        return make_response(jsonify({"status": "missing", "identity": key}), 404)
    state = row.get("state")
    if state == "producing":
        return make_response(jsonify({"status": "producing", "manifest": row}), 202)
    if state == "error":
        return make_response(jsonify({"status": "error", "manifest": row}), 503)
    if state == "ok":
        path = row.get("artifact_path")
        if not path:
            return make_response(jsonify({"status": "inconsistent", "manifest": row}), 500)
        p = Path(path)
        if not p.exists():
            return make_response(jsonify({"status": "inconsistent", "manifest": row, "error": "file missing"}), 500)
        try:
            obj = load_artifact(p)
        except Exception as e:
            return make_response(jsonify({"status": "error_loading", "error": str(e)}), 500)
        # if tabular, return as JSON records
        if hasattr(obj, "to_dict"):
            return jsonify({"status": "ok", "data": obj.to_dict(orient="records"), "manifest": row})
        return jsonify({"status": "ok", "data": obj, "manifest": row})
    return make_response(jsonify({"status": "unknown", "manifest": row}), 500)


if __name__ == "__main__":
    app.run(port=5002, debug=True)
