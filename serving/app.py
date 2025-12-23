from __future__ import annotations

from flask import Flask, request, jsonify, make_response
from typing import Optional

from .types import CanonicalIdentity, BadRequest
from .stores import InMemoryManifestStore, InMemoryIntentStore
from .decision import decide_request
from .types import ManifestRow
from .data_loader import ArtifactLoader, LocalPlaceholderLoader, LocalFileLoader
from .query_descriptors import get_descriptor
from datetime import timedelta
from typing import Tuple, Optional

# Hard limits for Stage 4 serving bounds enforcement
MAX_TIME_WINDOW_DAYS = 90
MAX_POLYGON_SPATIAL_INDICES = 10_000
MAX_RAW_ROWS = 1_000_000


def _parse_iso(ts: str):
    # naive ISO parser; rely on datetime.fromisoformat for strictness
    from datetime import datetime

    try:
        # fromisoformat accepts 'YYYY-MM-DDTHH:MM:SS' and with timezone
        return datetime.fromisoformat(ts)
    except Exception:
        # Let calling code treat parsing failures as validation errors
        return None


def _estimate_spatial_index_count_from_wkt(wkt: str) -> int:
    """Estimate the number of spatial indices a polygon would cover.

    Strategy:
    - If Shapely is present, compute geometry complexity via number of coords.
    - Otherwise, fall back to a cheap heuristic using WKT length.
    This is conservative but performant and allows serving to reject
    obviously huge polygons.
    """
    if not wkt:
        return 0
    try:
        from shapely import wkt as shapely_wkt  # type: ignore

        geom = shapely_wkt.loads(wkt)
        # Count coordinates in exterior + interiors
        count = 0
        if hasattr(geom, "exterior") and geom.exterior is not None:
            count += len(list(geom.exterior.coords))
        if hasattr(geom, "interiors") and geom.interiors is not None:
            for ring in geom.interiors:
                count += len(list(ring.coords))
        return max(1, count)
    except Exception:
        # Fallback heuristic: WKT length / 10
        return max(1, len(wkt) // 10)
from datetime import datetime, timezone
import uuid


def create_app(
    manifest_store: Optional[InMemoryManifestStore] = None,
    intent_store: Optional[InMemoryIntentStore] = None,
    loader: Optional[ArtifactLoader] = None,
) -> Flask:
    app = Flask(__name__)

    manifest = manifest_store or InMemoryManifestStore()
    intents = intent_store or InMemoryIntentStore()
    # Default loader for Stage 3: read bytes from the local filesystem.
    # Tests may still inject a different loader instance.
    artifact_loader = loader or LocalFileLoader()


    def parse_identity(args) -> CanonicalIdentity:
        # Accept query params: artifact_type, spatial_id, timestamp, version
        mapping = {
            "artifact_type": args.get("artifact_type"),
            "spatial_id": args.get("spatial_id"),
            "timestamp": args.get("timestamp"),
            "version": args.get("version"),
        }
        try:
            return CanonicalIdentity.from_mapping(mapping)
        except BadRequest as e:
            raise


    @app.route("/artifact", methods=["GET"])
    def get_artifact():
        # single request_id per incoming HTTP request
        client_req = request.args.get("client_request_id") or request.headers.get("X-Client-Request-ID")
        if client_req:
            request_id = str(client_req)
        else:
            request_id = uuid.uuid4().hex

        trigger = request.args.get("trigger_production", "false").lower() in ("1", "true", "yes")
        try:
            identity = parse_identity(request.args)
        except BadRequest as e:
            resp = make_response(jsonify({"error": str(e)}), 400)
            resp.headers["X-Request-ID"] = request_id
            return resp

        mr = manifest.get(identity)
        decision = decide_request(identity, mr, trigger, intents)

        # normalize headers in one place (do not overwrite existing Decision headers)
        hdrs = dict(decision.headers) if decision.headers is not None else {}
        # X-Request-ID: from request entry (do not overwrite if present)
        hdrs.setdefault("X-Request-ID", request_id)
        # X-Manifest-ID: include when a manifest row was involved
        if mr is not None:
            hdrs.setdefault("X-Manifest-ID", mr.manifest_id)
        # X-Intent-ID: include when decision body returned an intent
        intent_obj = None
        if isinstance(decision.body, dict):
            intent_obj = decision.body.get("intent")
        if intent_obj and isinstance(intent_obj, dict):
            iid = intent_obj.get("intent_id")
            if iid:
                hdrs.setdefault("X-Intent-ID", iid)

        # If decision is a successful serve, invoke the loader to obtain raw bytes.
        if decision.http_status == 200 and mr is not None and mr.state == "ok":
            # loader enforces the precondition that manifest_row.state == 'ok'
            try:
                data_bytes = artifact_loader.load(mr)
            except Exception as e:
                # Loader failures are internal server errors for now.
                resp = make_response(jsonify({"error": "loader_error", "detail": str(e)}), 500)
                resp.headers.update(hdrs)
                return resp

            resp = make_response(data_bytes, 200)
            resp.headers["Content-Type"] = "application/octet-stream"
            for k, v in hdrs.items():
                resp.headers[k] = v
            return resp

        resp = make_response(jsonify(decision.body), decision.http_status)
        for k, v in hdrs.items():
            resp.headers[k] = v
        return resp


    @app.route("/artifact/status", methods=["GET"])
    def status():
        # single request_id per incoming HTTP request
        client_req = request.args.get("client_request_id") or request.headers.get("X-Client-Request-ID")
        if client_req:
            request_id = str(client_req)
        else:
            request_id = uuid.uuid4().hex

        # status by identity or by intent_id
        intent_id = request.args.get("intent_id")
        if intent_id:
            intent = intents.get(intent_id)
            if not intent:
                resp = make_response(jsonify({"error": "intent not found"}), 404)
                resp.headers["X-Request-ID"] = request_id
                return resp
            resp = make_response(jsonify({"intent": intent.__dict__}), 200)
            resp.headers["X-Request-ID"] = request_id
            resp.headers.setdefault("X-Intent-ID", intent.intent_id)
            return resp

        try:
            identity = parse_identity(request.args)
        except BadRequest as e:
            resp = make_response(jsonify({"error": str(e)}), 400)
            resp.headers["X-Request-ID"] = request_id
            return resp
        mr = manifest.get(identity)
        if mr is None:
            resp = make_response(jsonify({"status": "missing", "identity": identity.__dict__}), 404)
            resp.headers["X-Request-ID"] = request_id
            return resp
        resp = make_response(jsonify({"manifest": mr.__dict__}), 200)
        resp.headers["X-Request-ID"] = request_id
        resp.headers.setdefault("X-Manifest-ID", mr.manifest_id)
        return resp

    @app.route("/query", methods=["GET"])
    def query_endpoint():
        # single request_id per incoming HTTP request
        client_req = request.args.get("client_request_id") or request.headers.get("X-Client-Request-ID")
        if client_req:
            request_id = str(client_req)
        else:
            request_id = uuid.uuid4().hex

        # parse descriptor kind
        qkind = request.args.get("query_kind")
        if not qkind:
            resp = make_response(jsonify({"error": "missing query_kind"}), 400)
            resp.headers["X-Request-ID"] = request_id
            return resp

        descriptor = get_descriptor(qkind)
        if descriptor is None:
            resp = make_response(jsonify({"error": "unsupported query_kind"}), 400)
            resp.headers["X-Request-ID"] = request_id
            return resp

        # parse identity
        try:
            identity = parse_identity(request.args)
        except BadRequest as e:
            resp = make_response(jsonify({"error": str(e)}), 400)
            resp.headers["X-Request-ID"] = request_id
            return resp

        # validate descriptor parameters
        params = {}
        for p in descriptor.params:
            val = request.args.get(p.name)
            if val is None:
                if p.required:
                    resp = make_response(jsonify({"error": f"missing parameter: {p.name}"}), 400)
                    resp.headers["X-Request-ID"] = request_id
                    return resp
                else:
                    params[p.name] = None
                    continue
            try:
                # only basic casting (str, int, float)
                if p.type is int:
                    params[p.name] = int(val)
                elif p.type is float:
                    params[p.name] = float(val)
                else:
                    params[p.name] = val
            except Exception:
                resp = make_response(jsonify({"error": f"invalid type for parameter: {p.name}"}), 400)
                resp.headers["X-Request-ID"] = request_id
                return resp

        # Now consult manifest and decision as usual
        mr = manifest.get(identity)
        decision = decide_request(identity, mr, False, intents)

        # normalize headers
        hdrs = dict(decision.headers) if decision.headers is not None else {}
        hdrs.setdefault("X-Request-ID", request_id)
        if mr is not None:
            hdrs.setdefault("X-Manifest-ID", mr.manifest_id)

        intent_obj = None
        if isinstance(decision.body, dict):
            intent_obj = decision.body.get("intent")
        if intent_obj and isinstance(intent_obj, dict):
            iid = intent_obj.get("intent_id")
            if iid:
                hdrs.setdefault("X-Intent-ID", iid)

        if decision.http_status != 200 or mr is None or mr.state != "ok":
            resp = make_response(jsonify(decision.body), decision.http_status)
            for k, v in hdrs.items():
                resp.headers[k] = v
            return resp
        # Enforce bounds before invoking the loader (time windows, polygon size, raw rows)
        # 1) Time window enforcement
        if any(p.name in ("start", "end") for p in descriptor.params):
            start = params.get("start")
            end = params.get("end")
            if start and end:
                s_dt = _parse_iso(start)
                e_dt = _parse_iso(end)
                if s_dt is None or e_dt is None:
                    resp = make_response(jsonify({"error": "invalid date format for start/end"}), 400)
                    resp.headers.update(hdrs)
                    return resp
                window_days = (e_dt - s_dt).total_seconds() / 86400.0
                if window_days < 0:
                    resp = make_response(jsonify({"error": "start must be before end"}), 400)
                    resp.headers.update(hdrs)
                    return resp
                if window_days > MAX_TIME_WINDOW_DAYS:
                    resp = make_response(jsonify({"error": "time_window_exceeded", "max_days": MAX_TIME_WINDOW_DAYS}), 422)
                    resp.headers.update(hdrs)
                    return resp

        # 2) Polygon spatial coverage enforcement
        if any(p.name == "wkt" for p in descriptor.params):
            wkt = params.get("wkt")
            if wkt:
                idx_count = _estimate_spatial_index_count_from_wkt(wkt)
                if idx_count > MAX_POLYGON_SPATIAL_INDICES:
                    resp = make_response(jsonify({"error": "polygon_too_large", "indices": idx_count, "max": MAX_POLYGON_SPATIAL_INDICES}), 422)
                    resp.headers.update(hdrs)
                    return resp

        # 3) Raw rows enforcement (descriptor-level)
        if descriptor.query_kind.name == "POLYGON_TIME_RANGE_RAW":
            declared = None
            if descriptor.safety_limits:
                declared = descriptor.safety_limits.get("max_rows")
            if declared is not None and declared > MAX_RAW_ROWS:
                resp = make_response(jsonify({"error": "raw_rows_exceed_limit", "declared": declared, "max": MAX_RAW_ROWS}), 422)
                resp.headers.update(hdrs)
                return resp

        # Ensure loader supports query()
        try:
            data_bytes = artifact_loader.query(mr, descriptor, params)
        except NotImplementedError:
            resp = make_response(jsonify({"error": "loader does not support queries"}), 501)
            resp.headers.update(hdrs)
            return resp
        except Exception as e:
            resp = make_response(jsonify({"error": "loader_error", "detail": str(e)}), 500)
            resp.headers.update(hdrs)
            return resp

        resp = make_response(data_bytes, 200)
        resp.headers["Content-Type"] = "application/octet-stream"
        for k, v in hdrs.items():
            resp.headers[k] = v
        return resp

    return app


if __name__ == "__main__":
    # run default app with empty in-memory stores
    app = create_app()
    app.run(port=5003, debug=True)
