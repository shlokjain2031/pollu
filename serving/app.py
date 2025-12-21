from __future__ import annotations

from flask import Flask, request, jsonify, make_response
from typing import Optional

from .types import CanonicalIdentity, BadRequest
from .stores import InMemoryManifestStore, InMemoryIntentStore
from .decision import decide_request
from .types import ManifestRow
from datetime import datetime, timezone
import uuid


def create_app(manifest_store: Optional[InMemoryManifestStore] = None, intent_store: Optional[InMemoryIntentStore] = None) -> Flask:
    app = Flask(__name__)

    manifest = manifest_store or InMemoryManifestStore()
    intents = intent_store or InMemoryIntentStore()


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

        # attach request id to decision headers (observability spine)
        decision.headers = dict(decision.headers) if decision.headers is not None else {}
        decision.headers.setdefault("X-Request-ID", request_id)

        resp = make_response(jsonify(decision.body), decision.http_status)
        for k, v in decision.headers.items():
            resp.headers[k] = v
        return resp


    @app.route("/artifact/status", methods=["GET"])
    def status():
        # status by identity or by intent_id
        intent_id = request.args.get("intent_id")
        if intent_id:
            intent = intents.get(intent_id)
            if not intent:
                return make_response(jsonify({"error": "intent not found"}), 404)
            return jsonify({"intent": intent.__dict__})

        try:
            identity = parse_identity(request.args)
        except BadRequest as e:
            return make_response(jsonify({"error": str(e)}), 400)
        mr = manifest.get(identity)
        if mr is None:
            return make_response(jsonify({"status": "missing", "identity": identity.__dict__}), 404)
        return jsonify({"manifest": mr.__dict__})

    return app


if __name__ == "__main__":
    # run default app with empty in-memory stores
    app = create_app()
    app.run(port=5003, debug=True)
