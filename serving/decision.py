from __future__ import annotations

from pathlib import Path
from typing import Optional

from .types import CanonicalIdentity, ManifestRow, Decision
from .stores import IntentStore


def decide_request(
    identity: Optional[CanonicalIdentity],
    manifest_row: Optional[ManifestRow],
    trigger_production: bool,
    intent_store: IntentStore,
) -> Decision:
    """Decide what to return for a client request.

    This function is pure-ish but it will create an intent row via `intent_store`
    when a production enqueue decision is required.
    """
    headers = {}
    if identity is None:
        return Decision(http_status=400, headers=headers, body={"error": "bad_request"})

    # manifest present
    if manifest_row is not None:
        headers.update({"X-Manifest-ID": manifest_row.manifest_id})
        if manifest_row.state == "ok":
            # verify artifact path exists
            if manifest_row.artifact_path:
                p = Path(manifest_row.artifact_path)
                if p.exists():
                    # Serve placeholder (serving does not stream actual bytes here)
                    return Decision(
                        http_status=200,
                        headers=headers,
                        body={"status": "ok", "manifest": manifest_row.__dict__, "data": "<artifact placeholder>"},
                    )
                else:
                    return Decision(
                        http_status=500,
                        headers=headers,
                        body={"status": "inconsistent", "error": "artifact file missing", "manifest": manifest_row.__dict__},
                    )
            else:
                return Decision(
                    http_status=500,
                    headers=headers,
                    body={"status": "inconsistent", "error": "artifact_path missing in manifest", "manifest": manifest_row.__dict__},
                )

        if manifest_row.state == "producing":
            return Decision(http_status=202, headers=headers, body={"status": "producing", "manifest": manifest_row.__dict__})

        if manifest_row.state == "error":
            if trigger_production:
                intent = intent_store.get_or_create(identity)
                headers.update({"X-Intent-ID": intent.intent_id})
                return Decision(http_status=202, headers=headers, body={"status": "queued", "intent": intent.__dict__})
            return Decision(http_status=503, headers=headers, body={"status": "error", "manifest": manifest_row.__dict__})

    # manifest missing
    if manifest_row is None:
        if trigger_production:
            intent = intent_store.get_or_create(identity)
            headers.update({"X-Intent-ID": intent.intent_id})
            return Decision(http_status=202, headers=headers, body={"status": "queued", "intent": intent.__dict__})
        return Decision(http_status=404, headers=headers, body={"status": "missing", "identity": identity.__dict__})

    # fallback internal error
    return Decision(http_status=500, headers=headers, body={"error": "internal"})
