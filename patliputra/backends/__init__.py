"""Backends used by Patliputra providers (Earth Engine, S3, local).

This package contains light wrappers over data backends. Implementations
should be small and defensive (clear errors if dependencies or auth are
missing). The orchestrator can accept a backend client and pass it to
providers via their config.
"""
