"""Signal providers for Patliputra ingestion module.

Each signal provider implements the same minimal interface so the pipeline
can iterate over signals for a city and a date range and perform the same
operations (fetch, preprocess, sample, cache).
"""

from .base import SignalProvider  # re-export
