from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple


class QueryKind(str, Enum):
    POINT_LATEST = "POINT_LATEST"
    POINT_TIME_RANGE = "POINT_TIME_RANGE"
    POLYGON_LATEST_AVG = "POLYGON_LATEST_AVG"
    POLYGON_TIME_RANGE_AVG = "POLYGON_TIME_RANGE_AVG"
    POLYGON_TIME_RANGE_RAW = "POLYGON_TIME_RANGE_RAW"


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: type
    required: bool = True
    # For clarity: description or units (optional)
    description: Optional[str] = None


@dataclass(frozen=True)
class QueryDescriptor:
    query_kind: QueryKind
    params: Tuple[ParamSpec, ...]
    # Informational only: expected output shape (e.g. 'point', 'timeseries', 'aggregate')
    output_shape: str = "opaque"
    # Safety limits (informational): e.g. max_time_window_days, max_rows
    safety_limits: Mapping[str, Any] = None


# Declarative set of descriptors. These are deliberately simple and do not
# contain any SQL nor any execution details. They only declare what serving
# accepts and what the loader will implement.
DESCRIPTORS: Dict[QueryKind, QueryDescriptor] = {
    QueryKind.POINT_LATEST: QueryDescriptor(
        query_kind=QueryKind.POINT_LATEST,
        params=(
            ParamSpec("spatial_id", str, True, "point identifier"),
            ParamSpec("as_of", str, True, "ISO-8601 timestamp; latest observation <= as_of"),
        ),
        output_shape="point",
        safety_limits={"max_rows": 1},
    ),
    QueryKind.POINT_TIME_RANGE: QueryDescriptor(
        query_kind=QueryKind.POINT_TIME_RANGE,
        params=(
            ParamSpec("spatial_id", str, True, "point identifier"),
            ParamSpec("start", str, True, "ISO-8601 start timestamp"),
            ParamSpec("end", str, True, "ISO-8601 end timestamp"),
        ),
        output_shape="timeseries",
        safety_limits={"max_time_window_days": 365},
    ),
    QueryKind.POLYGON_LATEST_AVG: QueryDescriptor(
        query_kind=QueryKind.POLYGON_LATEST_AVG,
        params=(
            ParamSpec("wkt", str, True, "WKT polygon string"),
            ParamSpec("as_of", str, True, "ISO-8601 timestamp; latest observations <= as_of"),
        ),
        output_shape="aggregate",
        safety_limits={"max_rows": 1},
    ),
    QueryKind.POLYGON_TIME_RANGE_AVG: QueryDescriptor(
        query_kind=QueryKind.POLYGON_TIME_RANGE_AVG,
        params=(
            ParamSpec("wkt", str, True, "WKT polygon string"),
            ParamSpec("start", str, True, "ISO-8601 start timestamp"),
            ParamSpec("end", str, True, "ISO-8601 end timestamp"),
        ),
        output_shape="aggregate",
        safety_limits={"max_time_window_days": 365},
    ),
    QueryKind.POLYGON_TIME_RANGE_RAW: QueryDescriptor(
        query_kind=QueryKind.POLYGON_TIME_RANGE_RAW,
        params=(
            ParamSpec("wkt", str, True, "WKT polygon string"),
            ParamSpec("start", str, True, "ISO-8601 start timestamp"),
            ParamSpec("end", str, True, "ISO-8601 end timestamp"),
        ),
        output_shape="timeseries",
        safety_limits={"max_time_window_days": 365, "max_rows": 10000},
    ),
}


def get_descriptor(kind: str) -> Optional[QueryDescriptor]:
    try:
        k = QueryKind(kind)
    except Exception:
        return None
    return DESCRIPTORS.get(k)
