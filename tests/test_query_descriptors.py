import sys
from pathlib import Path
import pytest

# Ensure repo root is importable
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from serving.query_descriptors import DESCRIPTORS, get_descriptor, QueryKind


def test_descriptors_complete_and_well_formed():
    """Ensure every descriptor declares its contract (pure, no I/O)."""
    assert isinstance(DESCRIPTORS, dict)
    for kind, desc in DESCRIPTORS.items():
        # query_kind exists and matches the dict key
        assert desc.query_kind == kind
        assert isinstance(desc.query_kind, QueryKind)

        # params is a non-empty tuple (at least some queries require params)
        assert hasattr(desc, "params")
        assert isinstance(desc.params, tuple)

        # output_shape metadata present
        assert hasattr(desc, "output_shape")


def test_descriptor_param_spec_typing_and_required_flags():
    # Each ParamSpec must have name, type and required flag
    for desc in DESCRIPTORS.values():
        for p in desc.params:
            assert hasattr(p, "name") and isinstance(p.name, str)
            assert hasattr(p, "type")
            assert hasattr(p, "required") and isinstance(p.required, bool)


def test_descriptor_immutability():
    # QueryDescriptor is frozen; attempts to mutate should raise
    for desc in DESCRIPTORS.values():
        with pytest.raises(Exception):
            # attempt to mutate a frozen dataclass field
            desc.output_shape = "mutated"


def test_get_descriptor_invalid_kind_returns_none():
    assert get_descriptor("NON_EXISTENT_KIND") is None
