"""Tests for mixed-window analysis and lowering."""

import pytest

from codestr.parser import parse
from codestr.planner import (
    MixedWindowBoundary,
    find_mixed_window_boundary,
)
from codestr.udf.registry import UDFRegistry


@pytest.fixture
def registry():
    return UDFRegistry.get_instance()


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "ts_mean(cs_moderate(x), 60)",
            MixedWindowBoundary("ts_mean", "ts", "cs_moderate", "cs"),
        ),
        (
            "cs_mean(ts_mean(x, 60))",
            MixedWindowBoundary("cs_mean", "cs", "ts_mean", "ts"),
        ),
        (
            "ts_mean(abs(cs_moderate(x)), 60)",
            MixedWindowBoundary("ts_mean", "ts", "cs_moderate", "cs"),
        ),
        (
            "ts_mean(add(left=cs_moderate(x), right=y), 60)",
            MixedWindowBoundary("ts_mean", "ts", "cs_moderate", "cs"),
        ),
    ],
)
def test_find_mixed_window_boundary(source, expected, registry):
    assert find_mixed_window_boundary(parse(source), registry) == expected


@pytest.mark.parametrize(
    "source",
    [
        "ts_mean(ts_mean(x, 5), 20)",
        "cs_rank(cs_moderate(x))",
        "cs_rank(x) + ts_mean(y, 20)",
        "abs(x) + log(y)",
    ],
)
def test_same_domain_and_sibling_windows_are_not_mixed(source, registry):
    assert find_mixed_window_boundary(parse(source), registry) is None


def test_unknown_call_is_opaque_to_window_analysis(registry):
    node = parse("ts_mean(not_registered(cs_moderate(x)), 60)")

    assert find_mixed_window_boundary(node, registry) is None
