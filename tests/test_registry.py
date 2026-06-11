"""Tests for UDF registry (udf/registry.py)."""

import pytest

from codestr.udf.registry import UDFMeta, UDFRegistry, udf


class TestUDFRegistry:
    def test_singleton(self):
        a = UDFRegistry.get_instance()
        b = UDFRegistry.get_instance()
        assert a is b

    def test_reset(self):
        a = UDFRegistry.get_instance()
        a.register(UDFMeta(name="test_fn", fn=lambda x: x))
        UDFRegistry.reset()
        b = UDFRegistry.get_instance()
        assert a is not b
        assert "test_fn" not in b

    def test_register_and_get(self):
        registry = UDFRegistry.get_instance()
        registry.register(UDFMeta(name="my_fn", fn=lambda x: x))
        assert "my_fn" in registry
        meta = registry["my_fn"]
        assert meta.name == "my_fn"

    def test_get_missing(self):
        registry = UDFRegistry.get_instance()
        assert registry.get("nonexistent") is None

    def test_getitem_missing_raises(self):
        registry = UDFRegistry.get_instance()
        with pytest.raises(KeyError):
            _ = registry["nonexistent"]

    def test_all(self):
        registry = UDFRegistry.get_instance()
        registry.register(UDFMeta(name="f1", fn=lambda x: x))
        registry.register(UDFMeta(name="f2", fn=lambda x, y: x + y))
        all_meta = registry.all()
        names = {m.name for m in all_meta}
        assert "f1" in names
        assert "f2" in names


class TestUDFDecorator:
    def test_decorator_basic(self):
        @udf(category="math")
        def double(x):
            return x * 2

        registry = UDFRegistry.get_instance()
        meta = registry["double"]
        assert meta.name == "double"
        assert meta.category == "math"
        assert meta.arity == 1

    def test_decorator_with_name_override(self):
        @udf(name="triple_fn", category="math")
        def something(x):
            return x * 3

        registry = UDFRegistry.get_instance()
        assert "triple_fn" in registry
        assert "something" not in registry

    def test_decorator_no_args(self):
        """@udf without parentheses"""

        @udf
        def quad(x):
            return x * 4

        registry = UDFRegistry.get_instance()
        meta = registry["quad"]
        assert meta.category == "math"
        assert meta.arity == 1

    def test_auto_arity(self):
        @udf(category="cs")
        def two_args(left, right):
            return left + right

        registry = UDFRegistry.get_instance()
        assert registry["two_args"].arity == 2

    def test_manual_arity(self):
        @udf(category="math", arity=3)
        def flexible(*args):
            return sum(args)

        registry = UDFRegistry.get_instance()
        assert registry["flexible"].arity == 3


class TestUDFMeta:
    def test_dataclass_fields(self):
        meta = UDFMeta(name="test", fn=lambda x: x, category="cs", arity=1)
        assert meta.name == "test"
        assert meta.category == "cs"
        assert meta.arity == 1
