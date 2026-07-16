"""Tests for AST node definitions (syntax.py)."""

import pytest

from codestr.syntax import (
    Call,
    Column,
    KeywordArg,
    Literal,
    common_subexprs,
    depth,
    descendants,
    node_count,
    pre_cal_items,
    to_rpn,
)
from codestr.tokens import TokenType


class TestColumn:
    def test_equality(self):
        assert Column("close") == Column("close")
        assert Column("close") != Column("volume")

    def test_hash(self):
        d = {Column("close"): 1, Column("volume"): 2}
        assert d[Column("close")] == 1

    def test_immutable(self):
        with pytest.raises(Exception):  # noqa: B017
            Column("close").name = "changed"  # type: ignore


class TestLiteral:
    def test_equality(self):
        assert Literal(5) == Literal(5)
        assert Literal(5.0) == Literal(5.0)
        assert Literal(5) != Literal(6)

    def test_hash(self):
        d = {Literal(5): "five"}
        assert d[Literal(5)] == "five"


class TestCall:
    def test_alias_rendering_binary(self):
        node = Call(fn_name="add", args=(Column("a"), Column("b")))
        assert node.alias == "(a+b)"

    def test_alias_rendering_unary(self):
        node = Call(fn_name="neg", args=(Column("x"),))
        assert node.alias == "-x"

    def test_alias_rendering_ternary(self):
        node = Call(fn_name="if_", args=(Column("a"), Column("b"), Column("c")))
        assert node.alias == "a?b:c"

    def test_alias_rendering_function(self):
        node = Call(fn_name="ts_mean", args=(Column("close"), Literal(5)))
        assert node.alias == "ts_mean(close, 5)"

    def test_hash_ignores_alias(self):
        a = Call(fn_name="add", args=(Column("x"), Column("y")))
        b = Call(fn_name="add", args=(Column("x"), Column("y")), _alias="custom")
        assert hash(a) == hash(b)
        assert a == b

    def test_frozen(self):
        with pytest.raises(Exception):  # noqa: B017
            Call(fn_name="add", args=(Column("x"),)).fn_name = "sub"  # type: ignore

    def test_rejects_mutable_argument_dicts(self):
        with pytest.raises(TypeError, match="Call arguments"):
            Call("clip", (Column("close"), {"lower_bound": Literal(0)}))  # type: ignore

    def test_keyword_argument_rendering(self):
        node = Call(
            "clip",
            (Column("close"), KeywordArg("lower_bound", Literal(0))),
        )

        assert node.alias == "clip(close, lower_bound=0)"


class TestDepth:
    def test_leaf_depth(self):
        assert depth(Column("x")) == 1
        assert depth(Literal(5)) == 1

    def test_simple_call(self):
        node = Call(fn_name="add", args=(Column("a"), Column("b")))
        assert depth(node) == 2

    def test_nested_call(self):
        # add(mul(a, b), c) → depth = 3
        inner = Call(fn_name="mul", args=(Column("a"), Column("b")))
        node = Call(fn_name="add", args=(inner, Column("c")))
        assert depth(node) == 3

    def test_keyword_argument_is_transparent(self):
        node = Call(
            "clip",
            (
                Call("abs", (Column("close"),)),
                KeywordArg("lower_bound", Literal(0)),
            ),
        )

        assert depth(node) == 3


class TestNodeCount:
    def test_leaf_count(self):
        assert node_count(Column("x")) == 1

    def test_call_count(self):
        node = Call(fn_name="add", args=(Column("a"), Column("b")))
        assert node_count(node) == 3  # 1 call + 2 leaves

    def test_nested_count(self):
        inner = Call(fn_name="mul", args=(Column("a"), Column("b")))
        node = Call(fn_name="add", args=(inner, Column("c")))
        assert node_count(node) == 5  # 2 calls + 3 leaves

    def test_keyword_argument_is_transparent(self):
        node = Call(
            "clip",
            (
                Call("abs", (Column("close"),)),
                KeywordArg("lower_bound", Literal(0)),
            ),
        )

        assert node_count(node) == 4


class TestRPN:
    def test_simple_rpn(self):
        """a + b → [Feat(a), Feat(b), Op(add)]"""
        node = Call(fn_name="add", args=(Column("a"), Column("b")))
        rpn = to_rpn(node)
        assert len(rpn) == 3
        assert rpn[0].type == TokenType.FEATURE
        assert rpn[1].type == TokenType.FEATURE
        assert rpn[2].type == TokenType.OPERATOR
        assert rpn[2].name == "add"

    def test_nested_rpn(self):
        """(a + b) * c → a b + c *"""
        inner = Call(fn_name="add", args=(Column("a"), Column("b")))
        node = Call(fn_name="mul", args=(inner, Column("c")))
        rpn = to_rpn(node)
        assert [t.name for t in rpn] == ["a", "b", "add", "c", "mul"]

    def test_keyword_argument_is_transparent(self):
        node = Call(
            "clip",
            (
                Call("abs", (Column("close"),)),
                KeywordArg("lower_bound", Literal(0)),
            ),
        )

        assert [token.name for token in to_rpn(node)] == ["close", "abs", "0", "clip"]


class TestDescendants:
    def test_flat_list(self):
        node = Call(fn_name="add", args=(Column("a"), Column("b")))
        desc = descendants(node)
        assert len(desc) == 3  # (a+b, 2), (a, 0), (b, 0)

    def test_keyword_argument_is_transparent(self):
        node = Call(
            "clip",
            (Column("close"), KeywordArg("lower_bound", Literal(0))),
        )

        assert descendants(node) == [
            ("clip(close, lower_bound=0)", 2),
            ("close", 0),
            ("0", 0),
        ]


class TestCommonSubexprs:
    def test_no_repetition(self):
        node = Call(fn_name="add", args=(Column("a"), Column("b")))
        result = common_subexprs(node)
        # Every sub-expr appears once
        assert all(v == 1 for v in result.values())


class TestPreCalItems:
    def test_not_frequent_enough(self):
        """Simple expression should return no pre-cal items."""
        node = Call(fn_name="add", args=(Column("a"), Column("b")))
        items = pre_cal_items(node, filter_value=3, least_depth=1)
        assert items == []
