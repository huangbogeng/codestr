"""Tests for DSL parser (parser.py)."""

import pytest

from codestr.errors import ParseError
from codestr.parser import parse
from codestr.syntax import Call, Column, Literal


class TestLeafNodes:
    def test_column_basic(self):
        node = parse("close")
        assert isinstance(node, Column)
        assert node.name == "close"
        assert node.alias == "close"

    def test_column_with_underscore(self):
        node = parse("_my_col")
        assert isinstance(node, Column)
        assert node.name == "_my_col"

    def test_column_with_dollar(self):
        node = parse("$close")
        assert isinstance(node, Column)
        assert node.name == "close"

    def test_int_literal(self):
        node = parse("42")
        assert isinstance(node, Literal)
        assert node.value == 42

    def test_float_literal(self):
        node = parse("3.14")
        assert isinstance(node, Literal)
        assert node.value == 3.14


class TestBinaryOperators:
    @pytest.mark.parametrize(
        "expr,fn_name",
        [
            ("close + volume", "add"),
            ("close - volume", "sub"),
            ("close * volume", "mul"),
            ("close / volume", "div"),
            ("close // volume", "floordiv"),
            ("close % volume", "mod"),
            ("close ** volume", "pow"),
            ("close & volume", "and_"),
            ("close | volume", "or_"),
            ("close > volume", "gt"),
            ("close >= volume", "ge"),
            ("close < volume", "lt"),
            ("close <= volume", "le"),
            ("close == volume", "eq"),
            ("close != volume", "neq"),
        ],
    )
    def test_binary_ops(self, expr, fn_name):
        node = parse(expr)
        assert isinstance(node, Call)
        assert node.fn_name == fn_name
        assert len(node.args) == 2

    def test_add_mul_precedence(self):
        """a + b * c → add(a, mul(b, c))"""
        node = parse("close + volume * high")
        assert node.fn_name == "add"
        assert node.args[1].fn_name == "mul"

    def test_mul_add_precedence(self):
        """a * b + c → add(mul(a, b), c)"""
        node = parse("close * volume + high")
        assert node.fn_name == "add"
        assert node.args[0].fn_name == "mul"

    def test_parens_override_precedence(self):
        """(a + b) * c → mul(add(a, b), c)"""
        node = parse("(close + volume) * high")
        assert node.fn_name == "mul"
        assert node.args[0].fn_name == "add"


class TestUnaryOperators:
    def test_neg_number(self):
        """-5 should fold to Literal(-5)"""
        node = parse("-5")
        assert isinstance(node, Literal)
        assert node.value == -5

    def test_neg_float(self):
        node = parse("-3.14")
        assert isinstance(node, Literal)
        assert node.value == -3.14

    def test_neg_column(self):
        """-close → neg(close)"""
        node = parse("-close")
        assert isinstance(node, Call)
        assert node.fn_name == "neg"

    def test_not_expr(self):
        """~close → not_(close)"""
        node = parse("~close")
        assert isinstance(node, Call)
        assert node.fn_name == "not_"

    def test_bang_not(self):
        """!close should be normalized to ~close → not_(close)"""
        node = parse("!close")
        assert isinstance(node, Call)
        assert node.fn_name == "not_"

    def test_bang_not_preserves_neq(self):
        """close != 5 should remain neq, not become =~"""
        node = parse("close != 5")
        assert node.fn_name == "neq"


class TestTernary:
    def test_ternary_basic(self):
        node = parse("close > 0 ? close : 0")
        assert node.fn_name == "if_"
        assert len(node.args) == 3
        assert str(node.args[0]) == "(close>0)"
        assert str(node.args[1]) == "close"
        assert str(node.args[2]) == "0"

    def test_nested_ternary(self):
        """Nested ternary requires parentheses: a ? (b ? c : d) : e"""
        node = parse("a ? (b ? c : d) : e")
        assert node.fn_name == "if_"
        # Outer: if_(a, inner, e)
        assert node.args[1].fn_name == "if_"


class TestFunctionCalls:
    def test_simple_function(self):
        node = parse("ts_mean(close, 5)")
        assert node.fn_name == "ts_mean"
        assert len(node.args) == 2
        assert str(node.args[0]) == "close"
        assert str(node.args[1]) == "5"

    def test_nested_function(self):
        node = parse("abs(ts_mean(close, 5))")
        assert node.fn_name == "abs"
        assert node.args[0].fn_name == "ts_mean"

    def test_keyword_arguments(self):
        node = parse("clip(close, lower_bound=0.1)")
        assert node.fn_name == "clip"
        assert len(node.args) == 2

    def test_normalize_if(self):
        """if(a, b, c) should normalize to if_(a, b, c)"""
        node = parse("if(close > 0, close, 0)")
        assert node.fn_name == "if_"

    def test_normalize_not(self):
        node = parse("not(close)")
        assert node.fn_name == "not_"

    def test_normalize_and(self):
        node = parse("and(close, volume)")
        assert node.fn_name == "and_"

    def test_normalize_or(self):
        node = parse("or(close, volume)")
        assert node.fn_name == "or_"


class TestImplicitMultiplication:
    def test_number_then_column(self):
        node = parse("5close")
        assert node.fn_name == "mul"
        assert str(node.args[0]) == "5"
        assert str(node.args[1]) == "close"

    def test_float_then_column(self):
        node = parse("3.14close")
        assert node.fn_name == "mul"
        assert str(node.args[0]) == "3.14"


class TestAttributeAccess:
    def test_single_dot(self):
        node = parse("df.close")
        assert isinstance(node, Column)
        assert node.name == "df.close"

    def test_chained_attr(self):
        node = parse("df.close.mean")
        assert isinstance(node, Column)
        assert node.name == "df.close.mean"


class TestAlias:
    def test_as_alias_on_call(self):
        node = parse("ts_mean(close, 5) as ma5")
        assert node.alias == "ma5"

    def test_no_alias(self):
        node = parse("close")
        assert node.alias == "close"


class TestParseErrors:
    def test_empty_string(self):
        with pytest.raises(ParseError):
            parse("")

    def test_unbalanced_parens(self):
        with pytest.raises(ParseError):
            parse("close + (volume * high")

    def test_incomplete_expr(self):
        with pytest.raises(ParseError):
            parse("close +")
