from __future__ import annotations

import re

from lark import Lark, Transformer, v_args

from codestr.errors import ParseError
from codestr.syntax import Call, Column, ExprNode, Literal

_GRAMMAR = r"""
    start: expr
    ?expr: ternary_expr
    ?ternary_expr: or_expr
         | or_expr "?" or_expr ":" ternary_expr -> ternary
    ?or_expr: and_expr
         | or_expr "|" and_expr -> or_
    ?and_expr: comp_expr
         | and_expr "&" comp_expr -> and_
    ?comp_expr: eq_expr
         | comp_expr "<" eq_expr -> lt
         | comp_expr ">" eq_expr -> gt
         | comp_expr "<=" eq_expr -> le
         | comp_expr ">=" eq_expr -> ge
    ?eq_expr: arith_expr
         | eq_expr "==" arith_expr -> eq
         | eq_expr "!=" arith_expr -> neq
    ?arith_expr: term
         | arith_expr "+" term -> add
         | arith_expr "-" term -> sub
    ?term: pow_expr
         | term "*" pow_expr -> mul
         | term "/" pow_expr -> div
         | term "//" pow_expr -> floordiv
         | term "%" pow_expr -> mod
    ?pow_expr: factor
         | factor "**" pow_expr -> pow
    ?factor: atom
         | "-" factor -> neg
         | "~" factor -> not_
    ?atom: function
         | NAME
         | NUMBER
         | FLOAT
         | "(" expr ")"
         | implicit_mul
         | attribute_access
    implicit_mul: (NUMBER | FLOAT) NAME -> implicit_mul
    attribute_access: atom "." NAME -> attribute_access
    function: NAME "(" expr_list ")" -> function
    keyword_arg: NAME "=" expr -> keyword_arg
    expr_list: (expr | keyword_arg) ("," (expr | keyword_arg))*
    NAME: /[a-zA-Z_$,][a-zA-Z0-9_$]*/
    NUMBER: /\d+/
    FLOAT: /\d+\.\d+/
    %import common.WS
    %ignore WS
"""


class ExprBuilder(Transformer):
    """Lark transformer producing ExprNode AST."""

    # ---- leaf tokens -------------------------------------------------------

    def NAME(self, name) -> Column:  # noqa: N802
        return Column(str(name))

    def NUMBER(self, number) -> Literal:  # noqa: N802
        return Literal(int(number))

    def FLOAT(self, number) -> Literal:  # noqa: N802
        return Literal(float(number))

    # ---- attribute access --------------------------------------------------

    def attribute_access(self, items) -> Column:
        parts = [i.name if isinstance(i, Column) else str(i) for i in items]
        return Column(".".join(parts))

    def keyword_arg(self, item) -> dict:
        key, val = item
        key = key.name if isinstance(key, Column) else str(key)
        return {key: val}

    # ---- top-level ----------------------------------------------------------

    def start(self, items):
        return items[0]

    # ---- ternary -----------------------------------------------------------

    @v_args(inline=True)
    def ternary(self, a, b, c) -> Call:
        return Call(fn_name="if_", args=(a, b, c))

    # ---- binary ops ---------------------------------------------------------

    @staticmethod
    def _make_bin(fn_name: str):
        def method(self, items):
            return Call(fn_name=fn_name, args=tuple(items))

        return method

    # ---- unary --------------------------------------------------------------

    def neg(self, items) -> ExprNode:
        item = items[0]
        if isinstance(item, Literal):
            return Literal(-item.value)
        return Call(fn_name="neg", args=(item,))

    def not_(self, items) -> Call:
        return Call(fn_name="not_", args=tuple(items))

    # ---- function call -----------------------------------------------------

    def function(self, items) -> Call:
        name = items[0]
        fn_name = name.name if isinstance(name, Column) else str(name)
        args = tuple(items[1]) if len(items) > 1 and items[1] else ()
        return Call(fn_name=fn_name, args=args)

    def expr_list(self, items):
        return items


# Generate binary-operator transformer methods
for _op in (
    "add",
    "sub",
    "mul",
    "div",
    "floordiv",
    "mod",
    "pow",
    "and_",
    "or_",
    "eq",
    "neq",
    "lt",
    "gt",
    "le",
    "ge",
):
    setattr(ExprBuilder, _op, ExprBuilder._make_bin(_op))
ExprBuilder.implicit_mul = ExprBuilder._make_bin("mul")


_parser: Lark | None = None


def _get_parser() -> Lark:
    global _parser
    if _parser is None:
        _parser = Lark(_GRAMMAR, parser="lalr", transformer=ExprBuilder())
    return _parser


def _normalize(expr: str) -> str:
    for old, new in {
        "if(": "if_(",
        "not(": "not_(",
        "and(": "and_(",
        "or(": "or_(",
        "$": "",
        "\n": "",
    }.items():
        expr = expr.replace(old, new)
    # Convert standalone ! (logical not) to ~, but preserve != (not-equal)
    expr = re.sub(r"(?<!!)!(?!=)", "~", expr)
    return expr


def parse(expression: str) -> ExprNode:
    """Parse a DSL expression string into an ExprNode AST."""
    expression = _normalize(expression)

    alias = None
    m = re.search(r"(?i)(.+?)\s+AS\s+(\w+)", expression)
    if m:
        expression = m.group(1).strip()
        alias = m.group(2).strip()

    try:
        result = _get_parser().parse(expression)
    except Exception as e:
        raise ParseError(f"{expression}\n{e}") from e

    # result is always an ExprNode now (Column / Literal / Call)
    if alias and isinstance(result, Call):
        object.__setattr__(result, "_alias", alias)

    return result
