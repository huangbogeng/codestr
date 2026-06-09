from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from codestr.tokens import Token, TokenType


# ---- Base class -----------------------------------------------------------

class ExprNode(ABC):
    """Abstract base for all AST nodes."""

    @property
    @abstractmethod
    def alias(self) -> str:
        """Human-readable string representation of this node."""
        ...

    @abstractmethod
    def __hash__(self) -> int:
        ...

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        ...


# ---- Leaf nodes -----------------------------------------------------------

@dataclass(frozen=True)
class Column(ExprNode):
    """A column reference, e.g. ``close``, ``volume``."""

    name: str

    @property
    def alias(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(("column", self.name))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Column):
            return NotImplemented
        return self.name == other.name


@dataclass(frozen=True)
class Literal(ExprNode):
    """A numeric literal, e.g. ``5``, ``3.14``."""

    value: int | float

    @property
    def alias(self) -> str:
        return str(self.value)

    def __str__(self) -> str:
        return str(self.value)

    def __hash__(self) -> int:
        return hash(("literal", self.value))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Literal):
            return NotImplemented
        return self.value == other.value


# ---- Internal node --------------------------------------------------------

_unary_map = {"neg": "-", "not_": "!"}
_binary_map = {
    "add": "+", "mul": "*", "div": "/", "sub": "-",
    "floordiv": "//", "mod": "%", "pow": "**",
    "and_": "&", "or_": "|",
    "gt": ">", "ge": ">=", "lt": "<", "le": "<=",
    "eq": "==", "neq": "!=",
}


def _render_call(fn_name: str, args: tuple[ExprNode, ...]) -> str:
    """Render a Call node's alias from its fn_name and args."""
    if fn_name == "if_":
        return f"{args[0]}?{args[1]}:{args[2]}"
    if fn_name in _unary_map:
        return f"{_unary_map[fn_name]}{args[0]}"
    if fn_name in _binary_map:
        return f"({_binary_map[fn_name].join(str(a) for a in args)})"
    return f"{fn_name}({', '.join(str(a) for a in args)})"


@dataclass(frozen=True)
class Call(ExprNode):
    """A function / operator invocation, e.g. ``ts_mean(close, 5)``."""

    fn_name: str
    args: tuple[ExprNode, ...] = ()
    _alias: str = ""          # cached rendering; set via __post_init__

    def __post_init__(self) -> None:
        if not self._alias:
            object.__setattr__(self, "_alias", _render_call(self.fn_name, self.args))

    @property
    def alias(self) -> str:
        return self._alias

    def __str__(self) -> str:
        return self._alias

    # alias 不参与 hash / eq，只由 fn_name + args 决定身份
    def __hash__(self) -> int:
        return hash((self.fn_name, self.args))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Call):
            return NotImplemented
        return self.fn_name == other.fn_name and self.args == other.args


# ---- Analysis helpers -----------------------------------------------------

def depth(node: ExprNode) -> int:
    """Max nesting depth.  Leaf nodes = 1."""
    if isinstance(node, (Column, Literal)):
        return 1
    if isinstance(node, Call):
        max_child = 0
        for arg in node.args:
            max_child = max(max_child, depth(arg))
        return 1 + max_child
    raise TypeError(f"Unknown node type: {type(node)}")


def node_count(node: ExprNode) -> int:
    """Total number of nodes in the tree."""
    if isinstance(node, (Column, Literal)):
        return 1
    if isinstance(node, Call):
        total = 1
        for arg in node.args:
            total += node_count(arg)
        return total
    raise TypeError(f"Unknown node type: {type(node)}")


def to_rpn(node: ExprNode) -> list[Token]:
    """Convert an AST to Reverse Polish Notation token list."""
    rpn: list[Token] = []

    def traverse(n: ExprNode) -> None:
        if isinstance(n, Call):
            for arg in n.args:
                traverse(arg)
            rpn.append(Token(
                type=TokenType.OPERATOR,
                name=n.fn_name,
                arity=len(n.args),
                value=n.fn_name,
            ))
        elif isinstance(n, Column):
            rpn.append(Token(
                type=TokenType.FEATURE,
                name=n.name,
                value=n.name,
            ))
        elif isinstance(n, Literal):
            rpn.append(Token(
                type=TokenType.CONSTANT,
                name=str(n.value),
                value=n.value,
            ))

    traverse(node)
    return rpn


def descendants(node: ExprNode) -> list[tuple[str, int]]:
    """Flatten the tree into (sub-expression-string, depth) pairs for frequency analysis."""
    result: list[tuple[str, int]] = []
    if isinstance(node, Call):
        result.append((str(node), depth(node)))
        for arg in node.args:
            result.extend(descendants(arg))
    else:
        result.append((str(node), 0))
    return result


def common_subexprs(node: ExprNode) -> dict[tuple[str, int], int]:
    """Count occurrences of each sub-expression.  Key is (expr_string, depth)."""
    from collections import Counter
    return dict(Counter(descendants(node)))


def pre_cal_items(node: ExprNode, filter_value: int = 3, least_depth: int = 3) -> list[str]:
    """Identify high-frequency sub-expressions that are worth caching.

    Only keeps sub-expressions with depth >= *least_depth* that appear at
    least *filter_value* times.
    """
    return [
        k[0]
        for k, v in common_subexprs(node).items()
        if v >= filter_value and k[1] >= least_depth
    ]
