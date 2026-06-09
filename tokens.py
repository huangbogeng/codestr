from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class TokenType(Enum):
    OPERATOR = auto()
    FEATURE = auto()
    CONSTANT = auto()
    PARAM = auto()
    WINDOW = auto()
    BINS = auto()
    GENERIC = auto()


@dataclass(frozen=True)
class Token:
    type: TokenType
    name: str
    value: Any = None
    arity: int = 0
    arg_types: tuple[TokenType, ...] = ()

    def __str__(self):
        if self.type == TokenType.CONSTANT:
            return f"Const({self.value})"
        elif self.type == TokenType.OPERATOR:
            return f"Op({self.name})"
        elif self.type == TokenType.FEATURE:
            return f"Feat({self.name})"
        return f"Token({self.name})"
