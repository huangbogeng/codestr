# -*- coding: utf-8 -*-

from dataclasses import dataclass


@dataclass
class ParseError(Exception):
    """解析错误"""
    message: str

    def __str__(self):
        return self.message

    def __repr__(self):
        return self.__str__()


@dataclass
class CalculateError(Exception):
    """计算错误"""
    message: str

    def __str__(self):
        return self.message

    def __repr__(self):
        return self.__str__()


@dataclass
class CompileError(Exception):
    """编译错误"""
    message: str

    def __str__(self):
        return self.message

    def __repr__(self):
        return self.__str__()


@dataclass
class PolarsError(Exception):
    """Polars 引擎错误"""
    message: str

    def __str__(self):
        return self.message

    def __repr__(self):
        return self.__str__()


@dataclass
class FailError:
    """失败错误信息容器"""
    expr: str
    error: Exception

    def __str__(self):
        return f"""
[失败表达式]: {self.expr}
[错误类型]: {self.error.__class__.__name__}
[错误信息]: \n{self.error}
"""

    def __repr__(self):
        return self.__str__()
