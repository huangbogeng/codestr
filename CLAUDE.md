# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

CodeStr 是一个专为量化因子挖掘设计的表达式计算引擎，基于 [Polars](https://pola.rs/) 构建，提供 DSL → Polars Expr 的高效转译与执行。

## 项目结构

```
codestr/
├── __init__.py    # 导出 CodeStr, Expr
├── engine.py      # CodeStr 引擎：compile() 纯编译 + sql() 交互式有状态
├── compiler.py    # DSL → Polars Expr 编译器
├── ast.py         # 抽象语法树定义
├── parser.py      # DSL 解析器
├── expr.py        # 表达式抽象层
├── types.py       # 类型系统
├── errors.py      # 错误处理
├── udf/           # 用户自定义函数
└── README.md      # 项目文档
```

## 核心架构

- CodeStr 编译为 Polars 表达式，最终由 `polars` 运行时执行
- 两种 API 模式：`compile()` 纯函数式，`sql()` 有状态交互式
- 表达式级缓存复用已有计算结果
- 支持自定义 UDF
