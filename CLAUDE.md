# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

CodeStr 是一个专为量化因子挖掘设计的表达式计算引擎，基于 [Polars](https://pola.rs/) 构建，提供 DSL → Polars Expr 的高效转译与执行。

## 项目结构

```
codestr/
├── pyproject.toml       # 项目元数据 + 依赖 (hatchling build)
├── .python-version      # Python 3.12 (uv 管理)
├── src/
│   └── codestr/
│       ├── __init__.py  # 导出 CodeStr, ExprNode, Call, Column, Literal
│       ├── engine.py    # CodeStr 引擎：compile() 纯编译 + sql() 交互式有状态
│       ├── compiler.py  # DSL → Polars Expr 编译器
│       ├── parser.py    # DSL 解析器 (Lark grammar)
│       ├── syntax.py    # AST 定义：ExprNode, Column, Literal, Call + 分析函数
│       ├── tokens.py    # Token/TokenType 定义
│       ├── errors.py    # 异常定义 (ParseError, CompileError, PolarsError...)
│       └── udf/
│           ├── registry.py   # UDF 注册中心 (单例 + @udf 装饰器)
│           ├── base_udf.py   # 基础算子 (一元/二元/三元)
│           ├── cs_udf.py     # 截面算子 (Cross-Section)
│           └── ts_udf.py     # 时序算子 (Time-Series)
└── tests/               # 测试目录
```

## 开发环境

```bash
uv venv               # 创建虚拟环境
uv pip install -e .   # 安装开发模式
```

## 核心架构

- CodeStr 编译为 Polars 表达式，最终由 `polars` 运行时执行
- 两种 API 模式：`compile()` 纯函数式，`sql()` 有状态交互式
- AST 三点层级：`ExprNode` → `Column` / `Literal` / `Call`
- 表达式级缓存复用已有计算结果，`Call.__hash__` 只看 `(fn_name, args)` 不受 alias 影响
- 支持通过 `@udf` 装饰器或 `engine.register_udf()` 注册自定义 UDF
