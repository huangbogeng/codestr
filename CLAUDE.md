# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

CodeStr 是一个专为量化因子挖掘设计的表达式计算引擎，基于 [Polars](https://pola.rs/) 构建，提供 DSL → Polars Expr 的高效转译与执行。

## 开发命令

```bash
uv venv                          # 创建虚拟环境
uv pip install -e .              # 安装开发模式

# 代码质量（与 CI 一致）
uvx ruff check src/              # lint 检查
uvx ruff format --check src/     # 格式检查（CI 模式）
uvx ruff format src/             # 自动格式化
```

项目使用 `hatchling` 构建，`ruff` 做 lint/格式化。CI 见 `.github/workflows/ci.yml`，在 push/PR 到 master 时运行 lint + format check，Python 3.12。

**目前没有测试**（`tests/` 目录只有 `__init__.py`，缺少 pytest 配置。添加测试后运行方式为 `uv run pytest`）。

## 项目结构

```
src/codestr/
├── __init__.py          # 公开 API：CodeStr, ExprNode, Call, Column, Literal
├── engine.py            # CodeStr 引擎：compile() 纯编译 + sql() 交互式有状态
├── compiler.py          # AST → Polars Expr 编译器（纯函数，无副作用）
├── parser.py            # DSL 解析器（Lark LALR grammar + Transformer → AST）
├── syntax.py            # AST 定义：ExprNode, Column, Literal, Call + 分析辅助函数
├── tokens.py            # Token/TokenType 定义（RPN 分析用）
├── errors.py            # 异常：ParseError, CompileError, PolarsError, FailError
└── udf/
    ├── registry.py      # UDF 注册中心（单例 + @udf 装饰器）
    ├── base_udf.py      # 基础算子：一元/二元/三元 + 算术/逻辑运算
    ├── cs_udf.py         # 截面算子 (Cross-Section)，如 cs_rank, cs_zscore
    └── ts_udf.py         # 时序算子 (Time-Series)，如 ts_mean, ts_delay
```

## 核心架构

### 数据流

```
DSL 字符串 → parser.parse() → AST (ExprNode) → compiler.compile() → pl.Expr → Polars 执行
```

- **Parser** (`parser.py`): 用 Lark LALR 解析器将 DSL 字符串转为 AST。`_normalize()` 预处理：`if(` → `if_(`，`!` → `~`（但保留 `!=`），移除 `$` 和换行。`as` 关键字后的别名会被注入 AST 的 `_alias` 字段。
- **Compiler** (`compiler.py`): 纯函数，AST → `pl.Expr`。`_resolve()` 对 Literal 返回 Python 标量，对 Column/Call 返回 `pl.Expr`。`_compile()` 总是返回 `pl.Expr`（Literal 包装为 `pl.lit()`）。函数从 UDFRegistry 查找，通过 `inspect.signature` 注入 `dims` 参数。
- **Engine** (`engine.py`): `CodeStr` 类封装了上述流程，额外管理状态（数据对齐、表达式缓存、惰性计算图）。

### 两种 API 模式

| 模式 | API | 副作用 | 适用场景 |
|------|-----|--------|---------|
| 纯编译 | `compile(expr_str) -> pl.Expr` | 无 | 批量/RL 评估 |
| 交互式 | `sql(expr_str, lazy=False) -> pl.DataFrame` | 有（更新内部状态） | 交互式研究/GP |

- `sql(lazy=True)` 返回 `pl.LazyFrame`，不物化，不更新内部状态（rollback `_data_`）。
- `sql(lazy=False)`（默认）collect 后物化，更新 `_expr_cache` 和 `data`。

### 表达式缓存策略

两层缓存，都基于 `Call.__hash__`（只看 `fn_name + args`，**别名不参与**）：

1. **`_expr_cache`** — 跨查询持久化缓存，只在 eager `sql()` 成功后合并入。
2. **`_cur_expr_cache`** — 单次查询临时缓存，查询成功后才 merge 到 `_expr_cache`。

缓存命中时直接 `pl.col(cached_alias).alias(new_alias)` 复用，避免重编译。`cover=True` 强制绕过缓存。

### CS vs TS 算子窗口模式

这是理解算子分类的核心：

| 类别 | 文件 | OVER 窗口 | 含义 |
|------|------|-----------|------|
| CS（截面） | `cs_udf.py` | `partition_by=["datetime"]`, `order_by=["asset"]` | 每个时间截面上对资产做排序/排名 |
| TS（时序） | `ts_udf.py` | `partition_by=["asset"]`, `order_by=["datetime"]` | 每个资产沿时间轴做滚动窗口 |

两个文件都有 `configure_over()` 允许运行时修改默认窗口。

### UDF 注册

`UDFRegistry` 是单例。三种注册方式：
1. `@udf(category="ts")` 装饰器（自动推断 arity）
2. `CodeStr.register_udf(func, name="...")` 实例方法
3. 直接 `UDFRegistry.get_instance().register(UDFMeta(...))`

编译器通过 `inspect.signature` 检查函数签名，如果接受 `dims` 参数则通过 `toolz.partial` 注入。

### DSL 语法

运算符优先级（从低到高）：三元 `?:` → 或 `|` → 与 `&` → 比较 `<>` `<=` `>=` `==` `!=` → 加减 `+` `-` → 乘除 `*` `/` `//` `%` → 幂 `**` → 一元 `-` `~` → 函数调用/属性访问

支持隐式乘法：`5close` → `5 * close`。支持属性链：`df.close` → `Column("df.close")`。
