# CodeStr Engine

CodeStr 是一个专为量化因子挖掘设计的表达式计算引擎，基于 Polars 构建，提供 DSL → Polars Expr 的高效转译与执行。

## 核心模式

### 纯编译模式（Batch-Lazy-Eval）

**API**：`CodeStr.compile(expr_str: str) -> pl.Expr`

- 无副作用，返回独立的 `pl.Expr`
- 面向 RL / 批量评估流程
- 支持“One Batch, One Collect”计算图合并

### 交互式/状态模式（Interactive-Stateful）

**API**：`CodeStr.sql(expr_str: str, ...)`

- 有副作用，结果列会注册到 CodeStr 内部数据集中
- 内置表达式级缓存，避免重复计算
- 面向 GP 与交互式研究流程

## CodeStr Contract

### `compile` 是纯编译接口

- `CodeStr.compile(expr_str)` 只负责把 DSL 转成 `pl.Expr`
- 不修改内部数据集，不写入缓存，不产生执行副作用
- 编译结果可被外部批量组装后统一执行（One Batch, One Collect）

### `sql` 的 lazy / eager 行为

- `CodeStr.sql(expr_str, lazy=True)` 返回 `pl.LazyFrame`，仅构建查询计划，不立即执行
- `CodeStr.sql(expr_str, lazy=False)`（默认）返回物化后的 `pl.DataFrame`
- `sql` 路径允许状态更新与表达式缓存复用，适合交互式增量计算

## 使用示例

```python
from codestr.engine import CodeStr

cs = CodeStr(data=df, index=("datetime", "asset"))

# 交互式执行
df_out = cs.sql("ts_mean(close, 5) as ma5", cover=True)

# 纯编译
expr = cs.compile("ts_mean(close, 5)")
```

## 目录结构

- engine.py：引擎入口
- expr.py：表达式解析与 AST 构建
- tokens.py：Token 定义
- udf/：算子实现（ts/cs/base）
