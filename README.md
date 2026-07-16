# CodeStr

[![CI](https://github.com/huangbogeng/codestr/actions/workflows/ci.yml/badge.svg)](https://github.com/huangbogeng/codestr/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

CodeStr 是一个专为量化因子挖掘设计的 DSL → Polars Expr 表达式计算引擎，提供高效的表达式转译、缓存与执行。

## 安装

```bash
git clone https://github.com/huangbogeng/codestr.git
cd codestr
uv sync --extra dev
```

## 快速开始

```python
import polars as pl
from codestr import CodeStr

# 标准面板数据 (time, entity)
df = pl.DataFrame({
    "datetime": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
    "asset":    ["A", "B", "A", "B"],
    "close":    [100.0, 200.0, 101.0, 198.0],
    "volume":   [1000.0, 2000.0, 1100.0, 1900.0],
})

cs = CodeStr(df, index=("datetime", "asset"))

# 交互式查询 — 结果自动缓存
result = cs.sql(
    "ts_mean(close, 5) as ma5",
    "cs_rank(close) as rank",
    "close / ts_delay(close, 1) - 1 as ret",
)
print(result)
```

## 两种 API 模式

| 模式 | API | 行为 |
|------|-----|------|
| **纯编译** | `cs.compile(expr) -> pl.Expr` | 无副作用，返回 Polars 表达式 |
| **交互式** | `cs.sql(expr, lazy=False) -> pl.DataFrame` | 有状态，自动缓存与复用 |

```python
# 纯编译 — 表达式可被任意 DataFrame 消费
expr = cs.compile("ts_mean(close, 5) as ma5")
other_df.with_columns(expr)

# 交互式 — 适合逐步构建因子
cs.sql("close + volume as total")
cs.sql("ts_mean(total, 5) as total_ma5")  # 复用上一步的 total
```

## 窗口配置

CodeStr 使用 `partition_by`（实体分组轴）和 `order_by`（时间排序轴）控制窗口算子：

```python
# 默认配置
cs = CodeStr(df)
# index=("datetime", "asset")
# → TS: over(partition_by=["asset"], order_by=["datetime"])
# → CS: over(partition_by=["datetime"], order_by=["asset"])

# 自定义列名
cs = CodeStr(df, index=("trade_date", "stock_code"))

# 多列窗口 — 按行业+股票分组，按日期+逐笔序号排序
cs = CodeStr(df,
    index=("trade_date", "stock_code"),
    partition_by=["industry", "stock_code"],
    order_by=["trade_date", "tick"],
)
```

| 算子类别 | 窗口规则 |
|---------|---------|
| **TS (时序)** | `over(partition_by=partition_by, order_by=order_by)` |
| **CS (截面)** | `over(partition_by=order_by, order_by=partition_by)` |

## 自定义算子

```python
from codestr.udf.registry import udf
import polars as pl

@udf(category="ts")
def ts_ewm(expr: pl.Expr, windows, partition_by=None, order_by=None):
    """指数加权移动平均"""
    return expr.ewm_mean(halflife=windows).over(
        partition_by=partition_by, order_by=order_by
    )

cs.sql("ts_ewm(close, 10) as ewm10")
```

## 内置算子

**基础算子** (`base_udf`)：`abs`, `log`, `sqrt`, `square`, `cube`, `sin`, `cos`, `tan`, `exp`, `sigmoid`, `sign`, `clip`, `trunc`, `between`, `cast`, `max`, `min`, `sum`, `mean`, `arg_max`, `arg_min`, `if_`, `fib` 等

**截面算子** (`cs_udf`)：`cs_rank`, `cs_zscore`, `cs_demean`, `cs_mean`, `cs_std`, `cs_var`, `cs_skew`, `cs_ic`, `cs_corr`, `cs_slope`, `cs_resid`, `cs_qcut`, `cs_midby`, `cs_meanby` 等

**时序算子** (`ts_udf`)：`ts_mean`, `ts_ema`, `ts_sum`, `ts_std`, `ts_var`, `ts_skew`, `ts_kurt`, `ts_max`, `ts_min`, `ts_mid`, `ts_delay`, `ts_delta`, `ts_mad` 等

## 项目结构

```
src/codestr/
├── engine.py            # CodeStr 引擎入口
├── compiler.py          # AST → Polars Expr 编译器
├── parser.py            # DSL 解析器 (Lark LALR grammar)
├── syntax.py            # AST 节点定义
├── tokens.py            # Token 定义
├── errors.py            # 异常类型
└── udf/
    ├── registry.py      # UDF 注册中心 (@udf 装饰器)
    ├── base_udf.py      # 基础算子
    ├── cs_udf.py         # 截面算子 (Cross-Section)
    └── ts_udf.py         # 时序算子 (Time-Series)
```
