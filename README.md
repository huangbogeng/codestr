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

## API 模式

| 模式 | API | 行为 |
|------|-----|------|
| **纯编译** | `cs.compile(expr) -> pl.Expr` | 无副作用，返回 Polars 表达式 |
| **交互式** | `cs.sql(expr, lazy=False) -> pl.DataFrame` | 有状态，自动缓存与复用 |
| **静态验证** | `cs.validate_expr(*exprs) -> list[dict]` | 无副作用，按当前 schema 干编译 |

```python
# 纯编译 — 表达式可被任意 DataFrame 消费
expr = cs.compile("ts_mean(close, 5) as ma5")
other_df.with_columns(expr)

# 交互式 — 适合逐步构建因子
cs.sql("close + volume as total")
cs.sql("ts_mean(total, 5) as total_ma5")  # 复用上一步的 total
```

### 静态验证

`validate_expr()` 复用 `sql()` 的 planner，在当前 LazyFrame schema 上解析
UDF、列引用、类型兼容性和混合窗口，但只调用 `collect_schema()`，不会
`collect()` 数据或修改引擎缓存：

```python
results = cs.validate_expr(
    "sin(1.0) as invalid",
    "ts_mean(cs_moderate(close), 5) as factor",
)
```

每个结果包含 `expr`、`valid`、`stage`、`error_type` 和 `message`。
失败阶段分为 `structural`、`compile` 和 `schema`。批量输入相互独立，
后一条表达式不能引用同批前一条表达式新建的别名。依赖实际数据值、
只在物化时出现的错误不属于该 API 的检查范围。

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

### 混合窗口

`CodeStr.sql()` 会自动把 TS/CS 混合窗口拆成连续的 lazy projection：

```python
cs.sql(
    "close * 2 as scaled",
    "ts_mean(cs_moderate(scaled), 60) as factor",
)
```

其语义等价于先生成 `cs_moderate(scaled)` 中间列，再沿资产时间轴
计算 `ts_mean`。中间列不会出现在返回结果中。

纯 `cs.compile()` 只能返回一个 `pl.Expr`，因此会明确拒绝需要多阶段
执行的 TS/CS 混合窗口。TS→TS 和 CS→CS 同域嵌套不受影响。

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

滚动统计与 `ts_ema` 支持关键字参数 `min_samples`。滚动统计省略该参数时沿用 Polars 默认值 `None`（需要完整窗口），`ts_ema` 则沿用 Polars 默认值 `1`：

```python
cs.sql("ts_mean(close, 20, min_samples=5) as ma20")
cs.sql("ts_ema(close, 10, min_samples=3) as ema10")
```

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
