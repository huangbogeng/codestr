# 算子参考手册

CodeStr 内置 **89 个算子**，按功能分为三大类：

| 类别 | 模块 | 数量 | 窗口注入 |
|------|------|:---:|---------|
| 基础算子 | `base_udf` | 55 | 无（逐元素计算） |
| 截面算子 | `cs_udf` | 21 | `over(partition_by=order_by, order_by=partition_by)` |
| 时序算子 | `ts_udf` | 13 | `over(partition_by=partition_by, order_by=order_by)` |

---

## 基础算子 (Math)

基础算子执行逐元素（element-wise）计算，不依赖窗口上下文。可直接用于标量、列表达式或可广播的混合操作。

### 算术运算

| 算子 | 签名 | 说明 | DSL 简写 |
|------|------|------|:---:|
| `add` | `(left, right)` | 加法 | `left + right` |
| `sub` | `(left, right)` | 减法 | `left - right` |
| `mul` | `(left, right)` | 乘法 | `left * right` |
| `div` | `(left, right)` | 除法 | `left / right` |
| `floordiv` | `(left, right)` | 整除 | `left // right` |
| `mod` | `(left, right)` | 取模 | `left % right` |
| `neg` | `(expr)` | 取相反数 | `-expr` |
| `abs` | `(expr)` | 绝对值 |  |

示例：
```python
cs.sql("close / volume as price_ratio")
cs.sql("(high - low) / close * 100 as volatility_pct")
cs.sql("-close as neg_close")
```

### 幂与根

| 算子 | 签名 | 说明 |
|------|------|------|
| `sqrt` | `(expr)` | 平方根 √x |
| `square` | `(expr)` | 平方 x² |
| `cube` | `(expr)` | 立方 x³ |
| `cbrt` | `(expr)` | 立方根 ∛x |
| `exp` | `(expr)` | 指数函数 eˣ |
| `log` | `(expr, base=e)` | 对数（默认自然对数） |
| `log1p` | `(expr)` | log(1 + x)，对小值更精确 |

示例：
```python
cs.sql("log(volume, 10) as log_volume")       # 以 10 为底
cs.sql("log(close) as log_close")              # 自然对数
cs.sql("sqrt(cube(close)) as close_pow")
```

### 三角函数

| 算子 | 说明 | 算子 | 说明 |
|------|------|------|------|
| `sin` | 正弦 | `arcsin` | 反正弦 |
| `cos` | 余弦 | `arccos` | 反余弦 |
| `tan` | 正切 | `arctan` | 反正切 |
| `cot` | 余切 | `sinh` | 双曲正弦 |
| `tanh` | 双曲正切 | `cosh` | 双曲余弦 |
| `arcsinh` | 反双曲正弦 | `arccosh` | 反双曲余弦 |
| `arctanh` | 反双曲正切 | `degrees` | 弧度转角度 |

### 逻辑运算

| 算子 | 签名 | 说明 | DSL 简写 |
|------|------|------|:---:|
| `and_` | `(left, right)` | 逻辑与 | `left & right` |
| `or_` | `(left, right)` | 逻辑或 | `left \| right` |
| `not_` | `(expr)` | 逻辑非 | `~expr` / `!expr` |

示例：
```python
cs.sql("close > 100 & volume < 5000 as signal")
cs.sql("not_(close.is_null()) as has_price")
```

### 比较运算

所有比较算子返回布尔值（`True` / `False`）：

| 算子 | 说明 | DSL 简写 |
|------|------|:---:|
| `eq` | 等于 | `left == right` |
| `neq` | 不等于 | `left != right` |
| `lt` | 小于 | `left < right` |
| `le` | 小于等于 | `left <= right` |
| `gt` | 大于 | `left > right` |
| `ge` | 大于等于 | `left >= right` |

示例：
```python
cs.sql("close >= ts_max(close, 20) as at_high")   # 收盘价是否等于或高于 20 日最高
cs.sql("volume != ts_delay(volume, 1) as vol_changed")
```

### 截断与裁剪

| 算子 | 签名 | 说明 |
|------|------|------|
| `clip` | `(expr, lower_bound=-inf, upper_bound=inf)` | 裁剪到 [lower, upper] 范围内 |
| `trunc` | `(expr, lower_bound=-inf, upper_bound=inf, left_closed=True, right_closed=True)` | 超出范围的值置为 null |
| `between` | `(expr, lower_bound=-inf, upper_bound=inf, close="both")` | 返回是否在范围内的布尔掩码 |

> **区别**：`clip` 将越界值钳位到边界上；`trunc` 将越界值设为 `null`。`between` 不修改值，只返回是否在范围内的布尔标记。

参数说明：
- `close`：`"both"`（默认）/ `"left"` / `"right"` / `"none"`，控制区间开闭
- `left_closed` / `right_closed`：控制截断边界的开闭

示例：
```python
cs.sql("clip(volume, 0, 100000) as capped_vol")          # 钳位成交量
cs.sql("trunc(close, lower_bound=0) as positive_close")  # 负值截为 null
cs.sql("between(close, 100, 200, close='both') as in_range")
```

### 水平聚合

对多列逐行取最大值/最小值/均值/求和，常用于多信号融合：

| 算子 | 签名 | 说明 |
|------|------|------|
| `max` | `(*exprs)` | 逐行最大值 |
| `min` | `(*exprs)` | 逐行最小值 |
| `sum` | `(*exprs)` | 逐行求和 |
| `mean` | `(*exprs)` | 逐行均值 |
| `arg_max` | `(*exprs)` | 最大值的列索引（0-based） |
| `arg_min` | `(*exprs)` | 最小值的列索引（0-based） |
| `concat` | `(*exprs)` | 拼接为列表 `[a, b, c, ...]` |

示例：
```python
cs.sql("max(signal_a, signal_b, signal_c) as best_signal")
cs.sql("arg_max(factor_1, factor_2, factor_3) as best_factor_idx")
cs.sql("concat(open, high, low, close) as ohlc")
```

### 其他一元算子

| 算子 | 签名 | 说明 |
|------|------|------|
| `sign` | `(expr)` | 符号函数：-1 / 0 / 1 |
| `sigmoid` | `(expr)` | Sigmoid：1/(1+e⁻ˣ) |
| `entropy` | `(expr)` | 信息熵 |
| `cast` | `(expr, dtype)` | 类型转换：`"int"` / `"float"` / `"cat"` / `"str"` |
| `null_type` | `(expr)` | 返回 `null`（忽略输入值，用于占位） |

示例：
```python
cs.sql("sign(ts_delta(close, 1)) as direction")        # 涨跌方向
cs.sql("sigmoid(cs_zscore(close)) as prob")            # 将 zscore 映射到 (0, 1)
cs.sql("cast(cs_qcut(close, 5), 'str') as group_str") # 分组标签转字符串
```

### 条件与三元运算

| 算子 | 签名 | 说明 | DSL 简写 |
|------|------|------|:---:|
| `if_` | `(cond, body, or_else)` | 条件分支 | `cond ? body : or_else` |
| `fib` | `(high, low, ratio=0.618)` | 斐波那契回调 |  |

> `if_` 在 DSL 中的别名是 `if`（自动规范化），支持 `?:` 三元语法糖。

示例：
```python
# 三目运算符
cs.sql("close > open ? 'up' : 'down' as direction")
# 函数调用形式
cs.sql("if_(volume > 0, close / volume, null) as vwap")
# 斐波那契回调
cs.sql("fib(high, low, 0.382) as fib_382")
cs.sql("fib(high, low, 0.618) as fib_618")
```

---

## 截面算子 (Cross-Section)

截面算子在每个时间截面上横向聚合所有实体，用于计算因子在横截面上的分布特征。

> 编译器自动注入窗口配置：`over(partition_by=datetime_cols, order_by=asset_cols)`

### 排名与标准化

| 算子 | 签名 | 说明 |
|------|------|------|
| `cs_rank` | `(expr)` | 截面排名（升序） |
| `cs_zscore` | `(expr)` | 截面 z-score：`(x - μ) / σ` |
| `cs_qcut` | `(expr, n_bins=10)` | 截面等分位数分组（返回 Int32 标签 1~n） |

示例：
```python
cs.sql("cs_rank(close) as close_rank")                # 按收盘价排名
cs.sql("cs_zscore(close) as close_z")                 # 截面标准化
cs.sql("cs_qcut(close, 5) as close_quintile")         # 分为 5 组
```

### 截面统计量

| 算子 | 签名 | 说明 |
|------|------|------|
| `cs_mean` | `(expr)` | 截面均值 |
| `cs_mid` | `(expr)` | 截面中位数 |
| `cs_std` | `(expr)` | 截面标准差 |
| `cs_var` | `(expr)` | 截面方差 |
| `cs_skew` | `(expr)` | 截面偏度 |
| `cs_max` | `(expr)` | 截面最大值 |
| `cs_min` | `(expr)` | 截面最小值 |

示例：
```python
cs.sql("cs_mean(close) as sector_avg")                # 截面平均股价
cs.sql("cs_std(cs_rank(factor)) as dispersion")       # 因子排名的截面离散度
```

### 去均值与偏离

| 算子 | 签名 | 说明 |
|------|------|------|
| `cs_demean` | `(expr)` | 去截面均值：`x - μ` |
| `cs_moderate` | `(expr)` | 偏离均值绝对值：`|x - μ|` |
| `cs_ufit` | `(expr)` | 偏离中位数绝对值（MAD）：`|x - median(x)|` |
| `cs_peakmax` | `(expr)` | 截面局部峰值 |
| `cs_peakmin` | `(expr)` | 截面局部谷值 |

示例：
```python
cs.sql("cs_demean(close) as close_dm")                # 去均值股价
cs.sql("cs_moderate(close) as close_dev")             # 偏离度
cs.sql("cs_ufit(close) as close_mad")                 # 稳健偏离度
```

### 截面相关与回归

| 算子 | 签名 | 说明 |
|------|------|------|
| `cs_ic` | `(left, right)` | Information Coefficient（Spearman 秩相关） |
| `cs_corr` | `(left, right)` | Pearson 线性相关 |
| `cs_slope` | `(left, right)` | 回归斜率：`ρ · σₗ / σᵣ` |
| `cs_resid` | `(left, right)` | 回归残差：`left - slope · right` |

示例：
```python
cs.sql("cs_ic(factor, ts_delta(close, 1) / ts_delay(close, 1)) as ic")
cs.sql("cs_slope(factor, return) as beta")
cs.sql("cs_resid(factor, beta) as alpha")              # 剥离 beta 后的 alpha
```

### 分组聚合

| 算子 | 签名 | 说明 |
|------|------|------|
| `cs_meanby` | `(expr, *by)` | 按 `by` 列分组的截面均值 |
| `cs_midby` | `(expr, *by)` | 按 `by` 列分组的截面中位数 |

> `cs_meanby` / `cs_midby` 在标准窗口配置的基础上，额外按 `*by` 列进行二级分组。

示例：
```python
cs.sql("cs_meanby(close, industry) as industry_avg")   # 行业内平均股价
cs.sql("cs_midby(factor, sector, cap_group) as group_median")
```

---

## 时序算子 (Time-Series)

时序算子在每个实体内部沿时间轴计算滚动窗口统计量。

> 编译器自动注入窗口配置：`over(partition_by=asset_cols, order_by=datetime_cols)`

滚动统计的通用签名为 `(expr, windows, min_samples=None)`，其中 `windows` 是整数，表示回溯窗口大小；`min_samples` 必须是正整数，省略时沿用 Polars 默认值 `None`（需要完整窗口）。`ts_ema` 的签名为 `(expr, span, min_samples=1)`，`span` 和 `min_samples` 都必须是正整数。滞后与差分算子仍使用 `(expr, windows)`，不接收 `min_samples`。

### 滚动统计

| 算子 | 说明 | 底层实现 |
|------|------|------|
| `ts_mean` | 滚动均值 | `rolling_mean(windows, min_samples=min_samples)` |
| `ts_ema` | 递归指数移动平均 | `ewm_mean(span=span, adjust=False, min_samples=min_samples)` |
| `ts_sum` | 滚动求和 | `rolling_sum(windows, min_samples=min_samples)` |
| `ts_std` | 滚动标准差 | `rolling_std(windows, min_samples=min_samples)` |
| `ts_var` | 滚动方差 | `rolling_var(windows, min_samples=min_samples)` |
| `ts_skew` | 滚动偏度 | `rolling_skew(windows, min_samples=min_samples)` |
| `ts_kurt` | 滚动峰度 | `rolling_kurtosis(windows, min_samples=min_samples)` |
| `ts_max` | 滚动最大值 | `rolling_max(windows, min_samples=min_samples)` |
| `ts_min` | 滚动最小值 | `rolling_min(windows, min_samples=min_samples)` |
| `ts_mid` | 滚动中位数 | `rolling_median(windows, min_samples=min_samples)` |
| `ts_mad` | 滚动 MAD | 内外两层 `rolling_median` 使用相同的 `min_samples` |

> `ts_mad` 计算 `median(|x - rolling_median(x)|) × 1.4826`。常数 1.4826 使其在正态分布下与标准差一致，内外两层滚动中位数共享同一个 `min_samples`。

示例：
```python
cs.sql("ts_mean(close, 5) as ma5")
cs.sql("ts_mean(close, 20, min_samples=5) as ma20_partial")
cs.sql("ts_ema(close, 10) as ema10")
cs.sql("ts_ema(close, 10, min_samples=3) as ema10_warmup")
cs.sql("ts_std(close, 20) as vol20")
cs.sql("ts_max(high, 20) as hh20")
cs.sql("ts_mad(close, 20, min_samples=5) as mad20")
```

### 滞后与差分

| 算子 | 说明 | 数学含义 |
|------|------|------|
| `ts_delay` | 滞后算子 | xₜ₋ₖ（`shift(k)`） |
| `ts_delta` | 差分算子 | xₜ − xₜ₋ₖ（`diff(k)`） |

示例：
```python
cs.sql("ts_delay(close, 1) as prev_close")
cs.sql("ts_delta(close, 5) as close_5d_chg")
cs.sql("close / ts_delay(close, 1) - 1 as ret")       # 收益率
cs.sql("ts_delta(ts_mean(close, 5), 1) as ma5_diff")  # 均线变化
```

---

## DSL 语法糖

以下 DSL 语法会被解析器自动转换为函数调用：

| 表达式 | 等价函数调用 | 说明 |
|--------|-------------|------|
| `a + b` | `add(a, b)` | 加法 |
| `a - b` | `sub(a, b)` | 减法 |
| `a * b` | `mul(a, b)` | 乘法 |
| `a / b` | `div(a, b)` | 除法 |
| `a // b` | `floordiv(a, b)` | 整除 |
| `a % b` | `mod(a, b)` | 取模 |
| `-a` | `neg(a)` | 取反 |
| `~a` / `!a` | `not_(a)` | 逻辑非 |
| `a & b` | `and_(a, b)` | 逻辑与 |
| `a \| b` | `or_(a, b)` | 逻辑或 |
| `a < b` | `lt(a, b)` | 小于 |
| `a <= b` | `le(a, b)` | 小于等于 |
| `a > b` | `gt(a, b)` | 大于 |
| `a >= b` | `ge(a, b)` | 大于等于 |
| `a == b` | `eq(a, b)` | 等于 |
| `a != b` | `neq(a, b)` | 不等于 |
| `a ? b : c` | `if_(a, b, c)` | 条件选择 |

注意事项：
- `$` 前缀会被自动移除（`$close` → `close`）
- `if(` / `and(` / `or(` / `not(` 会自动规范化为 `if_(` / `and_(` / `or_(` / `not_(`
- `**` 幂运算符暂不可用（`pow` 算子未注册），请使用 `square` / `cube` / `exp` / `log` 组合替代
- 支持隐式乘法：`5close` → `5 * close`
- 支持属性链：`df.close` → `Column("df.close")`

---

## 窗口配置

CodeStr 通过引擎构造参数控制算子窗口行为：

```python
cs = CodeStr(df,
    index=("datetime", "asset"),        # 数据对齐 & 结果选择
    partition_by=["asset", "industry"], # 实体分组轴
    order_by=["datetime", "tick"],      # 时间排序轴
)
```

| 算子类别 | 窗口等价调用 |
|---------|-------------|
| TS（时序） | `expr.rolling_*(w).over(partition_by=partition_by, order_by=order_by)` |
| CS（截面） | `expr.agg().over(partition_by=order_by, order_by=partition_by)` |
| Math（基础） | 无窗口（逐元素计算） |

CS 与 TS 的 `partition_by` / `order_by` **刚好交换**，分别实现"按时间分组，沿实体统计"（CS）和"按实体分组，沿时间统计"（TS）。

---

## 自定义算子

通过 `@udf` 装饰器注册自定义算子：

```python
from codestr.udf.registry import udf
import polars as pl

@udf(category="ts")
def ts_ewm(expr: pl.Expr, windows, partition_by=None, order_by=None):
    """指数加权移动平均"""
    return expr.ewm_mean(halflife=windows).over(
        partition_by=partition_by, order_by=order_by
    )

# 也可通过引擎实例注册
cs.register_udf(my_func, name="my_op")
```

`category` 决定窗口注入行为：
- `"math"` / `"user"`：不注入窗口参数
- `"ts"`：注入 `partition_by=partition_by, order_by=order_by`
- `"cs"`：注入 `partition_by=order_by, order_by=partition_by`
