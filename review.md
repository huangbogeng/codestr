# CodeStr 代码评审

> 评审日期：2026-06-09

---

## 🔴 严重 (P0) — 影响正确性 / 有 bug 风险 / 设计债务核心

### 1. `register_udf` 绕过注册中心，运行时不可用

**文件**：`engine.py:78-81`

```python
def register_udf(self, func, name=None):
    name = name if name is not None else func.__name__
    setattr(udf, name, func)   # ← monkey-patch 到模块，但 UDFRegistry 里没有 UDFMeta
```

**问题**：通过 `register_udf` 注册的函数被 `setattr` 到 `udf` 模块，但 `compiler.py` 编译时查的是 `UDFRegistry`（`node.fn_name not in registry`），registry 里根本没有这条记录，导致编译阶段报 `Unknown function`。

**修复方向**：
```python
def register_udf(self, func, name=None):
    from codestr.udf.registry import UDFRegistry, UDFMeta
    UDFRegistry.get_instance().register(UDFMeta(
        name=name or func.__name__,
        fn=func,
        category="user",
    ))
```

---

### 2. `Expr` 用 `object.__setattr__` 暴力破坏 `ExprNode` 的 frozen 契约

**文件**：`expr.py:33, 41, 49`

```python
@dataclass(frozen=True)
class ExprNode: ...        # ast.py — 宣称不可变

# expr.py — 但 Expr 直接穿墙:
@fn_name.setter
def fn_name(self, value):
    object.__setattr__(self._node, "fn_name", value)  # 绕过 frozen
```

**问题**：
- `ExprNode` 的 `frozen=True` 是为了保证 hash 稳定（做缓存 key），`Expr` 的 setter 让这个保证形同虚设
- `Expr(expr_str)` 的 `__init__` 先创建一个**空壳** `ExprNode()` 再解析替换，空壳 alias 为 `""`，在此期间 hash 值无意义
- 两套 API 同时暴露给用户，不知道什么时候该用哪个

**修复方向**：删除 `Expr` 类，把分析工具（`depth`, `node_count`, `to_rpn`, `common_map`, `pre_cal_items`）搬进 `ast.py` 作为 `ExprNode` 的纯函数。

---

## 🟠 高优先级 (P1) — 状态管理混乱 / 扩展性瓶颈

### 3. Engine 隐式状态机：5 个可空属性互相依赖

**文件**：`engine.py:17-65`

```python
class CodeStr:
    def __init__(self, data, ...):
        self.data: pl.LazyFrame = None       # 可能为 None
        self._data_: pl.LazyFrame = None     # 可能为 None
        self._last_query_cache = None        # 可能为 None
        self._expr_cache = {}
        self._cur_expr_cache = {}

    @property
    def cache_columns(self):
        if self.data is None:
            if self._last_query_cache is None:
                return []                    # 三层 if-else 推断状态
            else:
                return self._data_.collect_schema().names()
        else:
            return self.data.columns
```

**问题**：
- 状态转换无约束：`data` 和 `_data_` 何时互斥、何时共存、何时同步，均靠约定
- `sql()` 方法 60+ 行，兼任解析、编译、缓存查重、状态合并、错误收集、collect 执行六个职责
- 新增一种执行模式需要理解全部状态转换规则

**修复方向**：引入显式状态枚举，拆分为 pipeline：
```
RawData → AlignedData → CompilePlan → Execution
```

---

### 4. `sql()` 缓存策略有隐藏陷阱

**文件**：`engine.py:232-253`

```python
if lazy:
    self._expr_cache.update(self._cur_expr_cache)   # lazy 模式也更新了持久缓存
    return self._data_.select(...)

# eager 模式
self._last_query_cache = self._data_.select(...).collect()
self._expr_cache.update(self._cur_expr_cache)       # 成功后更新
```

**问题**：
- `lazy=True` 时 `_data_` 已累积了 `with_columns`（l146-168），但未 collect，下次调用会基于这个膨胀的 `_data_` 继续叠加，计算图无限膨胀
- eager 模式下 collect 失败会吞掉 cache 更新（l250-254），但 `_data_` 里的 `with_columns` 已经写进去了，下次重试不会重新编译而是带着上次的失败状态继续

---

## 🟡 中优先级 (P2) — 代码质量 / 一致性

### 5. 模块级可变单例，测试不友好

**文件**：`udf/registry.py:23-26`, `parser.py:123`

```python
class UDFRegistry:
    _instance: UDFRegistry | None = None   # 全局单例

_parser = Lark(_GRAMMAR, ...)              # 模块 import 时立即构建
```

**问题**：
- 测试中需要隔离注册表时无法 reset
- `_parser` 构建在 import 阶段，拖慢冷启动

**修复方向**：`UDFRegistry` 加 `reset()` 类方法；`_parser` 改为惰性初始化。

---

### 6. `_normalize` 制造死代码

**文件**：`parser.py:126-130, 40`

```python
def _normalize(expr: str) -> str:
    expr = expr.replace("!", "~")   # ← 所有 ! 先变成 ~

# grammar 中:
# | "!" factor -> not_    ← 永远不会被匹配，永远走下面那条
# | "~" factor -> not_
```

---

### 7. Import 风格不一致

**文件**：`engine.py:9-15, 120`

```python
from codestr import udf                           # 顶部 import
from codestr.compiler import compile as _pure_compile
# ... 6 个顶部 import ...

def _check_rpn(self, rpn, reasons):
    from codestr.types import TokenType            # ← 唯独这个在函数内 import
```

无理由的延迟导入，与其他 import 不一致。

---

### 8. `Expr.children` 返回类型不一致

**文件**：`expr.py:71-81`

```python
@property
def children(self):
    c = []
    for arg in self._node.args:
        if isinstance(arg, ExprNode):
            child = Expr()
            child._node = arg
            c.extend([(str(arg), _depth(arg)), *child.children])  # 中间节点返回 (str, int)
        else:
            c.append((str(arg), 0))                                # 叶子返回 (str, int)
    return c
```

**问题**：`children` 名义上应该返回子节点，实际却返回**展平后的全部后代节点**（因为 `extend` + `*child.children` 递归展开）。命名具有误导性。后面的 `common_map` 基于这个展平结果做 `Counter` 统计，语义模糊。

---

## 🔵 低优先级 (P3) — 改善性建议

### 9. `_bin` 方法可以通过装饰器简化

**文件**：`parser.py:84-109`

```python
def _bin(self, name, items):
    return ExprNode(fn_name=name, args=tuple(items))

def add(self, items): return self._bin("add", items)
def sub(self, items): return self._bin("sub", items)
def mul(self, items): return self._bin("mul", items)
# ... 重复 13 次
```

可以用元编程或 `__getattr__` 消除重复。

---

### 10. ts/cs 算子的 `over` 字典是模块级硬编码

**文件**：`ts_udf.py:10`, `cs_udf.py:11`

```python
over = dict(partition_by=["asset"], order_by=["datetime"])   # ts
over = dict(partition_by=["datetime"], order_by=["asset"])   # cs
```

如果未来需要支持不同的 index 列名（如 `stock_id` 代替 `asset`），需要修改源码。应该从 engine 传入或通过配置注入。

---

## 总结

| 等级 | 数量 | 关键词 |
|---|---|---|
| 🔴 P0 | 2 | register_udf 失效、Expr/ExprNode 双轨 |
| 🟠 P1 | 2 | 隐式状态机、缓存膨胀 |
| 🟡 P2 | 4 | 单例、死代码、import 不一致、children 命名 |
| 🔵 P3 | 2 | 样板代码、硬编码 over 字典 |

**建议修复顺序**：P0 → P1 → P2 → P3。P0 是正确性问题，P1 是扩展性瓶颈，P2/P3 是代码卫生。
