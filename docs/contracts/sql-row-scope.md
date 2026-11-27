# SQL 行范围：精确比较契约

## 用途与配置

行范围约束用于限制查询能够读取的授权取值。部署方通过
`sql_execution.row_filters` 配置受保护的表、列与授权值，通过
`sql_execution.row_scope_column_types` 声明已核实的列类型。
这两项配置由部署方维护，不能来自请求、模型或召回元数据。

以下示例将 `dim_region.region_name` 的授权值限定为“华北”，在核实目标数仓的
表结构和字符集后启用。未配置 `row_filters` 的表不受该行范围规则约束。

```yaml
sql_execution:
  max_rows: 1000
  row_filters:
    dim_region:
      region_name: ["华北"]
  row_scope_column_types:
    dim_region:
      region_name: varchar_utf8mb4
```

## 配置前提

`varchar_utf8mb4` 是当前唯一支持的类型声明。它要求：

- 目标基表位于 `db_dw.database` 指定的数据库。
- 实际列类型是 MySQL 8.0 的 `VARCHAR`，列字符集为 `utf8mb4`。
- 授权值是文本，按其存储文本的 UTF-8 字节精确匹配。

部署方应核实实际表、字段及字符集，并在变更 DDL 或执行身份时重新核实。
SQL 策略以这些声明为前提，不在运行时探测数据库，也不从仓库初始化 DDL
推断线上类型。

类型声明缺失、未知或不受支持时，策略拒绝以该条件证明授权范围。
数字授权值和需要数值隐式转换的谓词也不构成行范围证明。

## SQL 约束

每个读取受保护基表的 SELECT，都必须在自己的 `WHERE` 中为每个受保护列
提供独立的 `=` 或字面量 `IN` 条件。CTE 内部查询和每个 UNION 分支同样适用。
条件中的文本值必须属于该列配置的授权值集合。

## 改写与校验

策略将受保护基表绑定到配置的数据库名，并将用于证明行范围的条件改写为
`CAST(column AS BINARY)` 与授权文本 UTF-8 字节构成的十六进制二进制字面量比较。

MySQL 的[无长度二进制转换](https://dev.mysql.com/doc/refman/8.0/en/cast-functions.html)、
[二进制字面量](https://dev.mysql.com/doc/refman/8.0/en/hexadecimal-literals.html)和
[比较类型规则](https://dev.mysql.com/doc/refman/8.0/en/type-conversion.html)
使两侧按字节比较。比较不依赖 `utf8mb4_general_ci` 等字符排序规则，也不忽略
尾随空格。SQL 校验和执行阶段分别重新验证比较条件与授权值。

## 验证范围

[行范围回归](../../tests/test_sql_row_scope_literals.py)与
[SQL 执行边界回归](../../tests/test_sql_execution_boundary.py)检查发往 EXPLAIN
和执行适配器的 SQL，并验证缺少行范围证明时两个适配器均不被调用。
这些离线测试不覆盖实际部署的表结构、MySQL 执行结果、只读身份、超时或取消。

部署应保持声明的基表和列语义稳定，禁止用临时表替换同名基表。
对列做二进制转换可能影响索引使用；`LIMIT` 限制返回行数，不限制扫描成本。

[返回文档导航](../index.md)
