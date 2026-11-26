# SQL 行范围：精确比较契约

`sql_execution.row_filters` 与 `sql_execution.row_scope_column_types` 由部署方维护，不能来自请求、模型或召回元数据。示例（仅在核实目标数仓后启用）：

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

`varchar_utf8mb4` 是唯一支持的声明：它表示 `db_dw.database` 中指定基表的实际列是 MySQL 8.0 的 `VARCHAR`，列字符集为 `utf8mb4`，且授权值按存储文本的 UTF-8 字节精确匹配。部署方应核实实际表、字段及字符集，并在变更 DDL 或执行身份时重新核实。SQL 策略不会运行时探测数据库，也不会从仓库初始化 DDL 推断线上类型。未配置、未知或不支持的声明，以及数字授权值或需要数值隐式转换的谓词，均不构成行范围证明。未配置 `row_filters` 的表不受此规则约束。

每个读取受保护基表的 SELECT（包括 CTE 和每个 UNION 分支）必须在自己的 `WHERE` 中为每个受保护列提供独立的 `=` 或字面量 `IN` 条件。策略绑定受保护基表的数据库名，并把用来证明行范围的条件改写为 `CAST(column AS BINARY)` 与由授权文本 UTF-8 字节构成的十六进制二进制字面量比较。MySQL 的[无长度二进制转换](https://dev.mysql.com/doc/refman/8.0/en/cast-functions.html)、[二进制字面量](https://dev.mysql.com/doc/refman/8.0/en/hexadecimal-literals.html)和[比较类型规则](https://dev.mysql.com/doc/refman/8.0/en/type-conversion.html)使两侧按字节比较，而不是依赖 `utf8mb4_general_ci` 之类的字符排序规则，也不忽略尾随空格。校验和执行分别重验比较与授权值。

离线测试验证发往 EXPLAIN 和 run 的 SQL，以及缺少证明时两个适配器都不被调用；它们不验证实际部署的表结构、MySQL 执行结果、只读身份、超时或取消。部署应保持所声明的基表和列语义不变，不允许临时表替换同名基表。对列做转换可能影响索引使用；`LIMIT` 不限制扫描成本。