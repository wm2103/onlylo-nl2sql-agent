# 问数基线测试

下文提交号仅保留为上游实现的历史对照标识；独立发布快照不携带上游 Git 历史，因此不能在本仓库直接 checkout 该提交。测试运行的是当前代码与内存替身。

测试通过当前 LangGraph 运行，以实现基线 `0ab141086439e385a9cccf5f5acdf3c42b239e65` 为对照。固定问数请求「统计华北地区的销售总额」的成功用例使用单表 `fact_order`，包含 GMV 指标和华北字段值。另有 `dim_region` 夹具验证缺少权威关联路径的跨表需求会阻断。MySQL 8.0.36 为模拟环境；数仓替身预设返回 `[{"total_sales": 3200}]`，该值属于测试数据。

在仓库根目录运行以下 PowerShell 命令。使用 Python 3.14 和 `uv.lock` 中的项目依赖，测试框架使用标准库 `unittest`。`offline-fixture` 仅供项目配置加载，所有服务调用均由内存替身处理。

```powershell
$env:LLM_API_KEY = 'offline-fixture'
$env:LLM_MODEL = 'offline-fixture'
$env:LLM_BASE_URL = 'http://127.0.0.1:1/v1'
uv run --locked python -m unittest tests.test_baseline_harness
```

`await run_baseline_query()` 返回可 JSON 序列化的 trace，包括各路召回数量、候选和修正 SQL、校验与执行尝试数、结果行、流式事件，以及问数终态。每个依赖保留调用次数和每次操作的输入摘要、结果或异常。并行分支的事件顺序可以变化。

通过 `faults` 传入 `Exception` 或 `Fault`。例如，`{"qdrant.column_search": RuntimeError("unavailable")}` 只影响字段召回，`{"embedding": Fault(timeout_seconds=0.25)}` 影响共用 Embedding 的两个分支。也可用 `llm.recall_value` 定位字段值关键词扩展，或用 `llm` 作用于所有模型调用。

历史基线在召回分支抛出异常或超时时结束图执行，trace 记录 `error` 和异常类型，此时没有 SQL 执行。这是旧实现对照，不是当前图的行为承诺；当前实现按各路证据、隔离结果与策略决定继续、降级或阻断，见 `test_recall_isolation.py`、`test_recall_decision_graph.py` 与 `eval/README.md`。
