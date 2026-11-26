"""Deterministic adapters for exercising the committed baseline graph offline."""

import asyncio
import json
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.agent.context import DataAgentContext
from app.agent.graph import graph
from app.agent.nodes.validate_sql import DatabaseSyntaxError
from app.agent.state import DataAgentState
from app.entities.column_info import ColumnInfo
from app.entities.metric_info import MetricInfo
from app.entities.table_info import TableInfo
from app.entities.value_info import ValueInfo

BASELINE_COMMIT = "0ab141086439e385a9cccf5f5acdf3c42b239e65"
QUERY = "统计华北地区的销售总额"
SQL = (
    "SELECT SUM(order_amount) AS total_sales FROM fact_order WHERE region_name = '华北'"
)
EXECUTED_SQL = SQL + " LIMIT 1000"
ROWS = [{"total_sales": 3200}]
TABLES = {
    "fact_order": TableInfo("fact_order", "fact_order", "fact", "订单事实表"),
    "dim_region": TableInfo("dim_region", "dim_region", "dim", "地区维度表"),
}
COLUMNS = {
    column.id: column
    for column in (
        ColumnInfo(
            "fact_order.order_amount",
            "order_amount",
            "float",
            "measure",
            [],
            "订单金额",
            ["销售额"],
            "fact_order",
        ),
        ColumnInfo(
            "dim_region.region_name",
            "region_name",
            "varchar",
            "dimension",
            [],
            "地区名称",
            ["地区"],
            "dim_region",
        ),
        ColumnInfo(
            "fact_order.region_id",
            "region_id",
            "varchar",
            "foreign_key",
            [],
            "地区外键",
            [],
            "fact_order",
        ),
        ColumnInfo(
            "dim_region.region_id",
            "region_id",
            "varchar",
            "primary_key",
            [],
            "地区主键",
            [],
            "dim_region",
        ),
    )
}
SINGLE_TABLE_REGION = ColumnInfo(
    "fact_order.region_name",
    "region_name",
    "varchar",
    "dimension",
    [],
    "地区名称",
    ["地区"],
    "fact_order",
)
LLM_RESPONSES = (
    ("recall_column", "你是一名数据表字段推断专家", '["销售金额"]'),
    ("recall_metric", "你是一名指标语义扩展专家", '["销售总额"]'),
    ("recall_value", "你是一名业务语义解析专家", '["华北"]'),
    (
        "filter_table",
        "你是一名查询规划专家",
        json.dumps(
            {
                "fact_order": ["order_amount", "region_name"],
            }
        ),
    ),
    ("filter_metric", "你是一名指标筛选与查询规划专家", '["GMV"]'),
    ("generate_sql", "你是一个资深的数据库专家和数据分析师", SQL),
    ("correct_sql", "你是一个资深的数据库专家、SQL 调试专家", SQL),
)


@dataclass(frozen=True)
class Fault:
    """An injectable adapter failure, delay, or timeout."""

    exception: Exception | None = None
    delay_seconds: float = 0
    timeout_seconds: float | None = None


class RecordingAdapters:
    """Records dependency calls and applies named or shared dependency faults."""

    def __init__(self, faults: dict[str, Fault | Exception] | None = None):
        self.faults = faults or {}
        self.calls: dict[str, list[dict[str, Any]]] = {
            dependency: []
            for dependency in (
                "llm",
                "embedding",
                "qdrant",
                "elasticsearch",
                "meta_mysql",
                "dw_mysql",
            )
        }

    async def invoke(
        self, dependency: str, operation: str, inputs: Any, result: Any
    ) -> Any:
        call = {"operation": operation, "inputs": _summarize(inputs)}
        self.calls[dependency].append(call)
        fault = self.faults.get(
            f"{dependency}.{operation}", self.faults.get(dependency)
        )
        try:
            if isinstance(fault, Exception):
                raise fault
            if fault:
                if fault.timeout_seconds is not None:
                    delay = fault.delay_seconds or fault.timeout_seconds + 1
                    await asyncio.wait_for(
                        asyncio.sleep(delay), timeout=fault.timeout_seconds
                    )
                elif fault.delay_seconds:
                    await asyncio.sleep(fault.delay_seconds)
                if fault.exception:
                    raise fault.exception
            call["status"] = "success"
            call["result"] = _summarize(result)
            return deepcopy(result)
        except asyncio.CancelledError:
            call["status"] = "cancelled"
            raise
        except Exception as error:
            call["status"] = "timeout" if isinstance(error, TimeoutError) else "error"
            call["error"] = f"{type(error).__name__}: {error}"
            raise

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            dependency: {"count": len(calls), "calls": calls}
            for dependency, calls in self.calls.items()
        }


class FakeEmbedding:
    def __init__(self, adapters: RecordingAdapters):
        self.adapters = adapters

    async def aembed_query(self, text: str) -> list[float]:
        return await self.adapters.invoke("embedding", "aembed_query", text, [0.1, 0.2])


class FakeColumnRepository:
    def __init__(
        self, adapters: RecordingAdapters, columns: list[ColumnInfo] | None = None
    ):
        self.adapters = adapters
        self.columns = (
            columns
            if columns is not None
            else [
                COLUMNS["fact_order.order_amount"],
                COLUMNS["dim_region.region_name"],
            ]
        )

    async def search(self, embedding: list[float]) -> list[ColumnInfo]:
        return await self.adapters.invoke(
            "qdrant", "column_search", embedding, self.columns
        )


class FakeMetricRepository:
    def __init__(self, adapters: RecordingAdapters):
        self.adapters = adapters

    async def search(self, embedding: list[float]) -> list[MetricInfo]:
        metrics = [
            MetricInfo(
                id="total_sales",
                name="GMV",
                description="订单金额合计",
                relevant_columns=["fact_order.order_amount"],
                alias=["销售额"],
            )
        ]
        return await self.adapters.invoke("qdrant", "metric_search", embedding, metrics)


class FakeValueRepository:
    def __init__(
        self, adapters: RecordingAdapters, values: list[ValueInfo] | None = None
    ):
        self.adapters = adapters
        self.values = (
            values
            if values is not None
            else [
                ValueInfo(
                    id="region_north", value="华北", column_id="dim_region.region_name"
                )
            ]
        )

    async def search(self, keyword: str) -> list[ValueInfo]:
        return await self.adapters.invoke(
            "elasticsearch", "search", keyword, self.values
        )


class FakeMetaRepository:
    def __init__(self, adapters: RecordingAdapters):
        self.adapters = adapters

    async def get_column_info_by_id(self, column_id: str) -> ColumnInfo | None:
        column = (
            SINGLE_TABLE_REGION
            if column_id == SINGLE_TABLE_REGION.id
            else COLUMNS.get(column_id)
        )
        return await self.adapters.invoke(
            "meta_mysql", "get_column_info_by_id", column_id, column
        )

    async def get_key_columns_by_table_id(self, table_id: str) -> list[ColumnInfo]:
        return await self.adapters.invoke(
            "meta_mysql",
            "get_key_columns_by_table_id",
            table_id,
            [COLUMNS[f"{table_id}.region_id"]],
        )

    async def get_table_info_by_id(self, table_id: str) -> TableInfo:
        return await self.adapters.invoke(
            "meta_mysql", "get_table_info_by_id", table_id, TABLES[table_id]
        )


class FakeDWRepository:
    def __init__(
        self,
        adapters: RecordingAdapters,
        expected_sql: str | None = EXECUTED_SQL,
        validate_failures: int = 0,
    ):
        self.adapters = adapters
        self.expected_sql = expected_sql
        self.validate_failures = validate_failures

    async def get_db_info(self) -> dict[str, str]:
        return await self.adapters.invoke(
            "dw_mysql", "get_db_info", None, {"dialect": "mysql", "version": "8.0.36"}
        )

    async def validate(self, sql: str) -> None:
        await self.adapters.invoke("dw_mysql", "validate", sql, None)
        if self.validate_failures:
            self.validate_failures -= 1
            raise DatabaseSyntaxError("fixture syntax error")
        if self.expected_sql is not None and sql != self.expected_sql:
            raise ValueError(f"Unexpected fixture SQL: {sql}")

    async def run(self, sql: str) -> list[dict[str, int]]:
        return await self.adapters.invoke("dw_mysql", "run", sql, ROWS)


class FakeLLM:
    def __init__(
        self, adapters: RecordingAdapters, responses: dict[str, str] | None = None
    ):
        self.adapters = adapters
        self.responses = responses or {}
        self.runnable = RunnableLambda(self._respond)

    async def _respond(self, prompt: Any) -> AIMessage:
        text = prompt.to_string()
        for operation, marker, result in LLM_RESPONSES:
            if marker in text.splitlines()[1]:
                response = await self.adapters.invoke(
                    "llm", operation, text, self.responses.get(operation, result)
                )
                return AIMessage(content=response)
        raise ValueError(f"Unexpected fixture prompt: {text.splitlines()[1]}")


def _summarize(value: Any) -> Any:
    if is_dataclass(value):
        return _summarize(asdict(value))
    if isinstance(value, list):
        return [_summarize(item) for item in value]
    if isinstance(value, dict):
        return {key: _summarize(item) for key, item in value.items()}
    if isinstance(value, str) and len(value) > 180:
        return value[:177] + "..."
    return value


async def run_baseline_query(
    query: str = QUERY,
    faults: dict[str, Fault | Exception] | None = None,
    *,
    responses: dict[str, str] | None = None,
    expected_sql: str | None = EXECUTED_SQL,
    validate_failures: int = 0,
) -> dict[str, Any]:
    """Run one request through the compiled baseline graph with offline adapters."""

    adapters = RecordingAdapters(faults)
    llm_adapter = FakeLLM(adapters, responses)
    context = DataAgentContext(
        llm=llm_adapter.runnable,
        column_qdrant_repository=FakeColumnRepository(
            adapters,
            [COLUMNS["fact_order.order_amount"], SINGLE_TABLE_REGION],
        ),
        embedding_client=FakeEmbedding(adapters),
        metric_qdrant_repository=FakeMetricRepository(adapters),
        value_es_repository=FakeValueRepository(
            adapters,
            [ValueInfo("region_north", "华北", SINGLE_TABLE_REGION.id)],
        ),
        meta_mysql_repository=FakeMetaRepository(adapters),
        dw_mysql_repository=FakeDWRepository(adapters, expected_sql, validate_failures),
    )
    updates = {}
    events = []
    terminal = {"status": "success"}
    try:
        async for kind, chunk in graph.astream(
            DataAgentState(query=query),
            context=context,
            stream_mode=["updates", "custom"],
        ):
            if kind == "updates":
                updates.update(chunk)
            else:
                events.append(chunk)
    except Exception as error:
        terminal = {
            "status": "error",
            "exception_type": type(error).__name__,
            "message": str(error),
        }

    # Ticket 02 isolates branch failures inside the graph, so the graph itself
    # can finish without raising. Preserve a comparable terminal observation in
    # this offline trace for callers that use the baseline harness contract.
    dependencies = adapters.snapshot()
    outcomes = {}
    for source in ("column", "metric", "value"):
        branch = updates.get(f"recall_{source}", {})
        outcomes.update(branch.get("recall_outcomes", {}))
    failed = next(
        (
            (source, outcome)
            for source, outcome in outcomes.items()
            if outcome.get("status") in {"error", "timeout"}
        ),
        None,
    )
    if failed and terminal["status"] == "success":
        source, outcome = failed
        calls = dependencies["embedding"]["calls"] + dependencies["qdrant"]["calls"]
        if source == "value":
            calls = dependencies["elasticsearch"]["calls"]
        detail = next(
            (call for call in calls if call.get("status") in {"error", "timeout"}), None
        )
        error_type = (
            "TimeoutError" if outcome["status"] == "timeout" else "RuntimeError"
        )
        message = (detail or {}).get("error", "").split(": ", 1)[-1]
        terminal = {
            "status": "error",
            "exception_type": error_type,
            "message": message,
        }

    retrieval = {}
    for source, field in (
        ("column", "retrieved_column_infos"),
        ("metric", "retrieved_metric_infos"),
        ("value", "retrieved_value_infos"),
    ):
        update = updates.get(f"recall_{source}") or {}
        retrieval[f"{source}s"] = len(update[field]) if field in update else None

    generation = updates.get("generate_sql") or {}
    correction = updates.get("correct_sql") or {}
    validation = updates.get("validate_sql") or {}
    validate_calls = [
        call
        for call in dependencies["dw_mysql"]["calls"]
        if call["operation"] == "validate"
    ]
    execute_calls = [
        call for call in dependencies["dw_mysql"]["calls"] if call["operation"] == "run"
    ]
    correction_calls = [
        call for call in dependencies["llm"]["calls"] if call["operation"] == "correct_sql"
    ]
    terminal_events = [event for event in events if event.get("type") == "terminal"]
    return {
        "baseline_commit": BASELINE_COMMIT,
        "query": query,
        "retrieval": retrieval,
        "sql_generation": {
            "sql": generation.get("sql"),
            "corrected_sql": correction.get("sql"),
        },
        "validation": {
            "attempts": len(validate_calls),
            "error": validation.get("error"),
        },
        "execution": {
            "attempts": len(execute_calls),
            "sql": execute_calls[-1]["inputs"] if execute_calls else None,
            "rows": next(
                (event["data"] for event in events if event["type"] == "result"), None
            ),
        },
        "sql_loop": {
            "validation_attempts": len(validate_calls),
            "correction_attempts": len(correction_calls),
            "outcome": (
                "success"
                if any(event.get("status") == "success" for event in terminal_events)
                else (terminal_events[-1].get("reason_code") if terminal_events else terminal["status"])
            ),
        },
        "terminal": terminal,
        "events": events,
        "dependencies": dependencies,
    }
