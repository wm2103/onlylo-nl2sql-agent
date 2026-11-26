"""Scripted adapters for the compiled graph. Never read evaluation gold."""

from app.agent.context import DataAgentContext
from app.agent.graph import graph
from app.agent.sql_policy import SQLExecutionPolicy
from app.agent.state import DataAgentState
from app.conf.app_config import RecallBudget
from app.entities.value_info import ValueInfo
from tests.baseline_harness import (
    COLUMNS,
    SINGLE_TABLE_REGION,
    FakeColumnRepository,
    FakeDWRepository,
    FakeEmbedding,
    FakeLLM,
    FakeMetaRepository,
    FakeMetricRepository,
    FakeValueRepository,
    RecordingAdapters,
)

TOTAL = "SELECT SUM(order_amount) AS total_sales FROM fact_order"
NORTH = TOTAL + " WHERE region_name = '华北'"
WEST = (
    "SELECT region_name, SUM(order_amount) AS total_sales FROM fact_order "
    "WHERE region_name = '西北' GROUP BY region_name"
)
SCOPED_NORTH = (
    "SELECT region_name, SUM(order_amount) AS total_sales FROM `dw`.fact_order "
    "WHERE CAST(region_name AS BINARY) = x'E58D8EE58C97' GROUP BY region_name LIMIT 1000"
)
CANDIDATES = {
    "total-sales": TOTAL,
    "north-sales": NORTH,
    "empty-west": WEST,
    "delete-orders": "DELETE FROM fact_order WHERE region_name = '华北'",
    "readonly-denial": "DELETE FROM fact_order WHERE region_name = '华北'",
    "required-value-unavailable": NORTH,
    "optional-value-unavailable": TOTAL,
    "row-scope-denial": NORTH,
    "row-scope-allowed": WEST.replace("西北", "华北"),
    "correction-exhausted": NORTH,
    "correction-filter-drop": NORTH,
    "unsupported-filter-shape": TOTAL + " WHERE region_name IN ('华北')",
}


class ToyWarehouse(FakeDWRepository):
    """Recognize only specific forwarded SQL; compute from raw cents, not gold."""

    def __init__(self, adapters, orders, case_id):
        super().__init__(
            adapters, expected_sql=SCOPED_NORTH if case_id == "row-scope-allowed" else None,
            validate_failures={"correction-exhausted": 3, "correction-filter-drop": 1}.get(case_id, 0),
        )
        self.orders = orders

    async def run(self, sql):
        # Count even unexpected calls (including a leaked prohibited candidate).
        await self.adapters.invoke("dw_mysql", "run", sql, None)
        if sql == TOTAL + " LIMIT 1000":
            return [{"total_sales": sum(order["amount_cents"] for order in self.orders) / 100}]
        if sql in {NORTH + " LIMIT 1000", SCOPED_NORTH}:
            cents = sum(order["amount_cents"] for order in self.orders if order["region_name"] == "华北")
            row = {"total_sales": cents / 100}
            return [{"region_name": "华北", **row}] if sql == SCOPED_NORTH else [row]
        if sql == WEST + " LIMIT 1000":
            west = [order for order in self.orders if order["region_name"] == "西北"]
            return ([{"region_name": "西北", "total_sales": sum(o["amount_cents"] for o in west) / 100}]
                    if west else [])
        raise ValueError("Unexpected SQL at offline run adapter")


def make_adapters(case_id):
    faults = {}
    if case_id in {"required-value-unavailable", "optional-value-unavailable"}:
        faults["elasticsearch"] = RuntimeError("toy value index unavailable")
    return RecordingAdapters(faults)


def _policy(case_id):
    auth = {"row-scope-denial": "华南", "row-scope-allowed": "华北"}.get(case_id)
    if auth is None:
        return SQLExecutionPolicy("dw", 1000)
    return SQLExecutionPolicy(
        "dw", 1000, row_filters={"fact_order": {"region_name": (auth,)}},
        row_scope_column_types={"fact_order": {"region_name": "varchar_utf8mb4"}},
    )


async def graph_events(case_id, query, fixture, adapters):
    """Invoke the real graph once; callers get its own custom stream."""
    region = SINGLE_TABLE_REGION
    context = DataAgentContext(
        llm=FakeLLM(adapters, {
            "generate_sql": CANDIDATES.get(case_id, "SELECT 0"),
            "correct_sql": TOTAL if case_id == "correction-filter-drop" else NORTH,
            "recall_value": '["华北", "西北"]',
        }).runnable,
        column_qdrant_repository=FakeColumnRepository(
            adapters, [COLUMNS["fact_order.order_amount"], region]
        ),
        embedding_client=FakeEmbedding(adapters),
        metric_qdrant_repository=FakeMetricRepository(adapters),
        value_es_repository=FakeValueRepository(adapters, [
            ValueInfo(f"region-{name}", name, region.id) for name in fixture["regions"]
        ]),
        meta_mysql_repository=FakeMetaRepository(adapters),
        dw_mysql_repository=ToyWarehouse(adapters, fixture["orders"], case_id),
        recall_budget=RecallBudget(),
        sql_execution_policy=_policy(case_id),
        max_sql_correction_attempts=2,
    )
    async for event in graph.astream(
        DataAgentState(query=query, request_id=f"eval-{case_id}"),
        context=context,
        stream_mode="custom",
    ):
        yield event
