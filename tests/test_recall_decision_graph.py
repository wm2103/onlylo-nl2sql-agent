"""Exercise the real graph with deterministic, call-recording adapters."""

import json
import unittest

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.agent.context import DataAgentContext
from app.agent.graph import graph
from app.agent.state import DataAgentState
from app.entities.column_info import ColumnInfo
from app.entities.value_info import ValueInfo
from tests.baseline_harness import (
    COLUMNS,
    QUERY,
    FakeColumnRepository,
    FakeDWRepository,
    FakeEmbedding,
    FakeLLM,
    FakeMetaRepository,
    FakeMetricRepository,
    FakeValueRepository,
    RecordingAdapters,
)
from tests.baseline_harness import (
    SINGLE_TABLE_REGION as REGION,
)
from tests.baseline_harness import (
    SQL as SINGLE_TABLE_SQL,
)

CHANNEL = ColumnInfo(
    "fact_order.channel",
    "channel",
    "varchar",
    "dimension",
    ["华北"],
    "渠道",
    ["渠道"],
    "fact_order",
)
OTHER_REGION = ColumnInfo(
    "fact_order.other_region",
    "other_region",
    "varchar",
    "dimension",
    [],
    "地区",
    ["地区"],
    "fact_order",
)


class RecordingDW(FakeDWRepository):
    async def validate(self, sql):
        await self.adapters.invoke("dw_mysql", "validate", sql, None)


class GraphLLM(FakeLLM):
    def __init__(self, adapters, *, drop_filter=False, drop_metric=False):
        self.drop_filter = drop_filter
        self.drop_metric = drop_metric
        super().__init__(adapters)
        self.runnable = RunnableLambda(self._respond)

    async def _respond(self, prompt):
        text = prompt.to_string()
        if "你是一名查询规划专家" in text:
            fields = (
                ["order_amount"]
                if self.drop_filter
                else ["order_amount", "region_name"]
            )
            result = json.dumps({"fact_order": fields})
            return AIMessage(
                content=await self.adapters.invoke("llm", "filter_table", text, result)
            )
        if self.drop_metric and "你是一名指标筛选与查询规划专家" in text:
            return AIMessage(
                content=await self.adapters.invoke("llm", "filter_metric", text, "[]")
            )
        if "你是一个资深的数据库专家和数据分析师" in text:
            result = (
                SINGLE_TABLE_SQL
                if "region_name" in text
                else "SELECT SUM(order_amount) AS total_sales FROM fact_order"
            )
            return AIMessage(
                content=await self.adapters.invoke("llm", "generate_sql", text, result)
            )
        return await super()._respond(prompt)


async def run_graph(
    query=QUERY,
    faults=None,
    *,
    drop_filter=False,
    drop_metric=False,
    columns=None,
    values=None,
    **overrides,
):
    adapters = RecordingAdapters(faults)
    context = DataAgentContext(
        llm=GraphLLM(
            adapters, drop_filter=drop_filter, drop_metric=drop_metric
        ).runnable,
        column_qdrant_repository=FakeColumnRepository(
            adapters,
            [COLUMNS["fact_order.order_amount"], REGION]
            if columns is None
            else columns,
        ),
        metric_qdrant_repository=FakeMetricRepository(adapters),
        value_es_repository=FakeValueRepository(
            adapters,
            [ValueInfo("north", "华北", REGION.id)] if values is None else values,
        ),
        embedding_client=FakeEmbedding(adapters),
        meta_mysql_repository=FakeMetaRepository(adapters),
        dw_mysql_repository=RecordingDW(adapters),
    )
    context.update(overrides)
    state, events = {}, []
    async for kind, item in graph.astream(
        DataAgentState(query=query, request_id="graph-fixture"),
        context=context,
        stream_mode=["values", "custom"],
    ):
        if kind == "values":
            state = item
        else:
            events.append(item)
    return state, events, adapters


class RecallDecisionGraphTests(unittest.IsolatedAsyncioTestCase):
    def assert_no_sql(self, adapters):
        self.assertFalse(
            any(
                call["operation"] in {"generate_sql", "correct_sql"}
                for call in adapters.calls["llm"]
            )
        )
        self.assertFalse(
            any(
                call["operation"] in {"validate", "run"}
                for call in adapters.calls["dw_mysql"]
            )
        )

    async def test_value_on_a_different_column_cannot_satisfy_requested_filter(self):
        state, _, adapters = await run_graph(
            columns=[REGION, CHANNEL],
            values=[ValueInfo("north-channel", "华北", CHANNEL.id)],
        )
        self.assert_no_sql(adapters)
        self.assertEqual(state["recall_decision"]["status"], "blocked")
        self.assertIn("filter_value", state["recall_decision"]["missing"])

    async def test_value_hit_cannot_choose_between_ambiguous_filter_columns(self):
        state, _, adapters = await run_graph(columns=[REGION, OTHER_REGION])
        self.assert_no_sql(adapters)
        self.assertEqual(state["recall_decision"]["status"], "blocked")
        self.assertIn("filter_field", state["recall_decision"]["missing"])

    async def test_distinct_literals_that_normalize_to_the_same_value_block(self):
        state, _, adapters = await run_graph(
            values=[
                ValueInfo("north", "华北", REGION.id),
                ValueInfo("spaced", "华 北", REGION.id),
            ],
        )
        self.assert_no_sql(adapters)
        self.assertEqual(state["recall_decision"]["status"], "blocked")
        self.assertIn("filter_value", state["recall_decision"]["missing"])

    async def test_implicit_field_is_not_inferred_from_a_value_hit(self):
        state, _, adapters = await run_graph(query="统计华北的销售总额")
        self.assert_no_sql(adapters)
        self.assertEqual(state["recall_decision"]["status"], "blocked")
        self.assertIn("filter_field", state["recall_decision"]["missing"])

    async def test_cross_table_evidence_requires_an_authoritative_join_path(self):
        # Matching key names and primary/foreign-key roles are not a relationship.
        region = COLUMNS["dim_region.region_name"]
        for query in (QUERY, "按地区统计销售总额"):
            with self.subTest(query=query):
                state, _, adapters = await run_graph(
                    query=query,
                    columns=[*COLUMNS.values()],
                    values=[ValueInfo("north", "华北", region.id)],
                )
                self.assert_no_sql(adapters)
                decision = state["recall_decision"]
                self.assertEqual(decision["status"], "blocked")
                self.assertIn("join_path", decision["request"]["required_sources"])
                self.assertIn("join_path", decision["missing"])

    async def test_pruning_cannot_remove_verified_filter_or_metric_and_continue(self):
        for dropped, missing in (("filter", "filter_value"), ("metric", "metric")):
            with self.subTest(dropped=dropped):
                state, events, adapters = await run_graph(**{f"drop_{dropped}": True})
                operations = {call["operation"] for call in adapters.calls["llm"]}
                self.assertTrue({"filter_table", "filter_metric"} <= operations)
                self.assert_no_sql(adapters)
                decision = state["recall_decision"]
                self.assertEqual(decision["status"], "blocked")
                self.assertEqual(decision["reason_code"], "required_evidence_pruned")
                self.assertIn(missing, decision["missing"])
                self.assertEqual(
                    decision["required_filters"],
                    [
                        {"column_id": REGION.id, "operator": "eq", "value": "华北"},
                    ],
                )
                (result,) = [
                    event for event in events if event["type"] == "recall_decision"
                ]
                self.assertEqual(result["status"], "blocked")
                self.assertNotIn("required_filters", result)

    async def test_missing_required_filter_blocks_before_generation_and_execution(self):
        state, events, adapters = await run_graph(
            faults={"elasticsearch": RuntimeError("private-secret-endpoint")}
        )
        decision = state["recall_decision"]
        self.assertEqual(decision["status"], "blocked")
        self.assertIn("filter_value", decision["missing"])
        self.assert_no_sql(adapters)
        terminal = next(event for event in events if event["type"] == "recall_decision")
        self.assertEqual(terminal["status"], "blocked")
        self.assertEqual(terminal["request_id"], "graph-fixture")
        self.assertNotIn("private-secret-endpoint", json.dumps(events))
        self.assertNotIn("evidence", terminal)
        self.assertNotIn("query", terminal)

    async def test_metric_or_shared_embedding_failure_blocks_without_sql(self):
        for faults in (
            {"qdrant.metric_search": RuntimeError("private-metric"), "embedding": None},
            {"embedding": RuntimeError("private-embedding")},
        ):
            with self.subTest(faults=faults):
                state, events, adapters = await run_graph(faults=faults)
                self.assertEqual(state["recall_decision"]["status"], "blocked")
                self.assert_no_sql(adapters)
                self.assertNotIn("private-", json.dumps(events))

    async def test_optional_value_failure_degrades_but_executes_once(self):
        state, events, adapters = await run_graph(
            query="统计销售总额", faults={"elasticsearch": RuntimeError("private")}
        )
        self.assertEqual(state["recall_decision"]["status"], "degraded")
        self.assertEqual(
            sum(c["operation"] == "generate_sql" for c in adapters.calls["llm"]), 1
        )
        self.assertEqual(
            sum(c["operation"] == "run" for c in adapters.calls["dw_mysql"]), 1
        )
        self.assertTrue(
            any(
                e["type"] == "recall_decision" and e["status"] == "degraded"
                for e in events
            )
        )

    async def test_successful_recall_with_authoritative_matches_executes_once(self):
        for query in (QUERY, "统计地区为华 北的销售总额"):
            with self.subTest(query=query):
                state, _, adapters = await run_graph(query=query)
                decision = state["recall_decision"]
                self.assertEqual(decision["status"], "success")
                self.assertEqual(decision["missing"], [])
                self.assertEqual(
                    decision["required_filters"],
                    [
                        {"column_id": REGION.id, "operator": "eq", "value": "华北"},
                    ],
                )
                runs = [
                    call["inputs"]
                    for call in adapters.calls["dw_mysql"]
                    if call["operation"] == "run"
                ]
                self.assertEqual(runs, [SINGLE_TABLE_SQL + " LIMIT 1000"])


if __name__ == "__main__":
    unittest.main()
