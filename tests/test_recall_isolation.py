import json
import unittest

from langgraph.constants import END, START
from langgraph.graph import StateGraph

from app.agent.context import DataAgentContext
from app.agent.nodes.extract_keywords import extract_keywords
from app.agent.nodes.recall_column import recall_column
from app.agent.nodes.recall_metric import recall_metric
from app.agent.nodes.recall_value import recall_value
from app.agent.state import DataAgentState
from tests.baseline_harness import (
    QUERY,
    FakeColumnRepository,
    FakeDWRepository,
    FakeEmbedding,
    FakeLLM,
    FakeMetaRepository,
    FakeMetricRepository,
    FakeValueRepository,
    Fault,
    RecordingAdapters,
)


async def run_recall(adapters, **overrides):
    builder = StateGraph(DataAgentState, context_schema=DataAgentContext)
    builder.add_node("extract_keywords", extract_keywords)
    builder.add_edge(START, "extract_keywords")
    for node in (recall_column, recall_metric, recall_value):
        builder.add_node(node)
        builder.add_edge("extract_keywords", node.__name__)
        builder.add_edge(node.__name__, END)

    context_values = dict(
        llm=FakeLLM(adapters).runnable,
        column_qdrant_repository=FakeColumnRepository(adapters),
        embedding_client=FakeEmbedding(adapters),
        metric_qdrant_repository=FakeMetricRepository(adapters),
        value_es_repository=FakeValueRepository(adapters),
        meta_mysql_repository=FakeMetaRepository(adapters),
        dw_mysql_repository=FakeDWRepository(adapters),
    )
    context_values.update(overrides)
    context = DataAgentContext(**context_values)
    result, events = {}, []
    async for kind, chunk in builder.compile().astream(
        DataAgentState(query=QUERY), context=context, stream_mode=["values", "custom"]
    ):
        if kind == "values":
            result = chunk
        else:
            events.append(chunk)
    return result, events


class RecallIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_column_error_keeps_other_candidates_and_redacts_progress(self):
        private_detail = "fixture-connection-and-model-payload"
        adapters = RecordingAdapters(
            {"qdrant.column_search": RuntimeError(private_detail)}
        )

        state, events = await run_recall(adapters)

        self.assertEqual(state["retrieved_column_infos"], [])
        self.assertEqual([m.id for m in state["retrieved_metric_infos"]], ["total_sales"])
        self.assertEqual([v.id for v in state["retrieved_value_infos"]], ["region_north"])
        self.assertEqual(state["recall_outcomes"]["column"]["status"], "error")
        self.assertEqual(state["recall_outcomes"]["column"]["dependency"], "qdrant")
        self.assertEqual(state["recall_outcomes"]["metric"]["status"], "success")
        self.assertEqual(state["recall_outcomes"]["value"]["status"], "success")
        self.assertIn(
            {
                "type": "progress",
                "step": "召回字段信息",
                "status": "error",
                "reason_code": "retrieval_error",
                "request_id": state["request_id"],
            },
            events,
        )
        self.assertNotIn(private_detail, json.dumps(events))

    async def test_shared_embedding_failure_is_isolated_to_vector_sources(self):
        adapters = RecordingAdapters({"embedding": RuntimeError("private endpoint")})

        state, events = await run_recall(adapters)

        self.assertEqual(state["recall_outcomes"]["column"]["status"], "error")
        self.assertEqual(state["recall_outcomes"]["metric"]["status"], "error")
        self.assertEqual(state["recall_outcomes"]["value"]["status"], "success")
        self.assertNotIn("private endpoint", json.dumps(events))

    async def test_empty_value_search_is_reported_without_stopping_other_sources(self):
        class EmptyValueRepository(FakeValueRepository):
            async def search(self, keyword):
                await self.adapters.invoke("elasticsearch", "search", keyword, [])
                return []

        adapters = RecordingAdapters()
        state, _ = await run_recall(
            adapters, value_es_repository=EmptyValueRepository(adapters)
        )

        self.assertEqual(state["recall_outcomes"]["value"]["status"], "empty")
        self.assertEqual(state["recall_outcomes"]["value"]["reason_code"], "no_candidates")
        self.assertEqual(state["recall_outcomes"]["column"]["status"], "success")
        self.assertEqual(state["recall_outcomes"]["metric"]["status"], "success")

    async def test_keyword_expansion_timeout_falls_back_to_original_keywords(self):
        adapters = RecordingAdapters({"llm.recall_column": Fault(timeout_seconds=0.01)})

        state, events = await run_recall(adapters)

        outcome = state["recall_outcomes"]["column"]
        self.assertEqual(outcome["status"], "success")
        self.assertEqual(outcome["keyword_fallback_reason"], "keyword_expansion_timeout")
        self.assertTrue(
            any(
                event.get("reason_code") == "keyword_expansion_timeout"
                for event in events
            )
        )
        self.assertTrue(adapters.calls["embedding"])
