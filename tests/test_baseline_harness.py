import json
import unittest

from tests.baseline_harness import EXECUTED_SQL, Fault, run_baseline_query

EXPECTED_SQL = (
    "SELECT SUM(order_amount) AS total_sales FROM fact_order WHERE region_name = '华北'"
)


class BaselineHarnessTest(unittest.IsolatedAsyncioTestCase):
    async def test_fixed_ecommerce_query_runs_through_the_baseline_graph(self):
        trace = await run_baseline_query("统计华北地区的销售总额")

        self.assertEqual(
            trace["baseline_commit"], "0ab141086439e385a9cccf5f5acdf3c42b239e65"
        )
        self.assertEqual(trace["query"], "统计华北地区的销售总额")
        self.assertEqual(trace["retrieval"], {"columns": 2, "metrics": 1, "values": 1})
        self.assertEqual(trace["sql_generation"]["sql"], EXPECTED_SQL)
        self.assertEqual(trace["validation"], {"attempts": 1, "error": None})
        self.assertEqual(
            trace["execution"],
            {"attempts": 1, "sql": EXECUTED_SQL, "rows": [{"total_sales": 3200}]},
        )
        self.assertEqual(trace["terminal"], {"status": "success"})

        dependencies = trace["dependencies"]
        self.assertEqual(dependencies["llm"]["count"], 6)
        self.assertGreater(dependencies["embedding"]["count"], 0)
        self.assertEqual(
            {call["operation"] for call in dependencies["qdrant"]["calls"]},
            {"column_search", "metric_search"},
        )
        self.assertGreater(dependencies["elasticsearch"]["count"], 0)
        self.assertEqual(
            dependencies["dw_mysql"]["calls"][-1]["result"], [{"total_sales": 3200}]
        )
        json.dumps(trace)

    async def test_single_recall_failure_uses_remaining_evidence_when_sufficient(self):
        trace = await run_baseline_query(
            faults={"qdrant.column_search": RuntimeError("fixture column unavailable")}
        )

        self.assertEqual(trace["execution"]["attempts"], 1)
        self.assertTrue(
            any(
                call["operation"] == "column_search" and call["status"] == "error"
                for call in trace["dependencies"]["qdrant"]["calls"]
            )
        )

    async def test_shared_embedding_timeout_records_both_vector_recall_attempts(self):
        trace = await run_baseline_query(
            faults={"embedding": Fault(timeout_seconds=0.25)}
        )

        self.assertEqual(trace["terminal"]["exception_type"], "TimeoutError")
        self.assertEqual(trace["execution"]["attempts"], 0)
        self.assertGreaterEqual(trace["dependencies"]["embedding"]["count"], 2)
        self.assertTrue(
            any(
                call["status"] == "timeout"
                for call in trace["dependencies"]["embedding"]["calls"]
            )
        )
        self.assertEqual(trace["dependencies"]["qdrant"]["count"], 0)
