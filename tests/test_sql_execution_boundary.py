import unittest

from app.agent.nodes.validate_sql import DatabaseSyntaxError
from app.agent.sql_policy import SQLExecutionPolicy, SQLPolicyError, enforce_sql_policy
from tests.baseline_harness import run_baseline_query


class SQLExecutionBoundaryTest(unittest.IsolatedAsyncioTestCase):
    async def test_database_syntax_error_is_corrected_revalidated_before_one_run(self):
        trace = await run_baseline_query(
            expected_sql=None,
            responses={"generate_sql": "SELECT order_amount FROM fact_order WHERE region_name = '华北'", "correct_sql": "SELECT order_amount FROM fact_order WHERE region_name = '华北'"},
            faults={"dw_mysql.validate": DatabaseSyntaxError("fixture syntax error")},
        )
        self.assertEqual(trace["validation"]["attempts"], 3)
        self.assertEqual(trace["execution"]["attempts"], 0)
        self.assertEqual(trace["sql_loop"]["correction_attempts"], 2)
        self.assertEqual(trace["sql_loop"]["outcome"], "sql_correction_exhausted")

    async def test_policy_denial_never_attempts_correction(self):
        trace = await run_baseline_query(responses={"generate_sql": "DROP TABLE fact_order"})
        self.assertNotIn("correct_sql", [call["operation"] for call in trace["dependencies"]["llm"]["calls"]])

    async def test_one_explicit_database_syntax_failure_can_be_repaired_once(self):
        trace = await run_baseline_query(
            expected_sql=None,
            responses={"generate_sql": "SELECT order_amount FROM fact_order WHERE region_name = '华北'", "correct_sql": "SELECT order_amount FROM fact_order WHERE region_name = '华北'"},
            validate_failures=1,
        )
        self.assertEqual(trace["validation"]["attempts"], 2)
        self.assertEqual(trace["execution"]["attempts"], 1)
        self.assertEqual(trace["sql_loop"]["correction_attempts"], 1)
        self.assertEqual(trace["sql_loop"]["outcome"], "success")

    async def test_validation_infrastructure_failure_is_not_repaired(self):
        trace = await run_baseline_query(
            expected_sql=None,
            responses={"generate_sql": "SELECT order_amount FROM fact_order WHERE region_name = '华北'"},
            faults={"dw_mysql.validate": RuntimeError("fixture unavailable")},
        )
        self.assertEqual(trace["validation"]["attempts"], 1)
        self.assertEqual(trace["execution"]["attempts"], 0)
        self.assertEqual(trace["sql_loop"]["correction_attempts"], 0)
        self.assertEqual(trace["sql_loop"]["outcome"], "sql_validation_unavailable")

    async def test_dangerous_correction_is_rejected_by_the_same_policy(self):
        trace = await run_baseline_query(
            expected_sql=None,
            responses={"generate_sql": "SELECT order_amount FROM fact_order WHERE region_name = '华北'", "correct_sql": "DROP TABLE fact_order"},
            validate_failures=1,
        )
        self.assertEqual(trace["sql_loop"]["correction_attempts"], 1)
        self.assertEqual(trace["execution"]["attempts"], 0)
        self.assertEqual(trace["sql_loop"]["outcome"], "sql_policy_denied")

    async def test_execution_exception_has_a_sanitized_final_outcome(self):
        trace = await run_baseline_query(
            faults={"dw_mysql.run": RuntimeError("fixture execution failure")}
        )
        self.assertEqual(trace["execution"]["attempts"], 1)
        self.assertEqual(trace["sql_loop"]["outcome"], "sql_execution_failed")

    def test_each_select_including_union_and_cte_has_an_explicit_row_scope(self):
        tables = [
            {"name": "fact_order", "columns": [{"name": "order_amount"}]},
            {"name": "dim_region", "columns": [{"name": "region_id"}, {"name": "region_name"}]},
        ]
        policy = SQLExecutionPolicy(
            "dw", 1000, {"dim_region": {"region_name": ("north",)}},
            row_scope_column_types={"dim_region": {"region_name": "varchar_utf8mb4"}},
        )
        for candidate in (
            "SELECT region_id FROM dim_region",
            "SELECT region_id FROM dim_region WHERE region_name='north' UNION ALL SELECT region_id FROM dim_region",
            "WITH r AS (SELECT region_id FROM dim_region) SELECT region_id FROM r",
        ):
            with self.subTest(candidate=candidate), self.assertRaises(SQLPolicyError):
                enforce_sql_policy(candidate, tables, policy)

        accepted = enforce_sql_policy(
            "SELECT region_id FROM dim_region WHERE region_name='north' UNION ALL SELECT region_id FROM dim_region WHERE region_name='north'",
            tables,
            policy,
        )
        self.assertEqual(accepted.count("WHERE"), 2)

    async def test_non_readonly_candidate_never_reaches_the_database_or_correction(self):
        trace = await run_baseline_query(
            responses={"generate_sql": "DROP TABLE fact_order"}
        )

        self.assertIn(
            {"type": "terminal", "status": "blocked", "reason_code": "sql_policy_denied"},
            trace["events"],
        )
        self.assertEqual(trace["validation"]["attempts"], 0)
        self.assertEqual(trace["execution"]["attempts"], 0)
        self.assertNotIn(
            "correct_sql", [call["operation"] for call in trace["dependencies"]["llm"]["calls"]]
        )

    async def test_object_allowlist_resolves_the_actual_database_and_field(self):
        for candidate in (
            "SELECT order_amount FROM another_db.fact_order",
            "SELECT order_amount AS order_id, order_id FROM fact_order",
            "SELECT x.order_amount FROM dim_region AS x",
        ):
            with self.subTest(candidate=candidate):
                trace = await run_baseline_query(responses={"generate_sql": candidate})

                self.assertEqual(trace["events"][-1].get("reason_code"), "sql_policy_denied")
                self.assertEqual(trace["validation"]["attempts"], 0)
                self.assertEqual(trace["execution"]["attempts"], 0)

    async def test_server_limit_caps_the_sql_seen_by_explain_and_execution(self):
        for requested, permitted in (("", 1000), (" LIMIT 99999", 1000), (" LIMIT 12", 12)):
            with self.subTest(requested=requested):
                trace = await run_baseline_query(
                    responses={
                        "generate_sql": "SELECT order_amount FROM fact_order WHERE region_name = '华北'" + requested
                    },
                    expected_sql=None,
                )

                self.assertEqual(trace["execution"]["attempts"], 1)
                self.assertEqual(
                    trace["execution"]["sql"],
                    f"SELECT order_amount FROM fact_order WHERE region_name = '华北' LIMIT {permitted}",
                )
                self.assertEqual(trace["validation"]["attempts"], 1)
                self.assertEqual(trace["dependencies"]["dw_mysql"]["calls"][-2]["inputs"], trace["execution"]["sql"])
