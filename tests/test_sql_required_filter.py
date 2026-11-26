import unittest

from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.nodes.run_sql import run_sql
from app.agent.nodes.validate_sql import validate_sql
from app.agent.sql_policy import SQLExecutionPolicy, SQLPolicyError, enforce_sql_policy
from tests.baseline_harness import FakeDWRepository, RecordingAdapters

TABLES = [
    {
        "name": "fact_order",
        "columns": [
            {"id": "fact_order.order_amount", "name": "order_amount"},
            {"id": "fact_order.region_name", "name": "region_name"},
            {"id": "fact_order.tenant_id", "name": "tenant_id"},
        ],
    }
]
FILTER = [{"column_id": "fact_order.region_name", "operator": "eq", "value": "华北"}]
POLICY = SQLExecutionPolicy(database="dw", max_rows=1000)


class SQLRequiredFilterTest(unittest.IsolatedAsyncioTestCase):
    def test_filtered_business_query_is_permitted_from_retained_column_id(self):
        sql = "SELECT SUM(order_amount) AS total_sales FROM fact_order WHERE region_name = '华北'"

        permitted = enforce_sql_policy(sql, TABLES, POLICY, required_filters=FILTER)

        self.assertEqual(
            permitted,
            "SELECT SUM(order_amount) AS total_sales FROM fact_order "
            "WHERE region_name = '华北' LIMIT 1000",
        )

    def test_fixed_ordinary_unfiltered_query_is_rejected_before_execution_boundary(self):
        sql = "SELECT SUM(order_amount) AS total_sales FROM fact_order"

        with self.assertRaises(SQLPolicyError):
            enforce_sql_policy(sql, TABLES, POLICY, required_filters=FILTER)

    def test_required_filters_are_bound_to_retained_table_and_alias(self):
        sql = (
            "SELECT SUM(order_amount) FROM fact_order AS f "
            "WHERE f.region_name = '华北'"
        )
        with self.assertRaises(SQLPolicyError):
            enforce_sql_policy(sql, TABLES, POLICY, required_filters=FILTER)

        for wrong_sql in (
            "SELECT SUM(order_amount) FROM fact_order AS f WHERE region_name = '华北'",
            "SELECT SUM(order_amount) FROM fact_order WHERE region_name = '华北' AND region_name = '华南'",
        ):
            with self.subTest(sql=wrong_sql), self.assertRaises(SQLPolicyError):
                enforce_sql_policy(wrong_sql, TABLES, POLICY, required_filters=FILTER)

    async def test_validate_and_run_forward_the_same_request_filter_sql(self):
        source_sql = (
            "SELECT SUM(order_amount) AS total_sales FROM fact_order "
            "WHERE region_name = '华北'"
        )
        expected_sql = source_sql + " LIMIT 1000"
        for node in (validate_sql, run_sql):
            with self.subTest(node=node.__name__):
                adapters = RecordingAdapters()
                runtime = Runtime(
                    context=DataAgentContext(
                        sql_execution_policy=POLICY,
                        max_sql_correction_attempts=1,
                        dw_mysql_repository=FakeDWRepository(
                            adapters, expected_sql=expected_sql
                        ),
                    ),
                    stream_writer=lambda event: None,
                )
                state = {
                    "sql": source_sql,
                    "table_infos": TABLES,
                    "recall_decision": {"required_filters": FILTER},
                    "sql_validation_passed": True,
                }
                if node is validate_sql:
                    result = await node(state, runtime)
                    self.assertIsNone(result["error"])
                else:
                    await node(state, runtime)
                self.assertEqual(
                    adapters.snapshot()["dw_mysql"]["calls"][0]["inputs"], expected_sql
                )

    async def test_validate_and_run_reject_unfiltered_sql_before_adapter_calls(self):
        sql = "SELECT SUM(order_amount) AS total_sales FROM fact_order"
        for node in (validate_sql, run_sql):
            with self.subTest(node=node.__name__):
                adapters = RecordingAdapters()
                events = []
                runtime = Runtime(
                    context=DataAgentContext(
                        sql_execution_policy=POLICY,
                        dw_mysql_repository=FakeDWRepository(adapters, expected_sql=None),
                    ),
                    stream_writer=events.append,
                )
                result = await node(
                    {
                        "sql": sql,
                        "table_infos": TABLES,
                        "recall_decision": {"required_filters": FILTER},
                    },
                    runtime,
                )
                self.assertTrue(result["sql_policy_denied"])
                self.assertEqual(adapters.snapshot()["dw_mysql"]["count"], 0)
                self.assertEqual(events[-1]["reason_code"], "sql_policy_denied")

    def test_request_filter_and_deployment_authorization_compose_idempotently(self):
        policy = SQLExecutionPolicy(
            database="dw",
            max_rows=1000,
            row_filters={"fact_order": {"region_name": ("华北",)}},
            row_scope_column_types={"fact_order": {"region_name": "varchar_utf8mb4"}},
        )
        source_sql = "SELECT SUM(order_amount) FROM fact_order WHERE region_name = '华北'"
        expected_sql = (
            "SELECT SUM(order_amount) FROM `dw`.fact_order "
            "WHERE CAST(region_name AS BINARY) = x'E58D8EE58C97' LIMIT 1000"
        )

        first_policy_output = enforce_sql_policy(
            source_sql, TABLES, policy, required_filters=FILTER
        )
        second_policy_output = enforce_sql_policy(
            first_policy_output, TABLES, policy, required_filters=FILTER
        )

        self.assertEqual(first_policy_output, expected_sql)
        self.assertEqual(second_policy_output, expected_sql)

    async def test_validate_and_run_forward_composed_policy_sql_once_each(self):
        policy = SQLExecutionPolicy(
            database="dw",
            max_rows=1000,
            row_filters={"fact_order": {"region_name": ("华北",)}},
            row_scope_column_types={"fact_order": {"region_name": "varchar_utf8mb4"}},
        )
        source_sql = "SELECT SUM(order_amount) FROM fact_order WHERE region_name = '华北'"
        expected_sql = (
            "SELECT SUM(order_amount) FROM `dw`.fact_order "
            "WHERE CAST(region_name AS BINARY) = x'E58D8EE58C97' LIMIT 1000"
        )
        adapters = RecordingAdapters()
        runtime = Runtime(
            context=DataAgentContext(
                sql_execution_policy=policy,
                dw_mysql_repository=FakeDWRepository(adapters, expected_sql=expected_sql),
            ),
            stream_writer=lambda event: None,
        )
        state = {
            "sql": source_sql,
            "table_infos": TABLES,
            "recall_decision": {"required_filters": FILTER},
        }

        validated = await validate_sql(state, runtime)
        self.assertTrue(validated["sql_validation_passed"])
        executed = await run_sql({**state, **validated}, runtime)

        self.assertEqual(executed["sql"], expected_sql)
        self.assertEqual(
            [(call["operation"], call["inputs"]) for call in adapters.snapshot()["dw_mysql"]["calls"]],
            [("validate", expected_sql), ("run", expected_sql)],
        )

    def test_request_and_separate_deployment_scope_are_idempotent(self):
        policy = SQLExecutionPolicy(
            database="dw",
            max_rows=1000,
            row_filters={"fact_order": {"tenant_id": ("tenant-one",)}},
            row_scope_column_types={"fact_order": {"tenant_id": "varchar_utf8mb4"}},
        )
        source_sql = (
            "SELECT SUM(order_amount) FROM fact_order "
            "WHERE region_name = '华北' AND tenant_id = 'tenant-one'"
        )
        expected_sql = (
            "SELECT SUM(order_amount) FROM `dw`.fact_order "
            "WHERE region_name = '华北' AND "
            "CAST(tenant_id AS BINARY) = x'74656E616E742D6F6E65' LIMIT 1000"
        )

        first_policy_output = enforce_sql_policy(
            source_sql, TABLES, policy, required_filters=FILTER
        )
        second_policy_output = enforce_sql_policy(
            first_policy_output, TABLES, policy, required_filters=FILTER
        )

        self.assertEqual(first_policy_output, expected_sql)
        self.assertEqual(second_policy_output, expected_sql)

    async def test_validate_output_and_run_preserve_separate_scope_sql(self):
        policy = SQLExecutionPolicy(
            database="dw",
            max_rows=1000,
            row_filters={"fact_order": {"tenant_id": ("tenant-one",)}},
            row_scope_column_types={"fact_order": {"tenant_id": "varchar_utf8mb4"}},
        )
        source_sql = (
            "SELECT SUM(order_amount) FROM fact_order "
            "WHERE region_name = '华北' AND tenant_id = 'tenant-one'"
        )
        expected_sql = (
            "SELECT SUM(order_amount) FROM `dw`.fact_order "
            "WHERE region_name = '华北' AND "
            "CAST(tenant_id AS BINARY) = x'74656E616E742D6F6E65' LIMIT 1000"
        )
        adapters = RecordingAdapters()
        runtime = Runtime(
            context=DataAgentContext(
                sql_execution_policy=policy,
                dw_mysql_repository=FakeDWRepository(adapters, expected_sql=expected_sql),
            ),
            stream_writer=lambda event: None,
        )
        state = {
            "sql": source_sql,
            "table_infos": TABLES,
            "recall_decision": {"required_filters": FILTER},
        }

        validated = await validate_sql(state, runtime)
        self.assertTrue(validated["sql_validation_passed"])
        executed = await run_sql({**state, **validated}, runtime)

        self.assertEqual(executed["sql"], expected_sql)
        self.assertEqual(
            [(call["operation"], call["inputs"]) for call in adapters.snapshot()["dw_mysql"]["calls"]],
            [("validate", expected_sql), ("run", expected_sql)],
        )

    async def test_separate_request_and_scope_failures_reach_no_adapter(self):
        normalized_sql = (
            "SELECT SUM(order_amount) FROM `dw`.fact_order "
            "WHERE region_name = '华北' AND "
            "CAST(tenant_id AS BINARY) = x'74656E616E742D6F6E65' LIMIT 1000"
        )
        candidates = (
            (
                "SELECT SUM(order_amount) FROM fact_order "
                "WHERE region_name = '华南' AND tenant_id = 'tenant-one'",
                SQLExecutionPolicy(
                    database="dw",
                    max_rows=1000,
                    row_filters={"fact_order": {"tenant_id": ("tenant-one",)}},
                    row_scope_column_types={"fact_order": {"tenant_id": "varchar_utf8mb4"}},
                ),
            ),
            (
                "SELECT SUM(order_amount) FROM fact_order "
                "WHERE region_name = '华北' AND tenant_id = 'tenant-two'",
                SQLExecutionPolicy(
                    database="dw",
                    max_rows=1000,
                    row_filters={"fact_order": {"tenant_id": ("tenant-one",)}},
                    row_scope_column_types={"fact_order": {"tenant_id": "varchar_utf8mb4"}},
                ),
            ),
            (
                "SELECT SUM(order_amount) FROM fact_order WHERE region_name = '华北'",
                SQLExecutionPolicy(
                    database="dw",
                    max_rows=1000,
                    row_filters={"fact_order": {"tenant_id": ("tenant-one",)}},
                    row_scope_column_types={"fact_order": {"tenant_id": "varchar_utf8mb4"}},
                ),
            ),
            (
                "SELECT SUM(order_amount) FROM fact_order "
                "WHERE region_name = '华北' AND tenant_id = 'tenant-one'",
                SQLExecutionPolicy(
                    database="dw",
                    max_rows=1000,
                    row_filters={"fact_order": {"tenant_id": ("tenant-one",)}},
                    row_scope_column_types={},
                ),
            ),
            (
                normalized_sql,
                SQLExecutionPolicy(
                    database="dw",
                    max_rows=1000,
                    row_filters={"fact_order": {"tenant_id": ("tenant-two",)}},
                    row_scope_column_types={"fact_order": {"tenant_id": "varchar_utf8mb4"}},
                ),
            ),
            (
                normalized_sql,
                SQLExecutionPolicy(
                    database="dw",
                    max_rows=1000,
                    row_filters={"fact_order": {"tenant_id": ("tenant-one",)}},
                    row_scope_column_types={},
                ),
            ),
        )
        for sql, policy in candidates:
            for node in (validate_sql, run_sql):
                required_filters = (
                    [{"column_id": "fact_order.region_name", "operator": "eq", "value": "华南"}]
                    if sql == normalized_sql and policy.row_scope_column_types
                    else FILTER
                )
                with self.subTest(sql=sql, node=node.__name__):
                    adapters = RecordingAdapters()
                    runtime = Runtime(
                        context=DataAgentContext(
                            sql_execution_policy=policy,
                            dw_mysql_repository=FakeDWRepository(adapters, expected_sql=None),
                        ),
                        stream_writer=lambda event: None,
                    )
                    result = await node(
                        {
                            "sql": sql,
                            "table_infos": TABLES,
                            "recall_decision": {"required_filters": required_filters},
                        },
                        runtime,
                    )
                    self.assertTrue(result["sql_policy_denied"])
                    self.assertEqual(adapters.snapshot()["dw_mysql"]["count"], 0)

    async def test_changed_filter_authorization_and_unsupported_ast_reach_no_adapter(self):
        sql = "SELECT SUM(order_amount) FROM fact_order WHERE region_name = '华北'"
        cases = (
            (
                sql,
                [{"column_id": "fact_order.region_name", "operator": "eq", "value": "华南"}],
                ("华北",),
            ),
            (sql, FILTER, ("华南",)),
            (
                "SELECT SUM(order_amount) FROM fact_order "
                "WHERE CAST(region_name AS CHAR) = '华北'",
                FILTER,
                ("华北",),
            ),
        )
        for candidate, required_filters, authorized_values in cases:
            policy = SQLExecutionPolicy(
                database="dw",
                max_rows=1000,
                row_filters={"fact_order": {"region_name": authorized_values}},
                row_scope_column_types={"fact_order": {"region_name": "varchar_utf8mb4"}},
            )
            for node in (validate_sql, run_sql):
                with self.subTest(candidate=candidate, node=node.__name__):
                    adapters = RecordingAdapters()
                    runtime = Runtime(
                        context=DataAgentContext(
                            sql_execution_policy=policy,
                            dw_mysql_repository=FakeDWRepository(adapters, expected_sql=None),
                        ),
                        stream_writer=lambda event: None,
                    )
                    result = await node(
                        {
                            "sql": candidate,
                            "table_infos": TABLES,
                            "recall_decision": {"required_filters": required_filters},
                            "sql_validation_passed": True,
                        },
                        runtime,
                    )
                    self.assertTrue(result["sql_policy_denied"])
                    self.assertEqual(adapters.snapshot()["dw_mysql"]["count"], 0)

    def test_request_filter_does_not_replace_deployment_authorization(self):
        policy = SQLExecutionPolicy(
            database="dw",
            max_rows=1000,
            row_filters={"fact_order": {"region_name": ("华南",)}},
            row_scope_column_types={"fact_order": {"region_name": "varchar_utf8mb4"}},
        )
        sql = "SELECT SUM(order_amount) FROM fact_order WHERE region_name = '华北'"

        with self.assertRaises(SQLPolicyError):
            enforce_sql_policy(sql, TABLES, policy, required_filters=FILTER)

        unsupported = (
            "SELECT SUM(order_amount) FROM fact_order "
            "WHERE CAST(region_name AS CHAR) = '华北'",
            "SELECT SUM(order_amount) FROM fact_order "
            "WHERE CAST(region_name AS BINARY) = x'E58D8EE58D97'",
        )
        for candidate in unsupported:
            with self.subTest(candidate=candidate), self.assertRaises(SQLPolicyError):
                enforce_sql_policy(candidate, TABLES, policy, required_filters=FILTER)

    def test_only_supported_single_table_equality_shape_is_accepted(self):
        unsupported = (
            "SELECT SUM(order_amount) FROM fact_order "
            "WHERE region_name IN ('华北')",
            "SELECT SUM(order_amount) FROM fact_order "
            "WHERE region_name = '华北' OR order_amount > 0",
            "SELECT SUM(order_amount) FROM fact_order AS f "
            "JOIN dim_region AS r ON f.region_name = r.region_name "
            "WHERE f.region_name = '华北'",
        )
        for sql in unsupported:
            with self.subTest(sql=sql), self.assertRaises(SQLPolicyError):
                enforce_sql_policy(sql, TABLES, POLICY, required_filters=FILTER)


if __name__ == "__main__":
    unittest.main()
