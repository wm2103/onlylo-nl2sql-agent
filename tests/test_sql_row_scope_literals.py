import unittest
from dataclasses import replace
from unittest.mock import patch

from langgraph.runtime import Runtime
from omegaconf import OmegaConf

from app.agent.context import DataAgentContext
from app.agent.nodes.run_sql import run_sql
from app.agent.nodes.validate_sql import validate_sql
from app.agent.sql_policy import SQLExecutionPolicy, SQLPolicyError, enforce_sql_policy
from app.conf.app_config import SQLExecutionConfig, app_config
from tests.baseline_harness import FakeDWRepository, RecordingAdapters

TABLES = [
    {
        "name": "dim_region",
        "columns": [{"name": "region_id"}, {"name": "region_name"}],
    }
]
ROW_SCOPE_QUERIES = (
    "SELECT region_id FROM dim_region WHERE region_name='{value}'",
    "SELECT region_id FROM dim_region WHERE '{value}'=region_name",
    "SELECT region_id FROM dim_region WHERE region_name IN ('{allowed}', '{value}')",
    "SELECT r.region_id FROM dim_region AS r WHERE r.region_name='{value}'",
    "WITH r AS (SELECT region_id FROM dim_region WHERE region_name='{value}') "
    "SELECT region_id FROM r",
    "SELECT region_id FROM dim_region WHERE region_name='{allowed}' UNION ALL "
    "SELECT region_id FROM dim_region WHERE region_name='{value}'",
    "SELECT region_id FROM dim_region WHERE region_name='{value}' UNION ALL "
    "SELECT region_id FROM dim_region WHERE region_name='{allowed}'",
)


class SQLRowScopeLiteralTest(unittest.IsolatedAsyncioTestCase):
    def test_each_protected_branch_compiles_to_byte_comparisons(self):
        sql = (
            "WITH regions AS (SELECT r.region_id FROM dim_region AS r "
            "WHERE 'É ' = r.region_name AND r.region_id IS NOT NULL) "
            "SELECT region_id FROM regions UNION ALL "
            "SELECT region_id FROM dim_region WHERE region_name IN ('華', 'É ')"
        )
        expected = (
            "WITH regions AS (SELECT r.region_id FROM `dw`.dim_region AS r "
            "WHERE CAST(r.region_name AS BINARY) = x'C38920' AND NOT r.region_id IS NULL) "
            "SELECT region_id FROM regions UNION ALL "
            "SELECT region_id FROM `dw`.dim_region WHERE CAST(region_name AS BINARY) "
            "IN (x'E88FAF', x'C38920') LIMIT 1000"
        )
        policy = SQLExecutionPolicy(
            "dw", 1000, {"dim_region": {"region_name": ("É ", "華")}},
            row_scope_column_types={"dim_region": {"region_name": "varchar_utf8mb4"}},
        )
        compiled = enforce_sql_policy(sql, TABLES, policy)
        self.assertEqual(compiled, expected)
        self.assertEqual(enforce_sql_policy(compiled, TABLES, policy), expected)

    async def test_explain_and_execution_receive_the_same_byte_comparison(self):
        sql = "SELECT region_id FROM dim_region WHERE region_name IN ('華', 'É ')"
        expected = (
            "SELECT region_id FROM `dw`.dim_region WHERE CAST(region_name AS BINARY) "
            "IN (x'E88FAF', x'C38920') LIMIT 1000"
        )
        deployment = OmegaConf.merge(
            OmegaConf.structured(SQLExecutionConfig),
            {
                "row_filters": {"dim_region": {"region_name": ["É ", "華"]}},
                "row_scope_column_types": {"dim_region": {"region_name": "varchar_utf8mb4"}},
            },
        )
        config = replace(
            app_config,
            db_dw=replace(app_config.db_dw, database="dw"),
            sql_execution=OmegaConf.to_object(deployment),
        )
        adapters = RecordingAdapters()
        runtime = Runtime(
            context=DataAgentContext(
                dw_mysql_repository=FakeDWRepository(adapters, expected_sql=expected),
            ),
            stream_writer=lambda event: None,
        )
        state = {"sql": sql, "table_infos": TABLES}
        with patch("app.conf.app_config.app_config", config):
            validated = await validate_sql(state, runtime)
            self.assertIsNone(validated["error"])
            await run_sql({**state, **validated}, runtime)
        self.assertEqual(
            [(call["operation"], call["inputs"]) for call in adapters.snapshot()["dw_mysql"]["calls"]],
            [("validate", expected), ("run", expected)],
        )

    async def test_unproven_comparisons_never_reach_either_adapter(self):
        for column_type, allowed, literal in (
            (None, "north", "'north'"),
            ("char_utf8mb4", "north", "'north'"),
            ("int", 1, "1"),
            ("varchar_utf8mb4", 1, "1"),
            ("varchar_utf8mb4", "1", "1.0"),
        ):
            types = {"dim_region": {"region_name": column_type}} if column_type else {}
            policy = SQLExecutionPolicy(
                "dw", 1000, {"dim_region": {"region_name": (allowed,)}},
                row_scope_column_types=types,
            )
            sql = f"SELECT region_id FROM dim_region WHERE region_name={literal}"
            for node in (validate_sql, run_sql):
                with self.subTest(column_type=column_type, literal=literal, seam=node.__name__):
                    adapters = RecordingAdapters()
                    events = []
                    runtime = Runtime(
                        context=DataAgentContext(
                            sql_execution_policy=policy,
                            dw_mysql_repository=FakeDWRepository(adapters, expected_sql=None),
                        ),
                        stream_writer=events.append,
                    )
                    result = await node({"sql": sql, "table_infos": TABLES}, runtime)
                    self.assertEqual(adapters.snapshot()["dw_mysql"]["count"], 0)
                    self.assertTrue(result["sql_policy_denied"])
                    self.assertEqual(
                        events[-1],
                        {"type": "terminal", "status": "blocked", "reason_code": "sql_policy_denied"},
                    )

    async def test_case_mismatches_are_denied_without_database_calls(self):
        # Neither the authorization policy nor SQL proves case-insensitive equality.
        for allowed, candidate in (("north", "NORTH"), ("NORTH", "north"), ("İ", "i̇")):
            policy = SQLExecutionPolicy(
                "dw", 1000, {"dim_region": {"region_name": (allowed,)}},
                row_scope_column_types={"dim_region": {"region_name": "varchar_utf8mb4"}},
            )
            for template in ROW_SCOPE_QUERIES:
                sql = template.format(value=candidate, allowed=allowed)
                with self.subTest(sql=sql, allowed=allowed, seam="policy"):
                    with self.assertRaises(SQLPolicyError):
                        enforce_sql_policy(sql, TABLES, policy)
                for node in (validate_sql, run_sql):
                    with self.subTest(sql=sql, allowed=allowed, seam=node.__name__):
                        adapters = RecordingAdapters()
                        events = []
                        runtime = Runtime(
                            context=DataAgentContext(
                                sql_execution_policy=policy,
                                dw_mysql_repository=FakeDWRepository(adapters, expected_sql=None),
                            ),
                            stream_writer=events.append,
                        )
                        result = await node({"sql": sql, "table_infos": TABLES}, runtime)
                        self.assertEqual(adapters.snapshot()["dw_mysql"]["count"], 0)
                        self.assertTrue(result["sql_policy_denied"])
                        self.assertIn(
                            {"type": "terminal", "status": "blocked", "reason_code": "sql_policy_denied"},
                            events,
                        )
