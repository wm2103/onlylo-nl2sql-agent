"""Runtime adapter seams use substitutes; this module never calls live services."""

import unittest


class ConfigurationBootstrapTests(unittest.TestCase):
    def test_injected_runtime_loads_policy_without_default_model_environment(self):
        import os
        import subprocess
        import sys
        from pathlib import Path
        from textwrap import dedent

        env = {key: value for key, value in os.environ.items() if not key.startswith("LLM_")}
        result = subprocess.run(
            [sys.executable, "-c", dedent("""
                import asyncio
                from contextlib import asynccontextmanager
                from types import SimpleNamespace
                from eval.live_runtime import live_runtime

                class ResourcesReached(Exception):
                    pass

                @asynccontextmanager
                async def resources(config, args):
                    raise ResourcesReached
                    yield

                async def check():
                    try:
                        async with live_runtime(
                            {"db_dw": {"database": "dw"}},
                            SimpleNamespace(), resources_factory=resources,
                        ):
                            raise AssertionError("unexpected runtime")
                    except ResourcesReached:
                        print("policy bootstrap reached injected resources")

                asyncio.run(check())
            """)],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("policy bootstrap reached injected resources", result.stdout)


class PhysicalSchemaTests(unittest.IsolatedAsyncioTestCase):
    async def test_information_schema_labels_are_case_independent_and_not_fabricated(
        self,
    ):
        from eval.live_runtime import read_physical_schema

        class Rows:
            def mappings(self):
                return self

            def all(self):
                return [
                    {
                        "TABLE_NAME": "fact_order",
                        "COLUMN_NAME": "order_amount",
                        "COLUMN_TYPE": "decimal(12,2)",
                        "CHARACTER_SET_NAME": None,
                    },
                    {
                        "TABLE_NAME": "fact_order",
                        "COLUMN_NAME": "region_name",
                        "COLUMN_TYPE": "varchar(20)",
                        "CHARACTER_SET_NAME": "utf8mb4",
                    },
                ]

        class Session:
            async def execute(self, statement, params):
                self.sql, self.params = str(statement), params
                return Rows()

        session = Session()
        tables = await read_physical_schema(session, "dw")
        self.assertEqual(
            [(c["name"], c["type"]) for c in tables[0]["columns"]],
            [
                ("order_amount", "decimal(12,2)"),
                ("region_name", "varchar(20)"),
            ],
        )
        self.assertEqual(tables[0]["columns"][1]["charset"], "utf8mb4")
        self.assertEqual(session.params, {"database": "dw"})
        self.assertIn("INFORMATION_SCHEMA.COLUMNS", session.sql)
        self.assertNotIn("SELECT *", session.sql.upper())


class ResponseTests(unittest.TestCase):
    def test_unconfigured_bearer_tokens_are_removed_from_text_and_recording(self):
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, LLMResult

        from eval.live_runtime import ResponseRecorder, redact

        records, saved = [], []
        recorder = ResponseRecorder(records, redact, sink=saved.append)
        text = (
            "SELECT 1;\nAuthorization: Bearer synthetic-unconfigured-token\n  keep me  "
        )
        expected = "SELECT 1;\nAuthorization: [REDACTED]\n  keep me  "
        self.assertEqual(redact(text), expected)
        self.assertNotIn(
            "synthetic-unconfigured-token",
            redact('{"Authorization": "Bearer synthetic-unconfigured-token"}'),
        )
        recorder.on_llm_end(
            LLMResult(generations=[[ChatGeneration(message=AIMessage(content=text))]]),
            run_id="bearer",
        )
        self.assertEqual(records[0]["text"], expected)
        self.assertEqual(saved, records)
        self.assertIsNone(records[0]["response_model"])

    def test_response_text_is_preserved_except_secrets_and_response_model_is_not_guessed(
        self,
    ):
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, LLMResult

        from eval.live_runtime import ResponseRecorder, redact

        records = []
        recorder = ResponseRecorder(
            records, lambda value: redact(value, ("my-private-key",))
        )
        text = "```sql\nSELECT 1; -- my-private-key https://name:pass@host/api?token=hidden\n```"
        recorder.on_llm_end(
            LLMResult(generations=[[ChatGeneration(message=AIMessage(content=text))]]),
            run_id="first",
        )
        self.assertEqual(records[0]["response_model"], None)
        self.assertTrue(records[0]["text"].startswith("```sql\nSELECT 1;"))
        self.assertNotIn("my-private-key", records[0]["text"])
        self.assertNotIn("name:pass", records[0]["text"])
        self.assertNotIn("hidden", records[0]["text"])
        self.assertEqual(
            redact({"api_key": "arbitrary-other-key", "nested": {"password": "abc"}}),
            {"api_key": "[REDACTED]", "nested": {"password": "[REDACTED]"}},
        )
        recorder.on_llm_end(
            LLMResult(
                generations=[
                    [
                        ChatGeneration(
                            message=AIMessage(
                                content="  SELECT 2  ",
                                response_metadata={"model_name": "actual-model"},
                            )
                        )
                    ]
                ]
            ),
            run_id="second",
        )
        self.assertEqual(records[1]["text"], "  SELECT 2  ")
        self.assertEqual(records[1]["response_model"], "actual-model")


class LiveDWTests(unittest.IsolatedAsyncioTestCase):
    async def test_adapter_explains_then_reads_unique_cursor_projection_order(self):
        from eval.live_runtime import LiveDW

        class Result:
            def keys(self):
                return ["a", "b"]

            def mappings(self):
                return self

            def fetchall(self):
                return [{"a": "华北", "b": 10}]

        class Session:
            calls = []

            async def execute(self, statement):
                self.calls.append(str(statement))
                return Result()

        session = Session()
        dw = LiveDW(session)
        await dw.validate("SELECT 1 AS a, 2 AS b")
        self.assertEqual(
            await dw.run("SELECT 1 AS a, 2 AS b"), [{"a": "华北", "b": 10}]
        )
        self.assertEqual(dw.projection, ("a", "b"))
        self.assertEqual(
            session.calls, ["explain SELECT 1 AS a, 2 AS b", "SELECT 1 AS a, 2 AS b"]
        )
        Result.keys = lambda self: ["a", "A"]
        with self.assertRaisesRegex(ValueError, "projection"):
            await dw.run("SELECT 1 AS a, 2 AS A")


class ContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_business_context_joins_server_schema_without_inventing_columns(self):
        from types import SimpleNamespace

        from eval.live_runtime import read_business_context

        class Session:
            async def scalars(self, statement):
                entity = statement.column_descriptions[0]["entity"].__name__
                data = {
                    "TableInfoMySQL": [
                        SimpleNamespace(
                            id="t", name="fact_order", role="fact", description="订单"
                        )
                    ],
                    "ColumnInfoMySQL": [
                        SimpleNamespace(
                            id="c",
                            name="order_amount",
                            table_id="t",
                            role="measure",
                            description="订单金额",
                            alias=["销售额"],
                            examples=[],
                            type="WRONG-TYPE",
                        ),
                        SimpleNamespace(id="not-real", name="invented", table_id="t"),
                    ],
                    "MetricInfoMySQL": [
                        SimpleNamespace(
                            id="gmv",
                            name="GMV",
                            description="SUM(order_amount)",
                            relevant_columns=["c"],
                            alias=["成交总额"],
                        )
                    ],
                }
                return SimpleNamespace(all=lambda: data[entity])

        tables, metrics = await read_business_context(
            Session(),
            [
                {
                    "name": "fact_order",
                    "columns": [
                        {
                            "id": "physical",
                            "name": "order_amount",
                            "type": "decimal(12,2)",
                        }
                    ],
                }
            ],
        )
        self.assertEqual(tables[0]["description"], "订单")
        self.assertEqual(
            tables[0]["columns"],
            [
                {
                    "id": "physical",
                    "name": "order_amount",
                    "type": "decimal(12,2)",
                    "role": "measure",
                    "description": "订单金额",
                    "alias": ["销售额"],
                    "examples": [],
                }
            ],
        )
        self.assertEqual(metrics[0]["name"], "GMV")

    async def test_runtime_injects_user_clients_but_preserves_deployment_policy(self):
        from contextlib import asynccontextmanager
        from pathlib import Path
        from types import SimpleNamespace

        from app.conf.app_config import app_config
        from eval.live_runtime import live_runtime

        class Session:
            async def execute(self, statement, params=None):
                if "INFORMATION_SCHEMA" in str(statement):
                    row = {
                        "TABLE_NAME": "fact_order",
                        "COLUMN_NAME": "order_amount",
                        "COLUMN_TYPE": "decimal(12,2)",
                        "CHARACTER_SET_NAME": None,
                    }
                    return SimpleNamespace(
                        mappings=lambda: SimpleNamespace(all=lambda: [row])
                    )
                return None

            async def scalars(self, statement):
                return SimpleNamespace(all=lambda: [])

        @asynccontextmanager
        async def session():
            yield Session()

        closed = []

        @asynccontextmanager
        async def resources(config, args):
            try:
                yield SimpleNamespace(
                    dw_sessions=session,
                    meta_sessions=session,
                    embedding=object(),
                    qdrant=object(),
                    es=object(),
                )
            finally:
                closed.append(True)

        config = {
            "llm": {
                "model_name": "requested",
                "api_key": "private-key",
                "base_url": "https://user:url-password@x/private-key/v1?token=abc",
                "extra_endpoint": "https://unknown:extra-password@other/v1",
            },
            "db_dw": {
                "database": "dw",
                "host": "mysql://db-user:db-password@host/db",
                "connection_url": "mysql://unknown:extra-password@host/db",
            },
            "sql_execution": {"max_rows": 999999},
            "es": {"index_name": "value_index"},
        }
        args = SimpleNamespace(
            model_timeout=12, model_retries=0, max_tokens=100, output_dir=Path("unused")
        )
        async with live_runtime(config, args, resources_factory=resources) as runtime:
            self.assertEqual(runtime.policy.max_rows, app_config.sql_execution.max_rows)
            self.assertEqual(
                dict(runtime.policy.row_filters),
                dict(app_config.sql_execution.row_filters),
            )
            self.assertEqual(runtime.metadata["model"]["requested"], "requested")
            self.assertEqual(
                runtime.metadata["model"]["base_url"], "https://x/[REDACTED]/v1"
            )
            for secret in (
                "private-key",
                "url-password",
                "db-password",
                "extra-password",
            ):
                self.assertNotIn(secret, str(runtime.metadata))
            self.assertNotIn("extra_endpoint", runtime.metadata["safe_config"]["llm"])
            self.assertNotIn("connection_url", runtime.metadata["safe_config"]["db_dw"])
            self.assertEqual(runtime.physical_tables[0]["name"], "fact_order")
            self.assertEqual(
                runtime.correction_limit,
                app_config.sql_execution.max_correction_attempts,
            )
            async with runtime.context([]) as context:
                self.assertEqual(context["llm"].model_name, "requested")
                self.assertEqual(context["llm"].request_timeout, 12)
                self.assertEqual(context["llm"].max_retries, 0)
                self.assertEqual(context["llm"].temperature, 0)
                self.assertEqual(context["sql_execution_policy"], runtime.policy)
                self.assertEqual(
                    context["max_sql_correction_attempts"], runtime.correction_limit
                )
                self.assertEqual(
                    context["value_es_repository"].index_name, "value_index"
                )
            async with runtime.gold() as dw:
                self.assertIsNone(dw.projection)
        self.assertEqual(closed, [True])

    async def test_real_clients_can_be_constructed_and_closed_without_network_calls(
        self,
    ):
        from types import SimpleNamespace

        from eval.live_runtime import connect_services

        db = {
            "host": "127.0.0.1",
            "port": 9,
            "user": "offline",
            "password": "p@:/word",
            "database": "dw",
        }
        config = {
            "db_dw": db,
            "db_meta": {**db, "database": "meta"},
            "qdrant": {"host": "127.0.0.1", "port": 9},
            "es": {"host": "127.0.0.1", "port": 9},
            "embedding": {"host": "127.0.0.1", "port": 9, "model": "test-model"},
        }
        async with connect_services(
            config, SimpleNamespace(model_timeout=3)
        ) as clients:
            self.assertEqual(clients.dw_engine.url.password, "p@:/word")
            self.assertEqual(clients.dw_engine.url.database, "dw")
            self.assertEqual(clients.meta_engine.url.database, "meta")
            self.assertEqual(clients.embedding.model, "test-model")
