"""Exercise the live evaluator through graph events and DW adapters, offline."""

import unittest

from app.agent.sql_policy import SQLExecutionPolicy
from app.entities.value_info import ValueInfo
from tests.baseline_harness import (
    COLUMNS,
    QUERY,
    SQL,
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

PHYSICAL = [
    {
        "name": "fact_order",
        "columns": [
            {"name": "order_amount", "type": "decimal(12,2)"},
            {"name": "region_name", "type": "varchar(20)"},
            {"name": "physical_only", "type": "integer"},
        ],
    }
]


def fixture(*, sql=SQL, correction=SQL, faults=None, validate_failures=0):
    adapters = RecordingAdapters(faults)
    dw = FakeDWRepository(adapters, validate_failures=validate_failures)
    return adapters, dict(
        llm=FakeLLM(
            adapters, {"generate_sql": sql, "correct_sql": correction}
        ).runnable,
        column_qdrant_repository=FakeColumnRepository(
            adapters, [COLUMNS["fact_order.order_amount"], REGION]
        ),
        metric_qdrant_repository=FakeMetricRepository(adapters),
        value_es_repository=FakeValueRepository(
            adapters, [ValueInfo("north", "华北", REGION.id)]
        ),
        embedding_client=FakeEmbedding(adapters),
        meta_mysql_repository=FakeMetaRepository(adapters),
        dw_mysql_repository=dw,
        sql_execution_policy=SQLExecutionPolicy("dw", 1000),
        max_sql_correction_attempts=2,
    )


def dw_calls(adapters):
    return [
        (c["operation"], c["inputs"])
        for c in adapters.calls["dw_mysql"]
        if c["operation"] in {"validate", "run"}
    ]


class GraphEvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_uses_physical_prompt_but_executes_only_validated_graph_sql(self):
        from langchain_core.runnables import RunnableLambda

        from eval.live_evaluation import build_graph, observe

        adapters, context = fixture()
        responses = context["llm"]
        prompts = []

        async def capture(prompt):
            prompts.append(prompt.to_string())
            return await responses.ainvoke(prompt)

        context["llm"] = RunnableLambda(capture)
        result = await observe(
            build_graph("A", PHYSICAL), QUERY, context, group="A", timeout=5
        )
        self.assertEqual(result["terminal"], "success")
        self.assertEqual(result["run_calls"], 1)
        self.assertEqual(
            dw_calls(adapters),
            [
                ("validate", SQL + " LIMIT 1000"),
                ("run", SQL + " LIMIT 1000"),
            ],
        )
        (generation,) = [text for text in prompts if "资深的数据库专家" in text]
        self.assertIn("physical_only", generation)
        self.assertNotIn("name: GMV", generation)

    async def test_c_explains_once_without_repair_while_d_rechecks_repair(self):
        from eval.live_evaluation import build_graph, observe

        for group, expected in (
            ("C", [("validate", SQL + " LIMIT 1000")]),
            (
                "D",
                [
                    ("validate", SQL + " LIMIT 1000"),
                    ("validate", SQL + " LIMIT 1000"),
                    ("run", SQL + " LIMIT 1000"),
                ],
            ),
        ):
            with self.subTest(group=group):
                adapters, context = fixture(validate_failures=1)
                result = await observe(
                    build_graph(group, PHYSICAL), QUERY, context, group=group, timeout=5
                )
                self.assertEqual(dw_calls(adapters), expected)
                self.assertEqual(
                    result["terminal"], "blocked" if group == "C" else "success"
                )

    async def test_b_changes_generation_only_and_does_not_mutate_deployed_graph(self):
        from langchain_core.runnables import RunnableLambda

        from eval.live_evaluation import build_graph, observe

        adapters, context = fixture()
        prompts = []
        llm = context["llm"]

        async def capture(prompt):
            prompts.append(prompt.to_string())
            return await llm.ainvoke(prompt)

        context["llm"] = RunnableLambda(capture)
        for group in ("B", "D"):
            result = await observe(
                build_graph(group, PHYSICAL, PHYSICAL, [{"name": "business-sentinel"}]),
                QUERY,
                context,
                group=group,
                timeout=5,
            )
            self.assertEqual(result["terminal"], "success")
        generations = [text for text in prompts if "资深的数据库专家" in text]
        self.assertIn("business-sentinel", generations[0])
        self.assertIn("physical_only", generations[0])
        self.assertNotIn("business-sentinel", generations[1])
        self.assertNotIn("physical_only", generations[1])
        self.assertEqual(
            [op for op, sql in dw_calls(adapters)],
            ["validate", "run", "validate", "run"],
        )

    async def test_timeout_before_generation_never_dispatches_sql(self):
        import asyncio

        from langchain_core.runnables import RunnableLambda

        from eval.live_evaluation import build_graph, observe

        adapters, context = fixture()

        async def stalled(prompt):
            await asyncio.sleep(30)

        context["llm"] = RunnableLambda(stalled)
        result = await observe(
            build_graph("D", PHYSICAL), QUERY, context, group="D", timeout=0.05
        )
        self.assertEqual(result["terminal"], "timeout")
        self.assertEqual(result["run_calls"], 0)
        self.assertEqual(dw_calls(adapters), [])

    async def test_post_dispatch_timeout_keeps_run_count_and_no_result(self):
        import asyncio

        from eval.live_evaluation import build_graph, observe

        adapters, context = fixture()
        dw = context["dw_mysql_repository"]
        normal_run = dw.run

        async def slow_run(sql):
            await normal_run(sql)
            await asyncio.sleep(30)

        dw.run = slow_run
        result = await observe(
            build_graph("D", PHYSICAL), QUERY, context, group="D", timeout=2.5
        )
        self.assertEqual(result["terminal"], "timeout")
        self.assertIsNone(result["rows"])
        self.assertEqual(result["run_calls"], 1)
        self.assertEqual(
            dw_calls(adapters),
            [
                ("validate", SQL + " LIMIT 1000"),
                ("run", SQL + " LIMIT 1000"),
            ],
        )
        self.assertIsNone(result["run_completed"])

    async def test_all_groups_keep_recall_and_sql_policy_gates(self):
        from eval.live_evaluation import build_graph, observe

        omitted = SQL.split(" WHERE")[0]
        for group, params, reason, expected in (
            ("A", {"sql": omitted}, "sql_policy_denied", []),
            (
                "B",
                {"faults": {"elasticsearch": RuntimeError("private")}},
                "required_filter_unverified",
                [],
            ),
            ("C", {"sql": "DELETE FROM fact_order"}, "sql_policy_denied", []),
            (
                "D",
                {"correction": omitted, "validate_failures": 1},
                "sql_policy_denied",
                [("validate", SQL + " LIMIT 1000")],
            ),
        ):
            with self.subTest(group=group):
                adapters, context = fixture(**params)
                result = await observe(
                    build_graph(group, PHYSICAL, PHYSICAL),
                    QUERY,
                    context,
                    group=group,
                    timeout=5,
                )
                self.assertEqual(result["reason"], reason)
                self.assertEqual(result["terminal"], "blocked")
                self.assertEqual(dw_calls(adapters), expected)
                self.assertEqual(result["run_calls"], 0)
                self.assertEqual(
                    result["recall_decision"]["status"],
                    "blocked" if group == "B" else "success",
                )
                self.assertIn("required_sources", result["recall_decision"]["request"])
                self.assertIn("evidence", result["recall_decision"])


class GoldTests(unittest.IsolatedAsyncioTestCase):
    async def test_gold_is_independently_explained_after_policy_and_matches_by_position(
        self,
    ):
        from eval.live_evaluation import compare_projections, read_gold

        adapters, context = fixture()
        dw = context["dw_mysql_repository"]
        dw.projection = ("total_sales",)
        policy = context["sql_execution_policy"]
        result = await read_gold(SQL, PHYSICAL, policy, dw, timeout=5)
        self.assertEqual(
            dw_calls(adapters),
            [
                ("validate", SQL + " LIMIT 1000"),
                ("run", SQL + " LIMIT 1000"),
            ],
        )
        self.assertEqual(result["status"], "success")
        spec = {"columns": ["gmv"], "ordered": True, "abs_tolerance": 0.01}
        self.assertTrue(
            compare_projections(
                [{"renamed": 3200.01}],
                ("renamed",),
                result["rows"],
                result["projection"],
                spec,
            )
        )
        self.assertFalse(
            compare_projections(
                [],
                ("d", "D"),
                [],
                ("total_sales",),
                spec,
            )
        )
        self.assertFalse(
            compare_projections(
                [{"renamed": 0}],
                ("renamed",),
                [],
                ("total_sales",),
                spec,
            )
        )
        denied = await read_gold(
            "DELETE FROM fact_order", PHYSICAL, policy, dw, timeout=5
        )
        self.assertEqual((denied["status"], denied["run_calls"]), ("blocked", 0))
        self.assertEqual(len(dw_calls(adapters)), 2)


class CLITests(unittest.TestCase):
    def test_dataset_rejects_unaudited_or_ambiguous_projection_contracts(self):
        from copy import deepcopy

        from eval.live_evaluation import validate_dataset

        valid = {
            "version": "live-v1",
            "cases": [
                {
                    "id": "q",
                    "query": "total",
                    "label": "answerable",
                    "audit": "confirmed scalar sum",
                    "gold_sql": "SELECT SUM(amount) FROM orders",
                    "gold": {
                        "columns": ["total"],
                        "ordered": False,
                        "abs_tolerance": 0.01,
                    },
                }
            ],
        }
        validate_dataset(valid)
        for key, value in (
            ("audit", ""),
            ("gold_sql", ""),
            ("gold", {"columns": ["x", "X"], "ordered": True, "abs_tolerance": 0}),
            ("gold", {"columns": ["x"], "ordered": True, "abs_tolerance": -1}),
        ):
            invalid = deepcopy(valid)
            invalid["cases"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_dataset(invalid)

    def test_module_help_is_runnable_without_loading_application(self):
        import subprocess
        import sys

        process = subprocess.run(
            [sys.executable, "-m", "eval.live_evaluation", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(process.returncode, 0)
        self.assertIn("--config", process.stdout)
        self.assertIn("--case-timeout", process.stdout)
        self.assertNotIn("app_config", process.stderr)

    def test_missing_runtime_config_leaves_metadata_and_blocked_record(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from eval.live_evaluation import main

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.json"
            dataset.write_text(
                json.dumps(
                    {
                        "version": "test-v1",
                        "cases": [
                            {
                                "id": "total",
                                "query": "total sales",
                                "label": "answerable",
                                "audit": "audited against the test fact table",
                                "gold_sql": "SELECT SUM(order_amount) FROM fact_order",
                                "gold": {
                                    "columns": ["gmv"],
                                    "ordered": False,
                                    "abs_tolerance": 0,
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "out"
            self.assertEqual(
                main(
                    [
                        "--dataset",
                        str(dataset),
                        "--output-dir",
                        str(output),
                        "--config",
                        str(root / "password=must-not-appear.yaml"),
                    ]
                ),
                2,
            )
            meta = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            records = [
                json.loads(line)
                for line in (output / "records.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(meta["dataset_version"], "test-v1")
            self.assertEqual(records[0]["status"], "blocked")
            self.assertNotIn("must-not-appear", json.dumps([meta, records]))


class RunPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def run_substitute(self, runtime, *, answerable=False, timeout=5):
        import asyncio
        import json
        from contextlib import asynccontextmanager
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace

        from eval.live_evaluation import run_live

        @asynccontextmanager
        async def factory(*args):
            yield runtime

        case = {
            "id": "budget-case",
            "query": QUERY,
            "label": "answerable" if answerable else "needs_info",
            "gold_sql": SQL,
            "gold": {"columns": ["gmv"], "ordered": False, "abs_tolerance": 0},
        }
        with TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                output_dir=Path(tmp),
                groups=["C"],
                init_timeout=1,
                case_timeout=timeout,
                cleanup_timeout=0.02,
                model_timeout=1,
                model_retries=0,
                max_tokens=32,
            )
            runtime.args = args
            code = await asyncio.wait_for(
                run_live(
                    args,
                    {"version": "test-v1", "cases": [case]},
                    {},
                    {},
                    runtime_factory=factory,
                ),
                timeout=8 if timeout >= 1 else 5,
            )
            records = [
                json.loads(line)
                for line in (Path(tmp) / "records.jsonl").read_text().splitlines()
            ]
        return code, records

    async def test_readonly_transaction_entry_is_inside_case_budget(self):
        import asyncio
        from contextlib import asynccontextmanager
        from types import SimpleNamespace

        from eval.live_runtime import LiveRuntime, readonly_session

        entered = []

        class Session:
            async def execute(self, statement):
                entered.append(str(statement))
                await asyncio.Event().wait()

        @asynccontextmanager
        async def session():
            yield Session()

        runtime = LiveRuntime(
            metadata={},
            physical_tables=PHYSICAL,
            business_tables=PHYSICAL,
            business_metrics=[],
            policy=SQLExecutionPolicy("dw", 1000),
            config={
                "llm": {
                    "model_name": "offline",
                    "api_key": "synthetic",
                    "base_url": "http://localhost:9/v1",
                },
                "es": {"index_name": "offline"},
            },
            resources=SimpleNamespace(
                dw_sessions=lambda: readonly_session(session),
                meta_sessions=session,
                es=object(),
            ),
            redact=lambda value: value,
        )
        code, records = await self.run_substitute(
            runtime, answerable=True, timeout=0.02
        )
        self.assertEqual(code, 2)
        self.assertEqual(entered, ["SET TRANSACTION READ ONLY"] * 2)
        (gold,) = [record for record in records if record["type"] == "gold"]
        self.assertEqual(
            (gold["id"], gold["status"], gold["phase"]),
            ("budget-case", "timeout", "acquire"),
        )
        self.assertEqual((gold["validate_calls"], gold["run_calls"]), (0, 0))
        cases = [record for record in records if record["type"] == "case"]
        self.assertEqual(len(cases), 1)
        self.assertEqual((cases[0]["id"], cases[0]["group"]), ("budget-case", "C"))
        self.assertEqual(cases[0]["actual"]["terminal"], "timeout")
        self.assertEqual(cases[0]["actual"]["run_calls"], 0)
        self.assertEqual(cases[0]["actual"]["validate_calls"], 0)

    async def test_graph_and_gold_observations_survive_rollback_failure(self):
        import json
        from contextlib import asynccontextmanager
        from types import SimpleNamespace

        adapters, context = fixture()
        warehouse = context["dw_mysql_repository"]
        warehouse.projection = ("total_sales",)
        durable_before_exit = []
        runtime = SimpleNamespace(
            metadata={},
            physical_tables=PHYSICAL,
            business_tables=PHYSICAL,
            business_metrics=[],
            policy=context["sql_execution_policy"],
            redact=lambda value: value,
        )

        def assert_durable(kind):
            records = [
                json.loads(line)
                for line in (runtime.args.output_dir / "records.jsonl")
                .read_text()
                .splitlines()
            ]
            observed = [r for r in records if r["type"] == kind]
            self.assertEqual(len(observed), 1)
            result = observed[0]["actual"] if kind == "case" else observed[0]
            self.assertEqual(result["run_calls"], 1)
            durable_before_exit.append(kind)

        @asynccontextmanager
        async def gold():
            yield warehouse
            assert_durable("gold")
            raise RuntimeError("rollback failed password=synthetic")

        @asynccontextmanager
        async def graph_context(responses):
            yield context
            assert_durable("case")
            raise RuntimeError("rollback failed password=synthetic")

        runtime.gold, runtime.context = gold, graph_context
        code, records = await self.run_substitute(runtime, answerable=True)
        self.assertEqual(code, 2)
        self.assertEqual(durable_before_exit, ["gold", "case"])
        cases = [r for r in records if r["type"] == "case"]
        golds = [r for r in records if r["type"] == "gold"]
        self.assertEqual((len(cases), len(golds)), (1, 1))
        self.assertEqual((cases[0]["id"], cases[0]["group"]), ("budget-case", "C"))
        self.assertEqual(cases[0]["actual"]["terminal"], "success")
        self.assertEqual(golds[0]["status"], "success")
        self.assertTrue(cases[0]["sql_result_pass"])
        self.assertEqual(sum(op == "run" for op, _ in dw_calls(adapters)), 2)
        failures = [r for r in records if r.get("phase") == "teardown"]
        self.assertEqual(len(failures), 2)
        self.assertEqual({r["error_type"] for r in failures}, {"RuntimeError"})
        self.assertNotIn("synthetic", json.dumps(records))

    async def test_stalled_teardown_is_bounded_and_keeps_completed_results(self):
        import asyncio
        from contextlib import asynccontextmanager
        from types import SimpleNamespace

        _, context = fixture()
        warehouse = context["dw_mysql_repository"]
        warehouse.projection = ("total_sales",)
        cancelled = []

        @asynccontextmanager
        async def hanging(resource):
            yield resource
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(resource)

        runtime = SimpleNamespace(
            metadata={},
            physical_tables=PHYSICAL,
            business_tables=PHYSICAL,
            business_metrics=[],
            policy=context["sql_execution_policy"],
            redact=lambda value: value,
            gold=lambda: hanging(warehouse),
            context=lambda responses: hanging(context),
        )
        code, records = await self.run_substitute(runtime, answerable=True)
        self.assertEqual(code, 2)
        self.assertEqual(cancelled, [warehouse, context])
        for kind in ("gold", "case"):
            (observed,) = [r for r in records if r["type"] == kind]
            events = [
                r
                for r in records
                if r["type"] == "lifecycle" and r["record_id"] == observed["record_id"]
            ]
            self.assertEqual(
                [(r["phase"], r["status"]) for r in events],
                [("started", "started"), ("teardown", "timeout")],
            )
            self.assertTrue(all("run_calls" not in r for r in events))
            self.assertEqual(
                (observed["actual"] if kind == "case" else observed)["run_calls"], 1
            )

    async def test_inflight_gold_and_graph_timeouts_persist_dispatch_counts(self):
        import asyncio
        from contextlib import asynccontextmanager
        from types import SimpleNamespace

        adapters, context = fixture()
        warehouse = context["dw_mysql_repository"]
        run = warehouse.run

        async def stalled(sql):
            await run(sql)
            await asyncio.Event().wait()

        warehouse.run = stalled

        @asynccontextmanager
        async def opened(resource):
            yield resource

        runtime = SimpleNamespace(
            metadata={},
            physical_tables=PHYSICAL,
            business_tables=PHYSICAL,
            business_metrics=[],
            policy=context["sql_execution_policy"],
            redact=lambda value: value,
            gold=lambda: opened(warehouse),
            context=lambda responses: opened(context),
        )
        code, records = await self.run_substitute(runtime, answerable=True, timeout=2.5)
        self.assertEqual(code, 2)
        (gold,) = [r for r in records if r["type"] == "gold"]
        (case,) = [r for r in records if r["type"] == "case"]
        self.assertEqual(gold["status"], "timeout")
        self.assertEqual(case["actual"]["terminal"], "timeout")
        self.assertIsNone(case["sql_result_pass"])
        for result in (gold, case["actual"]):
            self.assertEqual((result["validate_calls"], result["run_calls"]), (1, 1))
            self.assertIsNone(result["run_completed"])
            self.assertIsNone(result["rows"])
        self.assertEqual([op for op, _ in dw_calls(adapters)], ["validate", "run"] * 2)

    async def test_initial_and_enriched_metadata_scrub_configured_and_url_credentials(
        self,
    ):
        import json
        from contextlib import asynccontextmanager
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from unittest.mock import patch

        from eval.live_evaluation import run_live

        config = {
            "llm": {
                "api_key": "synthetic-path-key",
                "model_name": "requested-model",
                "base_url": "https://name:url-pass@host/synthetic-path-key/v1?token=abc",
            }
        }
        snapshots = []

        @asynccontextmanager
        async def factory(config, args):
            snapshots.append(
                json.loads((args.output_dir / "metadata.json").read_text())
            )
            yield SimpleNamespace(
                metadata={
                    "extra": "mysql://other:extra-pass@host/db",
                    "path": "/synthetic-path-key/info",
                }
            )

        with TemporaryDirectory() as tmp:
            args = SimpleNamespace(output_dir=Path(tmp), init_timeout=1)
            with patch("eval.live_evaluation.evaluate_cases", return_value=0):
                code = await run_live(
                    args,
                    {},
                    config,
                    {"note": "synthetic-path-key"},
                    runtime_factory=factory,
                )
            snapshots.append(json.loads((Path(tmp) / "metadata.json").read_text()))
        self.assertEqual(code, 0)
        self.assertEqual(len(snapshots), 2)
        for metadata in snapshots:
            self.assertEqual(metadata["model"]["requested"], "requested-model")
            self.assertEqual(
                metadata["model"]["base_url"], "https://host/[REDACTED]/v1"
            )
            for secret in ("synthetic-path-key", "url-pass", "extra-pass"):
                self.assertNotIn(secret, json.dumps(metadata))

    async def test_runtime_init_timeout_is_not_a_sample_timeout(self):
        import asyncio
        import json
        from contextlib import asynccontextmanager
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace

        from eval.live_evaluation import run_live

        @asynccontextmanager
        async def stalled(*args):
            await asyncio.Event().wait()
            yield

        with TemporaryDirectory() as tmp:
            args = SimpleNamespace(output_dir=Path(tmp), init_timeout=0.02)
            code = await asyncio.wait_for(
                run_live(args, {}, {}, {}, runtime_factory=stalled), timeout=1
            )
            records = [
                json.loads(line)
                for line in (Path(tmp) / "records.jsonl").read_text().splitlines()
            ]
        self.assertEqual(code, 2)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["type"], "initialization")
        self.assertEqual(records[0]["status"], "timeout")
        self.assertEqual(records[0]["reason"], "initialization_budget_exhausted")

    async def test_runtime_init_failure_is_durable_and_never_runs_graph(self):
        import json
        from contextlib import asynccontextmanager
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace

        from eval.live_evaluation import run_live

        @asynccontextmanager
        async def unavailable(*args):
            raise RuntimeError("password=secret mysql://root:secret@host/db")
            yield

        with TemporaryDirectory() as tmp:
            args = SimpleNamespace(output_dir=Path(tmp), init_timeout=1)
            config = {
                "llm": {
                    "model_name": "requested-model",
                    "base_url": "https://user:pass@host/v1?token=secret",
                    "api_key": "secret",
                }
            }
            result = await run_live(
                args, {"cases": []}, config, {}, runtime_factory=unavailable
            )
            metadata = json.loads(
                (Path(tmp) / "metadata.json").read_text(encoding="utf8")
            )
            self.assertEqual(metadata["model"]["requested"], "requested-model")
            self.assertEqual(metadata["model"]["base_url"], "https://host/v1")
            record = json.loads(
                (Path(tmp) / "records.jsonl").read_text(encoding="utf8")
            )
            self.assertEqual(result, 2)
            self.assertEqual(record["status"], "blocked")
            self.assertNotIn("secret", json.dumps(record))

    async def test_completed_case_is_saved_before_next_case_fails(self):
        import json
        from contextlib import asynccontextmanager
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace

        from eval.live_evaluation import run_live

        adapters, context = fixture()
        warehouse = context["dw_mysql_repository"]
        warehouse.projection = ("total_sales",)

        test_policy = context["sql_execution_policy"]

        class SubstituteRuntime:
            metadata = {"runtime": "offline-substitute"}
            physical_tables = PHYSICAL
            business_tables = PHYSICAL
            business_metrics = []
            policy = test_policy

            @asynccontextmanager
            async def gold(self):
                yield warehouse

            @asynccontextmanager
            async def context(self, response_records):
                if getattr(self, "used", False):
                    raise RuntimeError("do not lose previous output")
                self.used = True
                yield context

            def redact(self, value):
                return value

        @asynccontextmanager
        async def factory(*args):
            yield SubstituteRuntime()

        case = {
            "id": "total",
            "query": QUERY,
            "label": "answerable",
            "audit": "test",
            "gold_sql": SQL,
            "gold": {"columns": ["gmv"], "ordered": False, "abs_tolerance": 0},
        }
        with TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                output_dir=Path(tmp), init_timeout=2, case_timeout=5, groups=["C"]
            )
            code = await run_live(
                args,
                {"version": "test-v1", "cases": [case, {**case, "id": "second"}]},
                {},
                {},
                runtime_factory=factory,
            )
            records = [
                json.loads(line)
                for line in (Path(tmp) / "records.jsonl")
                .read_text(encoding="utf8")
                .splitlines()
            ]
            self.assertEqual(code, 2)
            first = next(r for r in records if r.get("type") == "case")
            self.assertTrue(first["sql_result_pass"])
            self.assertEqual(first["actual"]["run_calls"], 1)
            # Gold and graph are separately accounted: no scoring replay.
            self.assertEqual(sum(op == "run" for op, _ in dw_calls(adapters)), 3)
            second = next(
                r for r in records if r["type"] == "case" and r["id"] == "second"
            )
            self.assertEqual(second["group"], "C")
            self.assertEqual(second["actual"]["terminal"], "error")
            self.assertEqual(second["actual"]["phase"], "acquire")
            self.assertEqual(second["actual"]["run_calls"], 0)
            self.assertEqual(second["actual"]["validate_calls"], 0)
