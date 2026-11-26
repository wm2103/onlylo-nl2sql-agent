"""Evaluate the full Text2SQL graph without replaying generated SQL."""

import argparse
import asyncio
import hashlib
import json
import math
import os
from contextlib import aclosing
from copy import copy
from datetime import UTC, datetime
from pathlib import Path

GROUPS = ("A", "B", "C", "D")
DEFAULT_CLEANUP_TIMEOUT = 2.0


def build_graph(group, physical_tables, business_tables=(), business_metrics=()):
    """Keep deployed topology; A/B replace only the generator's local inputs."""
    from langgraph.runtime import Runtime

    from app.agent.context import DataAgentContext
    from app.agent.graph import graph, graph_builder
    from app.agent.nodes.generate_sql import generate_sql
    from app.agent.state import DataAgentState

    if group not in GROUPS:
        raise ValueError("Unknown evaluation group")
    if group in {"C", "D"}:
        return graph

    async def evaluation_generate(
        state: DataAgentState, runtime: Runtime[DataAgentContext]
    ):
        tables = physical_tables if group == "A" else business_tables
        metrics = [] if group == "A" else business_metrics
        return await generate_sql(
            {**state, "table_infos": tables, "metric_infos": metrics}, runtime
        )

    # Fork the builder's node registry, not its edges or graph state. Only this
    # fork is compiled with a new node; the application's graph is unchanged.
    builder = copy(graph_builder)
    builder.nodes = dict(graph_builder.nodes)
    original = builder.nodes.pop("generate_sql")
    builder.compiled = False
    builder.add_node(
        "generate_sql",
        evaluation_generate,
        input_schema=original.input_schema,
        metadata=original.metadata,
        retry_policy=original.retry_policy,
        cache_policy=original.cache_policy,
        destinations=original.ends,
        defer=original.defer,
    )
    return builder.compile()


class RecordedDW:
    """Count adapter entries, including a run interrupted while in flight."""

    def __init__(self, repository, *, deadline=None):
        self.repository = repository
        self.deadline = deadline
        self.explained = None
        self.validate_calls = 0
        self.run_calls = 0
        self.run_sql = None
        self.run_completed = None

    async def get_db_info(self):
        self.check_deadline()
        return await self.repository.get_db_info()

    def check_deadline(self):
        if (
            self.deadline is not None
            and asyncio.get_running_loop().time() >= self.deadline
        ):
            raise TimeoutError

    async def validate(self, sql):
        self.check_deadline()
        self.explained = None
        self.validate_calls += 1
        await self.repository.validate(sql)
        self.explained = sql

    async def run(self, sql):
        self.check_deadline()
        if self.run_calls or self.explained != sql:
            raise RuntimeError("Expected one previously explained statement")
        self.run_calls += 1
        self.run_sql = sql
        rows = await self.repository.run(sql)
        self.run_completed = True
        return rows


async def observe(graph, query, context, *, group, timeout, request_id="live-eval"):
    """Consume the graph once and account for calls even without a terminal event."""
    repository = context["dw_mysql_repository"]
    dw = repository if isinstance(repository, RecordedDW) else RecordedDW(repository)
    context = {**context, "dw_mysql_repository": dw}
    if group != "D":
        context["max_sql_correction_attempts"] = 0
    recall, terminal, state, rows = {}, {}, {}, None
    interruption = None
    deadline = asyncio.timeout(timeout)
    try:
        async with (
            deadline,
            aclosing(
                graph.astream(
                    {"query": query, "request_id": request_id},
                    context=context,
                    stream_mode=["custom", "values"],
                )
            ) as events,
        ):
            async for kind, item in events:
                if kind == "values":
                    state = item
                elif item["type"] == "recall_decision":
                    recall = item
                elif item["type"] == "terminal":
                    terminal = item
                elif item["type"] == "result":
                    rows = item["data"]
    except Exception as error:
        interruption = (
            {"status": "timeout", "reason_code": "case_budget_exhausted"}
            if deadline.expired()
            else {
                "status": "error",
                "reason_code": "graph_error",
                "error_type": type(error).__name__,
            }
        )
    final = (
        interruption
        or terminal
        or (recall if recall.get("status") == "blocked" else {})
    )
    status = final.get("status", "error")
    if status == "success" and recall.get("status") == "degraded":
        status = "degraded"
    return {
        "terminal": status,
        "reason": final.get("reason_code", "graph_terminal_missing"),
        "error_type": final.get("error_type"),
        "recall_terminal": recall.get("status"),
        "recall_decision": state.get("recall_decision"),
        "recall_outcomes": state.get("recall_outcomes"),
        "terminal_event": terminal,
        "candidate_sql": state.get("sql"),
        "sql_failure_reason": state.get("sql_failure_reason"),
        "sql_terminal": terminal.get("status"),
        "sql_outcome": state.get("sql_final_outcome"),
        "rows": rows if status in {"success", "degraded"} else None,
        "validate_calls": dw.validate_calls,
        "run_calls": dw.run_calls,
        "run_sql": dw.run_sql,
        "run_completed": dw.run_completed,
        "correction_attempts": state.get("sql_correction_attempts", 0),
        "projection": getattr(dw.repository, "projection", None),
    }


async def read_gold(sql, tables, policy, repository, *, timeout):
    """Run an audited reference through policy and EXPLAIN on a separate adapter."""
    from app.agent.sql_policy import SQLPolicyError, SQLSyntaxError, enforce_sql_policy

    dw = repository if isinstance(repository, RecordedDW) else RecordedDW(repository)
    deadline = asyncio.timeout(timeout)
    rows, error_type = None, None
    try:
        async with deadline:
            sql = enforce_sql_policy(sql, tables, policy)
            await dw.validate(sql)
            rows = await dw.run(sql)
            status, reason = (
                ("error", "gold_row_cap_reached")
                if len(rows) >= policy.max_rows
                else ("success", "gold_executed")
            )
    except SQLPolicyError, SQLSyntaxError:
        status, reason = "blocked", "gold_policy_or_parse_denied"
    except Exception as error:
        status, reason = (
            ("timeout", "gold_budget_exhausted")
            if deadline.expired()
            else ("error", "gold_adapter_failed")
        )
        error_type = type(error).__name__
    return {
        "status": status,
        "reason": reason,
        "error_type": error_type,
        "rows": rows if status == "success" else None,
        "projection": getattr(dw.repository, "projection", None),
        "validate_calls": dw.validate_calls,
        "run_calls": dw.run_calls,
        "run_completed": dw.run_completed,
    }


def compare_projections(actual_rows, actual_keys, gold_rows, gold_keys, spec):
    """Use cursor projection order, not aliases, gold values, or guessed names."""
    from eval.trusted_evaluation import compare_results

    columns = spec["columns"]

    def canonical(rows, keys):
        if (
            not isinstance(rows, list)
            or keys is None
            or len(keys) != len(columns)
            or len({key.casefold() for key in keys}) != len(keys)
            or any(not isinstance(row, dict) or set(row) != set(keys) for row in rows)
        ):
            return None
        return [dict(zip(columns, (row[key] for key in keys))) for row in rows]

    actual = canonical(actual_rows, actual_keys)
    gold = canonical(gold_rows, gold_keys)
    return (
        actual is not None
        and gold is not None
        and compare_results(
            actual,
            {
                **spec,
                "rows": gold,
                "columns": {column: column for column in columns},
            },
        )
    )


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, default=str, allow_nan=False
        ).encode()
    ).hexdigest()


def persist(path, value, *, append=False):
    """Flush each independent record so completed cases survive interruption."""
    with path.open("a" if append else "w", encoding="utf-8") as stream:
        stream.write(
            json.dumps(value, ensure_ascii=False, default=str, allow_nan=False) + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


async def bounded(awaitable, timeout, cleanup_timeout):
    """Bound waiting and cancellation draining; never claim server cancellation."""
    task = asyncio.ensure_future(awaitable)
    try:
        done, _ = await asyncio.wait({task}, timeout=max(0, timeout))
        if done:
            return task.result()
        raise TimeoutError
    finally:
        if not task.done():
            task.cancel()
            await asyncio.wait({task}, timeout=cleanup_timeout)
        # Retrieve late exceptions, including cancellation, without waiting forever.
        task.add_done_callback(
            lambda completed: None if completed.cancelled() else completed.exception()
        )


async def scoped_observation(
    manager, operation, failure, save, lifecycle, *, timeout, cleanup_timeout
):
    """One acquire/observe deadline, durable observation, then bounded teardown."""
    clock = asyncio.get_running_loop().time
    deadline = clock() + timeout
    entered = False
    sealed = False
    lifecycle({"phase": "started", "status": "started"})

    async def execute():
        nonlocal entered
        resource = await manager.__aenter__()
        if sealed:
            # An uncooperative acquire yielded after its caller moved on.
            await bounded(manager.__aexit__(None, None, None), cleanup_timeout, 0)
            raise TimeoutError
        entered = True
        remaining = deadline - clock()
        if remaining <= 0:
            raise TimeoutError
        return await operation(resource, remaining)

    cleanup_failed = False
    error = None
    try:
        try:
            result = await bounded(execute(), deadline - clock(), cleanup_timeout)
        except Exception as exc:
            error = exc
            result = failure(exc, "observe" if entered else "acquire")
        save(result)  # Fsync BEFORE rollback/session close can raise or hang.
    finally:
        sealed = True
        if entered:
            try:
                # Teardown shares the deadline; grace covers cancellation only.
                exc_info = (
                    (type(error), error, error.__traceback__)
                    if error
                    else (None, None, None)
                )
                await bounded(
                    manager.__aexit__(*exc_info),
                    min(deadline - clock(), cleanup_timeout),
                    cleanup_timeout,
                )
                lifecycle({"phase": "teardown", "status": "success"})
            except Exception as exc:
                cleanup_failed = True
                lifecycle(
                    {
                        "phase": "teardown",
                        "status": "timeout"
                        if isinstance(exc, TimeoutError)
                        else "error",
                        "error_type": type(exc).__name__,
                    }
                )
    return result, cleanup_failed


async def evaluate_cases(args, dataset, runtime):
    """Persist exactly one observation per gold/case; lifecycle has no counts."""
    from time import monotonic

    from eval.trusted_evaluation import score_ordinary

    groups = dict.fromkeys(args.groups)
    graphs = {
        group: build_graph(
            group,
            runtime.physical_tables,
            runtime.business_tables,
            runtime.business_metrics,
        )
        for group in groups
    }
    cleanup_timeout = getattr(args, "cleanup_timeout", DEFAULT_CLEANUP_TIMEOUT)
    incomplete = False

    def write(record):
        persist(args.output_dir / "records.jsonl", runtime.redact(record), append=True)

    for case in dataset["cases"]:
        gold = None
        identity = {"id": case["id"], "dataset_version": dataset["version"]}

        def identifiers(kind, group=None):
            return {
                **identity,
                "record_id": fingerprint([dataset["version"], kind, case["id"], group]),
                "subject": kind,
                "group": group,
            }

        async def run_one(kind, manager, operation, save, tracker, *, group=None):
            ids = identifiers(kind, group)

            def failure(error, phase):
                dw = tracker.get("dw")
                return {
                    "terminal" if kind == "case" else "status": "timeout"
                    if isinstance(error, TimeoutError)
                    else "error",
                    "reason": f"{kind}_budget_exhausted"
                    if isinstance(error, TimeoutError)
                    else f"{kind}_{phase}_failed",
                    "error_type": type(error).__name__,
                    "phase": phase,
                    "rows": None,
                    "projection": None,
                    "validate_calls": dw.validate_calls if dw else 0,
                    "run_calls": dw.run_calls if dw else 0,
                    "run_sql": dw.run_sql if dw else None,
                    "run_completed": dw.run_completed if dw else None,
                }

            return await scoped_observation(
                manager,
                operation,
                failure,
                lambda result: save(ids, result),
                lambda event: write({"type": "lifecycle", **ids, **event}),
                timeout=args.case_timeout,
                cleanup_timeout=cleanup_timeout,
            )

        if case["label"] == "answerable":
            tracker = {}

            async def execute_gold(repository, remaining):
                tracker["dw"] = RecordedDW(
                    repository, deadline=asyncio.get_running_loop().time() + remaining
                )
                return await read_gold(
                    case["gold_sql"],
                    runtime.physical_tables,
                    runtime.policy,
                    tracker["dw"],
                    timeout=remaining,
                )

            gold, cleanup_failed = await run_one(
                "gold",
                runtime.gold(),
                execute_gold,
                lambda ids, result: write({"type": "gold", **ids, **result}),
                tracker,
            )
            incomplete |= gold["status"] != "success" or cleanup_failed

        for group in groups:
            responses, tracker = [], {}
            start = monotonic()

            async def execute_case(context, remaining):
                tracker["dw"] = RecordedDW(
                    context["dw_mysql_repository"],
                    deadline=asyncio.get_running_loop().time() + remaining,
                )
                return await observe(
                    graphs[group],
                    case["query"],
                    {**context, "dw_mysql_repository": tracker["dw"]},
                    group=group,
                    timeout=remaining,
                    request_id=f"live-{case['id']}-{group}",
                )

            def save_case(ids, actual):
                # Behavioral labels are unchanged; comparison never replays SQL.
                behavior = score_ordinary(
                    {
                        "label": case["label"],
                        "gold": {
                            "rows": [],
                            "columns": {},
                            "ordered": True,
                            "abs_tolerance": 0,
                        },
                    },
                    actual,
                )["behavior_pass"]
                result = None
                if gold and gold["status"] == "success":
                    result = behavior and compare_projections(
                        actual["rows"],
                        actual["projection"],
                        gold["rows"],
                        gold["projection"],
                        case["gold"],
                    )
                write(
                    {
                        "type": "case",
                        **ids,
                        "label": case["label"],
                        "actual": actual,
                        "responses": responses,
                        "behavior_pass": behavior,
                        "sql_result_pass": result,
                        "gold_status": gold["status"] if gold else None,
                        "latency_seconds": monotonic() - start,
                    }
                )

            actual, cleanup_failed = await run_one(
                "case",
                runtime.context(responses),
                execute_case,
                save_case,
                tracker,
                group=group,
            )
            incomplete |= actual["terminal"] in {"error", "timeout"} or cleanup_failed
    return 2 if incomplete else 0


async def run_live(args, dataset, config, metadata, *, runtime_factory=None):
    """Persist failures even if resource initialization cannot reach any service."""
    from contextlib import AsyncExitStack

    from eval.live_runtime import live_runtime, model_metadata, safe_metadata

    if runtime_factory is None:
        runtime_factory = live_runtime

    cleanup_timeout = getattr(args, "cleanup_timeout", DEFAULT_CLEANUP_TIMEOUT)
    stack = AsyncExitStack()
    phase = "initialization"
    code = 2
    abandoned = False

    def record_failure(kind, error):
        timed_out = isinstance(error, TimeoutError)
        persist(
            args.output_dir / "records.jsonl",
            {
                "type": kind,
                "status": "timeout" if timed_out else "blocked",
                "reason": f"{kind}_budget_exhausted" if timed_out else f"{kind}_failed",
                "error_type": type(error).__name__,
            },
            append=True,
        )

    async def initialize():
        runtime = await stack.enter_async_context(runtime_factory(config, args))
        if abandoned:
            await bounded(stack.aclose(), cleanup_timeout, 0)
            raise TimeoutError
        return runtime

    try:
        metadata["model"] = model_metadata(config, args)
        persist(args.output_dir / "metadata.json", safe_metadata(metadata, config))
        runtime = await bounded(initialize(), args.init_timeout, cleanup_timeout)
        phase = "run"
        metadata.update(runtime.metadata)
        persist(args.output_dir / "metadata.json", safe_metadata(metadata, config))
        code = await evaluate_cases(args, dataset, runtime)
    except Exception as error:
        record_failure(phase, error)
    finally:
        abandoned = True
        try:
            await bounded(stack.aclose(), cleanup_timeout, cleanup_timeout)
        except Exception as error:
            record_failure("runtime_teardown", error)
            code = 2
    return code


def validate_dataset(dataset):
    from decimal import Decimal

    if (
        not isinstance(dataset.get("version"), str)
        or not dataset["version"]
        or not dataset.get("cases")
    ):
        raise ValueError("A versioned nonempty dataset is required")
    ids = set()
    for case in dataset["cases"]:
        if (
            any(
                not isinstance(case.get(key), str) or not case[key].strip()
                for key in ("id", "query", "audit")
            )
            or case["id"] in ids
            or case.get("label") not in {"answerable", "needs_info", "refuse"}
        ):
            raise ValueError("Each unique case needs a query, label and audit")
        ids.add(case["id"])
        if case["label"] != "answerable":
            continue
        gold = case.get("gold", {})
        columns = gold.get("columns")
        tolerance = Decimal(str(gold.get("abs_tolerance", "NaN")))
        if (
            not isinstance(case.get("gold_sql"), str)
            or not case["gold_sql"].strip()
            or not isinstance(columns, list)
            or not columns
            or any(not isinstance(c, str) or not c for c in columns)
            or len({c.casefold() for c in columns}) != len(columns)
            or type(gold.get("ordered")) is not bool
            or not tolerance.is_finite()
            or tolerance < 0
        ):
            raise ValueError(
                "Audited SQL and explicit unique projection contract are required"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    parser.add_argument("--case-timeout", type=float, default=90)
    parser.add_argument(
        "--cleanup-timeout", type=float, default=DEFAULT_CLEANUP_TIMEOUT
    )
    parser.add_argument("--model-timeout", type=float, default=30)
    parser.add_argument("--init-timeout", type=float, default=30)
    parser.add_argument("--model-retries", type=int, choices=range(4), default=0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args(argv)
    for name in ("case_timeout", "cleanup_timeout", "model_timeout", "init_timeout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    # Never overwrite previous evidence, including interrupted runs.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.output_dir / "metadata.json"
    if meta_path.exists() or (args.output_dir / "records.jsonl").exists():
        parser.error("output directory already contains evaluation evidence")
    metadata = {
        "mode": "live",
        "started_at": datetime.now(UTC).isoformat(),
        "groups": args.groups,
        "dataset_version": None,
        "budgets": {
            key: getattr(args, key)
            for key in (
                "case_timeout",
                "cleanup_timeout",
                "model_timeout",
                "init_timeout",
                "model_retries",
                "max_tokens",
            )
        },
    }
    persist(meta_path, metadata)
    try:
        dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
        metadata.update(
            dataset_version=dataset["version"], dataset_hash=fingerprint(dataset)
        )
        persist(meta_path, metadata)
        validate_dataset(dataset)
        # Read user connection/model values only in memory, never output the config.
        from omegaconf import OmegaConf

        config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    except Exception as error:
        persist(
            args.output_dir / "records.jsonl",
            {
                "type": "initialization",
                "status": "blocked",
                "reason": "initialization_failed",
                "error_type": type(error).__name__,
            },
            append=True,
        )
        return 2
    return asyncio.run(run_live(args, dataset, config, metadata))


if __name__ == "__main__":
    raise SystemExit(main())
