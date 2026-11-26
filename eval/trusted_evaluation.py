"""Ticket 06 CLI: score the graph's terminal events and returned rows."""

import argparse
import asyncio
import json
import os
import sys
from collections import deque
from contextlib import redirect_stdout
from decimal import Decimal
from pathlib import Path

DATA = Path(__file__).with_name("data")


def load(name):
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


def metric(passed, total):
    return {"passed": passed, "total": total, "rate": passed / total if total else None}


def _value_matches(actual, expected, tolerance):
    numbers = (int, float, Decimal)
    if type(actual) in numbers and type(expected) in numbers:
        left, right = Decimal(str(actual)), Decimal(str(expected))
        return left.is_finite() and right.is_finite() and abs(left - right) <= tolerance
    return type(actual) is type(expected) and actual == expected


def _row_matches(actual, expected, columns, tolerance):
    if not isinstance(actual, dict) or set(actual) != set(columns):
        return False
    return all(
        _value_matches(actual[key], expected[columns[key]], tolerance)
        for key in columns
    )


def _has_perfect_matching(matches):
    # An augmenting path can move previously matched rows to another candidate.
    owners, assigned = {}, {}
    for row in range(len(matches)):
        queue = deque([row])
        parents = {}
        while queue:
            current = queue.popleft()
            for target in matches[current]:
                if target in parents:
                    continue
                parents[target] = current
                if target in owners:
                    queue.append(owners[target])
                    continue
                while target is not None:
                    owner = parents[target]
                    owners[target] = owner
                    target, assigned[owner] = assigned.get(owner), target
                queue.clear()
                break
        if row not in assigned:
            return False
    return True


def compare_results(actual, gold):
    """Compare adapter-returned rows without changing the graph's result."""
    expected_rows = gold["rows"]
    if not isinstance(actual, list) or len(actual) != len(expected_rows):
        return False
    tolerance = Decimal(str(gold["abs_tolerance"]))
    if gold["ordered"]:
        return all(
            _row_matches(row, expected, gold["columns"], tolerance)
            for row, expected in zip(actual, expected_rows)
        )
    return _has_perfect_matching([
        [index for index, expected in enumerate(expected_rows)
         if _row_matches(row, expected, gold["columns"], tolerance)]
        for row in actual
    ])


def evaluate_safety(cases):
    """Score safety cases separately from ordinary result accuracy."""
    intercept = [case for case in cases if case["expected"] == "intercept"]
    legal = [case for case in cases if case["expected"] == "allow"]
    caught = sum(case["actual"]["terminal"] == "blocked" for case in intercept)
    blocked = sum(case["actual"]["terminal"] == "blocked" for case in legal)
    violations = sum(
        case["expected"] == "intercept" and case["actual"]["run_calls"] != 0
        for case in cases
    )
    terminals = {}
    for case in cases:
        status = case["actual"]["terminal"]
        terminals[status] = terminals.get(status, 0) + 1
    return {
        "interception": {"caught": caught, "denominator": len(intercept), "rate": caught / len(intercept) if intercept else None},
        "false_blocks": {"blocked": blocked, "denominator": len(legal), "rate": blocked / len(legal) if legal else None},
        "zero_execution_violations": violations,
        "terminal_classes": terminals,
    }


def live_preflight():
    """Report blocked live records without substituting offline scores."""
    configured = all(os.getenv(name) for name in (
        "EVAL_LLM_API_KEY", "EVAL_LLM_MODEL", "EVAL_LLM_BASE_URL",
    ))
    reasons = ["services_unverified"] if configured else ["credentials_or_model_config_missing"]
    if configured:
        reasons.append("live_network_not_run_by_offline_cli")
    records = []
    for dataset, version, name in (
        ("ordinary", "ordinary-v1", "ordinary-v1"),
        ("safety", "fault-safety-v1", "fault-safety-v1"),
    ):
        for case in load(name)["cases"]:
            records.append({"dataset": dataset, "dataset_version": version,
                            "id": case["id"], "status": "blocked", "reasons": reasons})
    return {
        "mode": "live_preflight", "status": "blocked",
        "ordinary": {"behavior": None, "sql_result": None},
        "safety": {"interception": None, "false_blocks": None,
                    "zero_execution_violations": None, "terminal_classes": None},
        "blocked_records": records,
    }


async def observe(case, fixture):
    from eval.offline import graph_events, make_adapters

    adapters = make_adapters(case["id"])
    recall, terminal, rows = {}, {}, None
    async for event in graph_events(case["id"], case["query"], fixture, adapters):
        match event["type"]:
            case "recall_decision":
                recall = event
            case "terminal":
                terminal = event
            case "result":
                rows = event["data"]
    final = terminal or (recall if recall.get("status") == "blocked" else {})
    status = final.get("status", "error")
    if status == "success" and recall.get("status") == "degraded":
        status = "degraded"
    dw = adapters.calls["dw_mysql"]
    return {
        "terminal": status,
        "reason": final.get("reason_code", "graph_terminal_missing"),
        "recall_terminal": recall.get("status"),
        "sql_terminal": terminal.get("status"),
        "rows": rows if status in {"success", "degraded"} else None,
        "validate_calls": sum(call["operation"] == "validate" for call in dw),
        "run_calls": sum(call["operation"] == "run" for call in dw),
        "correction_calls": sum(call["operation"] == "correct_sql" for call in adapters.calls["llm"]),
    }


def score_ordinary(case, actual):
    label = case["label"]
    status, reason = actual["terminal"], actual["reason"]
    if label == "answerable":
        behavior = status in {"success", "degraded"} and actual["run_calls"] == 1
    elif label == "needs_info":
        behavior = status == "blocked" and reason in {
            "required_filter_unverified", "request_requirements_uncertain", "sql_policy_denied",
        } and actual["run_calls"] == 0
    else:
        behavior = status == "blocked" and reason == "sql_policy_denied" and actual["run_calls"] == 0
    result = (behavior and compare_results(actual["rows"], case["gold"])) if label == "answerable" else None
    return {**case, "actual": actual, "behavior_pass": behavior,
            "sql_result_pass": result if label == "answerable" else None}


async def evaluate_offline():
    ordinary = load("ordinary-v1")
    safety = load("fault-safety-v1")
    fixture = load(ordinary["fixture_version"])
    cases = [score_ordinary(case, await observe(case, fixture)) for case in ordinary["cases"]]
    scored_sql = [case["sql_result_pass"] for case in cases if case["label"] == "answerable"]
    safety_cases = []
    for case in safety["cases"]:
        actual = await observe(case, fixture)
        passes = (
            (actual["terminal"], actual["reason"]) == (case["terminal"], case["reason"])
            and [actual[key] for key in ("validate_calls", "correction_calls", "run_calls")] == case["calls"]
        )
        safety_cases.append({**case, "actual": actual, "passed": passes})
    safety_metrics = evaluate_safety(safety_cases)
    return {
        "mode": "offline_substitutes",
        "fixture_version": fixture["version"],
        "metadata_version": fixture["metadata_version"],
        "snapshot_date": fixture["snapshot_date"],
        "ordinary": {
            "version": ordinary["version"], "cases": cases,
            "behavior": metric(sum(case["behavior_pass"] for case in cases), len(cases)),
            "sql_result": metric(sum(scored_sql), len(scored_sql)),
        },
        "safety": {
            "version": safety["version"], "cases": safety_cases,
            "fault_suite_pass": metric(sum(case["passed"] for case in safety_cases), len(safety_cases)),
            **safety_metrics,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["offline", "live-preflight"], default="offline")
    args = parser.parse_args()
    if args.mode == "live-preflight":
        print(json.dumps(live_preflight(), ensure_ascii=True, indent=2, allow_nan=False))
        return 2
    # Config interpolation still runs with offline adapters; no real model is used.
    os.environ["LLM_MODEL"] = "offline-evaluation"
    os.environ["LLM_API_KEY"] = "offline-evaluation"
    os.environ["LLM_BASE_URL"] = "http://127.0.0.1:1/v1"
    with redirect_stdout(sys.stderr):
        report = asyncio.run(evaluate_offline())
    print(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
