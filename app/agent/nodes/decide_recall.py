"""Evidence gate between isolated retrieval and SQL context construction."""

from collections.abc import Callable
from typing import cast

from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.recall_decision import (
    EvidenceKind,
    RecallDecision,
    build_recall_decision,
    infer_required_sources,
)
from app.agent.state import DataAgentState
from app.core.log import logger


def report_recall_decision(decision: RecallDecision, writer: Callable[[dict], None]):
    """Do not publish request words, evidence payloads or dependency exceptions."""
    writer(
        {
            "type": "recall_decision",
            "request_id": decision["request"]["request_id"],
            "required_sources": decision["request"]["required_sources"],
            "missing": decision["missing"],
            "status": decision["status"],
            "reason_code": decision["reason_code"],
            "user_message": decision["user_message"],
        }
    )


async def decide_recall(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    outcomes = state.get("recall_outcomes", {})
    columns = (
        list(state.get("retrieved_column_infos", []))
        if outcomes.get("column", {}).get("status") == "success"
        else []
    )
    metrics = (
        list(state.get("retrieved_metric_infos", []))
        if outcomes.get("metric", {}).get("status") == "success"
        else []
    )
    values = (
        list(state.get("retrieved_value_infos", []))
        if outcomes.get("value", {}).get("status") == "success"
        else []
    )
    by_id = {column.id: column for column in columns}
    metadata_missing = False
    # Only ID lookups in the authoritative metadata repository. No warehouse
    # reads, value probing or scan fallback is permitted at this boundary.
    ids = {cid for metric in metrics for cid in metric.relevant_columns}
    ids.update(value.column_id for value in values)
    try:
        for column_id in sorted(ids - by_id.keys()):
            column = await runtime.context[
                "meta_mysql_repository"
            ].get_column_info_by_id(column_id)
            if column is None or column.id != column_id:
                metadata_missing = True
            else:
                by_id[column_id] = column
    except Exception:
        metadata_missing = True
        logger.warning(
            "Recall metadata unavailable request_id={}", state.get("request_id")
        )
    decision = build_recall_decision(
        request_id=state["request_id"],
        query=state["query"],
        requirements=infer_required_sources(state["query"]),
        branch_outcomes=outcomes,
        columns=list(by_id.values()),
        metrics=metrics,
        values=values,
    )
    if metadata_missing:
        decision.update(status="blocked", reason_code="metadata_dependency_unavailable")
        decision["reasons"].append("metadata_dependency_unavailable")
        decision["user_message"] = (
            "查询所需的权威元数据暂不可确认，请稍后重试；本次未生成或执行SQL。"
        )
    if decision["status"] == "blocked":
        report_recall_decision(decision, runtime.stream_writer)
    # Keep only executable evidence in generation context; original words stay
    # in the decision's separate low-confidence candidates field.
    approved_ids = {cid for item in decision["evidence"] for cid in item["column_ids"]}
    metric_ids = {
        item["identifier"] for item in decision["evidence"] if item["kind"] == "metric"
    }
    value_hits = {
        (item["identifier"], item["column_ids"][0], item["literal"])
        for item in decision["evidence"]
        if item["kind"] == "filter_value" and item["provenance"] == "value_index"
    }
    return {
        "recall_decision": decision,
        "retrieved_column_infos": [
            column for cid, column in by_id.items() if cid in approved_ids
        ],
        "retrieved_metric_infos": [
            metric for metric in metrics if metric.id in metric_ids
        ],
        "retrieved_value_infos": [
            value
            for value in values
            if (value.id, value.column_id, value.value) in value_hits
        ],
    }


def check_pruned_evidence(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    """Require all approved evidence to survive the two model-led selectors."""
    decision = cast(RecallDecision, state["recall_decision"])
    columns = {
        column["id"]: column
        for table in state["table_infos"]
        for column in table["columns"]
        if "id" in column
    }
    metric_ids = {metric.get("id") for metric in state["metric_infos"]}
    missing: set[EvidenceKind] = {
        item["kind"]
        for item in decision["evidence"]
        if any(column_id not in columns for column_id in item["column_ids"])
        or (item["kind"] == "metric" and item["identifier"] not in metric_ids)
    }
    for required in decision["required_filters"]:
        column = columns.get(required["column_id"])
        literal = required["value"]
        if column is None or not any(
            type(example) is type(literal) and example == literal
            for example in column["examples"]
        ):
            missing.add("filter_value")
    if missing:
        decision = decision.copy()
        decision.update(
            status="blocked",
            reason_code="required_evidence_pruned",
            missing=sorted(missing),
            reasons=[*decision["reasons"], "required_evidence_pruned"],
            user_message="筛选后的查询上下文缺少必要证据；本次未生成或执行SQL。",
        )
    report_recall_decision(decision, runtime.stream_writer)
    return {"recall_decision": decision}
