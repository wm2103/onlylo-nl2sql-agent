"""Run recall adapters with bounded keyword expansion and isolated outcomes."""

import asyncio
from collections.abc import Awaitable, Callable
from uuid import uuid4

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext, get_runtime_llm
from app.agent.state import (
    DataAgentState,
    RecallDependency,
    RecallOutcome,
    RecallReason,
    RecallSource,
    RecallStatus,
)
from app.conf.app_config import app_config
from app.core.log import logger
from app.entities.column_info import ColumnInfo
from app.entities.metric_info import MetricInfo
from app.entities.value_info import ValueInfo
from app.prompt.prompt_loader import load_prompt


async def run_recall_branch[Candidate: (ColumnInfo, MetricInfo, ValueInfo)](
    state: DataAgentState,
    runtime: Runtime[DataAgentContext],
    source: RecallSource,
    step: str,
    *,
    vector_search: Callable[[list[float]], Awaitable[list[Candidate]]] | None = None,
    text_search: Callable[[str], Awaitable[list[Candidate]]] | None = None,
) -> tuple[list[Candidate], RecallOutcome]:
    loop = asyncio.get_running_loop()
    budget = runtime.context.get("recall_budget", app_config.recall)
    started = loop.time()
    request_deadline = state.get("recall_deadline", started + budget.total_seconds)
    branch_deadline = started + {
        "column": budget.column_seconds,
        "metric": budget.metric_seconds,
        "value": budget.value_seconds,
    }[source]
    deadline = min(branch_deadline, request_deadline)
    budget_reason: RecallReason = (
        "request_budget_exhausted"
        if request_deadline <= branch_deadline
        else "branch_budget_exhausted"
    )
    request_id = state.get("request_id") or str(uuid4())

    def write_progress(step: str, status: str, reason_code: str) -> None:
        runtime.stream_writer(
            {
                "type": "progress",
                "step": step,
                "status": status,
                "reason_code": reason_code,
                "request_id": request_id,
            }
        )

    write_progress(step, "running", "recall_started")
    query = state["query"]
    keywords = [query, *state.get("keywords", [])]
    fallback_reason = None
    failed_dependency: RecallDependency | None = None
    candidates: dict[str, Candidate] = {}
    status: RecallStatus
    reason: RecallReason
    if loop.time() >= deadline:
        status, reason = "timeout", budget_reason
    else:
        expansion_deadline = min(loop.time() + budget.expansion_seconds, deadline)
        try:
            async with asyncio.timeout_at(expansion_deadline):
                prompt = PromptTemplate.from_template(
                    load_prompt(f"extend_keywords_for_{source}_recall")
                )
                chain = prompt | get_runtime_llm(runtime.context) | JsonOutputParser()
                expanded = await chain.ainvoke({"query": query})
                if not isinstance(expanded, list) or any(
                    not isinstance(term, str) or not term.strip() for term in expanded
                ):
                    raise ValueError("Keyword expansion must return an array of strings")
                keywords.extend(expanded)
        except TimeoutError:
            fallback_reason = "keyword_expansion_timeout"
        except Exception:
            fallback_reason = "keyword_expansion_error"

        if fallback_reason:
            write_progress(f"{step}：关键词扩展", "degraded", fallback_reason)

        active_dependency: RecallDependency | None = None
        scope = asyncio.timeout_at(deadline)
        try:
            if loop.time() >= deadline:
                raise TimeoutError
            async with scope:
                for keyword in dict.fromkeys(keywords):
                    if vector_search is not None:
                        active_dependency = "embedding"
                        vector = await runtime.context["embedding_client"].aembed_query(keyword)
                        active_dependency = "qdrant"
                        hits = await vector_search(vector)
                    else:
                        active_dependency = "elasticsearch"
                        assert text_search is not None
                        hits = await text_search(keyword)
                    for candidate in hits:
                        candidates.setdefault(candidate.id, candidate)
            status, reason = (
                ("success", "candidates_found") if candidates else ("empty", "no_candidates")
            )
        except TimeoutError:
            failed_dependency = active_dependency
            status = "timeout"
            reason = (
                budget_reason if scope.expired() or loop.time() >= deadline else "retrieval_timeout"
            )
        except Exception:
            failed_dependency = active_dependency
            status, reason = "error", "retrieval_error"

    if status in ("error", "timeout"):
        candidates.clear()
    outcome = RecallOutcome(
        status=status,
        reason_code=reason,
        dependency=failed_dependency,
        keyword_fallback_reason=fallback_reason,
    )
    write_progress(step, status, reason)
    if status in ("error", "timeout") or fallback_reason:
        logger.warning(
            "Recall request_id={} source={} status={} reason={} dependency={} fallback={}",
            request_id, source, status, reason, failed_dependency, fallback_reason,
        )
    if source == "column":
        from app.core.context import request_id_ctx_var

        request_id_ctx_var.set(request_id)
    return list(candidates.values()), outcome
