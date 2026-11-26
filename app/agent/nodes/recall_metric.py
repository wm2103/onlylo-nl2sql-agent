"""召回指标信息。"""

from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.recall import run_recall_branch
from app.agent.state import DataAgentState


async def recall_metric(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    candidates, outcome = await run_recall_branch(
        state,
        runtime,
        "metric",
        "召回指标信息",
        vector_search=runtime.context["metric_qdrant_repository"].search,
    )
    return {"retrieved_metric_infos": candidates, "recall_outcomes": {"metric": outcome}}
