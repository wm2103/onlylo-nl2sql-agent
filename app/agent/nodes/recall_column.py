"""召回字段信息。"""

from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.recall import run_recall_branch
from app.agent.state import DataAgentState


async def recall_column(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    candidates, outcome = await run_recall_branch(
        state,
        runtime,
        "column",
        "召回字段信息",
        vector_search=runtime.context["column_qdrant_repository"].search,
    )
    return {"retrieved_column_infos": candidates, "recall_outcomes": {"column": outcome}}
