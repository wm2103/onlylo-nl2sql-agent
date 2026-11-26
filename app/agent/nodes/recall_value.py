"""召回字段取值。"""

from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.recall import run_recall_branch
from app.agent.state import DataAgentState


async def recall_value(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    candidates, outcome = await run_recall_branch(
        state,
        runtime,
        "value",
        "召回字段取值",
        text_search=runtime.context["value_es_repository"].search,
    )
    return {"retrieved_value_infos": candidates, "recall_outcomes": {"value": outcome}}
