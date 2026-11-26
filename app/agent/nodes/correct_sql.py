"""Bounded SQL correction node."""

import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext, get_runtime_llm
from app.agent.state import DataAgentState
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt


async def correct_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    """Ask the model for one syntax correction, never for a policy exception."""
    writer = runtime.stream_writer
    step = "校正SQL"
    writer({"type": "progress", "step": step, "status": "running"})
    attempt = state.get("sql_correction_attempts", 0) + 1

    try:
        prompt = PromptTemplate(
            template=load_prompt("correct_sql"),
            input_variables=[
                "table_infos", "metric_infos", "date_info", "db_info", "query", "sql", "error"
            ],
        )
        chain = prompt | get_runtime_llm(runtime.context) | StrOutputParser()
        result = await chain.ainvoke(
            {
                "table_infos": yaml.dump(state["table_infos"], allow_unicode=True, sort_keys=False),
                "metric_infos": yaml.dump(state["metric_infos"], allow_unicode=True, sort_keys=False),
                "date_info": yaml.dump(state["date_info"], allow_unicode=True, sort_keys=False),
                "db_info": yaml.dump(state["db_info"], allow_unicode=True, sort_keys=False),
                "query": state["query"],
                "sql": state["sql"],
                "error": state["error"],
            }
        )
        logger.info("校正后的SQL：%s", result)
        writer({"type": "progress", "step": step, "status": "success"})
        return {
            "sql": result,
            "error": state["error"],
            "sql_correction_attempts": attempt,
            "sql_validation_passed": False,
            "sql_final_outcome": "pending",
        }
    except Exception as error:
        logger.error("%s failed: %s", step, type(error).__name__)
        writer({"type": "progress", "step": step, "status": "error"})
        writer({"type": "terminal", "status": "blocked", "reason_code": "sql_correction_unavailable"})
        return {
            "error": None,
            "sql_correction_attempts": attempt,
            "sql_validation_passed": False,
            "sql_failure_class": "infrastructure",
            "sql_failure_reason": "sql_correction_unavailable",
            "sql_final_outcome": "correction_unavailable",
        }
