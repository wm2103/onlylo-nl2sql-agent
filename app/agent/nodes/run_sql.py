"""SQL execution node; it accepts only a candidate passed by validation."""

from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext, get_sql_execution_policy
from app.agent.nodes.validate_sql import report_sql_policy_denial
from app.agent.sql_policy import SQLPolicyError, SQLSyntaxError, enforce_sql_policy
from app.agent.state import DataAgentState
from app.core.log import logger


async def run_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    """Execute exactly one SQL candidate after a successful validation pass."""
    writer = runtime.stream_writer
    step = "执行SQL"
    writer({"type": "progress", "step": step, "status": "running"})

    try:
        try:
            sql = enforce_sql_policy(
                state["sql"],
                state["table_infos"],
                get_sql_execution_policy(runtime.context),
                required_filters=state.get("recall_decision", {}).get("required_filters", ()),
            )
        except (SQLPolicyError, SQLSyntaxError):
            return report_sql_policy_denial(writer, step)

        if not state.get("sql_validation_passed", False):
            writer({"type": "progress", "step": step, "status": "error"})
            writer({"type": "terminal", "status": "blocked", "reason_code": "sql_not_validated"})
            return {
                "sql_validation_passed": False,
                "sql_final_outcome": "blocked",
                "sql_failure_class": "policy",
                "sql_failure_reason": "sql_not_validated",
            }

        result = await runtime.context["dw_mysql_repository"].run(sql)
        logger.info("SQL执行结果：%s", result)
        writer({"type": "progress", "step": step, "status": "success"})
        writer({"type": "result", "data": result})
        writer({"type": "terminal", "status": "success", "reason_code": "query_executed"})
        return {"sql": sql, "sql_result": result, "sql_final_outcome": "success"}
    except Exception as error:
        logger.error("%s failed: %s", step, type(error).__name__)
        writer({"type": "progress", "step": step, "status": "error"})
        writer({"type": "terminal", "status": "error", "reason_code": "sql_execution_failed"})
        return {
            "sql_validation_passed": True,
            "sql_final_outcome": "execution_failed",
            "sql_failure_class": "infrastructure",
            "sql_failure_reason": "sql_execution_failed",
        }
