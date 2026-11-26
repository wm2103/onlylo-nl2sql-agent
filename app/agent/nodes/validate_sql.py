"""Validate generated SQL at the policy and database adapter seam."""

from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext, get_sql_execution_policy
from app.agent.sql_policy import SQLPolicyError, SQLSyntaxError, enforce_sql_policy
from app.agent.state import DataAgentState
from app.core.log import logger


class DatabaseSyntaxError(Exception):
    """Adapter contract for an explicit database SQL syntax rejection."""


SQLDatabaseSyntaxError = DatabaseSyntaxError


_DATABASE_SYNTAX_NAMES = frozenset(
    {
        "databasesyntaxerror",
        "sqldatabasesyntaxerror",
        "mysqlsyntaxerror",
        "programmingerror",
    }
)


def is_database_syntax_error(error: Exception) -> bool:
    """Classify only explicitly identified database syntax failures as repairable."""
    if getattr(error, "sql_error_category", None) == "syntax":
        return True
    if getattr(error, "is_sql_syntax_error", False) is True:
        return True
    name = type(error).__name__.lower()
    if name in _DATABASE_SYNTAX_NAMES - {"programmingerror"}:
        return True
    if name == "programmingerror":
        original = getattr(error, "orig", None)
        code = getattr(original, "args", (None,))[0]
        return code in {1054, 1064, 1146, 1066, 1248}
    return False


def _terminal(writer, *, status: str, reason_code: str) -> None:
    writer({"type": "terminal", "status": status, "reason_code": reason_code})


def _syntax_failure(
    writer,
    state: DataAgentState,
    attempts: int,
    error: str,
    reason: str,
    limit: int,
):
    result = {
        "error": error,
        "sql_validation_attempts": attempts,
        "sql_validation_passed": False,
        "sql_failure_class": "syntax",
        "sql_failure_reason": reason,
        "sql_final_outcome": "pending",
        "sql_policy_denied": False,
    }
    if state.get("sql_correction_attempts", 0) >= limit:
        writer({"type": "progress", "step": "校验SQL", "status": "error"})
        _terminal(writer, status="blocked", reason_code="sql_correction_exhausted")
        result["sql_final_outcome"] = "correction_exhausted"
    return result


async def validate_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    """Run the complete server-owned validation for one SQL candidate."""
    writer = runtime.stream_writer
    step = "校验SQL"
    writer({"type": "progress", "step": step, "status": "running"})
    attempts = state.get("sql_validation_attempts", 0) + 1

    try:
        from app.agent.context import get_sql_correction_limit

        limit = get_sql_correction_limit(runtime.context)
        try:
            sql = enforce_sql_policy(
                state["sql"],
                state["table_infos"],
                get_sql_execution_policy(runtime.context),
                required_filters=state.get("recall_decision", {}).get("required_filters", ()),
            )
        except SQLPolicyError:
            writer({"type": "progress", "step": step, "status": "error"})
            _terminal(writer, status="blocked", reason_code="sql_policy_denied")
            return {
                "error": None,
                "sql_policy_denied": True,
                "sql_validation_attempts": attempts,
                "sql_validation_passed": False,
                "sql_failure_class": "policy",
                "sql_failure_reason": "sql_policy_denied",
                "sql_final_outcome": "blocked",
            }
        except SQLSyntaxError as error:
            return _syntax_failure(
                writer,
                state,
                attempts,
                str(error),
                "sql_parse_error",
                limit,
            )

        repository = runtime.context["dw_mysql_repository"]
        try:
            # The repository owns EXPLAIN; only the policy-approved SQL reaches it.
            await repository.validate(sql)
        except Exception as error:
            if is_database_syntax_error(error):
                return _syntax_failure(
                    writer,
                    state,
                    attempts,
                    "Database rejected SQL syntax",
                    "database_parse_error",
                    limit,
                )
            logger.error("SQL validation dependency failed: %s", type(error).__name__)
            writer({"type": "progress", "step": step, "status": "error"})
            _terminal(writer, status="error", reason_code="sql_validation_unavailable")
            return {
                "error": None,
                "sql_validation_attempts": attempts,
                "sql_validation_passed": False,
                "sql_failure_class": "infrastructure",
                "sql_failure_reason": "sql_validation_unavailable",
                "sql_final_outcome": "validation_unavailable",
            }

        writer({"type": "progress", "step": step, "status": "success"})
        logger.info("SQL语法正确")
        return {
            "error": None,
            "sql": sql,
            "sql_policy_denied": False,
            "sql_validation_attempts": attempts,
            "sql_validation_passed": True,
            "sql_failure_class": None,
            "sql_failure_reason": None,
            "sql_final_outcome": "pending",
        }
    except Exception as error:
        logger.error("%s failed: %s", step, type(error).__name__)
        writer({"type": "progress", "step": step, "status": "error"})
        _terminal(writer, status="error", reason_code="sql_validation_unavailable")
        return {
            "error": None,
            "sql_validation_attempts": attempts,
            "sql_validation_passed": False,
            "sql_failure_class": "infrastructure",
            "sql_failure_reason": "sql_validation_unavailable",
            "sql_final_outcome": "validation_unavailable",
        }


def report_sql_policy_denial(writer, step: str):
    writer({"type": "progress", "step": step, "status": "error"})
    _terminal(writer, status="blocked", reason_code="sql_policy_denied")
    return {"sql_policy_denied": True, "error": None}
