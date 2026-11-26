"""Server-owned checks for generated MySQL queries."""

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from sqlglot import exp, parse
from sqlglot.errors import (
    ErrorLevel,
    OptimizeError,
    ParseError,
    TokenError,
    UnsupportedError,
)
from sqlglot.optimizer.scope import Scope, traverse_scope

from app.agent.state import TableInfoState


class SQLPolicyError(ValueError):
    """The candidate violates the SQL policy and must not be repaired."""


class SQLSyntaxError(ValueError):
    """The parser could not read the candidate query."""


@dataclass(frozen=True)
class SQLExecutionPolicy:
    database: str
    max_rows: int
    row_filters: Mapping[str, Mapping[str, Sequence[str | int]]] = field(
        default_factory=dict
    )
    row_scope_column_types: Mapping[str, Mapping[str, str]] = field(default_factory=dict)


# Only constructs whose read-only behavior and name resolution are supported here.
_SAFE_NODES = frozenset({
    exp.Select, exp.Union, exp.With, exp.CTE, exp.Subquery,
    exp.From, exp.Join, exp.Table, exp.TableAlias, exp.Column, exp.Identifier,
    exp.Alias, exp.Literal, exp.HexString, exp.Null, exp.Boolean, exp.Star, exp.Distinct,
    exp.Where, exp.Group, exp.Having, exp.Order, exp.Ordered, exp.Limit, exp.Offset,
    exp.Paren, exp.Tuple, exp.In, exp.Exists, exp.Between, exp.Like, exp.Is,
    exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.And, exp.Or, exp.Not,
    exp.Add, exp.Sub, exp.Mul, exp.Div, exp.Mod, exp.Neg,
    exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max, exp.Round, exp.Abs,
    exp.Coalesce, exp.Nullif, exp.Case, exp.If, exp.Cast, exp.DataType,
    exp.CurrentDate, exp.Extract, exp.DateAdd, exp.DateSub, exp.DateDiff,
    exp.Year, exp.Month, exp.Day, exp.Interval, exp.Var, exp.TimeToStr,
    exp.Concat, exp.Substring, exp.Lower, exp.Upper, exp.Length, exp.Trim,
})
_SAFE_ARGS = {
    exp.Select: {"distinct", "expressions", "from_", "joins", "where", "group",
                 "having", "order", "limit", "offset", "with_"},
    exp.Union: {"this", "expression", "distinct", "order", "limit", "offset", "with_"},
    exp.Table: {"this", "db", "alias"},
    exp.Column: {"this", "table"},
    exp.TableAlias: {"this"},
    exp.With: {"expressions"},
    exp.CTE: {"this", "alias"},
    exp.Subquery: {"this", "alias"},
    exp.Join: {"this", "on", "side", "kind"},
    exp.Limit: {"expression"},
    exp.Offset: {"expression"},
}
_IDENTIFIER = re.compile(r"[\w$]+\Z")
_INTEGER = re.compile(r"[0-9]{1,19}\Z")
_HEX_BYTES = re.compile(r"(?:[0-9a-fA-F]{2})*\Z")
_BINARY_TYPE = exp.DataType(this=exp.DType.BINARY)
_DATE_UNITS = frozenset({"YEAR", "QUARTER", "MONTH", "WEEK", "DAY", "HOUR", "MINUTE", "SECOND"})


def enforce_sql_policy(
    sql: str,
    table_infos: list[TableInfoState],
    policy: SQLExecutionPolicy,
    required_filters: Sequence[Mapping[str, object]] = (),
) -> str:
    """Return SQL permitted to reach EXPLAIN and execution.

    Request filters are server-retained RequiredFilter values, not prompt text.
    Their supported SQL shape is a single-table SELECT with equality predicates
    joined only by AND. More complex request filters fail closed.
    """
    try:
        statements = [statement for statement in parse(sql, read="mysql") if statement]
    except (ParseError, TokenError) as error:
        raise SQLSyntaxError("SQL could not be parsed") from error

    if len(statements) != 1 or not isinstance(statements[0], (exp.Select, exp.Union)):
        raise SQLPolicyError("only one read-only query is allowed")

    query = statements[0]
    for node in query.walk():
        _check_node(node)

    if required_filters:
        _check_required_filters(query, table_infos, required_filters, policy)

    allowed_tables = {
        table["name"]: frozenset(col["name"].lower() for col in table["columns"])
        for table in table_infos
    }
    try:
        _check_scopes(query, allowed_tables, policy)
        requested = _limit_value(query)
        effective = policy.max_rows if requested is None else min(requested, policy.max_rows)
        if effective < 1:
            raise SQLPolicyError("result limit must be positive")
        query.set("limit", exp.Limit(expression=exp.Literal.number(effective)))
        return query.sql(dialect="mysql", comments=False, unsupported_level=ErrorLevel.RAISE)
    except (OptimizeError, UnsupportedError) as error:
        raise SQLPolicyError("unsupported query structure") from error


def _check_required_filters(
    query: exp.Select | exp.Union,
    table_infos: list[TableInfoState],
    required_filters: Sequence[Mapping[str, object]],
    policy: SQLExecutionPolicy,
) -> None:
    """Prove retained request filters in a deliberately small SQL shape.

    The column ID is resolved only through retained metadata. The deployment's
    row_filters remain a separate policy and are checked independently later.
    """
    if not isinstance(query, exp.Select):
        raise SQLPolicyError("required filters need a single SELECT")
    if any(
        isinstance(node, (exp.Join, exp.Subquery, exp.CTE, exp.With, exp.Union))
        for node in query.walk()
    ):
        raise SQLPolicyError("required filters do not support joins or subqueries")

    retained: dict[str, tuple[str, str, object]] = {}
    for item in required_filters:
        if not isinstance(item, Mapping) or set(item) != {"column_id", "operator", "value"}:
            raise SQLPolicyError("malformed required filter")
        column_id = item["column_id"]
        operator = item["operator"]
        value = item["value"]
        if not isinstance(column_id, str) or operator != "eq":
            raise SQLPolicyError("unsupported required filter")
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise SQLPolicyError("unsupported required filter value")
        if isinstance(value, float) and not math.isfinite(value):
            raise SQLPolicyError("unsupported required filter value")
        matches = [
            (table["name"], column["name"])
            for table in table_infos
            for column in table["columns"]
            if column.get("id") == column_id
        ]
        if len(matches) != 1 or column_id in retained:
            raise SQLPolicyError("required filter column is not authoritative")
        table_name, column_name = matches[0]
        retained[column_id] = (table_name, column_name, value)

    tables = list(query.find_all(exp.Table))
    if len(tables) != 1 or not retained:
        raise SQLPolicyError("required filters need one retained base table")
    source = tables[0]
    required_tables = {item[0] for item in retained.values()}
    if len(required_tables) != 1 or source.name not in required_tables:
        raise SQLPolicyError("required filter table is not authorized")
    if source.alias:
        raise SQLPolicyError("required filters do not support table aliases")

    where = query.args.get("where")
    if not where:
        raise SQLPolicyError("required filter is missing")
    predicates = _and_predicates(where.this)
    if not predicates or any(not isinstance(predicate, exp.EQ) for predicate in predicates):
        raise SQLPolicyError("required filters need equality predicates joined by AND")

    found: set[str] = set()
    for predicate in predicates:
        left, right = predicate.this, predicate.expression
        if isinstance(right, exp.Column) and not isinstance(left, exp.Column):
            left, right = right, left
        if (
            _is_exact_binary_cast(left)
            and isinstance(left.this, exp.Column)
            and isinstance(right, exp.HexString)
            and not right.args.get("is_integer")
        ):
            column = left.this
            column_matches = [
                (column_id, value)
                for column_id, (table_name, column_name, value) in retained.items()
                if table_name == source.name
                and column_name.casefold() == column.name.casefold()
                and isinstance(value, str)
                and right.this.upper() == _utf8_hex(value)
            ]
            if column.table and column.table != source.name:
                raise SQLPolicyError("required filter uses an unauthorized table alias")
            if not column_matches:
                if _is_authorized_row_scope_predicate(left, right, source.name, policy):
                    continue
                raise SQLPolicyError("required filter is missing or changed")
            if len(column_matches) > 1:
                raise SQLPolicyError("required filter is ambiguous")
            found.add(column_matches[0][0])
            continue
        if not isinstance(left, exp.Column) or not isinstance(right, exp.Literal):
            raise SQLPolicyError("required filters need column-to-literal equality")
        if left.table and left.table != source.name:
            raise SQLPolicyError("required filter uses an unauthorized table alias")
        column_matches = [
            (column_id, value)
            for column_id, (table_name, column_name, value) in retained.items()
            if table_name == source.name and column_name.casefold() == left.name.casefold()
        ]
        matches = [
            column_id
            for column_id, value in column_matches
            if _literal_matches(right, value)
        ]
        if column_matches and not matches:
            raise SQLPolicyError("required filter is missing or changed")
        if len(matches) > 1:
            raise SQLPolicyError("required filter is ambiguous")
        if matches:
            found.add(matches[0])
    if found != set(retained):
        raise SQLPolicyError("required filter is missing or changed")


def _and_predicates(condition: exp.Expr) -> list[exp.Expr]:
    if isinstance(condition, exp.And):
        return _and_predicates(condition.this) + _and_predicates(condition.expression)
    return [condition]


def _literal_matches(literal: exp.Literal, expected: str | int | float) -> bool:
    if isinstance(expected, str):
        return literal.is_string and literal.this == expected
    if not literal.is_number:
        return False
    try:
        return Decimal(literal.this) == Decimal(str(expected))
    except (InvalidOperation, ValueError):
        return False


def _is_exact_binary_cast(node: exp.Expr) -> bool:
    return (
        isinstance(node, exp.Cast)
        and node.to == _BINARY_TYPE
        and not any(node.args.get(key) for key in ("format", "safe", "action", "default"))
    )


def _is_authorized_row_scope_predicate(
    left: exp.Expr,
    right: exp.Expr,
    table_name: str,
    policy: SQLExecutionPolicy,
) -> bool:
    if not _is_exact_binary_cast(left) or not isinstance(right, exp.HexString):
        return False
    column = left.this
    if not isinstance(column, exp.Column) or (column.table and column.table != table_name):
        return False
    requirements = policy.row_filters.get(table_name)
    if not requirements:
        return False
    required_column = next(
        (name for name in requirements if name.casefold() == column.name.casefold()),
        None,
    )
    if required_column is None:
        return False
    if policy.row_scope_column_types.get(table_name, {}).get(required_column) != "varchar_utf8mb4":
        return False
    allowed_values = requirements[required_column]
    if not allowed_values or isinstance(allowed_values, str) or any(
        not isinstance(value, str) for value in allowed_values
    ):
        return False
    return right.this.upper() in {_utf8_hex(value) for value in allowed_values}


def _check_node(node: exp.Expr) -> None:
    kind = type(node)
    if kind not in _SAFE_NODES:
        raise SQLPolicyError("unsupported SQL construct or function")
    if kind in _SAFE_ARGS and any(
        key not in _SAFE_ARGS[kind] and value is not None and value is not False and value != []
        for key, value in node.args.items()
    ):
        raise SQLPolicyError("unsupported SQL clause")
    if isinstance(node, exp.Identifier) and not _IDENTIFIER.fullmatch(node.name):
        raise SQLPolicyError("unsupported identifier")
    if isinstance(node, exp.Literal) and node.is_string and any(
        char in node.this for char in "'\\\0"
    ):
        raise SQLPolicyError("unsupported string literal")
    if isinstance(node, exp.HexString) and (
        node.args.get("is_integer") or not _HEX_BYTES.fullmatch(node.this)
    ):
        raise SQLPolicyError("unsupported binary literal")
    if isinstance(node, exp.Var) and node.name.upper() not in _DATE_UNITS:
        raise SQLPolicyError("unsupported SQL keyword")
    if isinstance(node, exp.Star) and not isinstance(node.parent, exp.Count):
        raise SQLPolicyError("wildcard projections are not authorized")
    if isinstance(node, exp.Join) and (
        node.side not in {"", "LEFT", "RIGHT"}
        or node.kind not in {"", "INNER", "OUTER", "CROSS"}
    ):
        raise SQLPolicyError("unsupported join")


def _check_scopes(
    query: exp.Select | exp.Union,
    allowed_tables: Mapping[str, frozenset[str]],
    policy: SQLExecutionPolicy,
) -> None:
    checked_tables = set()
    checked_queries = set()
    for scope in traverse_scope(query):
        select = scope.expression
        checked_queries.add(id(select))
        columns = {}
        base_sources = []
        for alias, (reference, source) in scope.selected_sources.items():
            if not alias:
                raise SQLPolicyError("derived tables require an alias")
            if isinstance(reference, exp.Table):
                checked_tables.add(id(reference))
            if isinstance(source, Scope):
                if isinstance(reference, exp.Table) and reference.db:
                    raise SQLPolicyError("a CTE reference cannot also name a database")
                columns[alias] = frozenset(_output_columns(source.expression))
            elif isinstance(source, exp.Table):
                if source.db and source.db != policy.database:
                    raise SQLPolicyError("database is not authorized")
                if source.name not in allowed_tables:
                    raise SQLPolicyError("table is not authorized")
                columns[alias] = allowed_tables[source.name]
                base_sources.append((alias, source))
            else:
                raise SQLPolicyError("unsupported query source")

        if len({alias.lower() for alias in columns}) != len(columns):
            raise SQLPolicyError("ambiguous source alias")

        for node in scope.walk():
            if isinstance(node, exp.Column):
                _check_column(node, select, columns)
        for alias, source in base_sources:
            _enforce_row_scope(select, alias, source, policy)

    if checked_tables != {id(table) for table in query.find_all(exp.Table)} or (
        checked_queries != {id(select) for select in query.find_all(exp.Select, exp.Union)}
    ):
        raise SQLPolicyError("unresolved query source")


def _output_columns(query: exp.Expr) -> list[str]:
    return [
        item.output_name.lower() if isinstance(item, (exp.Alias, exp.Column)) else ""
        for item in query.selects
    ]


def _check_column(column: exp.Column, query: exp.Expr, columns: dict) -> None:
    name = column.name.lower()
    if column.table:
        if name not in columns.get(column.table, ()):
            raise SQLPolicyError("qualified column is not authorized")
        return

    # MySQL resolves a bare ORDER BY item against the output list first.
    if (
        isinstance(column.parent, exp.Ordered)
        and isinstance(column.parent.parent, exp.Order)
        and column.parent.parent.parent is query
        and _output_columns(query).count(name) == 1
    ):
        return

    # Metadata contains an authorized subset, so it cannot disambiguate bare
    # references across multiple sources. Qualify those references explicitly.
    if len(columns) != 1 or name not in next(iter(columns.values())):
        raise SQLPolicyError("unqualified column is not authorized or is ambiguous")


def _limit_value(query: exp.Expr) -> int | None:
    limit = query.args.get("limit")
    if not limit or not isinstance(limit.expression, exp.Literal) or not limit.expression.is_number:
        if limit:
            raise SQLPolicyError("limit must be a positive integer")
        return None
    if not _INTEGER.fullmatch(limit.expression.this):
        raise SQLPolicyError("limit must be a positive integer")
    return int(limit.expression.this)


def _enforce_row_scope(
    select: exp.Expr,
    alias: str,
    source: exp.Table,
    policy: SQLExecutionPolicy,
) -> None:
    table = source.name
    requirements = policy.row_filters.get(table)
    if requirements is None:
        # MySQL can resolve differently cased table names to the same base table.
        if any(name.casefold() == table.casefold() for name in policy.row_filters):
            raise SQLPolicyError("protected table must use its configured name")
        return
    if not requirements:
        raise SQLPolicyError("required row scope is not configured")
    where = select.args.get("where")
    if not where:
        raise SQLPolicyError("required row scope is missing")
    if not _IDENTIFIER.fullmatch(policy.database):
        raise SQLPolicyError("invalid database for row scope")
    source.set("db", exp.to_identifier(policy.database, quoted=True))
    for required_column, allowed_values in requirements.items():
        if policy.row_scope_column_types.get(table, {}).get(required_column) != "varchar_utf8mb4":
            raise SQLPolicyError("required row scope has unverified comparison semantics")
        if not allowed_values or isinstance(allowed_values, str) or any(
            not isinstance(value, str) for value in allowed_values
        ):
            raise SQLPolicyError("required row scope needs authorized text values")
        allowed_bytes = {_utf8_hex(value) for value in allowed_values}
        condition = where.this.unnest()
        conjuncts = condition.flatten() if isinstance(condition, exp.And) else [condition]
        for predicate in conjuncts:
            if isinstance(predicate, exp.EQ):
                operand, value = predicate.this, predicate.expression
                if isinstance(operand, exp.Literal):
                    operand, value = value, operand
                values = [value]
            elif isinstance(predicate, exp.In) and not any(
                predicate.args.get(key) for key in ("query", "unnest", "field", "is_global")
            ):
                operand, values = predicate.this, predicate.expressions
            else:
                continue
            # The next policy pass must prove the emitted form from scratch too.
            binary = _is_exact_binary_cast(operand)
            column = operand.this if binary else operand
            if not isinstance(column, exp.Column) or column.name.lower() != required_column.lower():
                continue
            if column.table and column.table != alias:
                continue
            if binary and all(isinstance(value, exp.HexString) for value in values):
                encoded = [value.this.upper() for value in values]
            elif not binary and all(isinstance(value, exp.Literal) and value.is_string for value in values):
                encoded = [_utf8_hex(value.this) for value in values]
            else:
                continue
            if encoded and allowed_bytes.issuperset(encoded):
                # A lengthless cast and binary literals compare bytes without padding
                # or a numeric operand. Build this AST rather than trusting input casts.
                left = exp.Cast(this=column.copy(), to=_BINARY_TYPE.copy())
                literals = [exp.HexString(this=value) for value in encoded]
                predicate.replace(
                    exp.EQ(this=left, expression=literals[0]) if isinstance(predicate, exp.EQ)
                    else exp.In(this=left, expressions=literals)
                )
                break
        else:
            raise SQLPolicyError("required row scope is missing or broadened")


def _utf8_hex(value: str) -> str:
    try:
        return value.encode("utf-8").hex().upper()
    except UnicodeEncodeError as error:
        raise SQLPolicyError("required row scope needs valid UTF-8 text") from error
