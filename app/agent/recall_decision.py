"""Explainable request evidence rules for the recall decision boundary."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Literal, NotRequired, TypedDict

from app.agent.state import RecallOutcome, RecallSource
from app.entities.column_info import ColumnInfo
from app.entities.metric_info import MetricInfo
from app.entities.value_info import ValueInfo

EvidenceKind = Literal[
    "metric",
    "analysis_field",
    "filter_field",
    "filter_value",
    "time_field",
    "join_path",
]


class RequiredSources(TypedDict):
    """Evidence requested by one query, independent of branch health."""

    required_sources: list[EvidenceKind]
    low_confidence_candidates: list[str]
    rules_applied: list[str]
    inference_uncertain: bool


_METRIC_MARKERS = (
    "销售额",
    "销售总额",
    "金额",
    "收入",
    "利润",
    "成本",
    "订单量",
    "数量",
    "总额",
    "总数",
    "平均",
    "占比",
    "GMV",
    "AOV",
    "ROI",
)
_FILTER_FIELD_MARKERS = (
    "地区",
    "区域",
    "省",
    "城市",
    "门店",
    "品牌",
    "商品",
    "渠道",
    "客户",
    "类别",
    "类型",
    "状态",
)
_GROUP_MARKERS = ("按", "按照", "分组", "各", "每个", "维度", "排行", "排名", "趋势")
_TIME_MARKERS = (
    "今天",
    "昨日",
    "昨天",
    "本周",
    "上周",
    "本月",
    "上月",
    "本季度",
    "上季度",
    "今年",
    "去年",
    "最近",
    "过去",
    "日期",
    "时间",
    "月份",
    "季度",
    "同比",
    "环比",
)
_JOIN_MARKERS = ("关联", "连接", "跨表", "多表", "join", "JOIN")
_VALUE_MARKERS = (
    "华北",
    "华东",
    "华南",
    "华中",
    "东北",
    "西北",
    "西南",
    "北京",
    "上海",
    "广州",
    "深圳",
)
_STOP_WORDS = {"统计", "查询", "查看", "帮我", "一下", "这个", "的", "请"}
_TOKEN_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9_]+")
_QUOTED_RE = re.compile(r"[‘’'\"“”]([^‘’'\"“”]+)[‘’'\"“”]")
_EXPLICIT_VALUE_RE = re.compile(
    r"(?:为|是|等于|包含|在|来自|位于)\s*([\u4e00-\u9fffA-Za-z0-9_]+)"
)


def infer_required_sources(query: str) -> RequiredSources:
    """Use lexical demand rules; unexplained modifiers become required filters."""
    text = query.strip()
    has_metric = any(marker.casefold() in text.casefold() for marker in _METRIC_MARKERS)
    has_analysis = any(marker in text for marker in _GROUP_MARKERS)
    field_mentioned = any(marker in text for marker in _FILTER_FIELD_MARKERS)
    has_time = any(marker in text for marker in _TIME_MARKERS) or bool(
        re.search(r"\d{4}[-年/]\d{1,2}|\d+[年月日天周]", text)
    )
    residue = _residual(
        text,
        [
            *_METRIC_MARKERS,
            *_FILTER_FIELD_MARKERS,
            *_GROUP_MARKERS,
            *_TIME_MARKERS,
            *_JOIN_MARKERS,
            *_VALUE_MARKERS,
        ],
    )
    has_value = bool(
        any(marker in text for marker in _VALUE_MARKERS)
        or _QUOTED_RE.search(text)
        or _EXPLICIT_VALUE_RE.search(text)
        or (has_metric and residue)
        or (field_mentioned and not has_analysis and not has_time)
    )
    required: list[EvidenceKind] = ["metric"]
    rules = ["metric_marker" if has_metric else "metric_unknown"]
    if has_analysis:
        required.append("analysis_field")
        rules.append("analysis_field_marker")
    if has_value:
        required.extend(["filter_field", "filter_value"])
        rules.extend(["filter_field_marker", "filter_value_marker"])
    if has_time:
        required.append("time_field")
        rules.append("time_field_marker")
    if any(marker.casefold() in text.casefold() for marker in _JOIN_MARKERS):
        required.append("join_path")
        rules.append("join_path_marker")
    if residue:
        rules.append("unexplained_modifier")
    candidates = [v for v in _VALUE_MARKERS if v in text]
    candidates.extend(m.group(1).strip() for m in _QUOTED_RE.finditer(text))
    if residue:
        candidates.append(residue)
    return RequiredSources(
        required_sources=required,
        low_confidence_candidates=list(dict.fromkeys(candidates)),
        rules_applied=rules,
        inference_uncertain=not has_metric,
    )


class RecallEvidence(TypedDict):
    kind: EvidenceKind
    identifier: str
    label: str
    provenance: Literal["metadata", "value_index"]
    matched_term: str
    column_ids: list[str]
    literal: NotRequired[str | int | float]


class RequiredFilter(TypedDict):
    """Internal handoff; keep the authoritative literal, not its search form."""

    column_id: str
    operator: Literal["eq"]
    value: str | int | float


class RecallDecision(TypedDict):
    request: dict[str, object]
    branch_statuses: dict[RecallSource, RecallOutcome]
    evidence: list[RecallEvidence]
    required_filters: list[RequiredFilter]
    low_confidence_candidates: list[str]
    missing: list[EvidenceKind]
    status: Literal["success", "degraded", "blocked"]
    reason_code: str
    reasons: list[str]
    user_message: str


def _normalize(text: str) -> str:
    # Unicode width, case and whitespace only; never fuzzy-match a database value.
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def _matching_term(query: str, terms: Sequence[str]) -> str | None:
    normalized = _normalize(query)
    return next(
        (
            term
            for term in sorted(terms, key=len, reverse=True)
            if term and _normalize(term) in normalized
        ),
        None,
    )


def _residual(query: str, terms: Sequence[str]) -> str:
    """Recognize only a small positive-query grammar; retain unsupported demands."""
    text = _normalize(query)
    grammar = (
        "按照",
        "统计",
        "查询",
        "查看",
        "帮我",
        "请问",
        "请",
        "的",
        "按",
        "分组",
        "各",
        "每个",
        "和",
        "以及",
        "与",
        "等于",
        "为",
    )
    for term in sorted(set([*terms, *grammar]), key=len, reverse=True):
        if term:
            text = text.replace(_normalize(term), "")
    return re.sub(r"[，,。？?、'\"‘’“”:：()（）]", "", text)


def _resolve_filter_field(
    query: str, metrics: Sequence[RecallEvidence], columns: Sequence[ColumnInfo]
) -> tuple[ColumnInfo, str, str] | None:
    """Resolve one named equality filter without consulting the value hits."""
    text = _normalize(query).rstrip("。?!")
    text = re.sub(r"^(?:请问|请|帮我|统计|查询|查看)*", "", text)
    matches: dict[tuple[str, str], tuple[ColumnInfo, str, str]] = {}
    for metric in metrics:
        metric_term = _normalize(metric["matched_term"])
        if not text.endswith(metric_term):
            continue
        modifier = text[: -len(metric_term)].removesuffix("的")
        for column in columns:
            if column.role != "dimension":
                continue
            for term in (column.name, *column.alias):
                name = _normalize(term)
                if not name:
                    continue
                field = re.escape(name)
                forms = (rf"{field}(?:为|等于)(.+)", rf"(.+){field}")
                for form in forms:
                    match = re.fullmatch(form, modifier)
                    if match:
                        value = match[1]
                        quoted = re.fullmatch(_QUOTED_RE, value)
                        if quoted:
                            value = quoted[1]
                        elif not re.fullmatch(r"[\w.\-]+", value):
                            continue
                        matches[column.id, value] = (column, term, value)
    return next(iter(matches.values())) if len(matches) == 1 else None


def collect_executable_evidence(
    query: str,
    *,
    required_sources: Sequence[EvidenceKind],
    columns: Sequence[ColumnInfo],
    metrics: Sequence[MetricInfo],
    values: Sequence[ValueInfo],
) -> list[RecallEvidence]:
    """Promote exact/normalized authoritative matches, never raw keywords.

    The inputs must be records from the metadata and controlled-value adapters.
    Similarity retrieval alone is not a semantic match. Existing entities have no
    authoritative join relationship or time-role mapping; those remain missing.
    """
    evidence: list[RecallEvidence] = []
    available_columns = {column.id for column in columns}
    for metric in metrics:
        terms = [metric.name, *metric.alias]
        # This is an explicit lexical rule, not an invented metric definition.
        terms += [
            term.replace("销售额", "销售总额")
            for term in metric.alias
            if "销售额" in term
        ]
        term = _matching_term(query, terms)
        if (
            "metric" in required_sources
            and term
            and metric.relevant_columns
            and available_columns.issuperset(metric.relevant_columns)
        ):
            evidence.append(
                RecallEvidence(
                    kind="metric",
                    identifier=metric.id,
                    label=metric.name,
                    provenance="metadata",
                    matched_term=term,
                    column_ids=list(metric.relevant_columns),
                )
            )
    if "analysis_field" in required_sources:
        for column in columns:
            term = _matching_term(
                query, [column.name, column.description, *column.alias]
            )
            if term and column.role == "dimension":
                evidence.append(
                    RecallEvidence(
                        kind="analysis_field",
                        identifier=column.id,
                        label=column.name,
                        provenance="metadata",
                        matched_term=term,
                        column_ids=[column.id],
                    )
                )
    if "filter_field" not in required_sources:
        return evidence
    binding = _resolve_filter_field(
        query, [item for item in evidence if item["kind"] == "metric"], columns
    )
    if binding is None:
        return evidence
    column, field_term, requested_value = binding
    evidence.append(
        RecallEvidence(
            kind="filter_field",
            identifier=column.id,
            label=column.name,
            provenance="metadata",
            matched_term=field_term,
            column_ids=[column.id],
        )
    )
    if "filter_value" not in required_sources:
        return evidence
    value_hits: list[RecallEvidence] = []
    for example in column.examples:
        if (
            type(example) in (str, int, float)
            and _normalize(str(example)) == requested_value
        ):
            value_hits.append(
                RecallEvidence(
                    kind="filter_value",
                    identifier=f"{column.id}:{example}",
                    label=str(example),
                    provenance="metadata",
                    matched_term=requested_value,
                    column_ids=[column.id],
                    literal=example,
                )
            )
    for value in values:
        if value.column_id == column.id and _normalize(value.value) == requested_value:
            value_hits.append(
                RecallEvidence(
                    kind="filter_value",
                    identifier=value.id,
                    label=value.value,
                    provenance="value_index",
                    matched_term=requested_value,
                    column_ids=[column.id],
                    literal=value.value,
                )
            )
    # Normalization can find multiple distinct stored literals. Do not choose one.
    if len({(type(hit["literal"]), hit["literal"]) for hit in value_hits}) == 1:
        evidence.extend(value_hits)
    return evidence


def build_recall_decision(
    *,
    request_id: str,
    query: str,
    requirements: RequiredSources,
    branch_outcomes: Mapping[RecallSource, RecallOutcome],
    columns: Sequence[ColumnInfo],
    metrics: Sequence[MetricInfo],
    values: Sequence[ValueInfo],
) -> RecallDecision:
    """Decide at the public recall seam, independently from branch health."""
    required = list(requirements["required_sources"])
    evidence = collect_executable_evidence(
        query,
        required_sources=required,
        columns=columns,
        metrics=metrics
        if branch_outcomes.get("metric", {}).get("status") == "success"
        else [],
        values=values
        if branch_outcomes.get("value", {}).get("status") == "success"
        else [],
    )
    required_column_ids = {cid for item in evidence for cid in item["column_ids"]}
    required_tables = {
        column.table_id for column in columns if column.id in required_column_ids
    }
    rules = list(requirements["rules_applied"])
    if len(required_tables) > 1 and "join_path" not in required:
        # Metadata exposes key roles, not relationships between key columns.
        required.append("join_path")
        rules.append("cross_table_evidence")
    found = {item["kind"] for item in evidence}
    unresolved = _residual(query, [item["matched_term"] for item in evidence])
    missing = [kind for kind in required if kind not in found]
    if unresolved and "filter_value" in required and "filter_value" not in missing:
        # A category hit does not cover every constraint in that category.
        missing.append("filter_value")
    branches = {
        source: dict(
            branch_outcomes.get(
                source,
                RecallOutcome(
                    status="insufficient",
                    reason_code="insufficient_evidence",
                    dependency=None,
                    keyword_fallback_reason=None,
                ),
            )
        )
        for source in ("column", "metric", "value")
    }
    for source, kinds in (
        ("metric", {"metric"}),
        ("column", {"analysis_field", "filter_field", "time_field", "join_path"}),
        ("value", {"filter_value"}),
    ):
        if set(missing) & kinds and branches[source]["status"] == "success":
            branches[source].update(
                status="insufficient", reason_code="insufficient_evidence"
            )
    reasons = [f"missing_{kind}" for kind in missing]
    if unresolved:
        reasons.append("unresolved_request_terms")
    if missing or unresolved or requirements["inference_uncertain"]:
        status = "blocked"
        reason = (
            "required_filter_unverified"
            if "filter_value" in missing
            else "required_evidence_unavailable"
        )
        if requirements["inference_uncertain"]:
            reason = "request_requirements_uncertain"
            reasons.append(reason)
        message = (
            "无法确认查询所需的证据，请补充指标、字段或过滤条件；本次未生成或执行SQL。"
        )
    elif any(
        o["status"] != "success" or o.get("keyword_fallback_reason")
        for o in branches.values()
    ):
        status, reason = "degraded", "optional_evidence_unavailable"
        reasons += [
            f"{source}_{o['status']}"
            for source, o in branches.items()
            if o["status"] != "success"
        ]
        message = "部分召回来源受限；必要证据已确认，按已确认的查询条件继续。"
    else:
        status, reason = "success", "required_evidence_complete"
        message = "查询所需证据完整。"
    value_evidence = next(
        (item for item in evidence if item["kind"] == "filter_value"), None
    )
    required_filters = (
        []
        if value_evidence is None
        else [
            RequiredFilter(
                column_id=value_evidence["column_ids"][0],
                operator="eq",
                value=value_evidence["literal"],
            )
        ]
    )
    return RecallDecision(
        request={
            "request_id": request_id,
            "query": query,
            **requirements,
            "required_sources": required,
            "rules_applied": rules,
            "unresolved_terms": [unresolved] if unresolved else [],
        },
        branch_statuses=branches,
        evidence=evidence,
        required_filters=required_filters,
        low_confidence_candidates=list(requirements["low_confidence_candidates"]),
        missing=missing,
        status=status,
        reason_code=reason,
        reasons=reasons,
        user_message=message,
    )
