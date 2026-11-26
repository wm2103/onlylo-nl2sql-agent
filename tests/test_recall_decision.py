import unittest

from app.agent.recall_decision import (
    build_recall_decision,
    collect_executable_evidence,
    infer_required_sources,
)
from app.agent.state import RecallOutcome
from app.entities.column_info import ColumnInfo
from app.entities.metric_info import MetricInfo
from app.entities.value_info import ValueInfo

QUERY = "统计华北地区的销售总额"
MEASURE = ColumnInfo(
    "fact_order.order_amount",
    "order_amount",
    "float",
    "measure",
    [],
    "订单金额",
    [],
    "fact_order",
)
REGION_ID = "fact_order.region_name"
COLUMNS = [
    MEASURE,
    ColumnInfo(
        REGION_ID,
        "region_name",
        "varchar",
        "dimension",
        [],
        "地区名称",
        ["地区"],
        "fact_order",
    ),
]
METRICS = [
    MetricInfo(
        "total_sales", "GMV", "订单金额合计", ["fact_order.order_amount"], ["销售额"]
    )
]


def outcome(status: str, reason: str = "candidates_found") -> RecallOutcome:
    return {
        "status": status,
        "reason_code": reason,
        "dependency": None,
        "keyword_fallback_reason": None,
    }


class RequiredSourcesTests(unittest.TestCase):
    def test_filter_words_are_required_but_not_executable_evidence(self):
        requirements = infer_required_sources(QUERY)

        self.assertEqual(
            requirements["required_sources"],
            ["metric", "filter_field", "filter_value"],
        )
        self.assertIn("华北", requirements["low_confidence_candidates"])
        self.assertNotIn("华北", requirements.get("executable_evidence", []))
        self.assertIn("metric_marker", requirements["rules_applied"])
        self.assertIn("filter_value_marker", requirements["rules_applied"])

    def test_unknown_request_is_conservative(self):
        requirements = infer_required_sources("帮我看看这个")

        self.assertEqual(requirements["required_sources"], ["metric"])
        self.assertTrue(requirements["inference_uncertain"])

    def test_value_index_match_is_evidence_but_unmatched_original_word_is_not(self):
        evidence = collect_executable_evidence(
            QUERY,
            required_sources=["metric", "filter_field", "filter_value"],
            columns=COLUMNS,
            metrics=METRICS,
            values=[ValueInfo("region_north", "华北", REGION_ID)],
        )

        self.assertEqual(
            {(item["kind"], item["identifier"]) for item in evidence},
            {
                ("metric", "total_sales"),
                ("filter_field", REGION_ID),
                ("filter_value", "region_north"),
            },
        )

        unmatched = collect_executable_evidence(
            QUERY,
            required_sources=["filter_value"],
            columns=[],
            metrics=[],
            values=[ValueInfo("red", "红色", "dim_product.color")],
        )
        self.assertEqual(unmatched, [])


class RecallDecisionTests(unittest.TestCase):
    def test_missing_required_filter_blocks_and_preserves_candidate_separation(self):
        requirements = infer_required_sources(QUERY)
        decision = build_recall_decision(
            request_id="request-1",
            query=QUERY,
            requirements=requirements,
            branch_outcomes={
                "column": outcome("success"),
                "metric": outcome("success"),
                "value": outcome("empty", "no_candidates"),
            },
            columns=COLUMNS,
            metrics=METRICS,
            values=[],
        )

        self.assertEqual(decision["status"], "blocked")
        self.assertEqual(decision["missing"], ["filter_value"])
        self.assertEqual(decision["reason_code"], "required_filter_unverified")
        self.assertIn("filter_value", decision["request"]["required_sources"])
        self.assertIn("华北", decision["low_confidence_candidates"])
        self.assertNotIn("华北", [item["label"] for item in decision["evidence"]])
        self.assertNotIn("no_candidates", decision["user_message"])

    def test_optional_failed_branch_degrades_when_required_evidence_is_complete(self):
        requirements = infer_required_sources("统计销售总额")
        decision = build_recall_decision(
            request_id="request-2",
            query="统计销售总额",
            requirements=requirements,
            branch_outcomes={
                "column": outcome("error", "retrieval_error"),
                "metric": outcome("success"),
                "value": outcome("empty", "no_candidates"),
            },
            columns=[MEASURE],
            metrics=METRICS,
            values=[],
        )

        self.assertEqual(decision["status"], "degraded")
        self.assertEqual(decision["missing"], [])
        self.assertEqual(decision["reason_code"], "optional_evidence_unavailable")
        self.assertIn("部分", decision["user_message"])


class ConservativeCoverageTests(unittest.TestCase):
    def test_unknown_or_partially_confirmed_filter_cannot_be_dropped(self):
        for query in (
            "统计天津地区的销售总额",
            "统计华北和华南地区的销售总额",
            "统计华北地区且订单金额大于100的销售总额",
            "统计未退款的销售总额",
        ):
            with self.subTest(query=query):
                decision = build_recall_decision(
                    request_id="coverage",
                    query=query,
                    requirements=infer_required_sources(query),
                    branch_outcomes={
                        s: outcome("success") for s in ("column", "metric", "value")
                    },
                    columns=COLUMNS,
                    metrics=METRICS,
                    values=[ValueInfo("north", "华北", REGION_ID)],
                )
                self.assertEqual(decision["status"], "blocked")
                self.assertIn("filter_value", decision["request"]["required_sources"])
                self.assertTrue(
                    decision["missing"] or decision["request"]["unresolved_terms"]
                )


class AuthorityTests(unittest.TestCase):
    def test_branch_health_does_not_authorize_stale_or_nonmatching_values(self):
        for status in ("success", "timeout", "error", "empty", "insufficient"):
            with self.subTest(status=status):
                decision = build_recall_decision(
                    request_id="status-matrix",
                    query=QUERY,
                    requirements=infer_required_sources(QUERY),
                    branch_outcomes={
                        "column": outcome("success"),
                        "metric": outcome("success"),
                        "value": outcome(status),
                    },
                    columns=COLUMNS,
                    metrics=METRICS,
                    values=[ValueInfo("north", "华北", REGION_ID)],
                )
                self.assertEqual(
                    decision["status"], "success" if status == "success" else "blocked"
                )
                self.assertEqual(decision["branch_statuses"]["value"]["status"], status)
                self.assertEqual(decision["request"]["request_id"], "status-matrix")

    def test_authoritative_metadata_examples_can_cover_unavailable_value_source(self):
        column = ColumnInfo(
            REGION_ID,
            "region_name",
            "varchar",
            "dimension",
            ["华北"],
            "地区",
            ["地区"],
            "fact_order",
        )
        decision = build_recall_decision(
            request_id="metadata",
            query="统计华 北地区的销售总额",
            requirements=infer_required_sources("统计华 北地区的销售总额"),
            branch_outcomes={
                "column": outcome("success"),
                "metric": outcome("success"),
                "value": outcome("error"),
            },
            columns=[MEASURE, column],
            metrics=METRICS,
            values=[],
        )
        self.assertEqual(decision["status"], "degraded")
        self.assertEqual(decision["missing"], [])
        self.assertTrue(
            any(
                e["kind"] == "filter_value" and e["provenance"] == "metadata"
                for e in decision["evidence"]
            )
        )

    def test_nonmatching_successful_branch_is_insufficient_not_executable(self):
        decision = build_recall_decision(
            request_id="nonmatching",
            query=QUERY,
            requirements=infer_required_sources(QUERY),
            branch_outcomes={
                s: outcome("success") for s in ("column", "metric", "value")
            },
            columns=COLUMNS,
            metrics=METRICS,
            values=[ValueInfo("south", "华南", REGION_ID)],
        )
        self.assertEqual(decision["status"], "blocked")
        self.assertEqual(decision["branch_statuses"]["value"]["status"], "insufficient")
        self.assertEqual(
            decision["branch_statuses"]["value"]["reason_code"], "insufficient_evidence"
        )

    def test_time_and_join_demands_without_authoritative_mapping_block(self):
        for query, missing in (
            ("本月销售总额", "time_field"),
            ("跨表统计销售总额", "join_path"),
            ("按地区统计销售总额", "analysis_field"),
        ):
            with self.subTest(query=query):
                decision = build_recall_decision(
                    request_id="unsupported",
                    query=query,
                    requirements=infer_required_sources(query),
                    branch_outcomes={
                        s: outcome("success") for s in ("column", "metric", "value")
                    },
                    columns=[MEASURE],
                    metrics=METRICS,
                    values=[],
                )
                self.assertEqual(decision["status"], "blocked")
                self.assertIn(missing, decision["missing"])


if __name__ == "__main__":
    unittest.main()
