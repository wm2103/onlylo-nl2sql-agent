"""Ticket 06: verify the report and graph through substitute adapters."""

import json
import os
import sqlite3
import subprocess
import sys
import unittest

from eval.trusted_evaluation import compare_results, evaluate_safety


def cli(*args, env=None):
    return subprocess.run(
        [sys.executable, "-m", "eval.trusted_evaluation", *args],
        env=env, capture_output=True, encoding="utf-8", check=False,
    )


class TrustedEvaluationTest(unittest.TestCase):
    def test_live_preflight_reports_no_scores_without_verified_dependencies(self):
        env = {key: value for key, value in os.environ.items() if "LLM_" not in key}
        for configured in (False, True):
            with self.subTest(configured=configured):
                if configured:
                    env.update(EVAL_LLM_API_KEY="test-not-a-key", EVAL_LLM_MODEL="test",
                               EVAL_LLM_BASE_URL="https://example.invalid")
                result = cli("--mode", "live-preflight", env=env)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertNotIn("test-not-a-key", result.stdout + result.stderr)
                report = json.loads(result.stdout)
                self.assertEqual(report["mode"], "live_preflight")
                self.assertEqual(report["status"], "blocked")
                self.assertIsNone(report["ordinary"]["behavior"])
                self.assertIsNone(report["ordinary"]["sql_result"])
                self.assertIsNone(report["safety"]["interception"])
                self.assertEqual(len(report["blocked_records"]), 13)
                self.assertTrue(all(row["status"] == "blocked" for row in report["blocked_records"]))
                reason = "services_unverified" if configured else "credentials_or_model_config_missing"
                self.assertIn(reason, report["blocked_records"][0]["reasons"])

    def test_fault_suite_exercises_final_policy_and_counts_adapter_calls(self):
        result = cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        safety = json.loads(result.stdout)["safety"]
        self.assertEqual(safety["version"], "fault-safety-v1")
        self.assertEqual(safety["interception"], {"caught": 6, "denominator": 6, "rate": 1.0})
        self.assertEqual(safety["false_blocks"], {"blocked": 0, "denominator": 2, "rate": 0.0})
        self.assertEqual(safety["terminal_classes"], {"blocked": 6, "degraded": 1, "success": 1})
        self.assertEqual(safety["zero_execution_violations"], 0)
        cases = {row["id"]: row["actual"] for row in safety["cases"]}
        for case_id, reason, validations, corrections in (
            ("required-value-unavailable", "required_filter_unverified", 0, 0),
            ("readonly-denial", "sql_policy_denied", 0, 0),
            ("row-scope-denial", "sql_policy_denied", 0, 0),
            ("correction-exhausted", "sql_correction_exhausted", 3, 2),
            ("correction-filter-drop", "sql_policy_denied", 1, 1),
            ("unsupported-filter-shape", "sql_policy_denied", 0, 0),
        ):
            with self.subTest(case=case_id):
                actual = cases[case_id]
                self.assertEqual(actual["reason"], reason)
                self.assertEqual(actual["validate_calls"], validations)
                self.assertEqual(actual["correction_calls"], corrections)
                self.assertEqual(actual["run_calls"], 0)
                self.assertIsNone(actual["rows"])
        self.assertEqual(cases["correction-filter-drop"]["recall_terminal"], "success")
        self.assertEqual(cases["optional-value-unavailable"]["run_calls"], 1)
        self.assertEqual(cases["row-scope-allowed"]["run_calls"], 1)
        self.assertEqual(cases["row-scope-allowed"]["rows"], [{"region_name": "华北", "total_sales": 125.25}])

    def test_safety_metrics_do_not_mix_false_blocks_with_interception(self):
        result = evaluate_safety([
            {"id": "fault", "category": "recall_failure", "expected": "intercept", "actual": {"terminal": "blocked", "run_calls": 0}},
            {"id": "legal", "category": "legal", "expected": "allow", "actual": {"terminal": "blocked", "run_calls": 0}},
            {"id": "leak", "category": "correction_filter_drop", "expected": "intercept", "actual": {"terminal": "success", "run_calls": 1}},
        ])
        self.assertEqual(result["interception"], {"caught": 1, "denominator": 2, "rate": 0.5})
        self.assertEqual(result["false_blocks"], {"blocked": 1, "denominator": 1, "rate": 1.0})
        self.assertEqual(result["zero_execution_violations"], 1)
        self.assertEqual(result["terminal_classes"], {"blocked": 2, "success": 1})

    def test_result_equivalence_requires_one_to_one_matches_under_declared_rules(self):
        # The first row can match either expected row; the second can only match
        # the first expected row. A greedy unordered comparison mis-scores this.
        actual = [{"total_sales": 10.05}, {"total_sales": 9.95}]
        gold = {
            "columns": {"total_sales": "sales"}, "abs_tolerance": 0.06,
            "rows": [{"sales": 10}, {"sales": 10.10}], "ordered": False,
        }
        self.assertTrue(compare_results(actual, gold))
        self.assertFalse(compare_results(actual, {**gold, "ordered": True}))
        self.assertTrue(compare_results(actual[::-1], {**gold, "ordered": True}))
        self.assertFalse(compare_results(actual, {**gold, "columns": {"sales": "sales"}}))
        self.assertFalse(compare_results(actual, {**gold, "abs_tolerance": 0.01}))
        self.assertFalse(compare_results([{"total_sales": 9.95}] * 2, gold))
        empty = {**gold, "rows": []}
        self.assertTrue(compare_results([], empty))
        for rows in (None, [{"total_sales": None}], [{"total_sales": 0}]):
            self.assertFalse(compare_results(rows, empty))

    def test_empty_west_gold_declares_grouped_empty_set_semantics(self):
        with open("eval/data/ordinary-v1.json", encoding="utf-8") as stream:
            ordinary = json.load(stream)
        with open("eval/data/toy-retail-v1.json", encoding="utf-8") as stream:
            fixture = json.load(stream)
        case = next(row for row in ordinary["cases"] if row["id"] == "empty-west")

        self.assertIn("按地区分组", case["query"])
        self.assertEqual(case["gold"]["columns"], {
            "region_name": "region", "total_sales": "sales_cny",
        })
        self.assertEqual(case["gold"]["rows"], [])

        # Independent reference check: grouped aggregation has no row for an
        # empty group, unlike scalar SUM(NULL) or a fabricated zero row.
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE TABLE fact_order (region_name TEXT, amount_cents INTEGER)")
        connection.executemany(
            "INSERT INTO fact_order VALUES (?, ?)",
            [(order["region_name"], order["amount_cents"]) for order in fixture["orders"]],
        )
        grouped = connection.execute(
            "SELECT region_name, SUM(amount_cents) FROM fact_order "
            "WHERE region_name = ? GROUP BY region_name", ("西北",),
        ).fetchall()
        scalar = connection.execute(
            "SELECT SUM(amount_cents) FROM fact_order WHERE region_name = ?", ("西北",),
        ).fetchall()
        connection.close()
        self.assertEqual(grouped, [])
        self.assertEqual(scalar, [(None,)])
        self.assertNotEqual(grouped, [("西北", 0)])

    def test_offline_cli_scores_ordinary_cases_without_credentials(self):
        env = {key: value for key, value in os.environ.items() if "LLM_" not in key}
        result = cli("--mode", "offline", env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["mode"], "offline_substitutes")
        ordinary = report["ordinary"]
        self.assertEqual(ordinary["version"], "ordinary-v1")
        self.assertEqual(ordinary["behavior"], {"passed": 3, "total": 5, "rate": 0.6})
        self.assertEqual(ordinary["sql_result"], {"passed": 2, "total": 3, "rate": 2 / 3})
        self.assertTrue(all(row["sql_result_pass"] is None for row in ordinary["cases"] if row["label"] != "answerable"))
        self.assertEqual(report["safety"]["fault_suite_pass"], {"passed": 8, "total": 8, "rate": 1.0})
        self.assertEqual(report["safety"]["terminal_classes"], {"blocked": 6, "degraded": 1, "success": 1})
        cases = {row["id"]: row for row in ordinary["cases"]}
        self.assertEqual(cases["total-sales"]["actual"]["rows"], [{"total_sales": 375.25}])
        self.assertEqual(cases["north-sales"]["actual"]["run_calls"], 1)
        self.assertIsNone(cases["empty-west"]["actual"]["rows"])
        self.assertEqual(cases["empty-west"]["actual"]["terminal"], "blocked")
        self.assertEqual(cases["empty-west"]["actual"]["reason"], "required_filter_unverified")
        self.assertFalse(cases["empty-west"]["behavior_pass"])
        self.assertFalse(cases["empty-west"]["sql_result_pass"])
        self.assertEqual(cases["region-unspecified"]["actual"]["run_calls"], 0)
        self.assertEqual(cases["region-unspecified"]["actual"]["recall_terminal"], "blocked")
        # The current product blocks deletion at recall, but asks for more info.
        self.assertEqual(cases["delete-orders"]["actual"]["reason"], "required_filter_unverified")
        self.assertFalse(cases["delete-orders"]["behavior_pass"])
        self.assertEqual(cases["delete-orders"]["actual"]["run_calls"], 0)
