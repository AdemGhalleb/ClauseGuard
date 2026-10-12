"""Synthetic tests for retrieval and deterministic-rule evaluation metrics."""

from importlib import import_module

import pytest

retrieval_eval = import_module("legal-ai.evaluation.retrieval")
retrieval_module = import_module("legal-ai.retrieval.contract_retriever")


def test_recall_at_k_for_labeled_demo_criteria():
    retriever = retrieval_module.InMemoryRuleRetriever()
    cases = [
        {"clause_type": "non_compete", "query": "duration months", "expected_rule_id": "demo.non_compete.duration_over_12_months"},
        {"clause_type": "termination_notice", "query": "notice days", "expected_rule_id": "demo.termination_notice.shorter_than_30_days"},
    ]
    assert retrieval_eval.recall_at_k(cases, retriever, k=1) == 1.0


def test_recall_at_k_empty_and_invalid_k():
    retriever = retrieval_module.InMemoryRuleRetriever()
    assert retrieval_eval.recall_at_k([], retriever) == 0.0
    with pytest.raises(ValueError):
        retrieval_eval.recall_at_k([], retriever, k=0)


def test_binary_rule_metrics_counts_confusion_matrix():
    result = retrieval_eval.binary_rule_metrics(
        [True, True, False, False], [True, False, True, False]
    )
    assert result == {"tp": 1, "fp": 1, "tn": 1, "fn": 1, "accuracy": 0.5}


def test_binary_rule_metrics_rejects_unaligned_inputs():
    with pytest.raises(ValueError):
        retrieval_eval.binary_rule_metrics([True], [])


def test_runnable_demo_evaluation_reports_fixture_metrics():
    demo = import_module("legal-ai.evaluation.run_concern_demo").run_demo()
    assert demo["recall_at_1"] == 1.0
    assert demo["rule_metrics"] == {"tp": 3, "fp": 0, "tn": 3, "fn": 0, "accuracy": 1.0}
    assert demo["dataset"].startswith("synthetic")
    assert demo["cases"] == 6
    assert demo["evaluated_cases"] == 6
    assert demo["unevaluable_cases"] == 0
    assert demo["unevaluable_rules"] == 0


def test_demo_case_does_not_count_unevaluable_rule_as_clean_negative():
    demo_module = import_module("legal-ai.evaluation.run_concern_demo")
    detector = import_module("legal-ai.rules.detector").ConcernDetector()

    prediction, unevaluable_rules = demo_module._evaluate_case(
        detector, "non_compete", {"exceptions": []}
    )

    assert prediction is None
    assert unevaluable_rules == 1
