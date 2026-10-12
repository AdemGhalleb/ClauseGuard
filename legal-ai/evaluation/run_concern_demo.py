"""Run labeled synthetic criteria-retrieval and rule-evaluation examples."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from importlib import import_module

metrics = import_module("legal-ai.evaluation.retrieval")
retrieval = import_module("legal-ai.retrieval.contract_retriever")
detector_module = import_module("legal-ai.rules.detector")


def _evaluate_case(detector, category: str, facts: dict[str, object]) -> tuple[bool | None, int]:
    """Return a concern prediction only when every applicable rule was evaluated."""
    result = detector.detect(category, facts)
    unevaluable_rules = sum(
        evaluation.outcome == "could_not_evaluate"
        for evaluation in result.evaluations
    )
    if unevaluable_rules:
        return None, unevaluable_rules
    return bool(result.findings), 0


def run_demo() -> dict[str, object]:
    """Return metrics for synthetic cases; extraction is supplied as known facts."""
    retriever = retrieval.InMemoryRuleRetriever()
    retrieval_cases = [
        {"clause_type": "non_compete", "query": "duration months", "expected_rule_id": "demo.non_compete.duration_over_12_months"},
        {"clause_type": "termination_notice", "query": "notice period days", "expected_rule_id": "demo.termination_notice.shorter_than_30_days"},
        {"clause_type": "confidentiality", "query": "duration months", "expected_rule_id": "demo.confidentiality.duration_over_60_months"},
    ]
    examples = [
        ("non_compete", {"duration": "five years", "exceptions": []}, True),
        ("non_compete", {"duration": "six months", "exceptions": []}, False),
        ("termination_notice", {"notice_period": "immediately", "conditions": []}, True),
        ("termination_notice", {"notice_period": "ninety days", "conditions": []}, False),
        ("confidentiality", {"duration": "six years", "exceptions": []}, True),
        ("confidentiality", {"duration": "five years", "exceptions": []}, False),
    ]
    detector = detector_module.ConcernDetector(retriever)
    predictions = []
    expected = []
    unevaluable_cases = 0
    unevaluable_rules = 0
    for category, facts, label in examples:
        prediction, case_unevaluable_rules = _evaluate_case(detector, category, facts)
        if prediction is None:
            unevaluable_cases += 1
            unevaluable_rules += case_unevaluable_rules
            continue
        predictions.append(prediction)
        expected.append(label)
    return {
        "dataset": "synthetic demonstration cases; extracted facts are supplied",
        "recall_at_1": metrics.recall_at_k(retrieval_cases, retriever, k=1),
        "rule_metrics": metrics.binary_rule_metrics(expected, predictions),
        "cases": len(examples),
        "evaluated_cases": len(predictions),
        "unevaluable_cases": unevaluable_cases,
        "unevaluable_rules": unevaluable_rules,
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2, sort_keys=True))
