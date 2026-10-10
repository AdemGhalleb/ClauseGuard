"""Compatibility integration from the current extractor JSON output to findings."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Mapping, Sequence

retrieval_module = import_module("legal-ai.retrieval.contract_retriever")
engine_module = import_module("legal-ai.rules.engine")


@dataclass(frozen=True)
class DetectionResult:
    findings: list[Any]
    retrieved_rule_ids: list[str]
    evaluations: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"findings": [finding.to_dict() for finding in self.findings],
                "retrieved_rule_ids": list(self.retrieved_rule_ids),
                "evaluations": [evaluation.to_dict() for evaluation in self.evaluations]}


def adapt_extraction_output(output: str | Mapping[str, Any]) -> dict[str, Any]:
    """Accept the extractor's JSON string or an already parsed extraction mapping."""
    if isinstance(output, str):
        try:
            facts = json.loads(output)
        except json.JSONDecodeError as exc:
            raise ValueError("extractor output must be valid JSON") from exc
    else:
        facts = dict(output)
    if not isinstance(facts, dict) or not all(isinstance(key, str) for key in facts):
        raise ValueError("extractor output must be a JSON object with string keys")
    normalizer = import_module("legal-ai.extraction.normalizer").Normalizer
    return normalizer.normalize_extraction("", facts)


class ConcernDetector:
    """Evaluate all applicable criteria deterministically, with optional supplementary retrieval.

    Rule selection identifies all rules applicable to the given clause type from the
    rule store, ensuring retrieval ranking or top_k filtering cannot cause an
    applicable rule to be skipped. Supplementary retrieval is retained to provide
    supporting context and rationale.
    """

    def __init__(
        self,
        retriever: Any | None = None,
        rule_store: Any | None = None,
        *,
        rules: Sequence[Any] | None = None,
    ):
        if rule_store is not None:
            self.rule_store = rule_store
        elif rules is not None:
            self.rule_store = engine_module.InMemoryRuleStore(rules)
        elif retriever is not None and hasattr(retriever, "get_rules_for_clause_type"):
            self.rule_store = retriever
        else:
            self.rule_store = engine_module.InMemoryRuleStore()
        self.retriever = retriever if retriever is not None else retrieval_module.InMemoryRuleRetriever()

    def detect(
        self,
        clause_type: str,
        extraction_output: str | Mapping[str, Any],
        *,
        clause_text: str | None = None,
        top_k: int = 5,
    ) -> DetectionResult:
        """Evaluate all applicable rules for clause_type, plus optional supplementary retrieval.

        Parameters:
            clause_type: The categorized type of the clause (e.g. 'non_compete').
            extraction_output: Raw JSON string or dictionary of extracted facts.
            clause_text: Optional full clause string used for extracting substring evidence.
            top_k: Maximum number of supplementary rule documents to retrieve for
                   supporting context and rationale. Does NOT limit rule evaluation.
        """
        facts = adapt_extraction_output(extraction_output)
        applicable_rules = self.rule_store.get_rules_for_clause_type(clause_type)
        evidence = _evidence_by_field(facts, clause_text)
        evaluations = engine_module.evaluate_all_rules(clause_type, facts, applicable_rules)
        findings = engine_module.evaluate_rules(
            clause_type, facts, applicable_rules, evidence=evidence
        )
        documents = []
        if self.retriever is not None and hasattr(self.retriever, "retrieve_for_facts"):
            documents = self.retriever.retrieve_for_facts(clause_type, facts, top_k=top_k)
        return DetectionResult(findings, [document.rule_id for document in documents], evaluations)


def _evidence_by_field(facts: Mapping[str, Any], clause_text: str | None) -> dict[str, str]:
    evidence = {}
    if not clause_text:
        return evidence
    folded = clause_text.casefold()
    for field, value in facts.items():
        raw = value.get("raw") if isinstance(value, Mapping) else value
        if not isinstance(raw, str) or not raw:
            continue
        start = folded.find(raw.casefold())
        if start >= 0:
            evidence[field] = clause_text[start:start + len(raw)]
    return evidence
