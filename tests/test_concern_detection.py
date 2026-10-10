"""End-to-end extraction adapter, criteria retrieval, and rule-evaluation tests."""

from importlib import import_module


ConcernDetector = import_module("legal-ai.rules.detector").ConcernDetector
ConcernRule = import_module("legal-ai.rules.engine").ConcernRule
InMemoryRuleStore = import_module("legal-ai.rules.engine").InMemoryRuleStore
InMemoryRuleRetriever = import_module("legal-ai.retrieval.contract_retriever").InMemoryRuleRetriever
RuleDocument = import_module("legal-ai.retrieval.contract_retriever").RuleDocument


def _make_rule_doc(rule: ConcernRule, text: str = "") -> RuleDocument:
    return RuleDocument(
        rule_id=rule.rule_id,
        text=text or f"{rule.clause_type} {rule.field} {rule.description}",
        metadata={"clause_type": rule.clause_type, "field": rule.field},
        source=rule.source,
        rule=rule,
    )



def test_extractor_json_to_retrieved_rule_and_finding():
    clause = "For five (5) years after termination, the employee shall not work for a competitor."
    result = ConcernDetector().detect(
        "non_compete", '{"duration": "five (5) years", "exceptions": []}', clause_text=clause
    ).to_dict()
    assert result["retrieved_rule_ids"] == ["demo.non_compete.duration_over_12_months"]
    assert len(result["findings"]) == 1
    finding = result["findings"][0]
    assert finding["extracted_value"] == "five (5) years"
    assert finding["evidence"] == "five (5) years"
    assert finding["source"] == "demo_configuration"
    assert "configured review policy" in finding["explanation"]


def test_non_trigger_and_exception_produce_no_finding():
    detector = ConcernDetector()
    below_threshold = detector.detect("non_compete", {"duration": "six months", "exceptions": []})
    excepted = detector.detect("non_compete", {"duration": "five years", "exceptions": ["prior consent"]})
    assert below_threshold.findings == []
    assert excepted.findings == []


def test_invalid_extractor_json_is_reported_as_adapter_error():
    import pytest

    with pytest.raises(ValueError, match="valid JSON"):
        ConcernDetector().detect("non_compete", "not JSON")


def test_top_k_miss_does_not_suppress_matching_rule():
    rule_a = ConcernRule(
        rule_id="nc.rule_a",
        clause_type="non_compete",
        field="geographical_scope",
        operator="equals",
        value="worldwide",
        finding_type="scope_review",
        severity="medium",
        description="Worldwide non-compete.",
        source="test",
    )
    rule_b = ConcernRule(
        rule_id="nc.rule_b",
        clause_type="non_compete",
        field="solicitation",
        operator="is_true",
        finding_type="solicitation_review",
        severity="medium",
        description="Customer solicitation non-compete.",
        source="test",
    )
    rule_c = ConcernRule(
        rule_id="nc.rule_c",
        clause_type="non_compete",
        field="duration",
        operator="greater_than",
        value=24,
        unit="month",
        finding_type="duration_review",
        severity="high",
        description="Duration over 24 months.",
        source="test",
    )
    rule_d = ConcernRule(
        rule_id="nc.rule_d",
        clause_type="non_compete",
        field="duration",
        operator="greater_than",
        value=12,
        unit="month",
        finding_type="duration_review",
        severity="high",
        description="Duration over 12 months.",
        source="test",
    )

    # doc_a has maximum lexical overlap with query tokens from facts: "non_compete duration 18 months local"
    doc_a = _make_rule_doc(rule_a, text="non_compete duration 18 months local geographical_scope")
    doc_b = _make_rule_doc(rule_b, text="non_compete solicitation")
    doc_c = _make_rule_doc(rule_c, text="non_compete other")
    doc_d = _make_rule_doc(rule_d, text="non_compete threshold")

    retriever = InMemoryRuleRetriever([doc_a, doc_b, doc_c, doc_d])
    detector = ConcernDetector(retriever=retriever)

    facts = {"duration": "18 months", "geographical_scope": "local"}
    result = detector.detect("non_compete", facts, top_k=1)

    # Retriever only returned rule A because top_k=1 and doc_a had highest lexical overlap
    assert result.retrieved_rule_ids == ["nc.rule_a"]
    # Despite retriever missing rule D, rule D was still evaluated and triggered a finding
    assert len(result.findings) == 1
    assert result.findings[0].rule_id == "nc.rule_d"
    assert result.findings[0].extracted_value == "18 months"


def test_positive_case_evaluates_all_applicable_rules_when_subset_retrieved():
    rule_a = ConcernRule(
        rule_id="nc.a",
        clause_type="non_compete",
        field="geographical_scope",
        operator="equals",
        value="worldwide",
        finding_type="scope_review",
        severity="medium",
        description="Worldwide non-compete.",
        source="test",
    )
    rule_b = ConcernRule(
        rule_id="nc.b",
        clause_type="non_compete",
        field="duration",
        operator="greater_than",
        value=36,
        unit="month",
        finding_type="duration_review",
        severity="high",
        description="Duration over 36 months.",
        source="test",
    )
    rule_c = ConcernRule(
        rule_id="nc.c",
        clause_type="non_compete",
        field="duration",
        operator="greater_than",
        value=12,
        unit="month",
        finding_type="duration_review",
        severity="high",
        description="Duration over 12 months.",
        source="test",
    )

    doc_a = _make_rule_doc(rule_a, text="non_compete duration 18 months local geographical_scope")
    doc_b = _make_rule_doc(rule_b, text="non_compete duration 18 months")
    doc_c = _make_rule_doc(rule_c, text="minimal")

    retriever = InMemoryRuleRetriever([doc_a, doc_b, doc_c])
    detector = ConcernDetector(retriever=retriever)

    result = detector.detect("non_compete", {"duration": "18 months", "geographical_scope": "local"}, top_k=2)
    # Retrieval returned only {nc.a, nc.b}, omitting nc.c
    assert result.retrieved_rule_ids == ["nc.a", "nc.b"]
    assert "nc.c" not in result.retrieved_rule_ids
    # Finding for rule C is still produced
    assert len(result.findings) == 1
    assert result.findings[0].rule_id == "nc.c"


def test_negative_case_all_applicable_rules_evaluated_none_match():
    rule_a = ConcernRule(
        rule_id="nc.a",
        clause_type="non_compete",
        field="duration",
        operator="greater_than",
        value=24,
        unit="month",
        finding_type="duration_review",
        severity="high",
        description="Over 24 months.",
        source="test",
    )
    rule_b = ConcernRule(
        rule_id="nc.b",
        clause_type="non_compete",
        field="geographical_scope",
        operator="equals",
        value="worldwide",
        finding_type="scope_review",
        severity="medium",
        description="Worldwide non-compete.",
        source="test",
    )
    store = InMemoryRuleStore([rule_a, rule_b])
    detector = ConcernDetector(rule_store=store)

    result = detector.detect("non_compete", {"duration": "6 months", "geographical_scope": "local"})
    assert result.findings == []


def test_unrelated_clause_rules_are_not_evaluated():
    rule_nc = ConcernRule(
        rule_id="nc.1",
        clause_type="non_compete",
        field="duration",
        operator="greater_than",
        value=24,
        unit="month",
        finding_type="duration_review",
        severity="high",
        description="Over 24 months.",
        source="test",
    )
    rule_term = ConcernRule(
        rule_id="term.1",
        clause_type="termination_notice",
        field="duration",
        operator="greater_than",
        value=12,
        unit="month",
        finding_type="notice_review",
        severity="medium",
        description="Notice duration over 12 months.",
        source="test",
    )
    store = InMemoryRuleStore([rule_nc, rule_term])
    detector = ConcernDetector(rule_store=store)

    # The facts {"duration": "18 months"} satisfy rule_term's condition (> 12 months),
    # but when analyzing "non_compete", termination_notice rules must NOT be evaluated.
    result = detector.detect("non_compete", {"duration": "18 months"})
    assert result.findings == []


def test_empty_retrieval_still_evaluates_applicable_rules():
    rule = ConcernRule(
        rule_id="nc.duration",
        clause_type="non_compete",
        field="duration",
        operator="greater_than",
        value=12,
        unit="month",
        finding_type="duration_review",
        severity="high",
        description="Duration over 12 months.",
        source="test",
    )
    missing_scope = ConcernRule(
        rule_id="nc.scope",
        clause_type="non_compete",
        field="geographical_scope",
        operator="equals",
        value="worldwide",
        finding_type="scope_review",
        severity="medium",
        description="Worldwide non-compete.",
        source="test",
    )
    store = InMemoryRuleStore([rule, missing_scope])
    detector = ConcernDetector(retriever=InMemoryRuleRetriever([]), rule_store=store)

    result = detector.detect("non_compete", {"duration": "18 months"})
    assert result.retrieved_rule_ids == []
    assert [finding.rule_id for finding in result.findings] == ["nc.duration"]
    outcomes = {item.rule_id: item.outcome for item in result.evaluations}
    assert outcomes == {"nc.duration": "matched", "nc.scope": "could_not_evaluate"}
    serialized = result.to_dict()
    assert serialized["evaluations"][0]["outcome"] in {"matched", "could_not_evaluate"}
    assert {item["outcome"] for item in serialized["evaluations"]} == {"matched", "could_not_evaluate"}

