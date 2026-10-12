"""Tests for typed concern rules and deterministic field evaluation."""

from importlib import import_module

import pytest

engine = import_module("legal-ai.rules.engine")
ConcernRule = engine.ConcernRule
rule_matches = engine.rule_matches
evaluate_rule = engine.evaluate_rule
evaluate_rules = engine.evaluate_rules
evaluate_all_rules = engine.evaluate_all_rules
load_rules = engine.load_rules
InMemoryRuleStore = engine.InMemoryRuleStore


def make_rule(**overrides):
    values = dict(rule_id="demo.duration", clause_type="non_compete", field="duration",
                  operator="greater_than", value=12, unit="month",
                  finding_type="duration_review", severity="high",
                  description="Demonstration review threshold.",
                  source="demo_configuration")
    values.update(overrides)
    return ConcernRule(**values)


def test_valid_rule_and_condition():
    rule = make_rule()
    assert rule.rule_id == "demo.duration"
    assert rule.rule_condition == "duration > 12 months"


@pytest.mark.parametrize("change", [
    {"rule_id": ""}, {"clause_type": ""}, {"field": ""},
    {"operator": "execute"}, {"severity": "critical"},
    {"value": None}, {"source": ""}, {"unit": "fortnight"},
])
def test_invalid_rule_rejected(change):
    with pytest.raises(ValueError):
        make_rule(**change)


@pytest.mark.parametrize("overrides", [
    {"operator": "greater_than", "value": "12", "unit": None},
    {"operator": "contains", "value": {"term": "client"}, "unit": None},
    {"operator": "in", "value": "worldwide", "unit": None},
    {"operator": "is_true", "value": False, "unit": None},
    {"operator": "is_false", "value": None, "unit": "month"},
])
def test_invalid_operator_operands_rejected(overrides):
    with pytest.raises(ValueError, match="value|unit"):
        make_rule(**overrides)


def test_numeric_duration_match_and_non_match():
    rule = make_rule()
    assert rule_matches(rule, {"duration": "five (5) years"})
    assert not rule_matches(rule, {"duration": "six months"})


def test_normalized_extractor_duration_is_supported():
    rule = make_rule()
    facts = {"duration": {"raw": "five (5) years", "normalized": {"value": 5, "unit": "year"}}}
    assert rule_matches(rule, facts)


def test_raw_duration_corrects_ambiguous_digit_in_number_word():
    rule = make_rule(field="notice_period", unit="day", value=30, operator="less_than")
    # The current extractor normalizer can read the "9" inside "ninety";
    # the raw extraction is retained, so rule comparison reparses that source.
    facts = {"notice_period": {"raw": "ninety days", "normalized": {"value": 9, "unit": "day"}}}
    assert not rule_matches(rule, facts)


def test_comparison_and_equality_operators():
    assert rule_matches(make_rule(operator="equals", value="worldwide", unit=None),
                        {"duration": "worldwide"})
    assert rule_matches(make_rule(operator="not_equals", value="local", unit=None),
                        {"duration": "worldwide"})
    assert rule_matches(make_rule(operator="less_than", value=24), {"duration": "18 months"})
    assert rule_matches(make_rule(operator="greater_than_or_equal", value=18),
                        {"duration": "18 months"})


def test_membership_boolean_and_nonempty_operators():
    assert rule_matches(make_rule(operator="contains", value="client", unit=None),
                        {"duration": ["client", "vendor"]})
    assert rule_matches(make_rule(operator="in", value=["worldwide", "global"], unit=None),
                        {"duration": "worldwide"})
    assert rule_matches(make_rule(operator="is_true", value=None, unit=None), {"duration": True})
    assert rule_matches(make_rule(operator="is_false", value=None, unit=None), {"duration": False})
    assert rule_matches(make_rule(operator="is_not_empty", value=None, unit=None), {"duration": ["x"]})


def test_matching_condition_is_matched():
    evaluation = evaluate_rule(make_rule(), {"duration": "18 months"})
    assert evaluation.outcome == "matched"
    assert evaluation.reason == "condition is satisfied"
    assert evaluation.extracted_value == "18 months"


def test_valid_non_matching_condition_is_not_matched():
    evaluation = evaluate_rule(make_rule(), {"duration": "six months"})
    assert evaluation.outcome == "not_matched"
    assert "condition is not satisfied" in evaluation.reason
    assert evaluation.extracted_value == "six months"
    assert not rule_matches(make_rule(), {"duration": "six months"})


def test_missing_field_could_not_evaluate():
    evaluation = evaluate_rule(make_rule(), {})
    assert evaluation.outcome == "could_not_evaluate"
    assert evaluation.reason == "required field 'duration' is missing"
    assert evaluation.extracted_value is None
    assert not rule_matches(make_rule(), {})


def test_null_field_could_not_evaluate():
    evaluation = evaluate_rule(make_rule(), {"duration": None})
    assert evaluation.outcome == "could_not_evaluate"
    assert evaluation.reason == "required field 'duration' is null"
    assert evaluation.extracted_value is None


def test_malformed_duration_could_not_evaluate():
    evaluation = evaluate_rule(make_rule(), {"duration": "an extended period"})
    assert evaluation.outcome == "could_not_evaluate"
    assert "malformed or unsupported" in evaluation.reason
    assert evaluation.extracted_value == "an extended period"


@pytest.mark.parametrize("overrides,facts,malformed", [
    ({"operator": "greater_than"}, {"duration": object()}, True),
    ({"operator": "greater_than_or_equal"}, {}, False),
    ({"operator": "less_than"}, {"duration": None}, False),
    ({"operator": "less_than_or_equal", "unit": None, "value": 10}, {"duration": {"nested": True}}, True),
    ({"operator": "equals", "value": "worldwide", "unit": None}, {}, False),
    ({"operator": "not_equals", "value": "local", "unit": None}, {"duration": None}, False),
    ({"operator": "contains", "value": "client", "unit": None}, {"duration": 12}, True),
    ({"operator": "in", "value": ["worldwide"], "unit": None}, {}, False),
    ({"operator": "is_true", "value": None, "unit": None}, {"duration": None}, False),
    ({"operator": "is_false", "value": None, "unit": None}, {}, False),
    ({"operator": "is_not_empty", "value": None, "unit": None}, {"duration": None}, False),
])
def test_operator_families_do_not_treat_insufficient_facts_as_not_matched(overrides, facts, malformed):
    evaluation = evaluate_rule(make_rule(**overrides), facts)
    assert evaluation.outcome == "could_not_evaluate"
    if malformed:
        assert "malformed or unsupported" in evaluation.reason
    elif "duration" not in facts:
        assert evaluation.reason == "required field 'duration' is missing"
    else:
        assert evaluation.reason == "required field 'duration' is null"


@pytest.mark.parametrize("operator,value,expected", [
    ("equals", {}, "worldwide"),
    ("not_equals", {}, "local"),
    ("is_true", "true", None),
    ("is_false", "false", None),
])
def test_scalar_and_boolean_operators_reject_malformed_extracted_values(
    operator, value, expected
):
    rule = make_rule(operator=operator, value=expected, unit=None)
    evaluation = evaluate_rule(rule, {"duration": value})
    assert evaluation.outcome == "could_not_evaluate"
    assert "malformed or unsupported" in evaluation.reason


@pytest.mark.parametrize("operator,value,expected", [
    ("greater_than", "eighteen months", 12),
    ("greater_than_or_equal", {}, 12),
    ("less_than", [], 12),
    ("less_than_or_equal", "unknown", 12),
    ("contains", {}, "client"),
    ("in", {}, ["worldwide"]),
])
def test_all_operator_families_reject_incompatible_extracted_values(
    operator, value, expected
):
    rule = make_rule(operator=operator, value=expected, unit=None)
    evaluation = evaluate_rule(rule, {"duration": value})
    assert evaluation.outcome == "could_not_evaluate"
    assert "malformed or unsupported" in evaluation.reason


def test_invalid_configured_operand_does_not_become_extracted_uncertainty():
    with pytest.raises(ValueError, match="numeric"):
        make_rule(operator="greater_than", value="twelve", unit=None)


def test_boolean_and_empty_present_values_are_not_matched_not_uncertain():
    assert evaluate_rule(make_rule(operator="is_true", value=None, unit=None), {"duration": False}).outcome == "not_matched"
    assert evaluate_rule(make_rule(operator="is_false", value=None, unit=None), {"duration": True}).outcome == "not_matched"
    assert evaluate_rule(make_rule(operator="is_not_empty", value=None, unit=None), {"duration": []}).outcome == "not_matched"


def test_multiple_rules_preserve_matched_not_matched_and_uncertain():
    matched = make_rule(rule_id="nc.matched", operator="greater_than", value=12)
    not_matched = make_rule(rule_id="nc.not_matched", field="geographical_scope", operator="equals",
                            value="worldwide", unit=None)
    uncertain = make_rule(rule_id="nc.uncertain", field="solicitation", operator="is_true", value=None, unit=None)
    evaluations = evaluate_all_rules(
        "non_compete",
        {"duration": "18 months", "geographical_scope": "local"},
        [matched, not_matched, uncertain],
    )
    by_id = {item.rule_id: item.outcome for item in evaluations}
    assert by_id == {
        "nc.matched": "matched",
        "nc.not_matched": "not_matched",
        "nc.uncertain": "could_not_evaluate",
    }
    findings = evaluate_rules("non_compete", {"duration": "18 months", "geographical_scope": "local"},
                              [matched, not_matched, uncertain])
    assert [finding.rule_id for finding in findings] == ["nc.matched"]


def test_rule_evaluation_is_deterministic():
    rule = make_rule()
    facts = {"duration": None}
    first = [evaluate_rule(rule, facts).to_dict() for _ in range(3)]
    second = [evaluate_rule(rule, facts).to_dict() for _ in range(3)]
    assert first == second
    assert evaluate_all_rules("non_compete", {"duration": "18 months"}, [rule]) == evaluate_all_rules(
        "non_compete", {"duration": "18 months"}, [rule]
    )


def test_evaluator_emits_structured_finding():
    rule = make_rule()
    findings = evaluate_rules("non_compete", {"duration": "five (5) years"}, [rule])
    assert len(findings) == 1
    result = findings[0].to_dict()
    assert result["rule_id"] == rule.rule_id
    assert result["extracted_value"] == "five (5) years"
    assert result["rule_condition"] == "duration > 12 months"
    assert result["evidence"] == "five (5) years"
    assert "potential concern" in result["explanation"]


def test_matching_rule_with_no_exception_field_is_not_configured():
    rule = make_rule(exception_field=None)
    findings = evaluate_rules("non_compete", {"duration": "five years"}, [rule])
    assert len(findings) == 1
    assert findings[0].exception_status == "not_configured"
    assert findings[0].exception_value is None
    assert findings[0].severity == "high"


def test_matching_rule_with_nonempty_exception_is_preserved():
    rule = make_rule()
    facts = {"duration": "five years", "exceptions": ["written consent"]}
    findings = evaluate_rules("non_compete", facts, [rule])
    assert len(findings) == 1
    finding = findings[0]
    assert finding.exception_status == "extracted_unverified"
    assert finding.exception_field == "exceptions"
    assert finding.exception_value == ["written consent"]
    assert finding.severity == "high"
    assert "applicability has not been established" in finding.explanation
    assert "does not dismiss" in finding.explanation
    assert finding.exception_status != "validated_applicable"


def test_matching_rule_with_empty_exception_is_empty_not_extracted():
    rule = make_rule()
    findings = evaluate_rules("non_compete", {"duration": "five years", "exceptions": []}, [rule])
    assert findings[0].exception_status == "empty"
    assert findings[0].exception_value == []


def test_matching_rule_with_missing_exception_field_is_not_extracted():
    rule = make_rule()
    findings = evaluate_rules("non_compete", {"duration": "five years"}, [rule])
    assert findings[0].exception_status == "not_extracted"
    assert findings[0].exception_value is None


def test_null_exception_could_not_evaluate_is_distinct_from_missing():
    rule = make_rule()
    findings = evaluate_rules("non_compete", {"duration": "five years", "exceptions": None}, [rule])
    assert findings[0].exception_status == "could_not_evaluate"
    assert findings[0].exception_status != "not_extracted"


def test_non_matching_rule_does_not_emit_finding_when_exception_is_populated():
    rule = make_rule()
    facts = {"duration": "six months", "exceptions": ["written consent"]}
    assert evaluate_rules("non_compete", facts, [rule]) == []
    evaluation = evaluate_rule(rule, facts)
    assert evaluation.outcome == "not_matched"
    assert evaluation.exception_status == "extracted_unverified"
    assert evaluation.exception_value == ["written consent"]


def test_wrong_category_is_not_evaluated():
    rule = make_rule()
    assert evaluate_rules("confidentiality", {"duration": "five years"}, [rule]) == []


def test_evaluator_keeps_empty_exceptions_and_supplies_source_evidence():
    rule = make_rule()
    findings = evaluate_rules("non_compete", {"duration": "five years", "exceptions": []}, [rule], evidence={"duration": "five (5) years following termination"})
    assert findings[0].evidence == "five (5) years following termination"


def test_load_rules_returns_configured_criteria():
    rules = load_rules()
    assert len(rules) == 3
    assert all(isinstance(rule, ConcernRule) for rule in rules)
    clause_types = {rule.clause_type for rule in rules}
    assert clause_types == {"non_compete", "termination_notice", "confidentiality"}


def test_in_memory_rule_store_filters_by_clause_type():
    rule_nc1 = make_rule(rule_id="nc.1", clause_type="non_compete")
    rule_nc2 = make_rule(rule_id="nc.2", clause_type="non_compete")
    rule_term = make_rule(rule_id="term.1", clause_type="termination_notice")
    store = InMemoryRuleStore([rule_nc1, rule_nc2, rule_term])

    nc_rules = store.get_rules_for_clause_type("non_compete")
    assert [r.rule_id for r in nc_rules] == ["nc.1", "nc.2"]

    term_rules = store.get_rules_for_clause_type("termination_notice")
    assert [r.rule_id for r in term_rules] == ["term.1"]

    assert store.get_rules_for_clause_type("confidentiality") == []


def test_in_memory_rule_store_defaults_to_all_fixture_rules():
    store = InMemoryRuleStore()
    nc_rules = store.get_rules_for_clause_type("non_compete")
    assert len(nc_rules) == 1
    assert nc_rules[0].rule_id == "demo.non_compete.duration_over_12_months"
