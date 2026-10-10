"""Typed, deterministic review-rule definitions and evaluation helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, Sequence


SUPPORTED_OPERATORS = {
    "equals", "not_equals", "greater_than", "greater_than_or_equal",
    "less_than", "less_than_or_equal", "contains", "in",
    "is_true", "is_false", "is_not_empty",
}
SEVERITIES = {"info", "low", "medium", "high"}
RuleOutcome = Literal["matched", "not_matched", "could_not_evaluate"]
MATCHED: RuleOutcome = "matched"
NOT_MATCHED: RuleOutcome = "not_matched"
COULD_NOT_EVALUATE: RuleOutcome = "could_not_evaluate"
EXCEPTION_NOT_CONFIGURED = "not_configured"
EXCEPTION_NOT_EXTRACTED = "not_extracted"
EXCEPTION_EMPTY = "empty"
EXCEPTION_COULD_NOT_EVALUATE = "could_not_evaluate"
EXCEPTION_EXTRACTED_UNVERIFIED = "extracted_unverified"


@dataclass(frozen=True)
class ConcernRule:
    """A configured review criterion; thresholds are policy, not legal facts."""

    rule_id: str
    clause_type: str
    field: str
    operator: str
    finding_type: str
    severity: str
    description: str
    source: str
    value: Any = None
    unit: str | None = None
    exception_field: str | None = "exceptions"

    def __post_init__(self) -> None:
        for name in ("rule_id", "clause_type", "field", "finding_type", "description", "source"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} is required")
        if self.operator not in SUPPORTED_OPERATORS:
            raise ValueError(f"unsupported operator: {self.operator}")
        if self.severity not in SEVERITIES:
            raise ValueError(f"unsupported severity: {self.severity}")
        if self.operator not in {"is_true", "is_false", "is_not_empty"} and self.value is None:
            raise ValueError(f"value is required for {self.operator}")
        if self.unit is not None and self.unit not in {"day", "month", "year"}:
            raise ValueError("unit must be day, month, or year")

    @property
    def rule_condition(self) -> str:
        symbols = {"equals": "==", "not_equals": "!=", "greater_than": ">",
                   "greater_than_or_equal": ">=", "less_than": "<",
                   "less_than_or_equal": "<=", "contains": "contains",
                   "in": "in", "is_true": "is true", "is_false": "is false",
                   "is_not_empty": "is not empty"}
        expected = "" if self.value is None else f" {self.value}"
        unit = f" {self.unit}s" if self.unit else ""
        return f"{self.field} {symbols[self.operator]}{expected}{unit}".strip()


def _duration_value(value: Any, target_unit: str | None) -> float:
    """Convert extractor Normalizer output or raw duration text to target units."""
    raw = value.get("raw") if isinstance(value, Mapping) else value
    parsed = _parse_duration(raw) if isinstance(raw, str) else None
    if isinstance(value, Mapping):
        if parsed is None:
            parsed = value.get("normalized", value)
    if isinstance(parsed, Mapping):
        unit = parsed.get("unit")
        amount = parsed.get("value")
    else:
        unit = amount = None
    if amount is None or unit is None:
        raise ValueError("malformed duration")
    factors = {"day": 1.0, "month": 30.0, "year": 365.0}
    target = target_unit or unit
    return float(amount) * factors[unit] / factors[target]


def _parse_duration(raw: str) -> dict[str, Any] | None:
    """Parse known duration words before digits (e.g. 'ninety' before its '9')."""
    text = raw.casefold().strip()
    if text == "immediately":
        return {"value": 0, "unit": "day"}
    unit = next((name for name in ("year", "month", "day") if name in text), None)
    if unit is None:
        return None
    words = {"twenty-four": 24, "eighteen": 18, "twelve": 12, "ninety": 90,
             "sixty": 60, "thirty": 30, "ten": 10, "nine": 9,
             "eight": 8, "seven": 7, "six": 6, "five": 5,
             "four": 4, "three": 3, "two": 2, "one": 1}
    for word, number in words.items():
        if re.search(rf"\b{re.escape(word)}\b", text):
            return {"value": number, "unit": unit}
    match = re.search(r"\b(\d+)\b", text)
    return {"value": int(match.group(1)), "unit": unit} if match else None


@dataclass(frozen=True)
class RuleEvaluation:
    """Outcome of evaluating one configured rule against extracted facts."""

    rule_id: str
    clause_type: str
    field: str
    outcome: RuleOutcome
    reason: str
    extracted_value: Any = None
    exception_status: str = EXCEPTION_NOT_CONFIGURED
    exception_field: str | None = None
    exception_value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def exception_state(rule: ConcernRule, facts: Mapping[str, Any]) -> tuple[str, str | None, Any]:
    """Describe extracted exception text without judging legal applicability."""
    field_name = rule.exception_field
    if not field_name:
        return EXCEPTION_NOT_CONFIGURED, None, None
    if field_name not in facts:
        return EXCEPTION_NOT_EXTRACTED, field_name, None
    value = facts[field_name]
    if value is None:
        return EXCEPTION_COULD_NOT_EVALUATE, field_name, None
    if not value:
        return EXCEPTION_EMPTY, field_name, value
    return EXCEPTION_EXTRACTED_UNVERIFIED, field_name, value


def evaluate_rule(rule: ConcernRule, facts: Mapping[str, Any]) -> RuleEvaluation:
    """Evaluate one rule. Missing or malformed facts are could_not_evaluate, not a negative match."""
    evaluation = _evaluate_condition(rule, facts)
    status, field_name, value = exception_state(rule, facts)
    return replace(
        evaluation,
        exception_status=status,
        exception_field=field_name,
        exception_value=value,
    )


def _evaluate_condition(rule: ConcernRule, facts: Mapping[str, Any]) -> RuleEvaluation:
    extracted = facts[rule.field] if rule.field in facts else None
    if rule.field not in facts:
        return RuleEvaluation(
            rule_id=rule.rule_id,
            clause_type=rule.clause_type,
            field=rule.field,
            outcome=COULD_NOT_EVALUATE,
            reason=f"required field '{rule.field}' is missing",
        )
    if facts[rule.field] is None:
        return RuleEvaluation(
            rule_id=rule.rule_id,
            clause_type=rule.clause_type,
            field=rule.field,
            outcome=COULD_NOT_EVALUATE,
            reason=f"required field '{rule.field}' is null",
            extracted_value=None,
        )
    actual = facts[rule.field]
    expected = rule.value
    op = rule.operator
    try:
        if rule.unit:
            actual = _duration_value(actual, rule.unit)
        if op == "equals":
            matched = actual == expected
        elif op == "not_equals":
            matched = actual != expected
        elif op == "greater_than":
            matched = actual > expected
        elif op == "greater_than_or_equal":
            matched = actual >= expected
        elif op == "less_than":
            matched = actual < expected
        elif op == "less_than_or_equal":
            matched = actual <= expected
        elif op == "contains":
            matched = expected in actual
        elif op == "in":
            matched = actual in expected
        elif op == "is_true":
            matched = actual is True
        elif op == "is_false":
            matched = actual is False
        elif op == "is_not_empty":
            matched = bool(actual)
        else:
            raise ValueError(f"unsupported operator: {op}")
    except ValueError as exc:
        if str(exc).startswith("unsupported operator"):
            raise
        return RuleEvaluation(
            rule_id=rule.rule_id,
            clause_type=rule.clause_type,
            field=rule.field,
            outcome=COULD_NOT_EVALUATE,
            reason=f"extracted value for '{rule.field}' is malformed or unsupported",
            extracted_value=extracted,
        )
    except (TypeError, KeyError):
        return RuleEvaluation(
            rule_id=rule.rule_id,
            clause_type=rule.clause_type,
            field=rule.field,
            outcome=COULD_NOT_EVALUATE,
            reason=f"extracted value for '{rule.field}' is malformed or unsupported",
            extracted_value=extracted,
        )
    if matched:
        return RuleEvaluation(
            rule_id=rule.rule_id,
            clause_type=rule.clause_type,
            field=rule.field,
            outcome=MATCHED,
            reason="condition is satisfied",
            extracted_value=extracted,
        )
    return RuleEvaluation(
        rule_id=rule.rule_id,
        clause_type=rule.clause_type,
        field=rule.field,
        outcome=NOT_MATCHED,
        reason="required facts are available and the condition is not satisfied",
        extracted_value=extracted,
    )


def rule_matches(rule: ConcernRule, facts: Mapping[str, Any]) -> bool:
    """Return True only when extracted facts are sufficient and satisfy the rule."""
    return evaluate_rule(rule, facts).outcome == MATCHED


def evaluate_all_rules(
    clause_type: str,
    facts: Mapping[str, Any],
    rules: Sequence[ConcernRule],
) -> list[RuleEvaluation]:
    """Evaluate every applicable configured rule independently of retrieval ranking."""
    return [evaluate_rule(rule, facts) for rule in rules if rule.clause_type == clause_type]


@dataclass(frozen=True)
class ConcernFinding:
    rule_id: str
    clause_type: str
    finding_type: str
    severity: str
    field: str
    extracted_value: Any
    rule_condition: str
    explanation: str
    evidence: Any
    source: str
    exception_status: str = EXCEPTION_NOT_CONFIGURED
    exception_field: str | None = None
    exception_value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def evaluate_rules(
    clause_type: str,
    facts: Mapping[str, Any],
    rules: Sequence[ConcernRule],
    *,
    evidence: Mapping[str, Any] | None = None,
) -> list[ConcernFinding]:
    """Emit findings for matched rules. Extracted exceptions are recorded, not used to dismiss matches."""
    findings = []
    for rule in rules:
        evaluation = evaluate_rule(rule, facts)
        if rule.clause_type != clause_type or evaluation.outcome != MATCHED:
            continue
        raw = facts[rule.field]
        extracted = raw.get("raw", raw) if isinstance(raw, Mapping) else raw
        explanation = (
            f"The extracted value for {rule.field} matches a configured review criterion. "
            "This is a potential concern according to the configured review policy."
        )
        if evaluation.exception_status == EXCEPTION_EXTRACTED_UNVERIFIED:
            explanation += (
                " An extracted exception is present, but its applicability has not been established "
                "and it does not dismiss this finding."
            )
        findings.append(ConcernFinding(
            rule_id=rule.rule_id,
            clause_type=clause_type,
            finding_type=rule.finding_type,
            severity=rule.severity,
            field=rule.field,
            extracted_value=extracted,
            rule_condition=rule.rule_condition,
            explanation=explanation,
            evidence=(evidence or {}).get(rule.field, extracted),
            source=rule.source,
            exception_status=evaluation.exception_status,
            exception_field=evaluation.exception_field,
            exception_value=evaluation.exception_value,
        ))
    return findings


class RuleStore(Protocol):
    """Protocol for rule-store implementations providing rules by clause type."""

    def get_rules_for_clause_type(self, clause_type: str) -> Sequence[ConcernRule]: ...


def load_rules(path: str | Path | None = None) -> list[ConcernRule]:
    """Load configured review rules from JSON fixture."""
    fixture = Path(path) if path else Path(__file__).parent / "review_rules.json"
    records = json.loads(fixture.read_text(encoding="utf-8"))
    return [ConcernRule(**{k: v for k, v in record.items() if k != "rationale"}) for record in records]


class InMemoryRuleStore:
    """Store for configured review rules partitioned by clause type."""

    def __init__(self, rules: Sequence[ConcernRule] | None = None):
        self.rules = list(rules if rules is not None else load_rules())

    def get_rules_for_clause_type(self, clause_type: str) -> list[ConcernRule]:
        """Return all configured rules applicable to a clause type."""
        return [rule for rule in self.rules if rule.clause_type == clause_type]

