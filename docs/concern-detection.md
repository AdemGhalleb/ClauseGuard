# Concern detection and review-criteria retrieval

ClauseGuard separates extraction from policy evaluation:

```text
classified clause
→ existing category-specific extractor
→ normalized extracted facts
→ all applicable configured review criteria for clause type
→ deterministic rule evaluation
→ structured findings
(optional supplementary criteria retrieval for supporting context)
```

The classifier currently returns a mapping with `category`, `label_id`, `confidence`, and `probabilities`. `NuExtractExtractor.extract(text, category)` returns a JSON string matching the category schema in `legal-ai/extraction/llm_extractor.py`. The detector accepts that existing string or a parsed mapping and uses the existing duration normalizer. The classifier and extractor remain unchanged.

## Responsibilities and limits

Extraction answers **what the clause says**. Concern detection answers **whether extracted facts match a configured review criterion**.

To prevent silent false negatives, rule selection is separated from retrieval:
1. **Rule selection**: Resolves **all** configured review rules applicable to the extracted `clause_type` from the rule store (`InMemoryRuleStore`). An applicable rule is never skipped due to lexical ranking, query mismatches, or top-k truncation.
2. **Deterministic evaluation**: Evaluates each applicable rule independently of retrieval ranking and `top_k`. The evaluator records a tri-state outcome from the extracted value and configured condition. There is no LLM judge, opaque risk score, external API, or vector database in this foundation.
3. **Supplementary retrieval**: Optional criteria retrieval provides supporting rule context, rationale, and source metadata. The `top_k` parameter controls the number of supplementary retrieved documents returned in `DetectionResult.retrieved_rule_ids` and does not limit the rules evaluated. Empty retrieval results do not skip evaluation.

The initial retrieval implementation is local and replaceable through the `RuleRetriever` protocol. It filters by clause type and optional metadata, then orders applicable documents by simple token overlap. `retrieve_for_facts` builds its query only from the clause category and extracted fields; it does not send an entire contract to retrieval. This is a transparent local retrieval baseline, not an embedding-based semantic search system.


## Rules and configuration

`legal-ai/rules/engine.py` defines `ConcernRule`. A rule includes a stable ID, clause type, field, operator, expected value, optional duration unit, finding type, severity, description, source, and optional exception field. Supported operations are equality/inequality, numeric comparisons, membership/containment, boolean checks, and non-empty checks. Rules are typed Python data; arbitrary expressions are never evaluated.

Rule selection is provided by `RuleStore` and implemented by `InMemoryRuleStore`, which loads rules via `load_rules()` from `review_rules.json` and partitions/filters them by `clause_type`. `InMemoryRuleRetriever` also implements `get_rules_for_clause_type()` directly from indexed documents.

The checked-in `legal-ai/rules/review_rules.json` entries are **demonstration configuration**, not legal thresholds or statements of law. Their IDs begin with `demo.` and their source is `demo_configuration`. Replace or supplement them with a team's documented review policy before relying on the results. Duration comparisons normalize to days using 30 days per month and 365 days per year; these are arithmetic comparison conventions, not legal definitions.

The same records are the small rule knowledge base. Retrieval documents include the rule ID, clause type, relevant field, condition, criterion description, rationale, severity, and source. This keeps retrieved rationale tied to the exact criterion evaluated.

To add a criterion:

1. Add a JSON record to `legal-ai/rules/review_rules.json` with a unique `rule_id`, extracted field, supported operator/value, finding metadata, rationale, and source.
2. Use a clearly identified demo source for examples. A production policy should identify its actual owner/source and should not imply legal authority without appropriate review.
3. Add a schema/evaluator test in `tests/test_concern_rules.py`, retrieval coverage in `tests/test_rule_retrieval.py`, and an end-to-end case in `tests/test_concern_detection.py` as applicable.

## Rule evaluation outcomes

`evaluate_rule` returns a `RuleEvaluation`. `evaluate_all_rules` evaluates every applicable configured rule for the clause type. `ConcernDetector.detect` exposes those records on `DetectionResult.evaluations` and in `DetectionResult.to_dict()["evaluations"]`. `rule_matches` remains a convenience that is true only for `matched`.

The three outcomes are:

| Outcome | Meaning |
| --- | --- |
| `matched` | Required facts are present and valid, and they satisfy the configured condition. |
| `not_matched` | Required facts are present and valid, and they do not satisfy the condition. |
| `could_not_evaluate` | Required facts are missing, null, malformed, or otherwise insufficient to decide. |

`could_not_evaluate` is not “no concern.” It means the engine could not determine whether the criterion applies. Treating it as a negative match would hide extraction gaps. Invalid rule configuration still raises `ValueError` during `ConcernRule` construction; that is not classified as a missing-fact outcome.

Worked example — missing duration, rule `duration > 12 months`:

- Duration `18 months` → `matched` (a finding is emitted).
- Duration `6 months` → `not_matched` (no finding; the rule was evaluated).
- Duration absent or `null` → `could_not_evaluate` (no finding; the evaluation record remains for review of extraction completeness).
- Duration `an extended period` (unparseable) → `could_not_evaluate`.

## Findings and exceptions

Each finding contains `rule_id`, `clause_type`, `finding_type`, `severity`, `field`, `extracted_value`, `rule_condition`, `explanation`, `evidence`, `source`, `exception_status`, `exception_field`, and `exception_value`. If the extracted value appears in the original clause text, the detector returns that exact substring as evidence. Otherwise it falls back to the extractor's raw value. The explanation says the value matches a configured review criterion and labels the result a potential concern according to the configured review policy.

A non-empty extracted exception does not suppress a matching rule. Existing category schemas extract exception lists but do not establish that an exception applies to a given condition. The evaluator therefore keeps the finding and records exception metadata:

| `exception_status` | Meaning |
| --- | --- |
| `not_configured` | The rule has no `exception_field`. |
| `not_extracted` | The configured exception field is absent from the extracted facts. |
| `empty` | The exception field is present and empty. |
| `could_not_evaluate` | The exception field is present but null, so exception content could not be read. |
| `extracted_unverified` | Exception text was extracted; applicability has not been established. |

There is no validated-applicable status in the current schema. Exception handling does not determine legal applicability, does not mark a finding resolved, and does not lower severity.

Worked example — extracted exception, same duration rule:

- Duration `five years`, `exceptions: ["written consent"]` → condition `matched`, finding preserved, `exception_status=extracted_unverified`.
- Duration `six months` with the same exception list → condition `not_matched`; no finding is created merely because exception text exists.

ClauseGuard surfaces potential concerns according to configured review criteria. It does not provide legal advice.

## Evaluation and tests

Run the focused tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_concern_rules.py tests/test_rule_retrieval.py tests/test_concern_detection.py tests/test_concern_evaluation.py -q
```

Run the labeled synthetic evaluation with:

```powershell
.\.venv\Scripts\python.exe legal-ai\evaluation\run_concern_demo.py
```

`legal-ai/evaluation/retrieval.py` provides `recall_at_k` over labeled retrieval cases and `binary_rule_metrics` (TP, FP, TN, FN, accuracy) over known extracted facts. The demo currently reports Recall@1 1.0 and rule counts TP=3, FP=0, TN=3, FN=0 across six supplied-fact examples. These are mechanics checks on hand-written synthetic cases only. Demonstration rules and synthetic test results do not establish real-world legal accuracy. Rule metrics assume extracted facts are correct and do not measure extraction quality. No representative contract evaluation or real policy quality result is claimed.

## Limitations

- Demo thresholds are illustrative configuration and are not authoritative legal standards.
- Rule evaluation is deterministic but only as accurate as the extracted fields and configured criteria.
- `could_not_evaluate` records an insufficient-fact outcome; it is not evidence that a concern is absent.
- Extracted exception text is stored as unverified metadata. The engine does not decide whether an exception is legally effective, applicable, or sufficient to eliminate a concern.
- Retrieval uses lexical token overlap and checked-in criteria only. No legal corpus or external source is retrieved. Retrieval ranking does not choose which rules run.
- Missing or malformed fields are not guessed by the rule engine; they surface as `could_not_evaluate`.
- Findings are review prompts, not legal advice or conclusions about legality.
