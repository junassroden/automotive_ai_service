# Automotive Chatbot Knowledge and Retrieval Evaluation

## Result and scope

The knowledge pipeline and troubleshooting answer structure were improved without
retraining the intent classifier or maintenance model, changing API routes, or
weakening the maintenance-model safeguard.

The fixed regression set contains **126 questions**. The deterministic evaluator
reports 1.0000 top-1 expected-topic/abstention accuracy, 1.0000 expected-topic
recall at six/abstention accuracy, 1.0000 abstention accuracy on unsupported
questions, 1.0000 presence of expected safety terms in 17 selected cases, and
1.0000 presence of explicit uncertainty language in six vehicle-specific cases.
These are curated retrieval and response-marker checks, not independent evidence
of mechanical correctness. The evaluator's manual review fields remain unscored.
See the machine-readable result at
[`chatbot_knowledge_evaluation.json`](chatbot_knowledge_evaluation.json) and
fixed questions at
[`chatbot_knowledge_eval_questions.json`](chatbot_knowledge_eval_questions.json).

All 45 Django tests pass, and `python manage.py check` reports no issues.

## Findings and root causes

- The original vector index used document title, keywords, and component functions,
  omitting the substantive causes, symptoms, checks, and diagnostic text. Relevant
  evidence therefore did not contribute fully to ranking.
- `retrieve_knowledge` returned only the single top document. The answer formatter
  then emitted a short generic list, preventing it from assembling a useful
  troubleshooting explanation from multiple relevant sections.
- The original overheating record listed causes generically and lacked
  distinguishing symptoms, test order, workshop checks, vehicle differences, and
  explicit evidence gaps.
- The API did not generate prose from a language model. Answer depth was limited
  by the knowledge record and deterministic formatter; there is no evidence that
  an output token limit caused the reported short answer.
- `retrieval_score` is a heuristic lexical/vector match score, not a probability.
  The existing `confidence` is the classifier's top `predict_proba` output, but it
  has not been calibrated or validated as a probability of correct intent or
  answer. The classifier was preserved.
- `source_count` counted references stored in the selected document; it did not
  mean those sources had been fetched or verified for the answer. The response
  now distinguishes stored pointers from fetched references and calls out a
  missing exact-vehicle source.

## Changes made

- [`ai_api/automotive_knowledge.py`](../../ai_api/automotive_knowledge.py):
  indexes all structured technical sections as passages, ranks by lexical and
  vector matches, applies query-keyword relevance checks, returns up to six
  diversified related passages, and formats readable sections with line breaks.
  Unrelated/unsupported questions are more reliably rejected.
- [`data/automotive_knowledge.json`](../automotive_knowledge.json): expanded
  overheating knowledge with mechanisms, distinguishing clues, user-safe checks,
  workshop diagnostic sequence, vehicle-architecture differences, urgency, and
  explicit unknowns. Added separate general-knowledge records for hybrid/EV
  high-voltage safety, HVAC, exhaust/emissions/DPF, and motorcycle final drives.
  New records do not contain fabricated manufacturer procedures or specifications.
- [`ai_api/chatbot_service.py`](../../ai_api/chatbot_service.py): keeps the
  existing `knowledge_id` and API shape while adding related topic IDs, passage
  count, source-pointer count, fetched-source count, and score explanations.
- [`ai_api/tests.py`](../../ai_api/tests.py): adds regression checks for detailed
  overheating output, uncalibrated-score descriptions, unsupported-query
  abstention, and high-voltage safety boundaries.
- [`data_pipeline/evaluate_knowledge.py`](../../data_pipeline/evaluate_knowledge.py):
  repeatable fixed-set retrieval, abstention, safety-marker, and uncertainty
  evaluator.
- [`README.md`](../../README.md): documents how to run the evaluation and explains
  score and source limitations.

## Knowledge organization and provenance

The knowledge base is organized by system/category, with records for engine
misfire and cooling, lubrication, fuel/ignition, timing, brakes, transmission and
drivetrain, tires, steering/suspension, electrical starting/charging, OBD-II,
fluids, filters and service, fleet operations, model/vehicle-health limits,
hybrid/EV high voltage, HVAC, exhaust/emissions, and motorcycle chain/belt/shaft
drive.

No manufacturer manual, official service procedure, publication date, or
vehicle-specific technical source was present in the inspected project knowledge
files for the new material. Attempts to retrieve NHTSA pages returned HTTP 403;
the attempted EPA OBD URL returned 404. No external content was therefore
represented as consulted or added as a verified citation. Overheating responses
now report `source_count: 0` and explicitly state the source gap. Other existing
records may display reference pointers, clearly marked as not fetched for the
response. Exact fluids, capacities, pressure thresholds, torque values, intervals,
pinouts, code-specific tests, and manufacturer procedures remain unavailable
unless the exact vehicle's source is supplied and verified.

## Evaluation details

Run from the AI service directory:

```powershell
python -m data_pipeline.evaluate_knowledge
```

The 126-case set covers existing systems, ambiguous and vehicle-specific
questions, OBD-II, safety-critical cases, hybrid/EV, HVAC, exhaust, motorcycle
drive systems, and unsupported/out-of-domain questions. It is intentionally
matched to this small fixed knowledge base; the perfect retrieval results do not
estimate performance on arbitrary user questions.

The evaluator checks expected topic IDs or abstention, top-six topic retrieval,
presence of explicitly specified safety phrases, and uncertainty language where
vehicle-specific information is unavailable. It does **not** automatically
verify factual accuracy, source correctness, diagnostic logic, or completeness.
The JSON report includes a 0–2 human-review rubric for relevance/correctness,
coverage, reasoning/uncertainty, source traceability, unsupported claims, safety,
and unknown handling; no human scores have been entered.

## Same-question before/after

**Before** (the reported response): intent `engine_cooling`, confidence `0.1846`,
retrieval score `0.32`, source count `1`; causes were summarized as low coolant or
a leak, restricted airflow, a fan/thermostat problem, a water-pump issue, or
another fault.

**After**, from the same local `answer_chat` path: intent remains
`engine_cooling`; classifier score remains `0.1846`; heuristic retrieval score is
`0.48`; six passages are returned for `engine_overheating`; source count is `0`
because no verified reference was retrieved. The answer now differentiates
coolant loss, radiator/airflow restriction, fan-control faults, thermostat and
pump circulation, trapped air/cap problems, internal leakage, and sensor error;
it also gives safe checks, workshop tests, urgency, architecture caveats, and
focused follow-up questions. The score change reflects the revised index and
matching aliases, not a calibrated confidence or proven factual improvement.

## Limitations and next steps

- This remains a deterministic retrieval assistant, not a generative or
  independently validated diagnostic system.
- The current knowledge is general. Cars, motorcycles, pickups, vans, and
  minivans differ by year, engine, market, and configuration; exact procedures
  require source documents for the specific vehicle.
- The 126 curated cases measure retrieval and selected safeguards, not technician
  agreement or broad real-world answer quality. Have qualified automotive
  reviewers score the documented rubric before making expert-quality claims.
- Add legally reusable official owner/service information with title, publisher,
  version/date, URL or local path, section/page, applicable vehicles, and source
  verification status. Keep unsupported systems and source gaps explicit.
- OBD-II answers remain generic. No DTC should be treated as proof of a failed
  part; exact code definitions and test procedures require the make/model/year/
  engine-specific source.
- The maintenance model registry remains `blocked`; predictive maintenance and
  fleet model outputs remain unavailable. This work only improves general
  knowledge retrieval.
