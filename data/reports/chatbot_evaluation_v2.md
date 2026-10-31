# Chatbot evaluation (v2)

Generated 2026-11-01T07:01:18+0800 with `python scripts/evaluate_chatbot.py`. Evaluation set: `data\evaluation\chatbot_eval_v2.json` (221 author-written cases + 15 DTC questions from the held-out chat validation split). Cases are evaluation-only; the training-overlap check is below.

| Metric | Current router | Previous core (baseline) |
|---|---:|---:|
| Pass rate (all checks) | 1.0 | 0.1525 (content checks only) |
| Critical-case pass rate | 1.0 | 0.2 |
| Permission leaks | 0 of 27 | 1 of 27 |
| Latency p50 / p95 (ms) | 167.9 / 251.8 | 93.1 / 313.7 |
| Mean lexical groundedness (reference answers) | 0.8947 over 50 | n/a |

Baseline rows apply the same content checks (required/forbidden text, abstention, intent) but not the evidence-type or citation checks, which the old core cannot produce.

## Retrieval

Cases with labelled sources: 29; recall@3 0.931, precision@3 0.3966, MRR 0.9483.

## By role

| Role | Cases | Router pass | Baseline pass |
|---|---:|---:|---:|
| admin | 55 | 1.0 | 0.1091 |
| customer | 181 | 1.0 | 0.1657 |

## By routing

`backend`: the production flow, with records retrieved by the Back-End's ChatRetrieval from the fixture database. `direct`: context injected as written, to test the AI service's own checks. `diagnostic_mode`: the Back-End's explicit diagnostic mode.

| Routing | Cases | Router pass | Baseline pass |
|---|---:|---:|---:|
| backend | 233 | 1.0 | 0.1502 |
| diagnostic_mode | 1 | 1.0 | 1.0 |
| direct | 2 | 1.0 | 0.0 |

## By category

| Category | Cases | Router pass | Baseline pass |
|---|---:|---:|---:|
| admin_operations | 11 | 1.0 | 0.0909 |
| diagnostics | 7 | 1.0 | 0.1429 |
| diagnostics_external | 15 | 1.0 | 0.0 |
| fleet_management | 18 | 1.0 | 0.0 |
| follow_up | 6 | 1.0 | 0.1667 |
| general | 5 | 1.0 | 0.2 |
| maintenance_knowledge | 23 | 1.0 | 0.8696 |
| model_limitations | 1 | 1.0 | 0.0 |
| out_of_scope | 5 | 1.0 | 1.0 |
| parts | 1 | 1.0 | 0.0 |
| permissions | 27 | 1.0 | 0.2222 |
| rental | 87 | 1.0 | 0.0115 |
| system_support | 30 | 1.0 | 0.0 |

## Router failures (0)

- none

## Training overlap

Compared with 2898 intent-training patterns: exact matches ['reg-greet-001', 'reg-greet-003', 'db-v-001', 'db-p-004']; near-duplicates (char-trigram Jaccard >= 0.9) none.

## Limits of this evaluation

- Questions and expected facts are author-written and checked against the code and references; they still need review by a second person.
- This set was also used while developing the router (routing rules and system-guide keywords were adjusted after earlier runs), so it is a regression/development set; add fresh, unseen questions to measure generalisation.
- Groundedness is a lexical overlap proxy, not a human judgement of factual support.
- Database answers are scored on fixture contexts shaped like the Back-End's payloads, not on the live MySQL database; retrieval runs the Back-End's ChatRetrieval rules on Back-End/tests/fixtures/chat_fixture_db.json, but the HTTP endpoint, session and SQL queries are not exercised here (see Back-End/tests/chat_integration.ps1 for that).
