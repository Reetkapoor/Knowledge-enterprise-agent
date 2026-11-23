# Evaluation data formats

Everything here marked `illustrative: true` is a **format example, not data**. `run_eval` and the verifier scripts skip
such rows. Do not report numbers computed from them.

## 1. Question dataset (`run_eval`, JSON list or `{"questions": [...]}`, or CSV)

| field | required | meaning |
|---|---|---|
| `id` | yes | unique |
| `question` | yes | |
| `answerability` | yes | `full` / `partial` / `unanswerable` / `conflicting` |
| `query_type` | yes | gold `SIMPLE` or `COMPLEX` (Adaptive RAG classification accuracy) |
| `reference_answer` | no | human answer (used by `--judge`) |
| `key_facts` | no | strings a correct answer must contain (deterministic key-fact recall) |
| `reference_sub_questions` | no | gold decomposition for COMPLEX questions |
| `relevant_evidence` | no | `[{"document": "file.pdf", "pages": [4]}]` gold passages (retrieval relevance / sufficiency) |
| `expected_citations` | no | same shape; gold citation agreement |
| `expected_claims` | no | `[{"text", "label"}]` human claim labels (documentation / for building claim sets) |

In CSV, the list/object columns (`key_facts`, `reference_sub_questions`, `relevant_evidence`, `expected_citations`,
`expected_claims`) hold JSON strings. `document` must equal the uploaded file name.

Suggested mix for ~150-200 questions: fully answerable (simple + complex), partially answerable, unanswerable,
conflicting-across-documents. Have a second person label ~20% to estimate annotator agreement.

## 2. Labelled claims (`eval_verifier`, `calibrate_nli`; JSONL / JSON / CSV)

`{"claim": "...", "evidence": ["passage", ...], "label": "SUPPORTED|CONTRADICTED|INSUFFICIENT"}`.
CSV: separate passages with ` || ` and use a `label` column. You can also fill the `human_label` column of the
`claims_for_annotation_*.csv` files that `run_eval` exports - these claims come from the system's own answers, which
is the right population for measuring unsupported-claim rate with human labels.
Use a held-out split: calibrate thresholds on one part, report on the other (`calibrate_nli` does this).
