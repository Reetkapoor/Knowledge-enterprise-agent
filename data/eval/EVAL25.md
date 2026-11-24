# 25-question E1–E5 automated diagnostic benchmark

This branch adds **25 provisional questions** for the user's university PDF corpus and a **checkpointed 125-execution ablation launcher**. The 25 questions are **not** a statistically sufficient basis for final research claims; their gold labels were drafted from **user-shared extracted snippets**, not independently checked against all five original PDFs.

## What is included?

- `data/eval/questions25.provisional.json`: 25 non-illustrative provisional cases (**20 full, 3 partial, 2 unanswerable**) covering AICTE TFW, scholarship rules, B.Sc. progression and degree rules, and cross-document comparisons.
- `scripts/run_eval25.py`: executes all **E1, E2, E3, E4, E5** using the repository's existing `ExperimentRunner` and `summarize`. Writes a durable checkpoint after every query and resumes without duplicate runs.
- `tests/test_eval25_setup.py`: checks dataset structure, count and resume/export behavior.

The dataset is grounded only in two files' evidence extracted from earlier real logs:

- `B_Sc_Acad_Sys_Aug23_ecddd80d28.pdf`, p. 15
- `Scholarship_AY_2026_27_website_Apr_1_6c85696a82.pdf`, pp. 2–3

The other three uploaded PDFs **were not available** for independent source validation. No conflicting-information question has been fabricated. Partial/unanswerable labels may change after corpus-wide review.

## How to execute in Windows PowerShell

From the repository root with the project's virtual environment activated and its existing `.env` configured **privately** (Gemini or other supported LLM):

```powershell
python -m pytest
python -m scripts.run_eval25 --check
python -m scripts.run_eval25 --limit 2 --out results/eval25-smoke
python -m scripts.run_eval25 --out results/eval25 --max-minutes 120
```

**The 2-question smoke test is 10 executions.** The full invocation performs up to **125 executions** in sequence, subject to the 120-minute cap. If stopped, run exactly the same command again to resume the remaining pairs:

```powershell
python -m scripts.run_eval25 --out results/eval25 --max-minutes 120
```

`--check` reads and lists dataset cases without executing model calls. If document filenames are not identical to the ones indexed, the full runner **fails before the first query**. The document registry, ChromaDB, local embedding/NLI models, and configured LLM must be available on the host. **Do not commit/upload `.env`, indexed PDF files containing sensitive data, or credentials.**

## Outputs

After each completed query:

- `results/eval25/completed.jsonl`: per-query checkpoint, including errors (immutable records).
- `results/eval25/per_question_E1.csv` through `per_question_E5.csv`: answers, traces-derived metrics and errors for completed cases.
- `results/eval25/summary.csv`: current summary by experiment; **may be partial**.
- `results/eval25/claims_for_annotation_E*.csv`: verifier-extracted claim candidates. **Human_label must be filled before claiming human-verification precision, recall or F1.**
- `results/eval25/run_config.json`: dataset checksum, indexed document IDs, model settings and experiment flags, used to prevent inconsistent resuming.
- `results/eval25/RUN_STATUS.json`: completion / error counts and caution labels.

All E1–E5 runs use the same original question wording, fresh session IDs, and one shared index. Claim metrics are NLI-estimated, including post-hoc measurements for modes with verification disabled; **these are not independently verified factual accuracy scores**. `key_fact_recall` is normalized *substring* matching, not semantic correctness. A partial answer can be right while not matching all reference substrings.

## What can and cannot be concluded

The runner measures:

- Route accuracy and decomposition similarity (Adaptive modes).
- Gold-page overlap precision/recall and corrective retry outcomes (where gold pages supplied).
- System-estimated unsupported claims, citation validation/coverage.
- Abstention decisions, latency, token usage, LLM calls, errors.

It **does not** provide validated claim-level precision/recall/F1 without human labels, or a defensible independently judged answer-correctness score. In particular the project's existing NLI verifier has shown **false contradictions between distinct scholarship sections and B.Sc. regulations**. Fix or flag the problem and re-run the same benchmark when testing a corrected verifier.

Do not compare 25-question results to the previously reported 3-question diagnostics as if they were from the same experiment, or label them as 125 *successful* executions if errors remain.

## Why execution wasn't started through GitHub

A GitHub app connection can edit and read repository files but **cannot execute a long-running FastAPI/ChromaDB local runtime or access the user's local `.env` and PDFs**. Running the command above on the configured machine is the necessary step for actual 125-query metrics. No numerical E1–E5 results are prefilled or fabricated in this branch.
