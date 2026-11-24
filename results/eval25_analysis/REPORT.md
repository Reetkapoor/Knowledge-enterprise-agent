# Knowledge Enterprise Agent V2 — E1–E5 Diagnostic Ablation Report

**Data source:** User-uploaded `eval25.zip` containing completed 25-question × 5-experiment outputs. All numbers below were calculated or checked against those files, not invented. **Status: preliminary**, not publication-grade gold-standard results.

## Execution integrity

- **125/125 attempts recorded**, of which **115 returned normal pipeline responses** and **10 raised API errors** (7 HTTP 503 and 3 HTTP 429).
- Each experiment attempted the same 25 questions. The errors are not uniformly distributed and some metrics exclude them, creating incomparable subsets.
- Original indexed corpus has five PDFs; the provisional benchmark uses annotated evidence from only two. `gold_labels_source_validated=false`.
- `judge_score` was not run; no independent answer correctness, human-verified citation accuracy, or claim-label F1 was measured.

## Observed ablation comparison

| Metric | E1 | E2 | E3 | E4 | E5 |
|---|---:|---:|---:|---:|---:|
| Successful / attempted | 24/25 | 22/25 | 23/25 | 24/25 | 22/25 |
| API errors | 1 | 3 | 2 | 1 | 3 |
| Router accuracy | N/A | 90.9% | N/A | N/A | 95.5% |
| Evidence recall after selection | 81.8% | 77.5% | 95.2% | 81.8% | 95.5% |
| Retrieval precision after selection | 39.1% | 35.7% | 86.7% | 40.9% | 86.4% |
| Key-fact exact-string recall | 58.2% | 56.2% | 68.7% | 29.2% | 44.2% |
| System-estimated unsupported-claim rate | 61.7% | 56.0% | 52.5% | 56.4% | 48.1% |
| System-estimated citation precision | 45.9% | 47.1% | 54.8% | 44.8% | 56.9% |
| Abstention decision accuracy | 85.7% | 80.0% | 95.0% | 61.9% | 90.0% |
| Excess abstention on full questions | 15.8% | 22.2% | 5.6% | 42.1% | 10.0% |
| Mean latency (s) | 11.9 | 18.4 | 70.6 | 34.5 | 172.4 |
| Median latency (s) | 4.1 | 8.6 | 25.3 | 13.4 | 31.5 |
| Avg LLM calls | 1.0 | 2.0 | 2.7 | 1.7 | 5.0 |

### Key conclusions

1. **E3 Corrective RAG had the strongest evidence coverage and output completeness among the available measurements.** Its gold-page evidence recall was **95.2%** vs **81.8%** in E1, and retrieval precision after filtering was **86.7%** vs **39.1%** in E1. It answered 17/25 compared with E1 16/25, but failed two queries (not the same failures), so do not call it universally more accurate.
2. **E5 matched E3's evidence recall (95.5%) and had 86.4% retrieval precision**, but had 3 API failures and a median latency of roughly 31.5 s. **No basis yet to say E5 is the best overall system.**
3. **E4 Verification-only over-abstained.** It abstained on **8 of 19 successfully processed fully answerable questions (42.1%)** versus **3/19 (15.8%)** for E1. E5 improved that to **2/20 (10%)**, but E5's two unanswerable questions both failed with HTTP 429, so its abstention accuracy is based on the fully answerable set only.
4. **E2 Adaptive routing**: 20 of 22 successfully routed cases agreed with expected SIMPLE/COMPLEX, or **90.9%**. E5 had 21/22, **95.5%**. Decomposition recall/precision of 100% reflects the automatic embedding-match metric on the subset with reference subquestions, not an independent quality judgment.
5. **Latency is strongly affected by extreme outliers and provider throttling.** E5 `ke-02` took **3,004.7 seconds (50.08 minutes)**, with **2,994.2 seconds classified in retrieval**. E3 `ke-24` took **641.1 seconds (10.69 minutes)**. Compare medians and per-stage traces, not only means.
6. **Key-fact recall is a brittle substring metric.** E1/E3/E5 answered `ke-02` correctly with “8.00 lakhs” but each scored **0.0** because the annotated key fact is “8”. Thus the exact-match metric underestimates some factually correct answers. The data also shows E4 removes or fails to verify answerable factual claims, contributing to its low output completeness.
7. **No human-labelled verification F1 or factual answer correctness** was produced. Unsupported claim and citation values are produced by the same NLI verifier known to make false contradictions; do not treat them as independent hallucination/citation correctness measurements.

### Errors requiring targeted retry

- E1 / ke-14 — HTTP 503
- E2 / ke-13 — HTTP 503
- E2 / ke-16 — HTTP 503
- E2 / ke-21 — HTTP 503
- E3 / ke-13 — HTTP 503
- E3 / ke-20 — HTTP 503
- E4 / ke-13 — HTTP 503
- E5 / ke-23 — HTTP 429
- E5 / ke-24 — HTTP 429
- E5 / ke-25 — HTTP 429

**Rerun caution:** Existing launcher marks these error rows completed and will skip them on resume; do **not** delete the entire results directory. Retry only the failed experiment/question pairs using a safe targeted rerun procedure, preserving the original run separately and including the new measurement date/config. Consider provider-rate-limited runs separately from RAG-quality failures.

## What is in this analysis package

- `clean_comparison.csv`: compact metrics plus denominator and latency diagnostics.
- `failed_executions.csv`: all 10 execution errors.
- `full_question_abstentions.csv`: the 18 full-answerable question/experiment combinations that abstained.
- `key_fact_metric_audit.csv`: a concrete false-negative case from the exact-string matching metric.
- PNG charts: quality and timing comparisons.

**Do not present these as final research statistics until reference labels have been checked, missing API failures rerun, and human-validation samples scored.**
