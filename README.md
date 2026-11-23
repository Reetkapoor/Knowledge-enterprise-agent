# Knowledge Enterprise Agent: Adaptive + Corrective RAG with Claim-level Verification

Upload PDFs, ask questions, get answers with **document + page citations**. Two modes: **Baseline RAG** and
**Full Agentic RAG** (Adaptive routing, Corrective retrieval, claim-level NLI verification), each mechanism
independently switchable for ablations (E1-E5).

```
Question → [rewrite follow-ups] → [Adaptive: SIMPLE / COMPLEX(+sub-questions) / CONVERSATIONAL]
  → dense retrieval per (sub-)question → [Corrective: evaluate relevance+sufficiency → ONE refined retry]
  → grounded generation ([S#] markers) → [Verification: claims → evidence windows → NLI → citation check → correction]
  → final answer + citations   (original answer, claims, verdicts, trace all returned)
```
Brackets = optional. Any agentic stage that fails falls back to baseline behaviour for that stage and is listed in
`trace.fallbacks` (a missing API key still errors loudly).

## The three mechanisms

| | Module | What it does |
|---|---|---|
| Adaptive RAG | `services/query_router.py` | LLM structured output: `SIMPLE` (one retrieval), `COMPLEX` (2..`MAX_SUB_QUESTIONS` independent sub-questions, retrieved separately, merged round-robin, one answer), `CONVERSATIONAL` (reply without documents/citations). A "complex" question with <2 usable sub-questions is downgraded to SIMPLE. |
| Corrective RAG | `services/corrective_rag.py` | Evaluator judges each retrieved passage + sufficiency (question vs. passages only; never sees an answer). Verdict rules in code: none relevant → `IRRELEVANT`; relevant but not sufficient → `INSUFFICIENT_EVIDENCE`; sufficient but ≤50% passages relevant → `PARTIALLY_RELEVANT`; else `RELEVANT`. Retries for `CORRECTIVE_RETRY_ON` verdicts, at most `MAX_CORRECTIVE_RETRIES` (1) times, with an LLM-refined query; useful earlier passages are kept; the combined evidence is re-judged against the original question. Still insufficient → evidence + a note go to the generator, which must say what is missing. Applied per sub-question for COMPLEX queries. No web search. |
| Verification | `services/verification/` | (1) LLM extracts atomic claims + the `[S#]` markers on their sentence. (2) Each retrieved chunk is split into sentence windows (NLI models are length-limited); top windows per claim by embedding similarity, plus the windows of each cited chunk. (3) A **separate local NLI model** (`NLI_MODEL`, premise = window, hypothesis = claim) scores entailment/neutral/contradiction; `judge_claim` applies configurable thresholds → `SUPPORTED` / `CONTRADICTED` / `INSUFFICIENT` (+ `conflict` flag when both exceed thresholds). A numeric guard demotes `SUPPORTED` if a number in the claim is absent from the evidence (NLI models are weak on numbers). (4) Citations are valid only if the chunk exists, document id/page range are valid per the registry, and the cited chunk itself entails the claim; otherwise the citation is replaced by the best supporting retrieved chunk (metadata only, flagged `reattributed`). (5) Deterministic correction, no LLM rewrite: keep supported claims, replace a contradicted claim by a quote of the contradicting passage only if reliable, drop the rest and say so, abstain if nothing survives. |

The generator never grades its own answer. NLI thresholds are **uncalibrated defaults**: calibrate them (below).

## Using it

API: `POST /chat` with `{"question": "...", "mode": "baseline" | "full" | "custom", "flags": {"adaptive": true, "corrective": false, "verification": true}}`.
Response adds `mode`, `mechanisms`, `original_answer`, `verification` (claims, verdicts, evidence, citation checks) and `trace`
(router plan, per-unit corrective attempts with `retry_reason` / `refined_query`, retrieval calls, timings, token usage, fallbacks).

UI: sidebar **Pipeline mode** (Baseline / Full / Custom); each answer has *Verification details* and *Retrieval trace* panels.

## Evaluation and ablations

```bash
# smoke test with the illustrative rows (format examples only - never report these numbers)
python -m scripts.run_eval --dataset data/eval/questions.sample.json --include-illustrative --limit 3
# real run on your annotated dataset (E1 baseline, E2 +adaptive, E3 +corrective, E4 +verification, E5 full)
python -m scripts.run_eval --dataset data/eval/questions.json --experiments E1 E2 E3 E4 E5 [--judge]
python -m scripts.calibrate_nli --claims data/eval/labelled_claims.jsonl      # thresholds chosen on dev, scored held-out
python -m scripts.eval_verifier --claims data/eval/labelled_claims.jsonl      # claim P/R/F1, confusion, contradiction recall
```
Outputs (`results/<run_id>/`): `summary.csv` (one row per experiment), `per_question_E*.csv`, `claims_for_annotation_E*.csv`, `run_config.json`.
Formats: `data/eval/README.md`. All formulas are documented once in `backend/evaluation/metrics.py`.

How to read the numbers honestly:
- Retrieval relevance/sufficiency are measured against **your gold pages** (`relevant_evidence`), not the LLM evaluator.
- Claim-level metrics in `summary.csv` (unsupported rate, citation precision/coverage) are **NLI-estimated** on the *draft* answer; E1-E3 are measured post hoc with the same verifier so experiments are comparable. For human-validated precision/recall/F1, fill `human_label` in the exported claim CSVs (or a labelled claim set) and run `eval_verifier`.
- `key_fact_recall` is deterministic; `--judge` adds an LLM judge (same-provider bias possible; spot-check by hand).
- Abstention accuracy covers `full` + `unanswerable` questions only. Token counts are the provider-reported usage; latency includes network.
- Questions that crash are recorded (`n_errors`), not hidden.

## Known limitations
- No NLI model or LLM was run in the development sandbox (no model downloads): unit/integration tests use scripted fake LLM/NLI. **Run `pytest`, then a real smoke test, on your machine before trusting any behaviour with real models.**
- NLI on long, table-heavy or numeric evidence is unreliable; the numeric guard is a heuristic. Claim extraction quality depends on the LLM.
- Window-based NLI can miss claims that need facts from two distant windows (reported INSUFFICIENT rather than SUPPORTED).
- Corrective evaluation and verification add LLM calls (see `avg_llm_calls`, token columns).
- Scanned PDFs need OCR (not implemented). Conversation history is in memory.

```
Ingestion (on upload)                       Query (on every question)
PDF → PyMuPDF text (per page)               Question (+ session history)
    → clean → page-aware chunks                 → QueryProcessor  (follow-up → standalone question)
    → Sentence Transformers embeddings          → RetrievalService (embed query → Chroma top-K → score filter)
    → ChromaDB (persistent)                     → AnswerGenerator (LLM, numbered sources)
                                                → citations rebuilt from chunk metadata → ChatResponse
```

## Quick start

Requires Python 3.10+ (developed on 3.12), ~1 GB disk for PyTorch + the embedding model.

```bash
cd knowledge-enterprise-agent
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then edit .env and set LLM_API_KEY
```

Terminal 1, backend (first start downloads the embedding model, ~130 MB, from Hugging Face):
```bash
uvicorn backend.main:app --reload --port 8000
```
Terminal 2, frontend:
```bash
streamlit run frontend/app.py
```
Open http://localhost:8501. Interactive API docs: http://localhost:8000/docs.

### LLM provider
Set in `.env`; the key is read from the environment only and never logged or returned by any endpoint.

| Provider | `LLM_PROVIDER` | Notes |
|---|---|---|
| Anthropic | `anthropic` (default) | `LLM_MODEL=claude-sonnet-5-5` (any Messages-API model works) |
| OpenAI / any OpenAI-compatible server | `openai` | set `LLM_MODEL`, and `LLM_BASE_URL` for non-OpenAI hosts (e.g. a local server) |

Adding a provider = one small class in `backend/services/llm_service.py` implementing `generate(system, user)`.

## Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `LLM_API_KEY`, `LLM_MODEL`, `LLM_PROVIDER`, `LLM_BASE_URL` | – | LLM access |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local Sentence Transformers model (512-token window, fits 500-token chunks) |
| `EMBEDDING_QUERY_PREFIX` | bge query instruction | Prepended to **queries only**; set empty for models that don't use one |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 500 / 100 | In *approximate tokens* (words × `TOKENS_PER_WORD`, default 1.3) |
| `TOP_K` | 5 | Chunks retrieved per question (overridable per request) |
| `MIN_SIMILARITY` | 0.35 | Cosine-similarity floor; weaker chunks never reach the LLM. **Tune for your model/corpus** |
| `MAX_CONTEXT_TOKENS` | 3500 | Upper bound on passages sent to the LLM |
| `CHROMA_DIR`, `UPLOAD_DIR`, `REGISTRY_PATH` | `data/...` | Persistence locations |
| `MAX_UPLOAD_MB` | 25 | Per-file upload limit |
| `HISTORY_TURNS` | 6 | Conversation turns remembered per session |

Changing `EMBEDDING_MODEL` after indexing requires re-uploading documents (vectors from different models are incompatible; the server logs a warning if it detects this).

## API

| Method & path | Description |
|---|---|
| `POST /documents/upload` | multipart field `files` (one or many PDFs). Returns a per-file result: `indexed`, `duplicate`, `failed` (with reason) or `rejected` (not a PDF / too large). One bad file never fails the batch. |
| `GET /documents` | All documents with `document_id`, `filename`, `upload_timestamp`, `num_pages`, `status` (`processing`/`indexed`/`failed`), `error`, `num_chunks`. |
| `DELETE /documents/{document_id}` | Removes the document's chunks, embeddings, stored file and metadata. 404 if unknown. |
| `POST /chat` | `{"question", "session_id"?, "top_k"?, "document_ids"?}` → answer, `status`, `citations`, `evidence`, `trace`. |
| `DELETE /chat/{session_id}` | Clear a conversation. |
| `GET /health` | Vector DB status, indexed docs, total chunks, embedding model (+ loaded?), LLM provider/model/configured. |

`/chat` errors: `422` invalid input, `502` LLM provider failure, `503` LLM key missing or embedding model unavailable.

```bash
curl -F "files=@Employee Handbook.pdf" -F "files=@Leave Policy.pdf" localhost:8000/documents/upload
curl -X POST localhost:8000/chat -H 'content-type: application/json' \
  -d '{"question": "What is the annual leave entitlement, and how many unused leaves can be carried forward?", "session_id": "demo"}'
```

Response shape (abridged):
```json
{
  "status": "answered",
  "answer": "Employees receive 20 days of annual leave [Leave Policy.pdf, p. 2]. Up to 5 unused days can be carried forward [Leave Policy.pdf, p. 3].",
  "citations": [{"filename": "Leave Policy.pdf", "page_start": 2, "page_end": 2, "page_label": "2", "chunk_id": "…:00001", "passage": "…"}],
  "evidence":  [{"rank": 1, "filename": "Leave Policy.pdf", "page_label": "2", "similarity": 0.71, "cited": true, "text": "…"}],
  "trace": {"query_rewritten": false, "candidates_considered": 5, "dropped_below_threshold": 2, "timings_ms": {…}}
}
```
`status` is `insufficient_information` (with a fixed, honest message and `reason`) when nothing relevant is retrieved, when the model says the sources are insufficient, or when its answer carries no valid citation.

## End-to-end demo workflow

1. Start backend and frontend. The sidebar shows 0 documents and the system panel (model, vector DB, LLM key status).
2. Upload `Employee Handbook.pdf`, `IT Security Policy.pdf`, `Leave Policy.pdf` → **Process uploads**. Each shows page and chunk counts; failures show their reason.
3. Ask: *"What is the annual leave entitlement, and how many unused leaves can be carried forward?"* → cited answer; open **Retrieved evidence** to see passages, files, pages and scores (passages actually cited are marked).
4. Follow up: *"Can they be carried forward?"* → the evidence panel shows the resolved query; evidence is retrieved again for the follow-up.
5. Ask something absent (*"What is the parental leave policy in France?"*) → "couldn't find enough information", no invented citation.
6. Delete a document → its chunks disappear from retrieval. Restart the backend → documents and index are still there.

The same flow without a browser: the `curl` commands above, or `pytest tests/test_api_e2e.py`.

## Tests

```bash
pytest                          # 59 tests (+1 optional real-model test, skipped by default); no network, API key or model download needed
RUN_REAL_MODEL_TESTS=1 pytest -m real_model   # optional: checks the real embedding model (needs download)
```

Tests build real PDFs with PyMuPDF and exercise real extraction, chunking, ChromaDB (persistence, delete, filters) and the FastAPI app. Only two things are replaced by test doubles in `tests/conftest.py`: the **embedding model** (a deterministic hashed bag-of-words embedder, so retrieval in tests is lexical) and the **LLM** (a scripted fake, which records prompts so tests can verify that retrieved text reached it). Coverage: extraction, page-number preservation, chunk metadata and page ranges, embedding batching/dedup, vector insert/retrieve/filter/delete/restart persistence, no-evidence retrieval, prompt/context construction, citation generation from metadata (including dropping fabricated `[S9]`-style references), invalid/empty/encrypted/text-less PDFs, duplicate uploads, oversize/non-PDF rejection, follow-up rewriting, LLM-failure and missing-key handling, and the full upload → answer pipeline.

## Design decisions

- **Citations are built by the server, not the LLM.** The prompt labels passages `[S1]…[Sn]`; the model may only reference those labels. Labels are mapped to the stored metadata of the retrieved chunk (filename, pages, chunk id); out-of-range labels are dropped. An answer with no valid citation is converted to "insufficient information".
- **Page-aware chunking.** Every word carries its source page, so a chunk spanning a page break reports an exact range (`page 2-3`) rather than a wrong single page. Chunks prefer to end at paragraph/sentence boundaries.
- **Document IDs are content hashes**, so re-uploading identical content is detected and skipped (no duplicate chunks); chunk IDs are deterministic and written with upsert.
- **Ingestion is separate from querying**: `DocumentService` runs at upload; `RAGPipeline` only reads the index.
- **Metadata registry (JSON) is separate from the vector index**, and failed documents are kept with their error rather than dropped. Records left in `processing` by a crash are marked failed on restart.
- **Follow-ups**: history lives in a session store (not in Chroma). An LLM call rewrites the follow-up into a standalone question for retrieval; history is never used as evidence, and answers are always generated from freshly retrieved passages.
- **No LLM call without evidence**: empty/low-similarity retrieval short-circuits to the insufficient-information message.

## Implemented features

PDF upload (multi-file, validation, size limit) · per-page extraction with empty-page and damaged-PDF handling · cleaning and page-aware chunking · local embeddings with batching and duplicate suppression · persistent ChromaDB with filtering, listing and deletion · dense top-K retrieval with scores · grounded generation with server-built citations · session conversations with follow-up resolution · FastAPI + Pydantic API · Streamlit UI (document manager, chat, citations, evidence panel, system info) · logging · Adaptive/Corrective/Verification mechanisms · evaluation module · 114 tests.

## Baseline limitations

- **No OCR**: scanned/image-only PDFs fail with a clear message. Tables, multi-column layouts and headers/footers are extracted as plain text blocks (page-number footers can appear in chunks).
- Chunk size is a word-based token *estimate*, not the model tokenizer's exact count.
- `MIN_SIMILARITY=0.35` is a starting point and has **not been calibrated** on real data; tune it (look at the scores in the evidence panel).
- Dense retrieval only (by design for this baseline): exact-term/ID queries and multi-hop questions can retrieve poorly.
- Ingestion is synchronous inside the upload request (large PDFs block that request); conversation history is in memory and lost on restart; no authentication or multi-user isolation.
- In baseline mode citation checking is structural only; semantic citation validation happens in verification mode.
- Two documents with the same filename but different content are both kept; citations then show the same filename.
- The default Anthropic model name is configurable; confirm it matches a model available to your key.
