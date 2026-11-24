"""Run the 25-question E1-E5 benchmark, with per-query checkpoints and resume.

Commands:
    python -m scripts.run_eval25 --check
    python -m scripts.run_eval25 --limit 2 --out results/eval25-smoke
    python -m scripts.run_eval25 --out results/eval25 --max-minutes 120
    python -m scripts.run_eval25 --out results/eval25  # resumes unfinished pairs

Read data/eval/EVAL25.md before reporting any result as final.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import pandas as pd

from backend.evaluation.dataset import load_dataset
from backend.evaluation.runner import EXPERIMENTS, ExperimentRunner, summarize
from scripts._common import build_container

DEFAULT_DATASET = Path("data/eval/questions25.provisional.json")
DEFAULT_OUT = Path("results/eval25")
EXPERIMENT_IDS = ("E1", "E2", "E3", "E4", "E5")


def _primitive(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, default=_primitive, allow_nan=True)


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Broken checkpoint {path}:{i}: {exc}") from exc
    return rows


def _export(out: Path, rows: list[dict], experiments: tuple[str, ...]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for exp in experiments:
        erows = [r for r in rows if r["experiment"] == exp]
        if not erows:
            continue
        df = pd.DataFrame(erows)
        claims = []
        for r in erows:
            for c in (r.get("claims_json") or []):
                if isinstance(c, dict):
                    claims.append({"experiment": exp, "question_id": r["id"], **c, "human_label": ""})
        if "claims_json" in df:
            df["claims_json"] = df["claims_json"].apply(
                lambda v: _json(v) if isinstance(v, list) else ""
            )
        df.to_csv(out / f"per_question_{exp}.csv", index=False)
        if claims:
            pd.DataFrame(claims).to_csv(out / f"claims_for_annotation_{exp}.csv", index=False)
        if (df["status"] != "error").any():
            summaries.append(summarize(df, exp))
        else:
            summaries.append({"experiment": exp, "name": exp, "n_questions": len(df),
                              "n_errors": len(df)})
    if summaries:
        pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)


def _preflight(questions, container) -> None:
    docs = container.registry.list()
    indexed = {d.filename for d in docs if d.status == "indexed"}
    required = {r.document for q in questions for r in q.relevant_evidence}
    missing = sorted(required - indexed)
    if missing:
        raise SystemExit(
            "STOP: indexed documents with EXACT filenames are missing: "
            + ", ".join(missing)
            + ". Upload the original PDFs first and verify their names."
        )
    if not indexed:
        raise SystemExit("STOP: no indexed PDFs found.")
    if not container.settings.llm_configured:
        raise SystemExit("STOP: set your LLM_API_KEY privately in local .env first.")
    if any(q.illustrative for q in questions):
        raise SystemExit("STOP: illustrative rows must not be benchmarked.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=25, help="unique questions to test (default: 25)")
    ap.add_argument("--max-minutes", type=float, default=0, help="stop after time budget, keeping completed rows; zero means no cap")
    ap.add_argument("--check", action="store_true", help="validate dataset without loading models or calling LLM")
    ap.add_argument("--no-draft-measure", action="store_true", help="skip post-hoc verifier in E1-E3 (not recommended)")
    args = ap.parse_args()
    if args.limit < 1 or args.max_minutes < 0:
        ap.error("--limit must be positive and --max-minutes must be nonnegative")
    questions = load_dataset(args.dataset)
    if len(questions) < args.limit:
        ap.error(f"Only {len(questions)} non-illustrative questions exist; requested {args.limit}.")
    questions = questions[:args.limit]
    expected = [(e, q.id) for e in EXPERIMENT_IDS for q in questions]
    print(f"Dataset: {len(questions)} unique questions, {len(expected)} planned executions.")
    print("Label status: PROVISIONAL (not independently source-validated).")
    if args.check:
        types = {}
        for q in questions:
            types[q.answerability] = types.get(q.answerability, 0) + 1
        print("Question categories:", types)
        for q in questions:
            print(f"  {q.id}: {q.query_type} / {q.answerability} | {q.question}")
        return

    container = build_container()
    _preflight(questions, container)
    runner = ExperimentRunner(
        container.pipeline, container.embedder,
        verifier=container.pipeline.verifier, measure_draft=not args.no_draft_measure,
    )
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "question_ids": [q.id for q in questions],
        "experiments": list(EXPERIMENT_IDS),
        "indexed_documents": sorted([{"id": d.document_id, "filename": d.filename}
                                     for d in container.registry.list() if d.status == "indexed"],
                                    key=lambda d: d["filename"]),
        "llm_provider": container.settings.llm_provider,
        "llm_model": container.settings.llm_model,
        "embedding_model": container.settings.embedding_model,
        "nli_model": container.settings.nli_model,
        "top_k": container.settings.top_k,
        "min_similarity": container.settings.min_similarity,
        "entail_threshold": container.settings.nli_entail_threshold,
        "contradict_threshold": container.settings.nli_contradict_threshold,
        "measure_draft": not args.no_draft_measure,
    }
    manifest_path = out / "run_config.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old != manifest:
            raise SystemExit("STOP: resume configuration changed. Choose a NEW --out directory.")
    else:
        manifest_path.write_text(_json(manifest) + "\n", encoding="utf-8")
    checkpoint = out / "completed.jsonl"
    rows = _load_rows(checkpoint)
    seen = [(r["experiment"], r["id"]) for r in rows]
    if len(seen) != len(set(seen)) or set(seen) - set(expected):
        raise SystemExit("STOP: checkpoint has duplicates or unexpected question IDs.")
    completed = set(seen)
    print(f"Resuming: {len(completed)}/{len(expected)} completed. Output: {out}")
    start = time.monotonic()
    for exp, question_id in expected:
        if (exp, question_id) in completed:
            continue
        if args.max_minutes and time.monotonic() - start >= args.max_minutes * 60:
            print("Time budget reached; completed results remain saved. Re-run the same command to resume.")
            break
        question = next(q for q in questions if q.id == question_id)
        print(f"[{len(completed)+1}/{len(expected)}] {exp} / {question_id}", flush=True)
        row = runner._run_one(exp, EXPERIMENTS[exp], question, None, None)
        # Write each result before the next API/model call. A crash loses at most one in-flight execution.
        with checkpoint.open("a", encoding="utf-8") as f:
            f.write(_json(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
        rows.append(row)
        completed.add((exp, question_id))
        print(f"  {row.get('status')} | {round(row.get('wall_ms', 0)/1000, 1)} sec")
        _export(out, rows, EXPERIMENT_IDS)
    done = len(completed) == len(expected)
    _export(out, rows, EXPERIMENT_IDS)
    (out / "RUN_STATUS.json").write_text(_json({
        "complete": done, "executions_completed": len(completed),
        "executions_planned": len(expected),
        "errors": sum(r.get("status") == "error" for r in rows),
        "gold_labels_source_validated": False,
        "ready_for_final_research_reporting": False,
        "note": "Review provisional gold labels, verifier false contradictions, and human claim annotations.",
    }) + "\n", encoding="utf-8")
    if done:
        print(f"Completed all {len(expected)} executions; review {out}/summary.csv.")
    else:
        print(f"Completed {len(completed)} of {len(expected)}; resume with same --out.")
    print("NOTICE: these are provisional metrics until independent evidence/claim labels are validated.")


if __name__ == "__main__":
    main()
