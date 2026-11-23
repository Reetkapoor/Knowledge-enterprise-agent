"""Run ablation experiments E1-E5 over a labelled dataset and export CSVs.

    python -m scripts.run_eval --dataset data/eval/questions.json --experiments E1 E2 E3 E4 E5 [--judge] [--limit 20]

Outputs in results/<run_id>/ : summary.csv (one row per experiment), per_question_<E>.csv, claims_for_annotation_<E>.csv,
run_config.json. Nothing is estimated or filled in: every number comes from running the pipeline on your dataset.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from backend.evaluation.dataset import load_dataset
from backend.evaluation.runner import EXPERIMENTS, ExperimentRunner, summarize
from scripts._common import build_container


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="JSON or CSV file (see data/eval/README.md)")
    ap.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    ap.add_argument("--out", default="results")
    ap.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    ap.add_argument("--judge", action="store_true", help="also grade answers with an LLM judge (extra API cost)")
    ap.add_argument("--no-draft-measure", action="store_true", help="skip post-hoc claim checks of E1-E3 answers")
    ap.add_argument("--include-illustrative", action="store_true", help="include rows marked illustrative (smoke tests only)")
    args = ap.parse_args()

    questions = load_dataset(args.dataset, include_illustrative=args.include_illustrative)
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        raise SystemExit("No questions to run (illustrative rows are skipped unless --include-illustrative).")

    c = build_container()
    runner = ExperimentRunner(
        c.pipeline, c.embedder, verifier=c.pipeline.verifier, judge_llm=c.llm if args.judge else None,
        measure_draft=not args.no_draft_measure,
    )
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) / run_id
    out.mkdir(parents=True, exist_ok=True)
    s = c.settings
    (out / "run_config.json").write_text(json.dumps({
        "dataset": args.dataset, "n_questions": len(questions), "experiments": args.experiments, "judge": args.judge,
        "llm": f"{s.llm_provider}/{s.llm_model}", "embedding_model": s.embedding_model, "nli_model": s.nli_model,
        "top_k": s.top_k, "min_similarity": s.min_similarity, "chunk_size": s.chunk_size, "chunk_overlap": s.chunk_overlap,
        "nli_entail_threshold": s.nli_entail_threshold, "nli_contradict_threshold": s.nli_contradict_threshold,
        "numeric_guard": s.numeric_guard, "max_corrective_retries": s.max_corrective_retries, "max_sub_questions": s.max_sub_questions,
        "indexed_documents": [d.filename for d in c.registry.list()],
    }, indent=2))

    summaries = []
    for exp in args.experiments:
        print(f"\n=== {exp} ({EXPERIMENTS[exp].label}) ===")
        df = runner.run(exp, questions, on_progress=lambda i, n: print(f"\r  {i}/{n}", end="", flush=True))
        print()
        claims = [{"experiment": exp, "question_id": r.id, **cl, "human_label": ""}
                  for r in df.itertuples() for cl in (getattr(r, "claims_json", None) or []) if isinstance(cl, dict)]
        if "claims_json" in df:
            df["claims_json"] = df["claims_json"].apply(lambda v: json.dumps(v) if isinstance(v, list) else "")
        df.to_csv(out / f"per_question_{exp}.csv", index=False)
        if claims:
            pd.DataFrame(claims).to_csv(out / f"claims_for_annotation_{exp}.csv", index=False)
        summaries.append(summarize(df, exp))
        print(f"  errors: {summaries[-1]['n_errors']}  fallbacks: {summaries[-1]['fallback_rate']}")

    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "summary.csv", index=False)
    print(f"\nWrote {out}/summary.csv\n")
    print(summary.set_index("experiment").T.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
