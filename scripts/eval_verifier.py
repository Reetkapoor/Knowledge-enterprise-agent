"""Score the claim verifier against human labels: per-label precision / recall / F1, confusion matrix.

    python -m scripts.eval_verifier --claims data/eval/labelled_claims.jsonl [--entail 0.6 --contradict 0.6]

Input: your labelled claims (JSONL/JSON/CSV; see data/eval/README.md) OR the claims_for_annotation_*.csv exported by
run_eval after you filled in the `human_label` column (then the stored system_label is scored, no NLI rerun).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backend.core.config import get_settings
from backend.evaluation.dataset import load_labelled_claims
from backend.evaluation.metrics import LABELS, verification_report
from backend.services.verification.evidence import window_texts
from backend.services.verification.nli import HFNLIModel
from backend.services.verification.verifier import judge_claim


def predict_with_nli(claims, nli, entail, contradict, numeric_guard, window_sentences, stride):
    pairs, owner = [], []
    for i, c in enumerate(claims):
        for passage in c.evidence:
            for w in window_texts(passage, window_sentences, stride):
                pairs.append((w, c.claim))
                owner.append(i)
    scores = nli.predict(pairs)
    per_claim: dict[int, list] = {i: [] for i in range(len(claims))}
    for (w, _), s, i in zip(pairs, scores, owner):
        per_claim[i].append((w, s))
    return per_claim


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--claims", required=True)
    ap.add_argument("--entail", type=float, default=s.nli_entail_threshold)
    ap.add_argument("--contradict", type=float, default=s.nli_contradict_threshold)
    ap.add_argument("--no-numeric-guard", action="store_true")
    ap.add_argument("--nli-model", default=s.nli_model)
    ap.add_argument("--out", default="results/verifier_eval")
    args = ap.parse_args()

    claims = load_labelled_claims(args.claims)
    if not claims:
        raise SystemExit("No labelled claims found (rows without a label are ignored).")
    if all(c.system_label for c in claims):
        pred, source = [c.system_label.upper() for c in claims], "stored system_label"
    else:
        nli = HFNLIModel(args.nli_model, s.nli_batch_size, s.nli_max_length)
        per_claim = predict_with_nli(claims, nli, args.entail, args.contradict, not args.no_numeric_guard,
                                     s.nli_window_sentences, s.nli_window_stride)
        pred = [judge_claim(c.claim, per_claim[i], args.entail, args.contradict, not args.no_numeric_guard)["verdict"].value
                for i, c in enumerate(claims)]
        source = f"{args.nli_model} (entail>={args.entail}, contradict>={args.contradict})"

    rep = verification_report([c.gold for c in claims], pred)
    print(f"Verifier: {source}\nClaims scored: {rep['n']}   accuracy: {rep['accuracy']:.3f}   macro-F1: {rep['macro_f1']:.3f}")
    rows = [{"label": lab, **rep["per_label"][lab]} for lab in LABELS]
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nConfusion (rows = human label, columns = system label):")
    print(pd.DataFrame(rep["confusion"]).T.to_string())
    print(f"\nContradiction detection rate (recall of CONTRADICTED): {rep['contradiction_detection_rate']:.3f}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "per_label.csv", index=False)
    pd.DataFrame({"id": [c.id for c in claims], "claim": [c.claim for c in claims], "gold": [c.gold for c in claims], "pred": pred}).to_csv(out / "predictions.csv", index=False)
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=float))
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
