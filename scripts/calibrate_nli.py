"""Pick NLI decision thresholds from labelled claims, with a held-out split so the reported score is honest.

    python -m scripts.calibrate_nli --claims data/eval/labelled_claims.jsonl [--seed 0 --dev-fraction 0.5]

The NLI model runs once per claim; thresholds are then swept offline. Thresholds are chosen on the dev part only
(maximising macro-F1) and the chosen pair is reported on the held-out part. Raw NLI probabilities are not
probabilities of truth - this sweep is exactly why the thresholds are configurable.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backend.core.config import get_settings
from backend.evaluation.dataset import load_labelled_claims
from backend.evaluation.metrics import verification_report
from backend.services.verification.nli import HFNLIModel
from backend.services.verification.verifier import judge_claim
from scripts.eval_verifier import predict_with_nli


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--claims", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dev-fraction", type=float, default=0.5)
    ap.add_argument("--no-numeric-guard", action="store_true")
    ap.add_argument("--out", default="results/calibration")
    args = ap.parse_args()

    claims = load_labelled_claims(args.claims)
    if len(claims) < 20:
        raise SystemExit(f"Only {len(claims)} labelled claims; need at least ~20 (ideally 100+) for a meaningful sweep.")
    nli = HFNLIModel(s.nli_model, s.nli_batch_size, s.nli_max_length)
    per_claim = predict_with_nli(claims, nli, 0, 0, True, s.nli_window_sentences, s.nli_window_stride)

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(claims))
    cut = int(len(claims) * args.dev_fraction)
    dev, held = list(idx[:cut]), list(idx[cut:])
    grid = np.round(np.arange(0.30, 0.96, 0.05), 2)

    def macro_f1(ids, e, c):
        pred = [judge_claim(claims[i].claim, per_claim[i], e, c, not args.no_numeric_guard)["verdict"].value for i in ids]
        return verification_report([claims[i].gold for i in ids], pred)

    rows = []
    for e in grid:
        for c in grid:
            r = macro_f1(dev, e, c)
            rows.append({"entail": e, "contradict": c, "dev_macro_f1": r["macro_f1"], "dev_accuracy": r["accuracy"]})
    df = pd.DataFrame(rows).sort_values("dev_macro_f1", ascending=False)
    best = df.iloc[0]
    held_r = macro_f1(held, best.entail, best.contradict)
    default_r = macro_f1(held, s.nli_entail_threshold, s.nli_contradict_threshold)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "threshold_grid.csv", index=False)
    print(f"Claims: {len(claims)} (dev {len(dev)} / held-out {len(held)}), NLI model: {s.nli_model}")
    print(f"Best on dev: NLI_ENTAIL_THRESHOLD={best.entail}  NLI_CONTRADICT_THRESHOLD={best.contradict}  (dev macro-F1 {best.dev_macro_f1:.3f})")
    print(f"Held-out macro-F1 with those thresholds : {held_r['macro_f1']:.3f}  (accuracy {held_r['accuracy']:.3f})")
    print(f"Held-out macro-F1 with current defaults : {default_r['macro_f1']:.3f}  (accuracy {default_r['accuracy']:.3f})")
    print(f"Grid written to {out}/threshold_grid.csv")


if __name__ == "__main__":
    main()
