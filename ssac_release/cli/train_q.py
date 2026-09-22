#!/usr/bin/env python3
from __future__ import annotations

import argparse
from sklearn.metrics import accuracy_score, f1_score

from ssac.io import read_jsonl
from ssac.scoring import AccentScorerQ, train_q


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--dev-manifest")
    p.add_argument("--output", required=True)
    p.add_argument("--model-id", default="openai/whisper-small")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    q = train_q(args.train_manifest, args.output, model_id=args.model_id, device=args.device)
    if args.dev_manifest:
        y_true, y_pred = [], []
        for row in read_jsonl(args.dev_manifest):
            wav = row.get("audio_path", row.get("wav_path"))
            probs = q.probabilities(wav)
            y_true.append(str(row["accent_label"]))
            y_pred.append(max(probs, key=probs.get))
        print(f"dev_acc={accuracy_score(y_true, y_pred):.6f}")
        print(f"dev_macro_f1={f1_score(y_true, y_pred, average='macro'):.6f}")


if __name__ == "__main__":
    main()
