#!/usr/bin/env python3
from __future__ import annotations

import argparse

from ssac.io import append_jsonl, read_jsonl
from ssac.scoring import AccentScorerQ, EcapaSpeakerEncoder, WhisperSmallASR, score_candidate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", required=True)
    p.add_argument("--q", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--ecapa-dir", required=True)
    p.add_argument("--whisper-cache")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    q = AccentScorerQ.load(args.q, device=args.device)
    asr = WhisperSmallASR(device=args.device, download_root=args.whisper_cache)
    spk = EcapaSpeakerEncoder(args.ecapa_dir, device=args.device)

    for row in read_jsonl(args.candidates):
        if row.get("error"):
            append_jsonl(args.output, row)
            continue
        try:
            scored = score_candidate(row, q, asr, spk)
        except Exception as exc:
            scored = dict(row)
            scored["error"] = f"scoring:{type(exc).__name__}: {exc}"
        append_jsonl(args.output, scored)


if __name__ == "__main__":
    main()
