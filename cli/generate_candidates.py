#!/usr/bin/env python3
from __future__ import annotations

import argparse
import torch

from ssac.generation import generate_candidate_bank
from ssac.student import AdvancedAccentStudent, StudentConfig
from ssac.vevo2_backend import Vevo2Decoder


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--conditions", required=True)
    p.add_argument("--ar-checkpoint", required=True)
    p.add_argument("--student-checkpoint", required=True)
    p.add_argument("--vevo2-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--results-jsonl", required=True)
    p.add_argument("--best-of-n", type=int, default=8)
    p.add_argument("--flow-steps", type=int, default=32)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=500)
    p.add_argument("--min-new-tokens", type=int, default=15)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    student = AdvancedAccentStudent.from_pretrained(
        args.ar_checkpoint,
        config=StudentConfig(),
        torch_dtype=torch.bfloat16,
        device=args.device,
    )
    student.load_checkpoint(args.student_checkpoint)
    student.eval()
    decoder = Vevo2Decoder(args.vevo2_root, device=args.device)

    generate_candidate_bank(
        student,
        decoder,
        args.conditions,
        args.output_dir,
        args.results_jsonl,
        n_candidates=args.best_of_n,
        base_seed=args.seed,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
        flow_steps=args.flow_steps,
    )


if __name__ == "__main__":
    main()
