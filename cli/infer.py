#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ssac.generation import set_attempt_seed
from ssac.student import AdvancedAccentStudent, StudentConfig
from ssac.vevo2_backend import Vevo2Decoder


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ar-checkpoint", required=True)
    p.add_argument("--student-checkpoint", required=True)
    p.add_argument("--vevo2-root", required=True)
    p.add_argument("--source-wav", required=True)
    p.add_argument("--source-text", required=True)
    p.add_argument("--target-accent", required=True)
    p.add_argument("--output-wav", required=True)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    set_attempt_seed(args.seed)
    student = AdvancedAccentStudent.from_pretrained(
        args.ar_checkpoint,
        config=StudentConfig(),
        torch_dtype=torch.bfloat16,
        device=args.device,
    )
    student.load_checkpoint(args.student_checkpoint)
    student.eval()
    codes = student.generate_advanced_codes(
        source_text=args.source_text,
        target_accent=args.target_accent,
        source_prosody_ids=[],
        max_new_tokens=500,
        min_new_tokens=15,
        top_k=25,
        top_p=0.8,
        temperature=1.0,
        do_sample=True,
    )
    decoder = Vevo2Decoder(args.vevo2_root, device=args.device)
    decoder.decode_codes_with_source_timbre(codes, args.source_wav, args.output_wav, flow_steps=32)
    print(Path(args.output_wav).resolve())


if __name__ == "__main__":
    main()
