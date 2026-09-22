#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ssac.student import AdvancedAccentStudent, StudentConfig
from ssac.training import TrainConfig, train_student


def main() -> None:
    p = argparse.ArgumentParser(
        description="Two-stage adaptation used to create the fixed accent-adapted Vevo2 candidate generator G_phi."
    )
    p.add_argument("--manifest", required=True, help="Earlier selected reference-conditioned pseudo-target trajectories")
    p.add_argument("--ar-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--prompt-epochs", type=int, default=5)
    p.add_argument("--lora-epochs", type=int, default=10)
    p.add_argument("--prompt-lr", type=float, required=True,
                   help="Not recovered from the historical audit; supply the value from the original run metadata")
    p.add_argument("--lora-lr", type=float, required=True,
                   help="Not recovered from the historical audit; supply the value from the original run metadata")
    p.add_argument("--micro-batch-size", type=int, required=True,
                   help="Not safely recovered for the 29,840-trajectory generator-adaptation run")
    p.add_argument("--grad-accum", type=int, required=True,
                   help="Not safely recovered for the 29,840-trajectory generator-adaptation run")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out = Path(args.output_dir)
    stage_a = out / "prompt_only"
    stage_b = out / "qkvo"

    # Stage A: continuous accent prompt only.
    student = AdvancedAccentStudent.from_pretrained(
        args.ar_checkpoint, StudentConfig(), torch_dtype=torch.bfloat16, device=args.device
    )
    cfg_a = TrainConfig(
        epochs=args.prompt_epochs,
        micro_batch_size=args.micro_batch_size,
        grad_accum=args.grad_accum,
        lr=args.lora_lr,
        prompt_lr=args.prompt_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        seed=args.seed,
    )
    train_student(student, args.manifest, stage_a, cfg_a, prompt_only_epochs=args.prompt_epochs)

    # Stage B: reload the final prompt-only checkpoint, then train prompt + shared QKVO LoRA.
    student = AdvancedAccentStudent.from_pretrained(
        args.ar_checkpoint, StudentConfig(), torch_dtype=torch.bfloat16, device=args.device
    )
    student.load_checkpoint(stage_a / f"epoch_{args.prompt_epochs}")
    cfg_b = TrainConfig(
        epochs=args.lora_epochs,
        micro_batch_size=args.micro_batch_size,
        grad_accum=args.grad_accum,
        lr=args.lora_lr,
        prompt_lr=args.prompt_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        seed=args.seed,
    )
    train_student(student, args.manifest, stage_b, cfg_b, prompt_only_epochs=0)


if __name__ == "__main__":
    main()
