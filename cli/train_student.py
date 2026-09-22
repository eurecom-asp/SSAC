#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from ssac.student import AdvancedAccentStudent, StudentConfig
from ssac.training import TrainConfig, train_student


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--ar-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--init-checkpoint")
    p.add_argument("--seed", type=int)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_cfg = StudentConfig(**cfg.get("model", {}))
    train_dict = dict(cfg.get("training", {}))
    if args.seed is not None:
        train_dict["seed"] = args.seed
    train_cfg = TrainConfig(**train_dict)

    dtype = torch.bfloat16 if train_cfg.bf16 else torch.float32
    student = AdvancedAccentStudent.from_pretrained(
        args.ar_checkpoint,
        config=model_cfg,
        torch_dtype=dtype,
        device=args.device,
    )
    if args.init_checkpoint:
        missing, unexpected = student.load_checkpoint(args.init_checkpoint)
        print(f"Loaded init checkpoint. missing={len(missing)} unexpected={len(unexpected)}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolved_config.json").write_text(
        json.dumps({"model": cfg.get("model", {}), "training": train_dict}, indent=2),
        encoding="utf-8",
    )
    train_student(student, args.manifest, out, train_cfg)


if __name__ == "__main__":
    main()
