from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from .io import read_jsonl
from .student import AdvancedAccentStudent, AdvancedStudentExample


class JsonlStudentDataset(Dataset):
    def __init__(self, path: str | Path):
        self.rows = list(read_jsonl(path))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> AdvancedStudentExample:
        row = self.rows[idx]
        ids = row.get("teacher_cs_ids", row.get("positive_cs_ids", row.get("content_style_ids")))
        if ids is None:
            raise ValueError(f"Row {idx} has no teacher/content-style ids")
        return AdvancedStudentExample(
            pair_id=str(row.get("pair_id", row.get("base_pair_id", idx))),
            source_text=str(row["source_text"]),
            source_prosody_ids=list(map(int, row.get("source_prosody_ids", []))),
            teacher_cs_ids=list(map(int, ids)),
            target_accent=str(row["target_accent"]),
            source_speaker=str(row.get("source_speaker", "")),
        )


def _collate(batch):
    return batch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    epochs: int = 3
    micro_batch_size: int = 8
    grad_accum: int = 6
    lr: float = 5e-6
    prompt_lr: float | None = None
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    bf16: bool = True
    seed: int = 1337
    num_workers: int = 0


def _optimizer(student: AdvancedAccentStudent, cfg: TrainConfig) -> AdamW:
    prompt_lr = cfg.lr if cfg.prompt_lr is None else cfg.prompt_lr
    prompt, lora = [], []
    for name, p in student.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("accent_prompt."):
            prompt.append(p)
        else:
            lora.append(p)
    groups = []
    if lora:
        groups.append({"params": lora, "lr": cfg.lr, "weight_decay": cfg.weight_decay})
    if prompt:
        groups.append({"params": prompt, "lr": prompt_lr, "weight_decay": 0.0})
    return AdamW(groups)


def train_student(
    student: AdvancedAccentStudent,
    manifest: str | Path,
    output_dir: str | Path,
    cfg: TrainConfig,
    *,
    prompt_only_epochs: int = 0,
) -> None:
    seed_everything(cfg.seed)
    ds = JsonlStudentDataset(manifest)
    loader = DataLoader(
        ds,
        batch_size=cfg.micro_batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=cfg.num_workers,
    )
    if len(loader) == 0:
        raise ValueError("Empty training manifest")

    optimizer = _optimizer(student, cfg)
    updates_per_epoch = math.ceil(len(loader) / cfg.grad_accum)
    total_steps = updates_per_epoch * cfg.epochs
    warmup_steps = int(round(total_steps * cfg.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, cfg.epochs + 1):
        student.train()
        prompt_only = epoch <= prompt_only_epochs
        for name, p in student.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                p.requires_grad = not prompt_only

        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, examples in enumerate(loader, 1):
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=cfg.bf16 and torch.cuda.is_available(),
            ):
                out = student(examples)
                loss = out.loss / cfg.grad_accum
            loss.backward()
            running += float(loss.detach().cpu()) * cfg.grad_accum

            if step % cfg.grad_accum == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in student.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        epoch_dir = output_dir / f"epoch_{epoch}"
        student.save_checkpoint(epoch_dir)
        (epoch_dir / "train_loss.txt").write_text(
            f"{running / len(loader):.10f}\n", encoding="utf-8"
        )
