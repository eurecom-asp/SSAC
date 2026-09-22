from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from .constants import (
    DEFAULT_BASE_SEED,
    DEFAULT_CANDIDATES,
    DEFAULT_FLOW_STEPS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MIN_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)
from .io import append_jsonl, read_jsonl, stable_int
from .student import AdvancedAccentStudent
from .vevo2_backend import Vevo2Decoder


def set_attempt_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_candidate_bank(
    student: AdvancedAccentStudent,
    decoder: Vevo2Decoder,
    conditions_manifest: str | Path,
    output_dir: str | Path,
    results_jsonl: str | Path,
    *,
    n_candidates: int = DEFAULT_CANDIDATES,
    base_seed: int = DEFAULT_BASE_SEED,
    top_k: int = DEFAULT_TOP_K,
    top_p: float = DEFAULT_TOP_P,
    temperature: float = DEFAULT_TEMPERATURE,
    min_new_tokens: int = DEFAULT_MIN_NEW_TOKENS,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    flow_steps: int = DEFAULT_FLOW_STEPS,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = Path(results_jsonl)

    for row in read_jsonl(conditions_manifest):
        base_pair_id = str(row.get("base_pair_id", row.get("pair_id", "")))
        if not base_pair_id:
            raise ValueError("Condition row has no pair_id/base_pair_id")
        for attempt in range(n_candidates):
            seed = base_seed + stable_int(f"{base_pair_id}::attempt={attempt}")
            set_attempt_seed(seed)
            candidate_id = f"{base_pair_id}__n{attempt:02d}"
            wav_path = output_dir / base_pair_id / f"attempt_{attempt:02d}.wav"
            result = {
                **row,
                "base_pair_id": base_pair_id,
                "candidate_id": candidate_id,
                "best_of_n_attempt": attempt,
                "candidate_seed": seed,
                "output_wav": str(wav_path),
            }
            try:
                codes = student.generate_advanced_codes(
                    source_text=str(row["source_text"]),
                    target_accent=str(row["target_accent"]),
                    source_prosody_ids=[],
                    min_new_tokens=min_new_tokens,
                    max_new_tokens=max_new_tokens,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    do_sample=True,
                )
                decoder.decode_codes_with_source_timbre(
                    codes,
                    source_wav=row["source_wav"],
                    output_wav=wav_path,
                    flow_steps=flow_steps,
                )
                result["content_style_ids"] = codes
                result["num_content_style_codes"] = len(codes)
            except Exception as exc:  # preserve failures for an auditable bank
                result["error"] = f"{type(exc).__name__}: {exc}"
            append_jsonl(results_jsonl, result)
