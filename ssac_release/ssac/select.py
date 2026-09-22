from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Any

from .constants import (
    MAX_DURATION_RATIO,
    MAX_WER,
    MIN_DURATION_RATIO,
    MIN_SPEAKER_SIM,
    MIN_TARGET_PROB_DELTA,
)

_REQUIRED_METRICS = (
    "wer",
    "speaker_sim",
    "duration_ratio",
    "target_hit",
    "target_prob",
    "target_prob_delta",
)


def _finite(row: dict[str, Any]) -> bool:
    for key in _REQUIRED_METRICS:
        try:
            if not math.isfinite(float(row[key])):
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return True


def feasible(row: dict[str, Any]) -> bool:
    """Final ICASSP feasibility rule used before Top-1 selection."""
    if row.get("error"):
        return False
    if not _finite(row):
        return False

    wer = float(row["wer"])
    sim = float(row["speaker_sim"])
    ratio = float(row["duration_ratio"])
    hit = int(round(float(row["target_hit"])))
    delta = float(row["target_prob_delta"])

    return (
        wer <= MAX_WER
        and sim >= MIN_SPEAKER_SIM
        and MIN_DURATION_RATIO <= ratio <= MAX_DURATION_RATIO
        and (hit == 1 or delta >= MIN_TARGET_PROB_DELTA)
    )


def rank_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    """Lexicographic accent-first ranking recovered from the final project audit."""
    return (
        float(row["target_prob"]),
        float(row["target_prob_delta"]),
        float(row["target_hit"]),
        float(row["speaker_sim"]),
        -float(row["wer"]),
    )


def select_top1(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        base_pair_id = str(row.get("base_pair_id", row.get("pair_id", "")))
        if not base_pair_id:
            raise ValueError("Every candidate needs base_pair_id or pair_id")
        grouped[base_pair_id].append(row)

    selected: list[dict[str, Any]] = []
    for base_pair_id, candidates in grouped.items():
        eligible = [row for row in candidates if feasible(row)]
        if not eligible:
            continue
        best = max(eligible, key=rank_key).copy()
        best["base_pair_id"] = base_pair_id
        best["selection_rule"] = "hard_top1"
        ids = best.get("content_style_ids")
        if ids is None:
            ids = best.get("teacher_cs_ids")
        if ids is None:
            raise ValueError(f"Selected candidate {base_pair_id} has no content-style token ids")
        best["teacher_cs_ids"] = ids
        best["positive_cs_ids"] = ids
        selected.append(best)
    return selected
