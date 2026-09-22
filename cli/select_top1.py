#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter

from ssac.io import read_jsonl, write_jsonl
from ssac.select import feasible, select_top1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    rows = list(read_jsonl(args.scores))
    selected = select_top1(rows)
    write_jsonl(args.output, selected)
    print(f"candidate_rows={len(rows)}")
    print(f"feasible_rows={sum(feasible(r) for r in rows)}")
    print(f"retained_conditions={len(selected)}")
    print("by_target=", dict(Counter(r["target_accent"] for r in selected)))


if __name__ == "__main__":
    main()
