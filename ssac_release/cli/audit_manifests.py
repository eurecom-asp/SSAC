#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter

from ssac.io import read_jsonl
from ssac.select import feasible


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--conditions")
    p.add_argument("--candidates")
    p.add_argument("--scores")
    p.add_argument("--selected")
    args = p.parse_args()

    if args.conditions:
        rows = list(read_jsonl(args.conditions))
        print(f"conditions={len(rows)} (paper construction: 29,754)")
        by_target = Counter(r["target_accent"] for r in rows)
        print("conditions_by_target=", dict(by_target))
        if len(rows) == 29754:
            expected = {a: 4959 for a in ("Arabic", "Chinese", "Hindi", "Korean", "Spanish", "Vietnamese")}
            if dict(by_target) != expected:
                raise SystemExit(f"ERROR: 29,754-row manifest is not target-balanced as expected: {dict(by_target)}")
        corpus_key = next((k for k in ("source_corpus", "corpus", "dataset") if rows and k in rows[0]), None)
        if corpus_key:
            print("conditions_by_corpus=", dict(Counter(str(r.get(corpus_key, "")) for r in rows)))
    if args.candidates:
        rows = list(read_jsonl(args.candidates))
        print(f"candidate_rows={len(rows)} (29,754 x 8 = 238,032 when complete)")
        print(f"candidate_errors={sum(bool(r.get('error')) for r in rows)}")
    if args.scores:
        rows = list(read_jsonl(args.scores))
        print(f"score_rows={len(rows)}")
        print(f"feasible_rows={sum(feasible(r) for r in rows)}")
    if args.selected:
        rows = list(read_jsonl(args.selected))
        print(f"selected_conditions={len(rows)} (paper HardTop1: 16,153)")
        print("selected_by_target=", dict(Counter(r["target_accent"] for r in rows)))


if __name__ == "__main__":
    main()
