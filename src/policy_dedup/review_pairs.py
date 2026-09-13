#!/usr/bin/env python3
"""
Optional. Build a stratified review pack from a pair-score CSV
so a human can confirm the Jaccard cutoff.

  python3 review_pairs.py --pairs near_dup_pair_scores.csv --policies-dir DIR --out-dir DIR
  python3 review_pairs.py --pairs crossdev_pair_scores.csv --policies-dir DIR --out-dir DIR --prefix xd
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

from common import jaccard_bin, load_csv, write_csv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="near_dup_pair_scores.csv or crossdev_pair_scores.csv")
    ap.add_argument("--policies-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pack-name", default="review_pack")
    ap.add_argument("--obvious-min", type=float, default=0.98)
    ap.add_argument("--manual-min", type=float, default=0.90)
    ap.add_argument("--per-bin", type=int, default=15)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    policies = Path(args.policies_dir).expanduser()
    out = Path(args.out_dir).expanduser()
    pack = out / args.pack_name
    pack.mkdir(parents=True, exist_ok=True)

    rows = load_csv(Path(args.pairs).expanduser())
    for r in rows:
        r["_j"] = float(r["estimated_jaccard"])
        r["jaccard_bin"] = jaccard_bin(r["_j"])
        r["path_a"] = r.get("path_a") or str(policies / r["file_a"])
        r["path_b"] = r.get("path_b") or str(policies / r["file_b"])

    counts = defaultdict(int)
    by_bin: dict[str, list] = defaultdict(list)
    for r in rows:
        counts[r["jaccard_bin"]] += 1
        if args.manual_min <= r["_j"]:
            by_bin[r["jaccard_bin"]].append(r)

    print("Pair counts by bin:")
    for k in ["0.99-1.00", "0.98-0.99", "0.96-0.98", "0.94-0.96",
              "0.92-0.94", "0.90-0.92", "below-0.90"]:
        print(f"  {k:12}  {counts.get(k, 0)}")

    rng = random.Random(args.seed)
    sample = []
    order = ["0.99-1.00", "0.98-0.99", "0.96-0.98", "0.94-0.96", "0.92-0.94", "0.90-0.92"]
    for band in order:
        items = by_bin.get(band, [])
        take = min(args.per_bin, len(items))
        chosen = rng.sample(items, take) if len(items) > take else list(items)
        for i, r in enumerate(chosen, 1):
            rec = dict(r)
            rec["sample_id"] = f"{args.prefix}{band}_{i:02d}"
            sample.append(rec)

    scorecard = []
    for rec in sample:
        dest = pack / rec["sample_id"]
        dest.mkdir(parents=True, exist_ok=True)
        sa, sb = Path(rec["path_a"]), Path(rec["path_b"])
        if sa.is_file():
            shutil.copy2(sa, dest / f"a_{sa.name}")
        if sb.is_file():
            shutil.copy2(sb, dest / f"b_{sb.name}")
        scorecard.append({
            "sample_id": rec["sample_id"],
            "band": rec["jaccard_bin"],
            "estimated_jaccard": rec["estimated_jaccard"],
            "file_a": rec["file_a"],
            "file_b": rec["file_b"],
            "same_policy": "",
            "notes": "",
        })
    write_csv(pack / "SCORECARD.csv", scorecard)
    print(f"sampled {len(sample)} folders -> {pack}")
    print(f"fill {pack / 'SCORECARD.csv'} same_policy=YES/NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
