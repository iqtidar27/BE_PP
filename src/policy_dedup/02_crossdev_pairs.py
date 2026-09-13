#!/usr/bin/env python3
"""
Step 2. MinHash/LSH over one representative per exact doc_id.
        Keep pairs whose developers differ.
        Writes crossdev_pair_scores.csv for step 3.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from common import (
    MinHasher,
    estimated_jaccard,
    extract_raw_text,
    load_csv,
    lsh_candidates,
    normalize_text,
    setup_logging,
    shingles,
    words_of,
    write_csv,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Cross-developer near-dup pair scores")
    ap.add_argument("--out-dir", required=True, help="Directory written by step 1")
    ap.add_argument("--min-jaccard", type=float, default=0.90)
    ap.add_argument("--num-perm", type=int, default=128)
    ap.add_argument("--lsh-bands", type=int, default=32)
    ap.add_argument("--lsh-rows", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir).expanduser()
    log = setup_logging(out, "crossdev")
    manifest = load_csv(out / "file_manifest.csv")
    nonempty = [r for r in manifest if str(r.get("empty_after_norm", "")).lower() != "true"]

    # one row per doc_id (longest text)
    best: dict[str, dict] = {}
    for r in nonempty:
        cur = best.get(r["doc_id"])
        if cur is None or int(r["norm_chars"]) > int(cur["norm_chars"]):
            best[r["doc_id"]] = r
    reps = list(best.values())
    if args.limit:
        reps = reps[: args.limit]
    log.info("Representatives to hash: %s", len(reps))

    hasher = MinHasher(num_perm=args.num_perm)
    items = []
    t0 = time.time()
    for n, row in enumerate(reps, 1):
        if n % 2000 == 0 or n == 1:
            log.info("Hashing %s / %s", n, len(reps))
        path = Path(row["file_path"])
        if not path.is_file():
            continue
        try:
            norm = normalize_text(extract_raw_text(path))
        except Exception as e:
            log.debug("skip %s: %s", path.name, e)
            continue
        if len(norm) < 40:
            continue
        items.append({
            "doc_id": row["doc_id"],
            "dev_key": row["dev_key"],
            "file_name": row["file_name"],
            "file_path": row["file_path"],
            "extension_id": row["extension_id"],
            "norm_chars": len(norm),
            "sig": hasher.signature(shingles(words_of(norm))),
        })
    log.info("Hashed %s docs in %.1fs", len(items), time.time() - t0)

    indexed = [(i, it["sig"]) for i, it in enumerate(items)]
    raw = lsh_candidates(indexed, args.lsh_bands, args.lsh_rows)
    log.info("LSH candidate pairs: %s", len(raw))

    rows = []
    for i, j in raw:
        a, b = items[i], items[j]
        if a["dev_key"] and b["dev_key"] and a["dev_key"] == b["dev_key"]:
            continue
        ja = estimated_jaccard(a["sig"], b["sig"])
        if ja < args.min_jaccard:
            continue
        rows.append({
            "estimated_jaccard": f"{ja:.4f}",
            "dev_key_a": a["dev_key"],
            "dev_key_b": b["dev_key"],
            "ext_a": a["extension_id"],
            "ext_b": b["extension_id"],
            "doc_id_a": a["doc_id"],
            "doc_id_b": b["doc_id"],
            "file_a": a["file_name"],
            "file_b": b["file_name"],
            "path_a": a["file_path"],
            "path_b": b["file_path"],
            "norm_chars_a": a["norm_chars"],
            "norm_chars_b": b["norm_chars"],
        })
    rows.sort(key=lambda r: -float(r["estimated_jaccard"]))
    write_csv(out / "crossdev_pair_scores.csv", rows)
    log.info("Cross-dev pairs J>=%s: %s", args.min_jaccard, len(rows))
    log.info("Wrote %s", out / "crossdev_pair_scores.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
