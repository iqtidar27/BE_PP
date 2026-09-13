#!/usr/bin/env python3
"""
Step 3. Union-find on doc_id using same-dev and cross-dev pairs
        with estimated Jaccard >= --threshold (default 0.92).
        Writes the unique-page list used by Stage A.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

from common import UnionFind, load_csv, write_csv


def main() -> int:
    ap = argparse.ArgumentParser(description="Export unique policy documents")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.92)
    args = ap.parse_args()
    out = Path(args.out_dir).expanduser()
    thr = args.threshold

    files = load_csv(out / "file_manifest.csv")
    same = load_csv(out / "near_dup_pair_scores.csv")
    cross_path = out / "crossdev_pair_scores.csv"
    if not cross_path.is_file():
        raise SystemExit(f"Missing {cross_path} — run 02_crossdev_pairs.py first")
    cross = load_csv(cross_path)

    nonempty = [r for r in files if str(r.get("empty_after_norm", "")).lower() != "true"]
    uf = UnionFind()
    for r in nonempty:
        uf.add(r["doc_id"])

    n_same = n_cross = 0
    for p in same:
        if float(p["estimated_jaccard"]) >= thr:
            uf.union(p["doc_id_a"], p["doc_id_b"])
            n_same += 1
    for p in cross:
        if float(p["estimated_jaccard"]) >= thr:
            uf.union(p["doc_id_a"], p["doc_id_b"])
            n_cross += 1

    by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in nonempty:
        by_doc[r["doc_id"]].append(r)

    comps: dict[str, list[str]] = defaultdict(list)
    for d in by_doc:
        comps[uf.find(d)].append(d)

    reps = []
    for docs in comps.values():
        recs = [x for d in docs for x in by_doc[d]]
        rep = max(recs, key=lambda r: (int(r.get("norm_chars") or 0), int(r.get("file_bytes") or 0)))
        cid = (
            f"solo:{docs[0][:16]}"
            if len(docs) == 1
            else "g:" + hashlib.sha256("|".join(sorted(docs)).encode()).hexdigest()[:24]
        )
        reps.append({
            "cluster_id": cid,
            "n_doc_ids": len(docs),
            "n_files": len(recs),
            "file_name": rep["file_name"],
            "file_path": rep["file_path"],
            "extension_id": rep.get("extension_id", ""),
            "doc_id": rep["doc_id"],
            "dev_key": rep.get("dev_key", ""),
        })

    stem = f"unique_files_j{thr:.2f}".replace(".", "p")
    names_txt = out / f"{stem}.txt"
    paths_txt = out / f"{stem}_paths.txt"
    names_txt.write_text("\n".join(r["file_name"] for r in reps) + "\n", encoding="utf-8")
    paths_txt.write_text("\n".join(r["file_path"] for r in reps) + "\n", encoding="utf-8")
    write_csv(out / f"{stem}.csv", reps)

    n_near = sum(1 for r in reps if int(r["n_doc_ids"]) >= 2)
    print(f"threshold                 {thr}")
    print(f"same-dev pairs merged     {n_same}")
    print(f"cross-dev pairs merged    {n_cross}")
    print(f"unique files              {len(reps)}")
    print(f"clusters size >= 2        {n_near}")
    print(f"names                     {names_txt}")
    print(f"paths                     {paths_txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
