#!/usr/bin/env python3
"""
Step 1. Extract + normalize every policy file.
        Global exact SHA-256 (doc_id).
        Same-developer MinHash/LSH pair scores.

Does not apply the final uniqueness cutoff. That is step 3.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from common import (
    POLICY_EXTS,
    MinHasher,
    extract_extension_id,
    extract_raw_text,
    load_extensions,
    lsh_candidates,
    normalize_dev_key,
    normalize_text,
    estimated_jaccard,
    setup_logging,
    shingles,
    words_of,
    write_csv,
)


@dataclass
class FileRecord:
    file_path: str
    file_name: str
    suffix: str
    file_bytes: int
    extension_id: str
    ext_name: str
    developer_name_raw: str
    developer_email: str
    developer_website: str
    users_count: str
    external_privacy_policy_url: str
    dev_key: str
    extract_ok: bool
    extract_error: str
    raw_chars: int
    norm_chars: int
    n_words: int
    n_shingles: int
    doc_id: str
    empty_after_norm: bool


def main() -> int:
    ap = argparse.ArgumentParser(description="Exact hash + same-developer near-dup pairs")
    ap.add_argument("--policies-dir", required=True)
    ap.add_argument("--extensions-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shingle-k", type=int, default=5)
    ap.add_argument("--num-perm", type=int, default=128)
    ap.add_argument("--lsh-bands", type=int, default=32)
    ap.add_argument("--lsh-rows", type=int, default=4)
    ap.add_argument("--min-norm-chars", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    policies_dir = Path(args.policies_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out_dir, "dedup_same_dev")

    if args.lsh_bands * args.lsh_rows > args.num_perm:
        log.error("lsh-bands * lsh-rows must be <= num-perm")
        return 2

    meta_by_id = load_extensions(Path(args.extensions_json).expanduser())
    log.info("Loaded %s extension records", len(meta_by_id))

    files = sorted(
        p for p in policies_dir.iterdir()
        if p.is_file() and p.suffix.lower() in POLICY_EXTS
    )
    if args.limit:
        files = files[: args.limit]
    log.info("Found %s policy files", len(files))

    hasher = MinHasher(num_perm=args.num_perm)
    records: list[FileRecord] = []
    signatures: list[list[int]] = []
    unmatched: list[dict] = []
    failures: list[dict] = []
    empty_docs: list[dict] = []

    t0 = time.time()
    for i, path in enumerate(files, 1):
        if i % 2000 == 0 or i == 1:
            log.info("Extracting %s / %s  (%s)", i, len(files), path.name)

        eid = extract_extension_id(path.name)
        meta = meta_by_id.get(eid)
        if not eid or meta is None:
            unmatched.append({
                "file_path": str(path),
                "file_name": path.name,
                "parsed_extension_id": eid,
                "reason": "no_id_in_filename" if not eid else "id_not_in_extensions_json",
            })
        dev_key = normalize_dev_key(meta, eid or path.stem)

        extract_ok, extract_error, raw = True, "", ""
        try:
            raw = extract_raw_text(path)
        except Exception as e:
            extract_ok, extract_error = False, str(e)
            failures.append({"file_path": str(path), "extension_id": eid, "error": extract_error})

        norm = normalize_text(raw) if extract_ok else ""
        empty = (not extract_ok) or (len(norm) < args.min_norm_chars)
        wds = words_of(norm) if not empty else []
        sh = shingles(wds, k=args.shingle_k) if not empty else set()
        if empty:
            doc_id = "empty:" + hashlib.sha256((path.name + extract_error).encode()).hexdigest()
            empty_docs.append({
                "file_path": str(path),
                "extension_id": eid,
                "extract_ok": extract_ok,
                "extract_error": extract_error,
                "norm_chars": len(norm),
            })
            sig: list[int] = []
        else:
            doc_id = hashlib.sha256(norm.encode("utf-8")).hexdigest()
            sig = hasher.signature(sh)
        signatures.append(sig)

        records.append(FileRecord(
            file_path=str(path.resolve()),
            file_name=path.name,
            suffix=path.suffix.lower(),
            file_bytes=path.stat().st_size,
            extension_id=eid,
            ext_name=(meta or {}).get("name") or "",
            developer_name_raw=(meta or {}).get("developer_name") or "",
            developer_email=(meta or {}).get("developer_email") or "",
            developer_website=(meta or {}).get("developer_website") or "",
            users_count=str((meta or {}).get("users_count") or ""),
            external_privacy_policy_url=(meta or {}).get("external_privacy_policy_url") or "",
            dev_key=dev_key,
            extract_ok=extract_ok,
            extract_error=extract_error,
            raw_chars=len(raw),
            norm_chars=len(norm),
            n_words=len(wds),
            n_shingles=len(sh),
            doc_id=doc_id,
            empty_after_norm=empty,
        ))

    log.info("Extraction done in %.1fs", time.time() - t0)

    by_doc: dict[str, list[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        if not rec.empty_after_norm:
            by_doc[rec.doc_id].append(idx)

    exact_groups = []
    for doc_id, members in sorted(by_doc.items(), key=lambda kv: -len(kv[1])):
        recs = [records[i] for i in members]
        exact_groups.append({
            "doc_id": doc_id,
            "n_files": len(members),
            "n_extensions": len({r.extension_id for r in recs if r.extension_id}),
            "n_dev_keys": len({r.dev_key for r in recs}),
            "norm_chars": recs[0].norm_chars,
            "representative_file": recs[0].file_name,
            "extension_ids": " ".join(sorted({r.extension_id for r in recs if r.extension_id})),
            "dev_keys": " | ".join(sorted({r.dev_key for r in recs})),
        })
    write_csv(out_dir / "exact_doc_groups.csv", exact_groups)
    write_csv(out_dir / "cross_developer_exact_docs.csv",
              [g for g in exact_groups if g["n_dev_keys"] > 1])

    doc_sig: dict[str, list[int]] = {}
    doc_rep: dict[str, int] = {}
    by_dev: dict[str, list[str]] = defaultdict(list)
    for idx, rec in enumerate(records):
        if rec.empty_after_norm:
            continue
        if rec.doc_id not in doc_sig:
            doc_sig[rec.doc_id] = signatures[idx]
            doc_rep[rec.doc_id] = idx
        by_dev[rec.dev_key].append(rec.doc_id)
    for dev in by_dev:
        by_dev[dev] = sorted(set(by_dev[dev]))

    pair_rows = []
    t1 = time.time()
    for dev_key, doc_ids in by_dev.items():
        if len(doc_ids) < 2:
            continue
        local = [(i, doc_sig[d]) for i, d in enumerate(doc_ids)]
        for a, b in sorted(lsh_candidates(local, args.lsh_bands, args.lsh_rows)):
            ja = estimated_jaccard(doc_sig[doc_ids[a]], doc_sig[doc_ids[b]])
            ra, rb = records[doc_rep[doc_ids[a]]], records[doc_rep[doc_ids[b]]]
            pair_rows.append({
                "dev_key": dev_key,
                "doc_id_a": doc_ids[a],
                "doc_id_b": doc_ids[b],
                "file_a": ra.file_name,
                "file_b": rb.file_name,
                "ext_a": ra.extension_id,
                "ext_b": rb.extension_id,
                "estimated_jaccard": f"{ja:.4f}",
                "norm_chars_a": ra.norm_chars,
                "norm_chars_b": rb.norm_chars,
            })
    pair_rows.sort(key=lambda r: -float(r["estimated_jaccard"]))
    write_csv(out_dir / "near_dup_pair_scores.csv", pair_rows)
    log.info("Same-dev LSH in %.1fs  candidate pairs=%s", time.time() - t1, len(pair_rows))

    write_csv(out_dir / "file_manifest.csv", [asdict(r) for r in records])
    write_csv(out_dir / "unmatched_files.csv", unmatched)
    write_csv(out_dir / "extract_failures.csv", failures)
    write_csv(out_dir / "empty_or_short_docs.csv", empty_docs)

    summary = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "n_files": len(records),
        "n_nonempty": sum(1 for r in records if not r.empty_after_norm),
        "n_exact_doc_ids": len(by_doc),
        "n_same_dev_pairs": len(pair_rows),
        "n_unmatched": len(unmatched),
        "n_empty": len(empty_docs),
    }
    (out_dir / "step1_summary.json").write_text(
        __import__("json").dumps(summary, indent=2), encoding="utf-8"
    )
    log.info("exact doc_ids=%s  same-dev pairs=%s", len(by_doc), len(pair_rows))
    log.info("Wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
