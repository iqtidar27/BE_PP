#!/usr/bin/env python3
"""Run CUS tuple extraction on segmented-policy JSONL.

Each input line looks like:

    {
      "source_file": "<extension-id>",
      "policy_path": "...",
      "segment_id": 0,
      "heading": "",
      "rule": "C1",
      "sentence_id": 11,
      "sentence": "...",
      "matched_quotes": [...],
      "n_quotes_in_segment": 10
    }

From the repo root:

    PYTHONPATH=src/CUS_VP_VET:$PYTHONPATH python src/CUS_VP_VET/run_cus_on_segments.py \\
        --input /path/to/sentences.jsonl \\
        --out   /path/to/cus_tuples.jsonl

Default output is JSONL (one record per input sentence). Use --json for a
single blob with meta + results.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if (HERE / "cus_extractor.py").exists():
    VPVET_DIR = HERE
    REPO_ROOT = HERE.parent.parent if HERE.parent.name == "src" else Path.cwd()
else:
    REPO_ROOT = Path.cwd()
    VPVET_DIR = REPO_ROOT / "src" / "CUS_VP_VET"

sys.path.insert(0, str(VPVET_DIR))

from cus_extractor import CUSExtractor  # noqa: E402

DEFAULT_NER = (
    REPO_ROOT
    / "src"
    / "NER_detection"
    / "saved_model"
    / "dataobj_ner_v5_focal1.5_len256_unf6_dataset_version_v6_augmented_final"
)

KEEP_FIELDS = (
    "source_file",
    "extension_id",
    "extension_name",
    "policy_path",
    "segment_id",
    "heading",
    "rule",
    "sentence_id",
    "sentence",
    "matched_quotes",
    "n_quotes_in_segment",
)


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Bad JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def pred_to_tuple(p):
    action = p.get("cus_or_not") or ""
    return {
        "entity": p.get("entity_phrase"),
        "party": p.get("party", "unspecified"),
        "action": action,  # collect | not_collect | share | not_share
        "data_object": p.get("dataobj_phrase"),
        "verb": p.get("cus_verb"),
        "negated": str(action).startswith("not_"),
    }


def pack_record(rec, tuples, err=None):
    out = {k: rec.get(k) for k in KEEP_FIELDS}
    out["pred_tuples"] = tuples
    out["n_pred"] = len(tuples)
    out["error"] = err
    return out


def main():
    ap = argparse.ArgumentParser(description="CUS extractor over segmented-policy JSONL")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ner-model", type=Path, default=DEFAULT_NER)
    ap.add_argument("--spacy-model", default="en_core_web_sm")
    ap.add_argument(
        "--extension-name",
        default=None,
        help="Optional first-party product name applied to every sentence",
    )
    ap.add_argument(
        "--use-source-as-name",
        action="store_true",
        help="Pass source_file as extension_name (usually an extension id)",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--json", action="store_true", help="Write one JSON object instead of JSONL")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input}")
    if not args.ner_model.exists():
        raise SystemExit(f"NER model not found: {args.ner_model}")

    records = load_jsonl(args.input)
    if args.limit:
        records = records[: args.limit]

    print(f"[load] {len(records)} sentences from {args.input}")
    print(f"[ner ] {args.ner_model}")
    t0 = time.time()
    extractor = CUSExtractor(
        ner_model_path=str(args.ner_model),
        spacy_model=args.spacy_model,
    )
    print(f"[init] extractor ready in {time.time() - t0:.1f}s")

    results = []
    n_pred = 0
    n_empty = 0
    t1 = time.time()

    out_fh = None
    if not args.json:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out_fh = args.out.open("w", encoding="utf-8")

    try:
        for i, rec in enumerate(records, 1):
            sentence = rec.get("sentence") or ""
            name = args.extension_name or rec.get("extension_name")
            if not name and args.use_source_as_name:
                name = rec.get("source_file") or rec.get("extension_id")
            err = None
            try:
                preds = extractor.extract(sentence, extension_name=name) if sentence else []
            except Exception as exc:
                preds = []
                err = f"{type(exc).__name__}: {exc}"
                print(f"[error row {i}] {err}")
                if args.debug:
                    raise
            tuples = [pred_to_tuple(p) for p in preds]
            n_pred += len(tuples)
            if not tuples:
                n_empty += 1
            packed = pack_record(rec, tuples, err)
            if out_fh is not None:
                out_fh.write(json.dumps(packed, ensure_ascii=False) + "\n")
            else:
                results.append(packed)
            if i % 50 == 0 or i == len(records):
                print(f"[run ] {i}/{len(records)}  preds={n_pred}  empty={n_empty}")
    finally:
        if out_fh is not None:
            out_fh.close()

    meta = {
        "input": str(args.input),
        "out": str(args.out),
        "ner_model": str(args.ner_model),
        "spacy_model": args.spacy_model,
        "n_sentences": len(records),
        "n_pred_tuples": n_pred,
        "n_sentences_with_no_pred": n_empty,
        "seconds": round(time.time() - t1, 2),
    }

    if args.json:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"meta": meta, "results": results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"[done] {args.out}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
