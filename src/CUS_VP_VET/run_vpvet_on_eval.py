#!/usr/bin/env python3
"""Run the PPAudit/VPVet CUS pipeline on cus_eval_sentences.jsonl.

From the repo root:

    conda activate privacy   # or whatever env has spacy + transformers
    python src/CUS_VP_VET/run_vpvet_on_eval.py

Writes:
    cus_vpvet_eval_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2] if Path(__file__).name == "run_vpvet_on_eval.py" else Path.cwd()
# src/CUS_VP_VET/run_vpvet_on_eval.py  -> parents[2] == repo root
# also allow running from repo root or from CUS_VP_VET/
HERE = Path(__file__).resolve().parent
if (HERE / "cus_extractor.py").exists():
    VPVET_DIR = HERE
    # file lives in src/CUS_VP_VET
    if HERE.parent.name == "src":
        REPO_ROOT = HERE.parent.parent
    else:
        REPO_ROOT = Path.cwd()
else:
    VPVET_DIR = REPO_ROOT / "src" / "CUS_VP_VET"

sys.path.insert(0, str(VPVET_DIR))

from cus_extractor import CUSExtractor  # noqa: E402

DEFAULT_INPUT = REPO_ROOT / "src" / "CUS" / "cus_eval_sentences.jsonl"
DEFAULT_NER = (
    REPO_ROOT
    / "src"
    / "NER_detection"
    / "saved_model"
    / "dataobj_ner_v5_focal1.5_len256_unf6_dataset_version_v6_augmented_final"
)
DEFAULT_OUT = REPO_ROOT / "cus_vpvet_eval_results.json"


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
    return {
        "entity": p.get("entity_phrase"),
        "party": p.get("party", "unspecified"),
        "action": p.get("cus_or_not"),
        "data_object": p.get("dataobj_phrase"),
        "verb": p.get("cus_verb"),
        "negated": str(p.get("cus_or_not", "")).startswith("not_"),
    }


def main():
    ap = argparse.ArgumentParser(description="VPVet CUS extractor over eval JSONL")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--ner-model", type=Path, default=DEFAULT_NER)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--spacy-model", default="en_core_web_sm")
    ap.add_argument("--extension-name", default=None, help="Optional first-party product name")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--debug", action="store_true", help="Raise on first extract error")
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
    for i, rec in enumerate(records, 1):
        sid = rec.get("id", f"row_{i}")
        sentence = rec.get("sentence", "")
        try:
            preds = extractor.extract(sentence, extension_name=args.extension_name)
            err = None
        except Exception as exc:
            preds = []
            err = f"{type(exc).__name__}: {exc}"
            if i <= 3:
                print("[error on {}] {}".format(sid, err))
            if args.debug:
                raise
        tuples = [pred_to_tuple(p) for p in preds]
        n_pred += len(tuples)
        if not tuples:
            n_empty += 1
        results.append({
            "id": sid,
            "source": rec.get("source"),
            "category": rec.get("category"),
            "has_cus_signal": rec.get("has_cus_signal"),
            "sentence": sentence,
            "gold_tuples": rec.get("gold_tuples") or [],
            "pred_tuples": tuples,
            "n_pred": len(tuples),
            "error": err,
        })
        if i % 25 == 0 or i == len(records):
            print(f"[run ] {i}/{len(records)}  preds={n_pred}  empty={n_empty}")

    payload = {
        "meta": {
            "input": str(args.input),
            "ner_model": str(args.ner_model),
            "spacy_model": args.spacy_model,
            "n_sentences": len(records),
            "n_pred_tuples": n_pred,
            "n_sentences_with_no_pred": n_empty,
            "seconds": round(time.time() - t1, 2),
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {args.out}")
    print(json.dumps(payload["meta"], indent=2))


if __name__ == "__main__":
    main()
