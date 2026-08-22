#!/usr/bin/env python3
"""
Aggressive entity-swapping augmentation focused on rare entities
to reduce O-class dominance.
"""

import json
import random
from pathlib import Path
from collections import Counter
from typing import List, Tuple

# ====================== CONFIG ======================
INPUT  = Path("BIO_NER_original_labeled.jsonl")
OUTPUT = Path("bio_augmented_rare50_balanced_v2.jsonl")

SEED = 42

# How many new versions to create from each sentence that has entities
NUM_AUG_PER_ORIGINAL = 5          # ← increased (was 2)

# Only use the rarest X% of entities as replacement targets
RARE_PERCENTILE = 0.50

# Sometimes replace 2 entities in the same sentence (helps I count)
PROB_MULTI_REPLACE = 0.35

MIN_ENTITY_LEN = 1
MAX_ENTITY_LEN = 12
# ====================================================

random.seed(SEED)

def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def extract_entities(tokens: List[str], tags: List[str]) -> List[Tuple[int, int, List[str]]]:
    entities = []
    i = 0
    n = len(tokens)
    while i < n:
        if tags[i] == "B":
            start = i
            i += 1
            while i < n and tags[i] == "I":
                i += 1
            end = i
            ent_tokens = tokens[start:end]
            if MIN_ENTITY_LEN <= len(ent_tokens) <= MAX_ENTITY_LEN:
                entities.append((start, end, ent_tokens))
        else:
            i += 1
    return entities

def make_bio(length: int) -> List[str]:
    if length <= 0:
        return []
    return ["B"] + ["I"] * (length - 1)

def replace_entity(tokens, tags, start, end, new_ent_tokens):
    new_tokens = tokens[:start] + new_ent_tokens + tokens[end:]
    new_tags   = tags[:start] + make_bio(len(new_ent_tokens)) + tags[end:]
    return new_tokens, new_tags

def main():
    data = load_jsonl(INPUT)
    print(f"Loaded {len(data)} examples")

    # -------------------------------------------------
    # Collect entity frequencies
    # -------------------------------------------------
    entity_counter = Counter()
    for ex in data:
        for _, _, ent_toks in extract_entities(ex["tokens"], ex["ner_tags"]):
            entity_counter[" ".join(ent_toks)] += 1

    print(f"Unique data objects: {len(entity_counter)}")

    # Rarest 50%
    sorted_entities = entity_counter.most_common()[::-1]          # rare → frequent
    cutoff = max(1, int(len(sorted_entities) * RARE_PERCENTILE))
    rare_entities = [s for s, _ in sorted_entities[:cutoff]]

    print(f"Using rarest {len(rare_entities)} entities for replacement")
    print("Sample rare entities:")
    for s in rare_entities[:12]:
        print(f"  {entity_counter[s]:3d}×  |  {s}")

    # -------------------------------------------------
    # Augment
    # -------------------------------------------------
    augmented = []
    original_with_entity = 0
    aug_count = 0

    for ex in data:
        tokens = ex["tokens"]
        tags   = ex["ner_tags"]
        ents   = extract_entities(tokens, tags)

        # Always keep original
        augmented.append({"tokens": tokens, "ner_tags": tags})

        if not ents:
            continue

        original_with_entity += 1

        for _ in range(NUM_AUG_PER_ORIGINAL):
            # Work on a copy so we can do multiple replacements
            cur_tokens = tokens[:]
            cur_tags   = tags[:]
            cur_ents   = extract_entities(cur_tokens, cur_tags)

            # Decide how many replacements to do in this sentence
            n_replace = 2 if (len(cur_ents) >= 2 and random.random() < PROB_MULTI_REPLACE) else 1
            n_replace = min(n_replace, len(cur_ents))

            # Replace from the end so indices stay valid
            chosen = random.sample(cur_ents, n_replace)
            chosen = sorted(chosen, key=lambda x: x[0], reverse=True)

            for start, end, old_ent in chosen:
                old_surface = " ".join(old_ent)
                candidates = [e for e in rare_entities if e != old_surface]
                if not candidates:
                    continue

                new_surface = random.choice(candidates)
                new_ent_tokens = new_surface.split()

                cur_tokens, cur_tags = replace_entity(
                    cur_tokens, cur_tags, start, end, new_ent_tokens
                )

            if len(cur_tokens) != len(cur_tags):
                continue

            augmented.append({
                "tokens": cur_tokens,
                "ner_tags": cur_tags,
            })
            aug_count += 1

    # -------------------------------------------------
    # Save
    # -------------------------------------------------
    random.shuffle(augmented)

    with OUTPUT.open("w", encoding="utf-8") as f:
        for ex in augmented:
            f.write(json.dumps({
                "tokens": ex["tokens"],
                "ner_tags": ex["ner_tags"]
            }, ensure_ascii=False) + "\n")

    # Stats
    new_counts = Counter()
    for ex in augmented:
        new_counts.update(ex["ner_tags"])

    total = sum(new_counts.values())
    print("\n===== Augmentation finished =====")
    print(f"Original examples              : {len(data)}")
    print(f"Sentences that had entities    : {original_with_entity}")
    print(f"New augmented examples         : {aug_count}")
    print(f"Total after augmentation       : {len(augmented)}")
    print(f"\nLabel distribution:")
    for lab in ["O", "B", "I"]:
        cnt = new_counts.get(lab, 0)
        print(f"  {lab}: {cnt:7d}  ({100*cnt/total:.1f}%)")
    print(f"\nSaved → {OUTPUT}")

if __name__ == "__main__":
    main()