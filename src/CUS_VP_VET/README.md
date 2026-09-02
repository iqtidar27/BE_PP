# CUS phrase-tuple extractor (PPAudit method, your lists / NER)

Exact VPVet/PPAudit algorithm from
`kalamoo/PPAudit/cus_extract/extract_cus_phrase_tuple.py`:

1. spaCy parse + exclusion-phrase merge + noun-phrase merge
2. Tag tokens: SOC verb (lemma gazetteer) / DATA (your NER BIO spans) / ENTITY (your list)
3. Shortest-path dependency subgraph over tagged tokens
4. Collapse conjugated verbs/entities, prune non-sharing verbs
5. Walk share/collect verbs → agent, data objects, entities, negation, except-clauses
6. PolicyTransformer: share vs collect rewrite → `(entity, collect|not_collect, dataobj, sentence, verb)`

The graph / exception / transformation code is unmodified. Only the three
tagging sources are injected.

## Inputs you must point at

| Flag | What |
|---|---|
| `--ner-model` | HF token-classification model dir (your data-object NER). Accepts `B`/`I` or `B-DATA`/`I-DATA`. |
| `--soc-verbs` | JSON `{"share": [...], "collect": [...]}` of **lemmas**. Share vs collect is required because the transformer treats them differently. |
| `--entities` | JSON list or one phrase per line. Unigrams + multi-word (`third party`). Matched on `text.lower()` and `lemma_`. |

`lists/*.example.json` show the original PPAudit verb inventory plus a starter entity gazetteer. Replace them with yours.

## Run

```bash
cd cus_extract_adapted

# one sentence per line
python extract_cus_phrase_tuple.py \
  --ner-model /path/to/your/dataobj_ner \
  --soc-verbs /path/to/your/soc_verbs.json \
  --entities  /path/to/your/entities.json \
  --sentences /path/to/cus_candidate_sentences.txt \
  --out tuples.json

# original PPAudit batch layout: dir of {hash: {"CUS": [sentences]}}
python extract_cus_phrase_tuple.py \
  --ner-model /path/to/your/dataobj_ner \
  --soc-verbs /path/to/your/soc_verbs.json \
  --entities  /path/to/your/entities.json \
  --pp-components /path/to/pp_components \
  --out-dir cus_phrase_tuples \
  --analytics-dir analytics
```

From Python:

```python
from extract_cus_phrase_tuple import build_extractor, extract_cus_tuples_from_sentences

ex, _ = build_extractor(
    ner_model="/path/to/your/dataobj_ner",
    soc_verbs_path="/path/to/soc_verbs.json",
    entity_list_path="/path/to/entities.json",
    spacy_model="en_core_web_sm",   # VPVet used NlpFinalModel_Policheck
)
rows = extract_cus_tuples_from_sentences(ex, [
    "We collect your email address and share it with advertisers."
])
# {sentence, entity_phrase, cus_or_not, dataobj_phrase, cus_verb}
```

## Environment (this is the sharp edge)

PPAudit comments: `conda install spacy=2.0.18`.

`lib/NounPhraseMerger.py` calls `span.merge(...)` and `lib/ExclusionPhraseMerger.py`
uses the spaCy 2 `Matcher.add(id, callback, pattern)` API. Those will not run
unchanged on spaCy 3. Use the same stack VPVet used if you want bit-identical
trees:

- spaCy 2.0.x + `en_core_web_sm` (or their `NlpFinalModel_Policheck`)
- `transformers` pipeline `"ner"`
- `networkx`

If your NER also labels ENTITY/ORG/PERSON spans, pass
`--also-tag-ner-entities`. Pass `--no-spacy-entities` to disable the original
fallback `token.ent_type_ in {PERSON, ORG}`.

## Tuple meaning (unchanged)

`(entity_phrase, cus_or_not, dataobj_phrase, sentence, cus_verb)`

- `cus_or_not`: `collect` or `not_collect` after PolicyTransformer
- first-party implicit agent becomes `we_implicit`
- missing third party on a share verb becomes `third_party_implicit`
- “you” as agent is dropped (same as original)
