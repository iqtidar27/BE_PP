# CUS evaluation sentence dataset (English only)

**Total sentences:** 220  
**Language:** en only (FR/PL removed)

## By source
- synthetic-extension-policy: 110
- f-secure-browsing-protection: 47
- sanity-negatives: 25
- git-master: 15
- joblog: 10
- speed-dial-2: 7
- generator-czcionek: 6

## Signal vs negative
- has_cus_signal=True: 152
- has_cus_signal=False: 68

## Categories (top)
- synthetic_extra: 80
- negative: 25
- collect_first_party: 22
- not_share: 13
- collect_third_party: 12
- data_list: 9
- sanity: 7
- use: 6
- share_third_party: 5
- rights: 5
- sanity_product: 4
- store_local: 3
- sanity_purpose: 3
- sanity_legal: 3
- sanity_security: 2
- share_sync: 2
- share_partner: 2
- not_collect: 2
- share_first_party: 2
- data_mention: 1
- sanity_advice: 1
- sanity_compliance: 1
- sanity_entity: 1
- retention: 1
- collect_and_not_store: 1

## Files
- `cus_eval_sentences.jsonl` — one JSON object per line
- `cus_eval_sentences.json` — full array

## Field schema
```
id, source, language, sentence, category, has_cus_signal, gold_tuples, notes
gold_tuples: [{entity, action, data_object, negated}, ...]
action: collect | share | not_collect | not_share
```

## Notes
- English only (joblog/generator paraphrases kept; authentic FR/PL dropped).
- Sanity negatives should produce no tuples.
- Gold labels are approximate; refine before formal P/R.
