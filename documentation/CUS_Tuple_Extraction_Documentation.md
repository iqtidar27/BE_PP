# CUS Tuple Extraction Pipeline: Design, Implementation, and Comparison with VPVet

**Document type:** Technical design and method description  
**Purpose:** Internal documentation and foundation for academic writing  
**Scope:** Stage B dependency-based Collect–Use–Share (CUS) tuple extraction for privacy-policy text, with emphasis on browser-extension / modern SaaS policies  
**Status:** Pipeline finalized except for targeted Data Object NER coverage improvements  

---

## 1. Introduction

Privacy policies describe how organizations collect, use, and share personal data. Automated analysis of these documents is central to privacy compliance research, usable privacy tools, and large-scale policy auditing. A common intermediate representation is the **CUS tuple**: a structured statement of the form

\[
(\textit{Entity},\; \textit{Action},\; \textit{Data Object})
\]

where:

- **Entity** is the actor (first party or third party) that performs the action;
- **Action** is a normalized privacy-relevant operation, typically *collect* or *share* (and their negations);
- **Data Object** is the type of information involved (e.g., *email address*, *device identifiers*, *browsing history*).

This document describes a **Stage B, dependency-based CUS extraction pipeline** designed for privacy-policy sentences. The pipeline integrates:

1. Expanded SOC (Subject–Object–Complement style) verb inventories, including multi-word expressions;
2. Domain-adapted entity recognition with explicit first-party and third-party lexicons;
3. A fine-tuned token-level Named Entity Recognition (NER) model for data objects;
4. Dependency-driven linking of entities, verbs, and data objects;
5. Negation handling and light post-filtering for noise reduction.

The design is deliberately modular and is compared throughout with the **VPVet / PPAudit** approach (KeyphraseTagger and related CUS extraction logic), which is a strong and widely referenced baseline in privacy-policy analysis.

---

## 2. Problem Definition

### 2.1 Input

A natural-language sentence \(s\) drawn from a privacy policy (or a similar legal/consumer-facing disclosure), for example:

> “Google Analytics collects anonymized usage data from our users.”

### 2.2 Output

A set of CUS tuples \(\mathcal{T}(s)\), each with:

| Field | Description | Example |
|-------|-------------|---------|
| `entity` | Surface form of the actor | `Google Analytics` |
| `entity_normalized` | Lower-cased entity | `google analytics` |
| `entity_type` | `first_party` \| `third_party` \| `unknown` | `third_party` |
| `action` | `collect` \| `share` \| `not_collect` \| `not_share` | `collect` |
| `verb` | Surface verb | `collects` |
| `verb_normalized` | Lemma | `collect` |
| `data_object` | Surface data phrase | `usage data` |
| `data_object_normalized` | Lower-cased data phrase | `usage data` |
| `negated` | Boolean | `False` |
| `sentence` | Original sentence | full text |

When no explicit actor is present (e.g., pure passive “No personal information is collected…”), the pipeline may emit a synthetic entity `unspecified` with type `unknown`.

### 2.3 Design goals

1. **High precision on domain actors** (analytics SDKs, payment processors, AI APIs common in extension and SaaS policies).
2. **Robustness to policy syntax** (active/passive, negation, coordination, auxiliary constructions).
3. **Modularity** so verb lists, entity lists, and the data-object model can be improved independently.
4. **Traceability** for academic reporting: each stage has a clear input/output contract.

---

## 3. Background: VPVet / PPAudit CUS Extraction

VPVet-style extraction (as reflected in PPAudit’s `extract_cus_phrase_tuple.py` and KeyphraseTagger-style logic) typically proceeds as follows:

### 3.1 Verb detection

- Fixed lemma lists for **collect** and **share** verbs.
- Matching is largely lemma- and dependency-driven.
- Multi-word verbal expressions are limited or handled less systematically than single lemmas.

### 3.2 Entity detection

- Pronouns such as *we*, *us*, *I*, *you*, *our*.
- Generic spaCy named-entity labels, primarily `PERSON` and `ORG`.
- Little or no domain lexicon for modern third parties (e.g., *PostHog*, *Mixpanel*, *Stripe*, *Sentry*).

### 3.3 Data objects

- Internal phrase / pattern mechanisms (not a task-specific fine-tuned PrivBERT-style sequence labeler in the same form as the present work).

### 3.4 Linking

- Dependency relations and graph-oriented processing, including coordination collapsing for complex multi-entity / multi-object structures.
- Negation is tracked, often with distance or structural cues.

### 3.5 Strengths of VPVet

- Principled dependency use.
- Coordination handling via graph collapsing.
- Established ontology-oriented downstream analysis in the broader VPVet line of work.

### 3.6 Limitations relevant to modern policies

- **Weak domain third-party coverage:** spaCy `ORG`/`PERSON` frequently miss or mislabel product/SDK names (*Mixpanel* as person-like, *Stripe* absent, etc.).
- **No explicit first-party generics:** phrases such as *the company*, *the extension*, *the app* are not first-class.
- **Static verb inventories** without systematic multi-word phrase matching or corpus-driven expansion.
- **Noisy actor candidates:** short technical tokens (e.g., *IP* from “IP address”) may be tagged as `ORG`.

These limitations motivate the design choices below.

---

## 4. System Overview

The pipeline is organized as a sequence of modular stages:

```
Sentence
   │
   ▼
spaCy Doc (tokenization, POS, dependency parse)
   │
   ├──► VerbMatcher          → SOC verbs (collect / share)
   ├──► EntityRecognizer     → first_party / third_party spans
   └──► DataObjectAligner    → data-object spans (via fine-tuned NER)
   │
   ▼
SentenceTagger               → tagged tokens + span lists
   │
   ▼
TupleExtractor               → candidate CUS tuples (dependency linking)
   │
   ▼
TupleFilter                  → noise reduction / possessive rewrite
   │
   ▼
List[CUSTuple]
```

**Orchestrator:** `CUSExtractor`  
**Primary implementation file:** `cus_extractor.py`  
**Supporting resources:** `soc_verbs.py`, `entitiy_list.py`, `NER_detection/inference_ner.py`

---

## 5. Component Design

### 5.1 Linguistic preprocessing

- Model: `en_core_web_sm` (spaCy).
- Provides tokenization, lemmatization, POS tags, and dependency trees used by verb filtering and relation heuristics.
- An `EntityRuler` is inserted **before** the statistical NER so domain third-party patterns take priority.

### 5.2 VerbMatcher (SOC verbs)

#### 5.2.1 Resources

- `COLLECT_VERBS`: lemma inventory for collection-like actions  
  (e.g., *collect*, *obtain*, *process*, *store*, *receive*, *access*, *log*, *set*, *place*, …).
- `SHARE_VERBS`: lemma inventory for disclosure/sharing  
  (e.g., *share*, *disclose*, *sell*, *transmit*, *send*, *provide*, …).
- `MULTI_WORD_COLLECT` / `MULTI_WORD_SHARE`: phrase lists matched with spaCy `PhraseMatcher` (`attr="LOWER"`).

Verb lists combine:

1. Seeds aligned with VPVet-style collect/share inventories;
2. Corpus-driven candidates mined from a labeled privacy-policy sentence set (~5.5k BIO sentences) via dependency patterns (e.g., verbs governing data-object spans);
3. Manual curation (multi-word phrases and domain-critical lemmas such as *set*/*place* for cookies).

#### 5.2.2 Matching procedure

1. **Multi-word matches first** via `PhraseMatcher`; the syntactic head of the matched span is retained as the representative token.
2. **Single-word matches** on lemmas against collect/share sets, excluding tokens already covered by multi-word matches.
3. **Robust acceptance filter** to reduce noun/verb ambiguity:
   - Always accept `pos == VERB`.
   - For known SOC lemmas mis-tagged as `NOUN`/`PROPN` (observed for forms such as *collects* under some entity contexts), accept when dependency is among `{ROOT, ccomp, xcomp, relcl, advcl, conj, nsubj}`.
   - Ordinary nouns are accepted only under clearly clausal dependencies.

This filter addresses a concrete failure mode: spaCy occasionally tags a true collect verb as `NOUN` with `nsubj` in sentences headed by multi-token product names (e.g., “Google Analytics collects …”).

#### 5.2.3 Output

List of `(Token, verb_type)` with `verb_type ∈ {collect, share}`.

---

### 5.3 EntityRecognizer

#### 5.3.1 First-party inventory

Includes:

- Pronouns: *we, us, our, I, me, you, your*;
- Short generics: *the company, the app, the application, the service, the extension, the platform, the website, …*;
- Longer forms: *the service provider, the data controller, the data processor, …*.

Multi-word first-party phrases are matched with `PhraseMatcher`, **sorted by length descending**, so longer phrases are preferred over nested shorter ones when the matcher and `seen`-token set are applied correctly.

#### 5.3.2 Third-party inventory

A curated lexicon of services frequently named in extension and SaaS policies, grouped by function, for example:

- Analytics / monitoring: *Google Analytics, PostHog, Mixpanel, Sentry, Firebase Crashlytics, Meta Pixel, …*
- Payments: *Stripe, PayPal, Razorpay, …*
- Cloud / infra: *Cloudflare, Supabase, Vercel, Google Cloud, …*
- AI / LLM: *OpenAI, Anthropic, Gemini, …*
- Additional high-value names retained from coverage analysis and VPVet-style candidate mining.

Patterns are registered with the EntityRuler in both original and lower-cased forms.

#### 5.3.3 Recognition order

1. Third-party spans from EntityRuler / `doc.ents`;
2. First-party multi-word phrases (longest first);
3. First-party single tokens.

Overlaps are suppressed via a token-index `seen` set.

#### 5.3.4 Output

`EntitySpan` objects with full surface text (not only the syntactic head), normalized text, type, token list, and root token.

**Full-span preservation** is intentional: multi-word names such as *Google Analytics* and *Firebase Crashlytics* must remain intact for downstream tuple quality.

---

### 5.4 DataObjectAligner

#### 5.4.1 Model

A fine-tuned token classification model (PrivBERT-based) trained on BIO-labeled privacy-policy data for **data objects** only. Labels: `O`, `B`, `I`.

Inference is exposed through `DataObjectNER.predict(text)`, returning character-level spans.

#### 5.4.2 Alignment

For each predicted character span \([start, end)\):

1. Collect spaCy tokens overlapping the span;
2. Keep the **full surface string** from the NER output (or equivalent token join);
3. Choose a root token (typically the last overlapping token, often the nominal head).

Output: `DataObjectSpan` list.

This design fixed an early failure mode in which only the final token of a multi-word data object (e.g., *address* from *IP address*) was retained.

#### 5.4.3 Known coverage gaps (deferred work)

Empirical evaluation showed residual false negatives for certain phrases even when they exist in the training distribution, notably:

- *user behavior*
- *settings* / *preferences* in some co-occurrence contexts
- *cookies* under *set* / *place* constructions
- *crash reports* in some passive “sent to X” sentences
- bare *data* / *information* (intentionally low priority due to noise)

A small curated BIO batch was prepared for these gaps; integrating it is **out of scope** of the finalized extraction logic and is treated as NER model improvement.

---

### 5.5 SentenceTagger

Aggregates:

- Verb flags and types on tokens;
- Entity span list;
- Data-object span list;
- Negation word flags (`not`, `n't`, `never`, `no`, `neither`, `nor`).

Returns `(tagged_tokens, entity_spans, data_spans)`.

---

### 5.6 TupleExtractor (linking)

For each detected SOC verb:

1. Resolve **main verb** (follow `aux` / `auxpass` to the governing lexical verb when needed).
2. Detect **negation** (window around the verb; also sentence-initial *No* / *None* / *Never*).
3. Map to action: `collect` / `share` or `not_collect` / `not_share`.
4. Find **related data objects** and **related entities**.
5. If data objects exist but entities do not, insert synthetic entity `unspecified` (`unknown`).
6. Emit the Cartesian product of related entities × related data objects for that verb.

#### 5.6.1 Entity–verb relation

Preference order:

1. Overlap with syntactic subject / passive subject (`nsubj`, `nsubjpass`) or agent phrase (`agent` → prepositional object);
2. Nearby **third-party** entities (distance threshold);
3. Nearby non-possessive **first-party** entities.

This reduces the combinatorial explosion of pairing every pronoun in the sentence with every verb—an early noise source when all first-party mentions were treated equally.

#### 5.6.2 Data–verb relation

Tolerant heuristics:

- Any data-object token in the main verb’s subtree;
- Main verb among ancestors of the data-object root;
- Proximity thresholds;
- Data object appearing after the verb within a bounded distance (common linear order: *collects usage data*).

#### 5.6.3 Passive constructions

Passive collect/share sentences without an explicit *by*-agent (e.g., “No personal information is collected through this service”) are supported via:

- Verb detection on the passive participle;
- Negation from leading *No* or nearby negators;
- Synthetic `unspecified` entity when no actor span is found.

---

### 5.7 TupleFilter (post-processing)

Two light rules:

1. **Possessive-only actors** (*your*, *our*, *my*, …): rewrite entity to `unspecified` rather than dropping the tuple.  
   Rationale: the data object and action may still be valid; the actor is simply not a named party in the local span.

2. **Weak secondary pronoun tuples:** if a tuple’s entity is a weak pronoun (*you*, *we*, …) and another tuple for the **same data object** has a stronger entity (third party or non-weak first party), drop the weak one.  
   Rationale: suppresses noise such as *you – collect – prompts* when *OpenAI – collect – prompts* already exists.

---

## 6. End-to-End Algorithm (Summary)

**Input:** sentence string \(s\), path to data-object NER model  

**Steps:**

1. `doc ← nlp(s)`
2. `verbs ← VerbMatcher.find_verbs(doc)`
3. `entities ← EntityRecognizer.find_entities(doc)`
4. `data_objects ← DataObjectAligner.find_data_objects(doc)`
5. Mark negation tokens on the token sequence
6. For each verb \(v\):
   - Determine action and polarity
   - `D ← related data objects for v`
   - `E ← related entities for v` (or `{unspecified}` if \(E=\emptyset\) and \(D\neq\emptyset\))
   - Emit tuples for each \((e,d) \in E \times D\)
7. Apply `TupleFilter`
8. Return filtered tuple list

---

## 7. Comparison with VPVet

### 7.1 Qualitative comparison

| Dimension | VPVet / PPAudit-style | This pipeline |
|-----------|----------------------|---------------|
| Collect/share verbs | Fixed lemma lists | Expanded lists + multi-word phrases + corpus mining + robustness to POS errors |
| Entities | Pronouns + spaCy PERSON/ORG | Curated first-party generics + large third-party lexicon + EntityRuler |
| Entity typing | Implicit / generic | Explicit `first_party` / `third_party` / `unknown` |
| Data objects | Internal patterns | Task-specific fine-tuned BIO NER + full-span alignment |
| Multi-word names | Limited | Full surface spans for entities and data objects |
| Pure passive without agent | Varies | Explicit `unspecified` entity |
| Coordination | Stronger graph collapsing | Simpler subtree + proximity (adequate for many cases; complex coordination deferred) |
| Domain fit (extensions/SaaS) | Generic | Explicitly optimized |

### 7.2 Empirical entity-recognition comparison (controlled sentence set)

On a held set of privacy-policy-style test sentences used during development:

- Entities found by **both** methods: majority of overlapping pronouns and some well-known ORGs.
- Entities found **only by this pipeline:** substantially more (domain third parties such as *Stripe*, *Mixpanel*, *Sentry*, *Anthropic*, and first-party generics such as *the company*, *the extension*, *the app*).
- Entities found **only by VPVet-style spaCy ORG:** few, and partly **noise** (e.g., *IP* from “IP address” labeled as organization).

**Conclusion:** For actor identification in modern policy text, the curated EntityRuler approach dominates generic PERSON/ORG tagging, while retaining pronoun coverage comparable to VPVet.

### 7.3 Why the pipeline is better suited to the target domain

1. **Third-party density:** Extension and SaaS policies repeatedly name SDKs and processors that general NER does not reliably label.
2. **First-party generics:** Legal style uses *the company* / *the service* rather than only *we*.
3. **Action inventory:** Cookie and telemetry language uses verbs (*set*, *place*, *log*) underrepresented in minimal collect lists.
4. **Data-object specialization:** A dedicated sequence labeler trained on policy BIO data fits the long-tailed inventory of data types better than generic patterns alone.
5. **Engineering transparency:** Modular stages map cleanly onto ablation experiments and error analysis for academic reporting.

### 7.4 Where VPVet may still be stronger

- **Heavy coordination** across multiple verbs, objects, and parties in a single sentence (full dependency-graph collapsing).
- **Downstream ontology mapping** from surface tuples to higher-level policy concepts (present in broader VPVet-style analysis stacks).

These are acknowledged as future enhancements rather than blockers for the current baseline.

---

## 8. Implementation Notes

### 8.1 Repository layout (representative)

```text
src/
  CUS/
    cus_extractor.py      # orchestrator + all Stage B components
    soc_verbs.py          # collect/share lemmas and multi-word phrases
    entitiy_list.py       # FIRST_PARTY, THIRD_PARTY
    __init__.py
  NER_detection/
    inference_ner.py      # DataObjectNER
    saved_model/...       # fine-tuned weights
    NER_dataset/...       # BIO training data
    __init__.py
```

### 8.2 Import and execution

Because `CUS` and `NER_detection` are sibling packages under `src`:

```python
from NER_detection.inference_ner import DataObjectNER
```

Run from repository root:

```bash
PYTHONPATH=src python -m CUS.cus_extractor
```

Alternatively, a small `sys.path` bootstrap can support direct script execution.

### 8.3 Public API

```python
from CUS.cus_extractor import CUSExtractor

extractor = CUSExtractor(ner_model_path="path/to/dataobj_ner_model")
tuples = extractor.extract("Stripe processes your payment information on our behalf.")
for t in tuples:
    print(t.entity, t.action, t.data_object, t.negated)
```

---

## 9. Evaluation Methodology (Used During Development)

### 9.1 Smoke and regression sentences

A fixed list of policy-like sentences covering:

- First-party active collect/share;
- Third-party subjects;
- Negation and leading *No*;
- Passive with and without *by*-agent;
- Coordination of entities;
- Possessive-only subjects;
- Known NER stress phrases (*user behavior*, *settings*, *cookies* with *set*/*place*).

### 9.2 Component-level debug protocol

For failures, intermediate dumps were recorded:

- Detected verbs (surface, type, POS, dependency);
- Entity spans and types;
- Data-object spans;
- Per-verb related entity/data sets after linking;
- Final tuples after filtering.

This protocol separated **NER misses** from **linking** and **filter** errors and guided the decision that remaining empty outputs on hard sentences were predominantly data-object recall issues.

### 9.3 Dataset inventory analysis

Frequency analysis over the BIO corpus (~6.7k data-object mentions) confirmed that high-value phrases such as *cookies*, *preferences*, and *settings* appear often in text, while bare *information*/*data* dominate miss counts but are noisy targets. Improvements were therefore scoped to high-precision phrases rather than blind corpus expansion.

---

## 10. Limitations

1. **Data Object NER recall** remains the main source of empty extractions for a subset of phrases and phrasings.
2. **Coordination** of multiple verbs with multiple objects is handled heuristically, not with full VPVet-style graph closure.
3. **First-party multi-word matching** can still prefer a shorter nested phrase in edge cases (e.g., *the service* vs *the service provider*) depending on matcher application order and tokenization.
4. **Secondary clause verbs** (e.g., *create* in “when you create an account”) can still surface despite weak-secondary filtering.
5. **spaCy parse quality** affects subject/agent detection; robustness filters mitigate but do not eliminate parse errors.
6. **Ontology normalization** of data objects (mapping surface forms to canonical categories) is not performed in Stage B.

---

## 11. Future Work

1. Integrate the curated BIO gap-fill set and retrain/fine-tune the data-object NER.
2. Optional merge of high-precision spaCy `ORG` candidates not already in the third-party lexicon.
3. Richer coordination expansion (graph collapsing) for long policy sentences.
4. Post-extraction ontology mapping for cross-policy aggregation.
5. Document-level coreference so *the company* and *we* can be unified when desired.
6. Systematic precision/recall evaluation against a gold CUS-annotated test set.

---

## 12. Conclusion

This document specified a modular, dependency-based CUS tuple extraction pipeline tailored to modern privacy-policy language. Relative to VPVet-style extraction, the principal advances are:

- **Domain entity recognition** with explicit first- and third-party inventories;
- **Expanded and multi-word SOC verb coverage** with defenses against POS mis-tagging;
- **Task-specific data-object NER** with full-span alignment;
- **Explicit handling** of pure passives and possessive-only local subjects;
- **Light post-filters** that reduce pronoun-induced noise without discarding valid actions and objects.

The extraction logic is finalized for engineering use. Remaining quality gains are expected primarily from targeted improvements to data-object NER coverage, not from further restructuring of the linking architecture.

---

## Appendix A — Tuple schema (reference)

```text
CUSTuple(
  entity: str,
  entity_normalized: str,
  entity_type: "first_party" | "third_party" | "unknown",
  action: "collect" | "share" | "not_collect" | "not_share",
  verb: str,
  verb_normalized: str,
  data_object: str,
  data_object_normalized: str,
  negated: bool,
  sentence: str,
)
```

## Appendix B — Example outputs (illustrative)

**Input:** `Google Analytics collects anonymized usage data from our users.`  
**Output:**  
`(Google Analytics, collect, usage data)` with `entity_type=third_party`, `negated=False`.

**Input:** `No personal information is collected through this service.`  
**Output:**  
`(unspecified, not_collect, personal information)` with `entity_type=unknown`, `negated=True`.

**Input:** `We do not sell or share your personal information with advertisers.`  
**Output:**  
`(We, not_share, personal information)` for verbs *sell* and *share* (both under share-type inventory with negation).

**Input:** `Stripe processes your payment information on our behalf.`  
**Output:**  
`(Stripe, collect, payment information)` with `entity_type=third_party`.

---

## Appendix C — File checklist for reproducibility

| Artifact | Role |
|----------|------|
| `src/CUS/cus_extractor.py` | Stage B pipeline |
| `src/CUS/soc_verbs.py` | Verb inventories |
| `src/CUS/entitiy_list.py` | Entity inventories |
| `src/NER_detection/inference_ner.py` | Data-object NER inference API |
| `src/NER_detection/saved_model/...` | Model weights |
| BIO JSONL datasets | NER training/evaluation data |

---

*End of document.*
