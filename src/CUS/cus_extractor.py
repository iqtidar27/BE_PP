#!/usr/bin/env python3
"""
CUS Tuple Extractor (Stage B – dependency based)

Modular design:
  - VerbMatcher        : single-word + multi-word SOC verbs
  - EntityRecognizer   : first-party + third-party entities
  - DataObjectAligner  : aligns DataObjectNER spans to spaCy tokens
  - SentenceTagger     : produces a tagged representation of a sentence
  - DependencyHelper   : small utilities for dependency navigation
  - TupleExtractor     : extracts (entity, action, data_object) tuples
  - TupleFilter        : light post-filters to reduce noise
  - CUSExtractor       : thin orchestrator
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Token

from CUS.entity_list import FIRST_PARTY, THIRD_PARTY

from pathlib import Path
import sys

_SRC_DIR = Path(__file__).resolve().parents[1]  # .../src
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from NER_detection.inference_ner import DataObjectNER
from CUS.soc_verbs import (
    COLLECT_VERBS,
    MULTI_WORD_COLLECT,
    MULTI_WORD_SHARE,
    SHARE_VERBS,
)


# ===========================================================================
# 1. Data classes
# ===========================================================================

@dataclass
class TaggedToken:
    token: Token
    is_entity: bool = False
    entity_type: Optional[str] = None
    is_verb: bool = False
    verb_type: Optional[str] = None
    is_data_object: bool = False
    is_negation: bool = False


@dataclass
class CUSTuple:
    entity: str
    entity_normalized: str
    entity_type: str
    action: str
    verb: str
    verb_normalized: str
    data_object: str
    data_object_normalized: str
    negated: bool
    sentence: str


@dataclass
class DataObjectSpan:
    text: str
    text_normalized: str
    tokens: List[Token]
    root: Token


@dataclass
class EntitySpan:
    text: str
    text_normalized: str
    entity_type: str
    tokens: List[Token]
    root: Token


# ===========================================================================
# 2. VerbMatcher
# ===========================================================================

class VerbMatcher:
    def __init__(self, nlp: spacy.language.Language):
        self.nlp = nlp
        self.collect_lemmas = {v.lower() for v in COLLECT_VERBS}
        self.share_lemmas = {v.lower() for v in SHARE_VERBS}

        self.matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        self._add_phrases(MULTI_WORD_COLLECT, "COLLECT_MWE")
        self._add_phrases(MULTI_WORD_SHARE, "SHARE_MWE")

    def _add_phrases(self, phrases: List[str], label: str) -> None:
        patterns = [self.nlp.make_doc(p) for p in phrases]
        self.matcher.add(label, patterns)

    def find_verbs(self, doc: Doc) -> List[Tuple[Token, str]]:
        results: List[Tuple[Token, str]] = []
        matched_spans = set()

        # Multi-word first
        for match_id, start, end in self.matcher(doc):
            label = self.nlp.vocab.strings[match_id]
            verb_type = "collect" if "COLLECT" in label else "share"
            head = doc[start]
            for t in doc[start:end]:
                if t.head == t or t.dep_ == "ROOT":
                    head = t
                    break
            results.append((head, verb_type))
            matched_spans.update(range(start, end))

        # Single-word
        for token in doc:
            if token.i in matched_spans:
                continue

            lemma = token.lemma_.lower()
            if lemma not in self.collect_lemmas and lemma not in self.share_lemmas:
                continue

            # Robust filter (handles spaCy mis-tagging of known SOC verbs)
            if token.pos_ == "VERB":
                accept = True
            elif lemma in self.collect_lemmas or lemma in self.share_lemmas:
                accept = token.dep_ in {
                    "ROOT", "ccomp", "xcomp", "relcl", "advcl", "conj", "nsubj"
                }
            elif token.pos_ in {"NOUN", "PROPN"}:
                accept = token.dep_ in {
                    "ROOT", "ccomp", "xcomp", "relcl", "advcl", "conj"
                }
            else:
                accept = True

            if not accept:
                continue

            if lemma in self.collect_lemmas:
                results.append((token, "collect"))
            else:
                results.append((token, "share"))

        return results


# ===========================================================================
# 3. EntityRecognizer
# ===========================================================================

class EntityRecognizer:
    def __init__(self, nlp: spacy.language.Language):
        self.nlp = nlp
        self.first_party = {e.lower() for e in FIRST_PARTY}
        self.third_party = {e.lower() for e in THIRD_PARTY}

        ruler = nlp.add_pipe(
            "entity_ruler", before="ner", config={"overwrite_ents": True}
        )
        patterns = [{"label": "THIRD_PARTY", "pattern": name} for name in THIRD_PARTY]
        patterns += [
            {"label": "THIRD_PARTY", "pattern": name.lower()} for name in THIRD_PARTY
        ]
        ruler.add_patterns(patterns)

    def find_entities(self, doc: Doc) -> List[EntitySpan]:
        results: List[EntitySpan] = []
        seen = set()

        # A. Third-party
        for ent in doc.ents:
            lower = ent.text.lower()
            if ent.label_ == "THIRD_PARTY" or lower in self.third_party:
                tokens = list(ent)
                if not tokens or any(t.i in seen for t in tokens):
                    continue
                results.append(
                    EntitySpan(
                        text=ent.text,
                        text_normalized=lower,
                        entity_type="third_party",
                        tokens=tokens,
                        root=ent.root,
                    )
                )
                seen.update(t.i for t in tokens)

        # B. First-party multi-word (prefer longer spans)
        first_party_multi = [p for p in self.first_party if " " in p]
        if first_party_multi:
            matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
            patterns = [self.nlp.make_doc(p) for p in first_party_multi]
            matcher.add("FIRST_PARTY_MWE", patterns)

            # Important: sort matches by length DESC so longer phrases win
            raw_matches = list(matcher(doc))
            raw_matches.sort(key=lambda m: (m[2] - m[1], -m[1]), reverse=True)

            for _, start, end in raw_matches:
                if any(i in seen for i in range(start, end)):
                    continue
                span = doc[start:end]
                results.append(
                    EntitySpan(
                        text=span.text,
                        text_normalized=span.text.lower(),
                        entity_type="first_party",
                        tokens=list(span),
                        root=span.root,
                    )
                )
                seen.update(range(start, end))

        # C. First-party single tokens
        for token in doc:
            if token.i in seen:
                continue
            lower = token.text.lower()
            if lower in self.first_party:
                results.append(
                    EntitySpan(
                        text=token.text,
                        text_normalized=lower,
                        entity_type="first_party",
                        tokens=[token],
                        root=token,
                    )
                )
                seen.add(token.i)

        return results


# ===========================================================================
# 4. DataObjectAligner
# ===========================================================================

class DataObjectAligner:
    def __init__(self, ner_model: DataObjectNER):
        self.ner = ner_model

    def find_data_objects(self, doc: Doc) -> List[DataObjectSpan]:
        result = self.ner.predict(doc.text)
        spans: List[DataObjectSpan] = []

        for ent in result["entities"]:
            start_char, end_char = ent["start"], ent["end"]
            surface = ent["text"].strip()
            if not surface:
                continue

            matching_tokens = [
                t
                for t in doc
                if t.idx < end_char and (t.idx + len(t.text)) > start_char
            ]
            if not matching_tokens:
                continue

            spans.append(
                DataObjectSpan(
                    text=surface,
                    text_normalized=surface.lower(),
                    tokens=matching_tokens,
                    root=matching_tokens[-1],
                )
            )
        return spans


# ===========================================================================
# 5. SentenceTagger
# ===========================================================================

class SentenceTagger:
    NEGATION_WORDS = {"not", "n't", "never", "no", "neither", "nor"}

    def __init__(
        self,
        verb_matcher: VerbMatcher,
        entity_recognizer: EntityRecognizer,
        data_aligner: DataObjectAligner,
    ):
        self.verb_matcher = verb_matcher
        self.entity_recognizer = entity_recognizer
        self.data_aligner = data_aligner

    def tag(
        self, doc: Doc
    ) -> Tuple[List[TaggedToken], List[EntitySpan], List[DataObjectSpan]]:
        tagged = [TaggedToken(token=t) for t in doc]

        for token, vtype in self.verb_matcher.find_verbs(doc):
            tagged[token.i].is_verb = True
            tagged[token.i].verb_type = vtype

        entity_spans = self.entity_recognizer.find_entities(doc)
        data_spans = self.data_aligner.find_data_objects(doc)

        for token in doc:
            if (
                token.text.lower() in self.NEGATION_WORDS
                or token.lemma_.lower() in self.NEGATION_WORDS
            ):
                tagged[token.i].is_negation = True

        return tagged, entity_spans, data_spans


# ===========================================================================
# 6. DependencyHelper
# ===========================================================================

class DependencyHelper:
    @staticmethod
    def get_subtree_tokens(token: Token) -> List[Token]:
        return list(token.subtree)

    @staticmethod
    def find_negation_nearby(
        token: Token, tagged: List[TaggedToken], window: int = 5
    ) -> bool:
        if tagged and tagged[0].token.text.lower() in {"no", "none", "never"}:
            return True
        start = max(0, token.i - window)
        end = min(len(tagged), token.i + window + 1)
        return any(tagged[i].is_negation for i in range(start, end))

    @staticmethod
    def get_ancestors(token: Token) -> List[Token]:
        return list(token.ancestors)


# ===========================================================================
# 7. TupleExtractor
# ===========================================================================

class TupleExtractor:
    def __init__(self):
        self.helper = DependencyHelper()

    def extract(
        self,
        doc: Doc,
        tagged: List[TaggedToken],
        entity_spans: List[EntitySpan],
        data_spans: List[DataObjectSpan],
    ) -> List[CUSTuple]:
        tuples: List[CUSTuple] = []
        verbs = [t for t in tagged if t.is_verb]

        if not verbs or not data_spans:
            return []

        for v in verbs:
            verb_token = v.token
            verb_type = v.verb_type
            negated = self.helper.find_negation_nearby(verb_token, tagged)
            action = f"not_{verb_type}" if negated else verb_type

            related_data = self._find_related_data(verb_token, data_spans)
            related_entities = self._find_related_entities(verb_token, entity_spans)

            if not related_data:
                continue

            if not related_entities:
                related_entities = [
                    EntitySpan(
                        text="unspecified",
                        text_normalized="unspecified",
                        entity_type="unknown",
                        tokens=[],
                        root=verb_token,
                    )
                ]

            for ent in related_entities:
                for data in related_data:
                    tuples.append(
                        CUSTuple(
                            entity=ent.text,
                            entity_normalized=ent.text_normalized,
                            entity_type=ent.entity_type,
                            action=action,
                            verb=verb_token.text,
                            verb_normalized=verb_token.lemma_.lower(),
                            data_object=data.text,
                            data_object_normalized=data.text_normalized,
                            negated=negated,
                            sentence=doc.text,
                        )
                    )
        return tuples

    def _find_related_entities(
        self, verb: Token, entity_spans: List[EntitySpan]
    ) -> List[EntitySpan]:
        related: List[EntitySpan] = []
        main_verb = verb
        if verb.dep_ in ("aux", "auxpass") and verb.head.pos_ == "VERB":
            main_verb = verb.head

        subject_tokens = set()
        for child in main_verb.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                subject_tokens.update(child.subtree)
            elif child.dep_ == "agent":
                for g in child.children:
                    subject_tokens.update(g.subtree)

        for span in entity_spans:
            if any(t in subject_tokens for t in span.tokens):
                related.append(span)

        if related:
            return related

        for span in entity_spans:
            if span.entity_type == "third_party" and abs(span.root.i - main_verb.i) <= 10:
                related.append(span)

        if not related:
            for span in entity_spans:
                if (
                    span.entity_type == "first_party"
                    and span.text_normalized not in {"your", "our", "my"}
                    and abs(span.root.i - main_verb.i) <= 8
                ):
                    related.append(span)
        return related

    def _find_related_data(
        self, verb: Token, data_spans: List[DataObjectSpan]
    ) -> List[DataObjectSpan]:
        related: List[DataObjectSpan] = []
        main_verb = verb
        if verb.dep_ in ("aux", "auxpass") and verb.head.pos_ == "VERB":
            main_verb = verb.head

        verb_subtree = set(self.helper.get_subtree_tokens(main_verb))

        for span in data_spans:
            if any(t in verb_subtree for t in span.tokens):
                related.append(span)
                continue
            if main_verb in self.helper.get_ancestors(span.root):
                related.append(span)
                continue
            if abs(span.root.i - main_verb.i) <= 12:
                related.append(span)
                continue
            if span.root.i > main_verb.i and (span.root.i - main_verb.i) <= 15:
                related.append(span)
        return related


# ===========================================================================
# 8. TupleFilter
# ===========================================================================

class TupleFilter:
    """
    Light post-filters:
      1. Pure possessive pronoun entities → rewrite to 'unspecified'
      2. Weak secondary pronoun tuples → drop when a stronger tuple exists
    """

    POSSESSIVE_PRONOUNS = {"your", "our", "my", "his", "her", "their"}
    WEAK_PRONOUNS = {"you", "your", "our", "my", "we"}

    def filter(self, tuples: List[CUSTuple]) -> List[CUSTuple]:
        if not tuples:
            return []

        rewritten: List[CUSTuple] = []
        for t in tuples:
            if t.entity_normalized in self.POSSESSIVE_PRONOUNS:
                rewritten.append(
                    CUSTuple(
                        entity="unspecified",
                        entity_normalized="unspecified",
                        entity_type="unknown",
                        action=t.action,
                        verb=t.verb,
                        verb_normalized=t.verb_normalized,
                        data_object=t.data_object,
                        data_object_normalized=t.data_object_normalized,
                        negated=t.negated,
                        sentence=t.sentence,
                    )
                )
            else:
                rewritten.append(t)

        cleaned: List[CUSTuple] = []
        for t in rewritten:
            if self._is_weak_secondary(t, rewritten):
                continue
            cleaned.append(t)
        return cleaned

    def _is_weak_secondary(self, t: CUSTuple, all_tuples: List[CUSTuple]) -> bool:
        if t.entity_normalized not in self.WEAK_PRONOUNS:
            return False
        for other in all_tuples:
            if other is t:
                continue
            if other.data_object_normalized != t.data_object_normalized:
                continue
            if (
                other.entity_type in {"third_party", "first_party"}
                and other.entity_normalized not in self.WEAK_PRONOUNS
            ):
                return True
        return False


# ===========================================================================
# 9. CUSExtractor (orchestrator)
# ===========================================================================

class CUSExtractor:
    def __init__(self, ner_model_path: str, spacy_model: str = "en_core_web_sm"):
        self.nlp = spacy.load(spacy_model)

        self.verb_matcher = VerbMatcher(self.nlp)
        self.entity_recognizer = EntityRecognizer(self.nlp)
        self.data_aligner = DataObjectAligner(DataObjectNER(ner_model_path))
        self.tagger = SentenceTagger(
            self.verb_matcher,
            self.entity_recognizer,
            self.data_aligner,
        )
        self.tuple_extractor = TupleExtractor()
        self.tuple_filter = TupleFilter()

    def extract(self, sentence: str) -> List[CUSTuple]:
        if not sentence or not sentence.strip():
            return []

        doc = self.nlp(sentence)
        tagged, entity_spans, data_spans = self.tagger.tag(doc)
        raw_tuples = self.tuple_extractor.extract(
            doc, tagged, entity_spans, data_spans
        )
        return self.tuple_filter.filter(raw_tuples)

    def extract_batch(self, sentences: List[str]) -> List[List[CUSTuple]]:
        return [self.extract(s) for s in sentences]


# ===========================================================================
# 10. Minimal smoke test
# ===========================================================================

if __name__ == "__main__":
    NER_MODEL_PATH = (
        "/home/ayon/git/BE_PP/src/NER_detection/saved_model/"
        "dataobj_ner_v5_focal1.5_len256_unf6"
    )
    extractor = CUSExtractor(ner_model_path=NER_MODEL_PATH)

    samples = [
        "We collect your IP address and email address when you visit our website.",
        "Google Analytics collects anonymized usage data from our users.",
        "No personal information is collected through this service.",
        "Stripe processes your payment information on our behalf.",
        "The extension does not transmit any browsing history to external servers.",
        "PostHog and Mixpanel collect analytics data from users.",
        "Your email address will not be sold to third parties.",
    ]

    print("=" * 80)
    print("CUS Tuple Extraction – Smoke Test")
    print("=" * 80)

    for i, sentence in enumerate(samples, 1):
        print(f"\n[{i}] {sentence}")
        print("-" * 80)
        tuples = extractor.extract(sentence)
        if not tuples:
            print("  (no tuples extracted)")
            continue
        for t in tuples:
            print(f"  Entity     : {t.entity:<25} ({t.entity_type})")
            print(f"  Action     : {t.action}")
            print(f"  Verb       : {t.verb}")
            print(f"  Data Object: {t.data_object}")
            print(f"  Negated    : {t.negated}")
            print()