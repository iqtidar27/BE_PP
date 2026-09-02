#!/usr/bin/env python3
"""
CUS Tuple Extractor (Stage B – dependency based)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple
import sys

import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Token

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from CUS.entity_list import FIRST_PARTY, THIRD_PARTY
from CUS.soc_verbs import (
    COLLECT_VERBS,
    MULTI_WORD_COLLECT,
    MULTI_WORD_SHARE,
    SHARE_VERBS,
    SOC_VERB_BLOCKLIST,
)
from NER_detection.inference_ner import DataObjectNER


FIRST_PARTY_ALWAYS = {
    "this extension",
    "our extension",
    "the extension",
    "this service",
    "the research team",
    "our team",
}

ENTITY_BLOCKLIST = {
    "gdpr",
    "ccpa",
    "the gdpr",
    "the ccpa",
    "general data protection regulation",
    "the general data protection regulation",
    "california consumer privacy act",
    "the california consumer privacy act",
    "privacy policy",
    "this privacy policy",
    "cookies policy",
    "user data policy",
    "chrome web store user data policy",
    "google api services user data policy",
    "terms of service",
    "users",
}

ORG_REJECT_SUBSTR = (
    "privacy policy",
    "cookies policy",
    "data policy",
    "regulation",
    "consumer privacy act",
    "legal basis",
    "data processing",
    "data transfer",
    "data deletion",
    "your rights",
    "website access",
)

ORG_NOISE = {
    "ip", "api", "sms", "url", "http", "https", "pdf", "id", "uid",
    "pii", "gdpr", "ccpa", "cpu", "gpu", "os", "app", "web", "ai",
    "sdk", "cdn", "dns", "ssl", "tls", "html", "css", "js",
}

SITEISH = {
    "site", "website", "service", "services",
    "policy", "page", "account", "accounts",
}

WE_FALLBACK = {
    "we", "us", "the company", "the extension",
    "the service", "our extension",
}

USED_TO_XCOMP = {
    "measure", "analyze", "analyse", "track", "monitor",
    "assess", "evaluate", "process", "collect", "store",
    "record", "log",
}


def first_party_aliases(extension_name: str) -> Set[str]:
    if not extension_name or not extension_name.strip():
        return set()
    name = " ".join(extension_name.strip().split())
    variants = {
        name,
        name.lower(),
        f"the {name}",
        f"the {name} extension",
        f"the {name} app",
        f"{name} extension",
        f"{name} app",
        f"{name} service",
    }
    return {v.lower() for v in variants if v.strip()}


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _is_blocked(text: str) -> bool:
    n = _norm(text)
    if n in ENTITY_BLOCKLIST or n in ORG_NOISE:
        return True
    if any(s in n for s in ORG_REJECT_SUBSTR):
        return True
    if len(n.split()) > 6:
        return True
    return False


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


class VerbMatcher:
    def __init__(self, nlp: spacy.language.Language):
        self.nlp = nlp
        self.collect_lemmas = {v.lower() for v in COLLECT_VERBS}
        self.share_lemmas = {v.lower() for v in SHARE_VERBS}
        self.soc = self.collect_lemmas | self.share_lemmas

        self.matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        self._add_phrases(MULTI_WORD_COLLECT, "COLLECT_MWE")
        self._add_phrases(MULTI_WORD_SHARE, "SHARE_MWE")

    def _add_phrases(self, phrases: List[str], label: str) -> None:
        patterns = [self.nlp.make_doc(p) for p in phrases]
        self.matcher.add(label, patterns)

    def find_verbs(self, doc: Doc) -> List[Tuple[Token, str]]:
        results: List[Tuple[Token, str]] = []
        matched_spans = set()

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

        for token in doc:
            if token.i in matched_spans:
                continue

            lemma = token.lemma_.lower()
            text_l = token.text.lower()
            if lemma in SOC_VERB_BLOCKLIST or text_l in SOC_VERB_BLOCKLIST:
                continue
            if lemma in {"view", "report"} and token.pos_ != "VERB":
                continue
            if lemma in {"view", "report"} and token.dep_ in {
                "compound", "amod", "nsubj", "dobj", "npadvmod", "pobj"
            }:
                continue

            if lemma not in self.collect_lemmas and lemma not in self.share_lemmas:
                continue
            if lemma in {"log", "access"} and token.dep_ in {"amod", "acl"}:
                continue
            if lemma == "use":
                dobjs = [c for c in token.children if c.dep_ in {"dobj", "pobj"}]
                if dobjs and dobjs[0].lemma_.lower() in SITEISH:
                    continue

            if lemma in {"allow", "enable", "permit"}:
                has_soc = any(
                    ch.dep_ in {"xcomp", "ccomp"}
                    and ch.lemma_.lower() in (self.soc | USED_TO_XCOMP)
                    for ch in token.children
                )
                if not has_soc:
                    continue

            if lemma == "receive":
                dobjs = [c for c in token.children if c.dep_ == "dobj"]
                if not dobjs:
                    continue
                if any(
                    d.lemma_.lower()
                    in {"protection", "level", "treatment", "safeguard"}
                    for d in dobjs
                ):
                    continue

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


class EntityRecognizer:
    def __init__(self, nlp: spacy.language.Language):
        self.nlp = nlp
        self.base_first_party = {e.lower() for e in FIRST_PARTY} | FIRST_PARTY_ALWAYS
        self.extra_first_party: Set[str] = set()
        self.third_party = {e.lower() for e in THIRD_PARTY}
        self.soc_lemmas = {v.lower() for v in COLLECT_VERBS} | {
            v.lower() for v in SHARE_VERBS
        }

        if "entity_ruler" not in nlp.pipe_names:
            ruler = nlp.add_pipe(
                "entity_ruler", before="ner", config={"overwrite_ents": True}
            )
            patterns = [{"label": "THIRD_PARTY", "pattern": name} for name in THIRD_PARTY]
            patterns += [
                {"label": "THIRD_PARTY", "pattern": name.lower()} for name in THIRD_PARTY
            ]
            ruler.add_patterns(patterns)

    def _it_is_soc_subject(self, token: Token) -> bool:
        if token.text.lower() != "it":
            return False
        if token.dep_ not in {"nsubj", "nsubjpass"}:
            return False
        head = token.head
        lemma = head.lemma_.lower()
        if lemma in self.soc_lemmas:
            return True
        if lemma in {"be", "use"}:
            for ch in head.children:
                if ch.dep_ in {"xcomp", "acomp"} and ch.lemma_.lower() in self.soc_lemmas:
                    return True
        return False

    def set_extra_first_party(self, names: Set[str]) -> None:
        self.extra_first_party = {n.lower() for n in names}

    @property
    def first_party(self) -> Set[str]:
        return self.base_first_party | self.extra_first_party

    def _role(self, text: str) -> Optional[str]:
        n = _norm(text)
        if _is_blocked(text):
            return None
        if n in self.first_party:
            return "first_party"
        if n in self.third_party:
            return "third_party"
        return None

    def find_entities(self, doc: Doc) -> List[EntitySpan]:
        results: List[EntitySpan] = []
        seen = set()

        def add(text: str, tokens: List[Token], root: Token, etype: str) -> None:
            if not tokens or any(t.i in seen for t in tokens):
                return
            if _is_blocked(text):
                return
            results.append(
                EntitySpan(
                    text=text,
                    text_normalized=_norm(text),
                    entity_type=etype,
                    tokens=tokens,
                    root=root,
                )
            )
            seen.update(t.i for t in tokens)

        for ent in doc.ents:
            lower = ent.text.lower()
            if lower in self.first_party:
                add(ent.text, list(ent), ent.root, "first_party")
                continue
            if ent.label_ == "THIRD_PARTY" or lower in self.third_party:
                role = self._role(ent.text)
                if role == "third_party":
                    add(ent.text, list(ent), ent.root, "third_party")

        first_party_multi = [p for p in self.first_party if " " in p]
        if first_party_multi:
            matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
            matcher.add(
                "FIRST_PARTY_MWE",
                [self.nlp.make_doc(p) for p in first_party_multi],
            )
            raw = list(matcher(doc))
            raw.sort(key=lambda m: (m[2] - m[1], -m[1]), reverse=True)
            for _, start, end in raw:
                if any(i in seen for i in range(start, end)):
                    continue
                span = doc[start:end]
                add(span.text, list(span), span.root, "first_party")

        for token in doc:
            if token.i in seen:
                continue
            lower = token.text.lower()
            if lower == "it":
                if self._it_is_soc_subject(token):
                    add(token.text, [token], token, "first_party")
                continue
            if lower in self.first_party and not _is_blocked(token.text):
                add(token.text, [token], token, "first_party")

        for ent in doc.ents:
            if ent.label_ not in {"ORG", "PERSON"}:
                continue
            tokens = list(ent)
            if not tokens or any(t.i in seen for t in tokens):
                continue
            surface = ent.text.strip()
            lower = surface.lower()
            if _is_blocked(surface) or lower in ORG_NOISE:
                continue
            if len(surface) < 3:
                continue
            if len(tokens) == 1 and not tokens[0].text[:1].isupper():
                continue
            if lower in self.first_party:
                add(surface, tokens, ent.root, "first_party")
            elif lower in self.third_party:
                continue
            else:
                add(surface, tokens, ent.root, "third_party")

        return results


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


class SentenceTagger:
    NEGATION_WORDS = {
        "not", "n't", "never", "no", "neither", "nor",
        "don't", "dont", "doesn't", "doesnt",
        "didn't", "didnt", "won't", "wont",
        "cannot", "can't",
    }

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

        neg_words = {
            "not", "n't", "never", "no", "neither", "nor",
            "don't", "dont", "doesn't", "doesnt",
            "didn't", "didnt", "won't", "wont",
            "cannot", "can't",
        }

        for ch in token.children:
            if ch.dep_ == "neg" or ch.text.lower() in neg_words:
                return True
        if token.head is not token:
            for ch in token.head.children:
                if ch.dep_ == "neg" or ch.text.lower() in neg_words:
                    return True

        start = max(0, token.i - window)
        end = min(len(tagged), token.i + window + 1)
        return any(tagged[i].is_negation for i in range(start, end))

    @staticmethod
    def get_ancestors(token: Token) -> List[Token]:
        return list(token.ancestors)


class TupleExtractor:
    def __init__(self):
        self.helper = DependencyHelper()
        self.collect_lemmas = {v.lower() for v in COLLECT_VERBS}
        self.share_lemmas = {v.lower() for v in SHARE_VERBS}

    def extract(
        self,
        doc: Doc,
        tagged: List[TaggedToken],
        entity_spans: List[EntitySpan],
        data_spans: List[DataObjectSpan],
    ) -> List[CUSTuple]:
        tuples: List[CUSTuple] = []
        verbs = [t for t in tagged if t.is_verb]
        if not verbs:
            return []

        for v in verbs:
            verb_token = v.token
            verb_type = v.verb_type
            negated = self.helper.find_negation_nearby(verb_token, tagged)
            action = f"not_{verb_type}" if negated else verb_type

            if self._is_hypothetical_if_collect(verb_token):
                continue

            related_data = self._find_related_data(verb_token, data_spans)
            prefer_tokens: Set[Token] = set()

            if verb_token.lemma_.lower() == "use" or verb_token.text.lower() in {
                "used",
                "use",
            }:
                xcomps = [
                    ch
                    for ch in verb_token.children
                    if ch.dep_ == "xcomp"
                    and ch.lemma_.lower()
                    in (self.collect_lemmas | self.share_lemmas | USED_TO_XCOMP)
                ]
                if xcomps:
                    for ch in verb_token.children:
                        if ch.dep_ in {"nsubj", "nsubjpass"}:
                            prefer_tokens.update(ch.subtree)
                    extra: List[DataObjectSpan] = []
                    extra.extend(self._find_related_data(verb_token, data_spans))
                    for xc in xcomps:
                        extra.extend(self._find_related_data(xc, data_spans))
                    if extra:
                        related_data = extra
                    else:
                        related_data = self._fallback_used_to_object(verb_token)

            if (
                verb_token.dep_ == "xcomp"
                and verb_token.head.lemma_.lower() == "use"
            ):
                continue

            if not related_data:
                if any(
                    t.text.lower() in {"neither", "nor"} for t in verb_token.doc
                ) and verb_token.lemma_.lower() in {
                    "share",
                    "sell",
                    "transfer",
                    "disclose",
                }:
                    related_data = list(data_spans)

            if not related_data and verb_token.lemma_.lower() in {
                "disclose",
                "share",
                "report",
            }:
                related_data = self._fallback_legal_disclosure(
                    verb_token, data_spans
                )

            if not related_data:
                continue

            related_entities = self._find_related_entities(
                verb_token,
                entity_spans,
                prefer_tokens=prefer_tokens or None,
            )

            if verb_token.dep_ in {"advcl", "relcl"}:
                related_entities = [
                    e
                    for e in related_entities
                    if e.text_normalized not in {"you", "your", "my"}
                ]

            if not related_entities:
                if self._is_passive(verb_token) and not self._has_explicit_agent(
                    verb_token
                ):
                    we_spans = [
                        e
                        for e in entity_spans
                        if e.text_normalized in WE_FALLBACK
                    ]
                    if we_spans:
                        related_entities = [we_spans[0]]
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

    def _fallback_legal_disclosure(
        self, verb: Token, data_spans: List[DataObjectSpan]
    ) -> List[DataObjectSpan]:
        sent = verb.doc.text.lower()
        if not any(
            p in sent
            for p in (
                "law enforcement",
                "legal authority",
                "legal authorities",
                "court",
                "subpoena",
                "required by law",
            )
        ):
            return []
        if data_spans:
            return list(data_spans)
        hits = [
            t
            for t in verb.doc
            if t.lemma_.lower() in {"information", "data"}
            and t.pos_ in {"NOUN", "PROPN"}
        ]
        if not hits:
            return []
        return [
            DataObjectSpan(
                text=hits[0].text,
                text_normalized=hits[0].text.lower(),
                tokens=[hits[0]],
                root=hits[0],
            )
        ]

    def _fallback_used_to_object(self, verb: Token) -> List[DataObjectSpan]:
        hints = {
            "traffic",
            "conversion",
            "conversions",
            "performance",
            "request",
            "requests",
            "interaction",
            "usage",
        }
        hits = [
            t
            for t in verb.doc
            if t.lemma_.lower() in hints and t.pos_ in {"NOUN", "PROPN"}
        ]
        if not hits:
            return []
        return [
            DataObjectSpan(
                text=hits[0].text,
                text_normalized=hits[0].text.lower(),
                tokens=[hits[0]],
                root=hits[0],
            )
        ]

    def _is_hypothetical_if_collect(self, verb: Token) -> bool:
        if verb.lemma_.lower() not in self.collect_lemmas:
            return False
        cur = verb
        for _ in range(6):
            if cur.dep_ in {"ccomp", "advcl", "relcl"}:
                head = cur.head
                if head.lemma_.lower() in {"discover", "find", "learn", "realize"}:
                    return True
                for ch in head.children:
                    if ch.dep_ == "mark" and ch.text.lower() == "if":
                        if head.lemma_.lower() in {"discover", "find", "learn"}:
                            return True
            if cur.head is cur:
                break
            cur = cur.head
        return False

    def _is_passive(self, verb: Token) -> bool:
        if any(c.dep_ in {"auxpass", "nsubjpass"} for c in verb.children):
            return True
        if verb.tag_ == "VBN" and any(
            c.dep_ == "aux" and c.lemma_ == "be" for c in verb.children
        ):
            return True
        return False

    def _has_explicit_agent(self, verb: Token) -> bool:
        return any(c.dep_ == "agent" for c in verb.children)

    def _find_related_entities(
        self,
        verb: Token,
        entity_spans: List[EntitySpan],
        prefer_tokens: Optional[Set[Token]] = None,
    ) -> List[EntitySpan]:
        related: List[EntitySpan] = []

        if prefer_tokens:
            preferred = [
                span
                for span in entity_spans
                if any(t in prefer_tokens for t in span.tokens)
            ]
            if preferred:
                return preferred

        main_verb = verb
        if verb.dep_ in ("aux", "auxpass") and verb.head.pos_ == "VERB":
            main_verb = verb.head

        subj_source = main_verb
        if main_verb.dep_ == "conj" and main_verb.head.pos_ == "VERB":
            subj_source = main_verb.head

        subject_tokens = set()
        for child in subj_source.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                subject_tokens.update(child.subtree)
            elif child.dep_ == "agent":
                for g in child.children:
                    subject_tokens.update(g.subtree)

        if subj_source is not main_verb:
            for child in main_verb.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subject_tokens.update(child.subtree)
                elif child.dep_ == "agent":
                    for g in child.children:
                        subject_tokens.update(g.subtree)

        if not subject_tokens and main_verb.dep_ in {"advcl", "xcomp", "pcomp"}:
            parent = main_verb.head
            for ch in parent.children:
                if ch.dep_ in {"nsubj", "nsubjpass"}:
                    subject_tokens.update(ch.subtree)

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

        soc = self.collect_lemmas | self.share_lemmas
        soc_idx = [
            t.i
            for t in verb.doc
            if t.lemma_.lower() in soc and t.i != main_verb.i
        ]
        verb_subtree = set(self.helper.get_subtree_tokens(main_verb))

        for span in data_spans:
            dist = abs(span.root.i - main_verb.i)
            if soc_idx:
                other = min(abs(span.root.i - i) for i in soc_idx)
                if other < dist:
                    continue
            if any(t in verb_subtree for t in span.tokens):
                related.append(span)
                continue
            if main_verb in self.helper.get_ancestors(span.root):
                related.append(span)

        if not related:
            for span in data_spans:
                head = span.root.head
                if head == main_verb or head.head == main_verb:
                    related.append(span)
        return related


class TupleFilter:
    POSSESSIVE_PRONOUNS = {"your", "our", "my", "his", "her", "their"}
    WEAK_SECONDARY = {"you", "your", "my"}

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

        seen = set()
        unique: List[CUSTuple] = []
        for t in cleaned:
            key = (
                t.entity_normalized,
                t.action,
                t.data_object_normalized,
                t.negated,
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(t)
        return unique

    def _is_weak_secondary(self, t: CUSTuple, all_tuples: List[CUSTuple]) -> bool:
        if t.entity_normalized not in self.WEAK_SECONDARY:
            return False
        for other in all_tuples:
            if other is t:
                continue
            if other.data_object_normalized != t.data_object_normalized:
                continue
            if other.entity_type == "third_party":
                return True
            if (
                other.entity_type == "first_party"
                and other.entity_normalized not in self.WEAK_SECONDARY
            ):
                return True
        return False


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

    def set_extension_name(self, extension_name: Optional[str]) -> None:
        self.entity_recognizer.set_extra_first_party(
            first_party_aliases(extension_name or "")
        )

    def extract(
        self,
        sentence: str,
        extension_name: Optional[str] = None,
    ) -> List[CUSTuple]:
        if not sentence or not sentence.strip():
            return []
        if extension_name is not None:
            self.set_extension_name(extension_name)

        doc = self.nlp(sentence)
        tagged, entity_spans, data_spans = self.tagger.tag(doc)
        tagged, entity_spans = self._drop_data_overlaps(
            tagged, entity_spans, data_spans
        )
        raw = self.tuple_extractor.extract(doc, tagged, entity_spans, data_spans)
        return self.tuple_filter.filter(raw)

    @staticmethod
    def _drop_data_overlaps(tagged, entity_spans, data_spans):
        """A token inside a data-object span is not an actor and not a SOC verb."""
        data_ids = {t.i for span in data_spans for t in span.tokens}
        data_norms = {span.text_normalized for span in data_spans}

        for i in data_ids:
            if i < len(tagged):
                tagged[i].is_verb = False
                tagged[i].verb_type = None

        cleaned = []
        for e in entity_spans:
            if any(t.i in data_ids for t in e.tokens):
                continue
            en = e.text_normalized
            if any(
                en == d or d.startswith(en + " ") or en in d.split()
                for d in data_norms
            ):
                continue
            cleaned.append(e)
        return tagged, cleaned

    def extract_batch(self, sentences: List[str]) -> List[List[CUSTuple]]:
        return [self.extract(s) for s in sentences]


if __name__ == "__main__":
    NER_MODEL_PATH = (
        "/home/ayon/git/BE_PP/src/NER_detection/saved_model/"
        "dataobj_ner_v5_focal1.5_len256_unf6_dataset_version_v4_augmented_final"
    )
    extractor = CUSExtractor(ner_model_path=NER_MODEL_PATH)

    samples = [
        ("Crashlytics is used to collect crash reports.", "Algo"),
        ("Google Analytics is used to measure traffic.", "Algo"),
        ("PostHog is used to analyze how you interact with the app.", "Algo"),
        ("Pixels are used to track conversions.", "Algo"),
        ("Cloudflare is used to process requests on our behalf.", "Algo"),
        (
            "Google Analytics and Meta Pixel are used to measure advertising performance.",
            "Algo",
        ),
        ("We enable Google Analytics to measure site traffic.", "Algo"),
        (
            "We may disclose your information to law enforcement if required by law.",
            "Algo",
        ),
        ("We will not disclose your information except as required by law.", "Algo"),
        ("We collect your email address.", "Algo"),
        (
            "Google Analytics collects anonymized usage data from our users.",
            "Algo",
        ),
        ("This privacy policy explains our practices.", "Algo"),
    ]

    print("=" * 80)
    print("CUS smoke – used-to / disclose")
    print("=" * 80)
    for sent, name in samples:
        print(f"\n[{name}] {sent}")
        ts = extractor.extract(sent, extension_name=name)
        if not ts:
            print("  (none)")
            continue
        for t in ts:
            print(
                f"  {t.entity} ({t.entity_type}) | {t.action} | {t.verb} | {t.data_object} | neg={t.negated}"
            )
