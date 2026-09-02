"""
Sentence-level CUS tuple extractor.

Same algorithm as kalamoo/PPAudit/cus_extract/extract_cus_phrase_tuple.py
(MixExtractor + dependency-graph collapse + PolicyTransformer). Wired to:

  - NER: src/NER_detection/saved_model/dataobj_ner_v5_focal1.5_len256_unf6_dataset_version_v6_augmented_final
  - SOC verbs + multi-word phrases + blocklist from resources.py
  - FIRST_PARTY | THIRD_PARTY + first_party_aliases(extension_name)

Usage
-----
    from cus_extractor import CUSExtractor

    ex = CUSExtractor()                          # uses default paths/lists
    rows = ex.extract("We collect your email address and share it with advertisers.")
    rows = ex.extract("JobLog stores crash logs on Sentry.", extension_name="JobLog")
"""

import re
from pathlib import Path

from extract_cus_phrase_tuple import (
    Analytics,
    MixExtractor,
    val,
    valTxt,
)
from resources import (
    COLLECT_VERBS,
    DATA_HINTS,
    DEFAULT_NER_MODEL,
    FIRST_PARTY,
    FIRST_PARTY_ALWAYS,
    MULTI_WORD_COLLECT,
    MULTI_WORD_SHARE,
    SHARE_VERBS,
    SITEISH,
    SOC_VERB_BLOCKLIST,
    THIRD_PARTY,
    WE_FALLBACK,
    all_entities,
    first_party_aliases,
    is_blocked_entity,
)

_IMPLICIT = {"we_implicit", "third_party_implicit"}
_POSSESSIVE = {"your", "our", "my", "you"}
_ENTITY_LAST = {
    "server": "servers",
    "game": "games",
    "service": "services",
    "advertiser": "advertisers",
    "partner": "partners",
    "provider": "providers",
    "vendor": "vendors",
}


_LEMMA_TO_SURFACE = {
    "datum": "data",
    "token": "tokens",
    "cookie": "cookies",
    "cooky": "cookies",
    "statistic": "statistics",
    "timestamp": "timestamps",
    "address": "addresses",
    "type": "types",
    "click": "clicks",
}


_DATA_PREFIX = re.compile(
    r'^(strictly\s+)?(just\s+|only\s+|merely\s+|certain\s+)?',
    re.I,
)


def _surface_data(text):
    t = (text or "").strip()
    if not t:
        return t
    t = _DATA_PREFIX.sub('', t).strip()
    low = t.lower()
    if low in _LEMMA_TO_SURFACE:
        return _LEMMA_TO_SURFACE[low]
    parts = low.split()
    if parts and parts[-1] in _LEMMA_TO_SURFACE:
        return " ".join(parts[:-1] + [_LEMMA_TO_SURFACE[parts[-1]]])
    return t


_STOP_DATA = {
    "and", "or", "as", "the", "a", "an", "of", "to", "for", "in", "on", "at",
    "we", "you", "it", "they", "this", "that", "these", "those", "such",
    "other", "any", "all", "also", "only", "follow", "following",
}
_BAD_DATA_CHARS = set(".,;:|#*_[](){}<>/\\")
_WEAK_VERBS = {
    "set", "display", "provide", "associate", "lease", "contain",
    "include", "enable", "allow", "permit", "help", "make",
}
_CORE_VERBS = {
    "collect", "share", "store", "save", "receive", "disclose", "sell",
    "track", "record", "process", "transfer", "transmit", "obtain",
    "gather", "log", "access", "use", "analyze", "analyse", "monitor",
    "keep", "retain", "combine",
}
_PARTY_ENT = {e.lower() for e in FIRST_PARTY} | {e.lower() for e in FIRST_PARTY_ALWAYS} | {
    e.lower() for e in THIRD_PARTY
} | {e.lower() for e in WE_FALLBACK} | {"unspecified", "we", "us"}

# PolicyLint / PoliCheck entity roles (Andow et al.).
# First party = the app / publisher. Third party = anyone else named or generic.
_FIRST_PARTY_LABELS = {
    e.lower() for e in FIRST_PARTY
} | {e.lower() for e in FIRST_PARTY_ALWAYS} | {e.lower() for e in WE_FALLBACK} | {
    "we", "us", "our",
    "the company", "the app", "the application", "the extension",
    "this extension", "our extension", "our app", "our company",
    "our servers", "my servers", "the developer", "the developers",
    "login form",
}
_FIRST_PARTY_LABELS -= {"you", "your", "i", "me"}

_THIRD_PARTY_LABELS = {e.lower() for e in THIRD_PARTY} | {
    "third party", "third parties", "third-party",
    "advertiser", "advertisers",
    "partner", "partners", "affiliate", "affiliates",
    "vendor", "vendors",
    "service provider", "service providers",
    "processor", "processors",
    "these services", "those services", "their services",
    "google analytics", "google analytic",
}

_THIRD_GENERIC_RE = re.compile(
    r"\b(advertisers?|partners?|affiliates?|vendors?|"
    r"third[ -]?part(y|ies)|analytics|processors?|"
    r"service providers?)\b",
    re.I,
)

_EXTRA_FIRST_PARTY = set()


def classify_party(entity, extension_name=None):
    """Return 'first_party' | 'third_party' | 'unspecified'.

    Same role split PolicyLint/PoliCheck use on the entity ontology:
    the first party is the service the policy is about; everyone else
    named or referred to generically is third party.
    """
    e = " ".join((entity or "").lower().split())
    if not e or e in {"unspecified", "unknown"}:
        return "unspecified"
    aliases = {a.lower() for a in first_party_aliases(extension_name)}
    if extension_name and e == " ".join(str(extension_name).lower().split()):
        return "first_party"
    if e in _FIRST_PARTY_LABELS or e in aliases or e in _EXTRA_FIRST_PARTY:
        return "first_party"
    if e.endswith(" servers") and e.split()[0] in {"our", "my", "the"}:
        return "first_party"
    if e in _THIRD_PARTY_LABELS or _THIRD_GENERIC_RE.search(e):
        return "third_party"
    if e in {"they", "them", "it"}:
        return "third_party"
    return "third_party"


def _apply_extension_name(rows, extension_name):
    """Rewrite first-party actors to the product name. Third parties unchanged."""
    if not extension_name or not rows:
        return rows
    name = " ".join(str(extension_name).split()).strip()
    if not name:
        return rows
    out = []
    for r in rows:
        r = dict(r)
        party = r.get("party") or classify_party(r.get("entity_phrase"), extension_name)
        if party == "first_party":
            r["entity_phrase"] = name
            r["party"] = "first_party"
        out.append(r)
    return out


def _is_junk_data(data):
    d = (data or "").strip().lower()
    if not d or len(d) <= 1:
        return True
    if d in _STOP_DATA or d in _BAD_DATA_CHARS:
        return True
    if d.endswith("#") or d.startswith("#"):
        return True
    if all(ch in _BAD_DATA_CHARS or ch.isspace() for ch in d):
        return True
    if d in {"browse", "history"} and " " not in d:
        return True
    return False


_SOC_FIRST = {v.lower() for v in COLLECT_VERBS} | {v.lower() for v in SHARE_VERBS}


_KEEP_SERVICE_ACTORS = {
    "these services", "those services", "their services", "the services",
}


def _rewrite_bad_entity(entity):
    """Purpose NPs and bare site-ish nouns are not actors.

    Keep 'these services may collect data'. Still rewrite 'the site' / 'site visitor'.
    """
    e = (entity or "").strip().lower()
    if not e or e == "unspecified":
        return "unspecified"
    if e in _PARTY_ENT or e in _KEEP_SERVICE_ACTORS:
        return entity
    parts = e.split()
    first, last = parts[0], parts[-1]
    if first in _SOC_FIRST:
        return "unspecified"
    if first in SITEISH:
        return "unspecified"
    if e in SITEISH:
        return "unspecified"
    if last in SITEISH and first not in {"these", "those", "their"}:
        return "unspecified"
    return entity


def _is_data_like_entity(entity):
    e = (entity or "").strip().lower()
    if not e or e == "unspecified":
        return False
    words = set(e.replace("/", " ").split())
    if words & {h.lower() for h in DATA_HINTS}:
        if e not in _PARTY_ENT and not any(p in e for p in ("google", "analytics", "server")):
            return True
    return False


def _clean_rows(rows, extension_name=None):
    """Drop combinatorial / non-CUS noise after the PPAudit walk."""
    if not rows:
        return rows

    kept = []
    for r in rows:
        entity = _rewrite_bad_entity(
            _recover_demonstrative_actor(
                r["entity_phrase"], r.get("sentence") or ""
            )
        )
        r = dict(r)
        r["entity_phrase"] = entity
        r["party"] = classify_party(entity, extension_name)
        data = r["dataobj_phrase"]
        verb = (r.get("cus_verb") or "").lower()
        action = r["cus_or_not"]

        if _is_junk_data(data):
            continue
        if entity.lower() == data.lower():
            continue
        if _is_data_like_entity(entity) and entity.lower() not in _PARTY_ENT:
            continue
        # Markdown / URL fragments ("[data processing addendum](/legal/dpa)")
        dl = data.lower()
        if re.search(r'\[|\(\s*/|https?:|www\.|\.html|\]\(', data) or "/legal" in dl:
            continue
        # Party NP tagged as DATA ("our third-party partner")
        if re.search(r'\b(third[ -]?party )?partners?\b', dl) and not re.search(
            r'\b(data|information|cookie|email|ip)\b', dl
        ):
            continue
        # "pages the site visitor accesses" — visitor is not the collector
        if verb in {"access", "accesses", "visit", "visits"}:
            el = entity.lower()
            if el in {"unspecified", "site visitor", "visitor", "user", "the user"}:
                continue
            if el not in _PARTY_ENT | {"we", "us"}:
                continue
        # Heading glued on: "How We Use Cookies … # We collect usage information"
        sent = (r.get("sentence") or "").lower()
        if "#" in sent and verb == "use" and "cookie" in dl and sent.lstrip().startswith("how we"):
            continue
        # "X sets this cookie" → keep only cookie-shaped data
        if verb == "set":
            if "cookie" not in data.lower() and "cooky" not in data.lower():
                continue
        if verb in _WEAK_VERBS and verb != "set":
            if action not in {"not_collect", "not_share"} and verb not in {"provide"}:
                continue
            if verb == "provide" and "cookie" not in data.lower() and "data" not in data.lower() and "information" not in data.lower():
                continue
        kept.append(r)

    # Same actor+data only. Do not kill contrastive third-party collects.
    denied_pair = {
        (r["entity_phrase"].lower(), r["dataobj_phrase"].lower())
        for r in kept if r["cus_or_not"] in {"not_collect", "not_share"}
    }
    if denied_pair:
        kept = [
            r for r in kept
            if not (
                r["cus_or_not"] in {"collect", "share"}
                and (r["entity_phrase"].lower(), r["dataobj_phrase"].lower()) in denied_pair
            )
        ]
    kept = [
        r for r in kept
        if not (
            r["cus_or_not"] in {"collect", "share"}
            and re.search(r'\b(no|none|neither)\b', r["dataobj_phrase"].lower())
        )
    ]

    # Per verb: if a real party entity exists, drop unspecified clones
    by_vd = {}
    for r in kept:
        key = (r["cus_verb"], r["cus_or_not"], r["dataobj_phrase"].lower())
        by_vd.setdefault(key, []).append(r)
    out = []
    for group in by_vd.values():
        parties = [r for r in group if r["entity_phrase"].lower() in _PARTY_ENT or r["entity_phrase"].lower() == "we"]
        if parties:
            # one row per distinct entity in the party set
            seen_e = set()
            for r in parties:
                e = r["entity_phrase"].lower()
                if e in seen_e:
                    continue
                seen_e.add(e)
                out.append(r)
        else:
            out.extend(group)

    # Cap fan-out: max 6 tuples per sentence after cleaning
    if len(out) > 6:
        def _score(r):
            v = (r.get("cus_verb") or "").lower()
            e = r["entity_phrase"].lower()
            d = r["dataobj_phrase"]
            s = 0
            if r["cus_or_not"] in {"not_collect", "not_share"}:
                s += 5
            if v in _CORE_VERBS:
                s += 3
            if e in _PARTY_ENT or e == "we":
                s += 2
            if " " in d:
                s += 1
            return s
        out = sorted(out, key=_score, reverse=True)[:6]
    return out


def _norm_entity(text):
    if not text:
        return "unspecified"
    n = " ".join(str(text).lower().split())
    if n in _IMPLICIT or n in _POSSESSIVE or is_blocked_entity(text):
        return "unspecified"
    parts = n.split()
    if parts and parts[-1] in _ENTITY_LAST:
        parts[-1] = _ENTITY_LAST[parts[-1]]
        n = " ".join(parts)
    return n


def _recover_demonstrative_actor(entity, sentence):
    """Walk often leaves 'These services' untagged → unspecified."""
    if entity not in {"unspecified", ""}:
        return entity
    m = re.match(
        r'^\s*(these|those|their)\s+services\b',
        sentence or "",
        flags=re.I,
    )
    if m:
        return m.group(0).strip().lower()
    return entity


class CUSExtractor:
    def __init__(
        self,
        ner_model_path=DEFAULT_NER_MODEL,
        spacy_model="en_core_web_sm",
        nlp=None,
        use_spacy_ner_entities=True,
        also_tag_ner_entities=False,
        extra_entities=None,
        extra_share_verbs=None,
        extra_collect_verbs=None,
    ):
        share = {v.lower() for v in SHARE_VERBS}
        collect = {v.lower() for v in COLLECT_VERBS}
        if extra_share_verbs:
            share.update(v.lower() for v in extra_share_verbs)
        if extra_collect_verbs:
            collect.update(v.lower() for v in extra_collect_verbs)

        entities = all_entities()
        if extra_entities:
            entities = list(entities) + [e.lower() for e in extra_entities]

        self._nlp = nlp if nlp is not None else __import__("spacy").load(spacy_model)
        self._analytics = Analytics()
        self._ext = MixExtractor(
            self._nlp,
            self._analytics,
            str(ner_model_path),
            share_verbs=sorted(share),
            collect_verbs=sorted(collect),
            entity_list=entities,
            use_spacy_ner_entities=use_spacy_ner_entities,
            also_tag_ner_entities=also_tag_ner_entities,
            verb_blocklist=SOC_VERB_BLOCKLIST,
            multi_word_share=MULTI_WORD_SHARE,
            multi_word_collect=MULTI_WORD_COLLECT,
        )
        self._known_aliases = set()

    def set_extension_name(self, extension_name=None):
        """Register the product as first party (PolicyLint extra-first-party)."""
        global _EXTRA_FIRST_PARTY
        aliases = first_party_aliases(extension_name or "")
        _EXTRA_FIRST_PARTY |= aliases
        _FIRST_PARTY_LABELS.update(aliases)
        new_aliases = aliases - self._known_aliases
        if new_aliases:
            self._ext.tagger.add_entities(new_aliases)
            self._known_aliases |= new_aliases

    def extract(self, sentence, extension_name=None):
        """Return PPAudit-style CUS tuples for one sentence.

        extension_name: optional product name (e.g. "adblocker").
          - Added to the first-party gazetteer so "Adblocker collects X" matches.
          - First-party actors (we, us, our servers, this extension, login form)
            are rewritten to this name and labelled party=first_party.
          Third-party rows are not renamed.

        Each item:
            {
              "sentence": str,
              "entity_phrase": str,      # actor / recipient
              "party": str,              # first_party | third_party | unspecified
              "cus_or_not": str,         # "collect" | "not_collect" | "share" | "not_share"
              "dataobj_phrase": str,     # data object from your NER + graph
              "cus_verb": str,           # lemma (or merged multi-word phrase)
            }
        Empty list if the sentence has no CUS statement under this method.
        """
        if sentence is None:
            return []
        sentence = str(sentence).strip()
        if not sentence:
            return []

        if extension_name:
            self.set_extension_name(extension_name)

        doc_id = 'sent'
        self._analytics.startDoc(doc_id)
        try:
            raw = self._ext.mix_extract(sentence)
        finally:
            self._analytics.endDoc()
        if not raw:
            return []
        rows = []
        seen = set()
        for ent, col, dat, sen, action_lemma in raw:
            entity = _recover_demonstrative_actor(
                _norm_entity(val(ent)), sentence
            )
            data = _surface_data(val(dat))
            action = val(col)
            key = (entity.lower(), action, data.lower())
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "sentence": valTxt(sen),
                "entity_phrase": entity,
                "party": classify_party(entity, extension_name),
                "cus_or_not": action,
                "dataobj_phrase": data,
                "cus_verb": val(action_lemma),
            })
        return _apply_extension_name(_clean_rows(rows, extension_name), extension_name)

    def extract_many(self, sentences, extension_name=None):
        out = []
        for s in sentences:
            out.extend(self.extract(s, extension_name=extension_name))
        return out


def extract_cus_from_sentence(sentence, extension_name=None, extractor=None, **kwargs):
    """One-shot helper. Pass a long-lived extractor= to avoid reloading the NER."""
    if extractor is None:
        extractor = CUSExtractor(**kwargs)
    return extractor.extract(sentence, extension_name=extension_name)


if __name__ == "__main__":
    import json
    import sys

    text = " ".join(sys.argv[1:]) or (
        "We collect your email address and share it with advertisers."
    )
    # Construction will fail here if the NER dir / spaCy model are missing;
    # that is expected on this sandbox. Call CUSExtractor() from the project root.
    print(json.dumps({"hint": "from cus_extractor import CUSExtractor", "example": text}, indent=2))
