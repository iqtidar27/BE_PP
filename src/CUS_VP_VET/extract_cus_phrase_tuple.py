from enum import Enum
import os
from pathlib import Path
import pickle
import re
import sys
import json
import time
import networkx as nx
from transformers import pipeline
# pip install networkx
import lib.ExclusionPhraseMerger as epm
import lib.NounPhraseMerger as npm
import lib.ExclusionDetector as eh
import spacy
try:
    from resources import DATA_HINTS, SITEISH, USED_TO_XCOMP, is_blocked_entity
except Exception:
    DATA_HINTS, SITEISH, USED_TO_XCOMP = set(), set(), set()

    def is_blocked_entity(text):
        return False
# Original VPVet/PPAudit used spaCy 2.0.18 (token.pos == symbols.*, span.merge).
# Extraction algorithm below is unchanged. Only three inputs are swapped:
#   1) your NER model  2) your SOC verb lists  3) your entity list


def _read_json_or_lines(path):
    path = Path(path)
    text = path.read_text(encoding='utf-8').strip()
    if not text:
        return []
    if path.suffix.lower() == '.json':
        return json.loads(text)
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith('#')]


def load_soc_verbs(path):
    """
    Accepted formats
      {"share": [...], "collect": [...]}
      {"share_verbs": [...], "collect_verbs": [...]}
      ["collect", "share", ...]  -> treated as COLLECT; prefer the two-key form
    """
    raw = _read_json_or_lines(path)
    share, collect = [], []
    if isinstance(raw, dict):
        share = raw.get('share') or raw.get('share_verbs') or raw.get('SHARE') or []
        collect = raw.get('collect') or raw.get('collect_verbs') or raw.get('COLLECT') or raw.get('use') or []
        both = raw.get('both') or raw.get('share_and_collect') or []
        share = list(share) + list(both)
        collect = list(collect) + list(both)
    elif isinstance(raw, list):
        collect = list(raw)
    else:
        raise ValueError('SOC verb list must be a JSON object or a list of strings')

    def _norm(xs):
        out, seen = [], set()
        for v in xs:
            v = str(v).strip().lower()
            if v and v not in seen:
                seen.add(v)
                out.append(v)
        return out
    return _norm(share), _norm(collect)


def load_entity_list(path):
    raw = _read_json_or_lines(path)
    if isinstance(raw, dict):
        raw = raw.get('entities') or raw.get('entity') or raw.get('ENTITY') or raw.get('list') or []
    out, seen = [], set()
    for e in raw:
        e = re.sub(r'\s+', ' ', str(e).strip().lower())
        if e and e not in seen:
            seen.add(e)
            out.append(e)
    out.sort(key=lambda s: (-len(s.split()), -len(s)))
    return out


def _normalize_bio_label(label):
    if label is None:
        return 'O', None
    lab = str(label)
    if '-' in lab and not lab.startswith('LABEL_'):
        pre, typ = lab.split('-', 1)
        return pre.upper(), typ.upper()
    return lab.upper(), None


class DataObjNERHelper:
    """Same BIO span reconstruction as PPAudit, with typed-label support.

    Your model may emit untyped B/I (original dataobj_ner), typed B-DATA/I-DATA,
    or LABEL_n if config.json id2label is present.
    """
    DEFAULT_DATA_TYPES = {'', 'DATA', 'DAT', 'DATAOBJ', 'DATA_OBJ', 'DATA-OBJECT', 'INFORMATION'}
    DEFAULT_ENTITY_TYPES = {'ENTITY', 'ENT', 'ORG', 'PERSON', 'ACTOR', 'PARTY'}

    def __init__(self, dataobj_path, data_label_types=None, entity_label_types=None, id2label=None):
        self.classifier = pipeline('ner', model=str(dataobj_path))
        self.data_label_types = {t.upper() for t in (data_label_types or self.DEFAULT_DATA_TYPES)}
        self.entity_label_types = {t.upper() for t in (entity_label_types or self.DEFAULT_ENTITY_TYPES)}
        self.id2label = dict(id2label or {})
        cfg = getattr(getattr(self.classifier, 'model', None), 'config', None)
        if cfg is not None and getattr(cfg, 'id2label', None):
            self.id2label = {int(k) if str(k).isdigit() else k: v for k, v in cfg.id2label.items()}

    def _label_of(self, item):
        raw = item.get('entity') or item.get('entity_group') or item.get('label') or 'O'
        if raw in self.id2label:
            raw = self.id2label[raw]
        if str(raw).startswith('LABEL_') and str(raw)[6:].isdigit():
            idx = int(str(raw)[6:])
            raw = self.id2label.get(idx, raw)
        return raw

    def _is_begin(self, lab, wanted_types):
        pre, typ = _normalize_bio_label(lab)
        if pre in ('O', 'OUTSIDE'):
            return False
        if pre in ('B', 'S', 'U') and (typ in wanted_types or typ is None):
            return True
        return False

    def _is_inside(self, lab, wanted_types):
        pre, typ = _normalize_bio_label(lab)
        return pre in ('I', 'E', 'L') and (typ in wanted_types or typ is None)

    def _spans_of(self, text, results, wanted_types):
        pos_results = []
        if not results:
            return pos_results
        token_pt = 0
        while token_pt < len(results):
            lab = self._label_of(results[token_pt])
            starts = self._is_begin(lab, wanted_types)
            if not starts and self._is_inside(lab, wanted_types):
                prev_inside = token_pt > 0 and self._is_inside(self._label_of(results[token_pt - 1]), wanted_types)
                starts = not prev_inside
            if starts:
                work_pt = token_pt + 1
                while work_pt < len(results) and self._is_inside(self._label_of(results[work_pt]), wanted_types):
                    work_pt += 1
                work_pt -= 1
                start_index = results[token_pt]['start']
                end_index = results[work_pt]['end']
                pos_results.append((start_index, end_index, text[start_index:end_index]))
                token_pt = work_pt + 1
            else:
                token_pt += 1
        return pos_results

    def infer_dataobj(self, text):
        results = self.classifier(text)
        return self._spans_of(text, results, self.data_label_types)

    def infer_entities(self, text):
        results = self.classifier(text)
        return self._spans_of(text, results, self.entity_label_types)


def cleanupUnicodeErrors(term):
    # Cleanup from mistakes before... this should really be fixed during the intial parsing of the document...
    t = re.sub('\\ufffc', ' ', term)
    t = re.sub('â€œ', '', t)
    t = re.sub('â€\\u009d', '', t)
    t = re.sub('â\\u0080\\u0094', '', t)
    t = re.sub('â\\u0080\\u009d', '', t)
    t = re.sub('â\\u0080\\u009c', '', t)
    t = re.sub('â\\u0080\\u0099', '', t)
    t = re.sub('â€', '', t)
    t = re.sub('äë', '', t)
    t = re.sub('ä', '', t)
    t = re.sub('\\u0093', '', t)
    t = re.sub('\\u0092', '', t)
    t = re.sub('\\u0094', '', t)
    t = re.sub('\\u00a7', '', t)  # Section symbol
    t = re.sub('\\u25cf', '', t)  # bullet point symbol
    t = re.sub('´', '\'', t)
    t = re.sub('\\u00ac', '', t)
    t = re.sub('\\u00ad', '-', t)
    t = re.sub('\\u2211', '', t)
    t = re.sub('\\ufb01', 'fi', t)
    t = re.sub('\\uff0c', ', ', t)
    t = re.sub('\\uf0b7', '', t)
    t = re.sub('\\u037e', ';', t)
    return t


class Analytics:
    def __init__(self):
        self.dataStore = {}
        self.currentDoc = None

    def recordPolicyStatementAnalytics(self, policyStatement):
        # negation_distance and exceptImpact
        if self.currentDoc is None:
            self.startDoc('_adhoc_')
        sentenceText = cleanupUnicodeErrors(policyStatement['original_sentence'])
        if 'exceptImpact' in policyStatement and policyStatement['exceptImpact']:
            if sentenceText not in self.dataStore[self.currentDoc]['exceptions']:
                self.dataStore[self.currentDoc]['exceptions'][sentenceText] = 0
            self.dataStore[self.currentDoc]['exceptions'][sentenceText] += 1
        if 'negation_distance' in policyStatement and policyStatement['negation_distance'] >= 0:
            if sentenceText not in self.dataStore[self.currentDoc]['negations']:
                self.dataStore[self.currentDoc]['negations'][sentenceText] = []
            self.dataStore[self.currentDoc]['negations'][sentenceText].append((policyStatement['negation_distance'], policyStatement['action'][1].i))
        if sentenceText not in self.dataStore[self.currentDoc]['all']:
            self.dataStore[self.currentDoc]['all'][sentenceText] = 0
        self.dataStore[self.currentDoc]['all'][sentenceText] += 1

    def startDoc(self, filename):
        self.currentDoc = filename
        self.dataStore[self.currentDoc] = {
            'performance': {
                'startTime': time.time(),
                'endTime': 0,
            },
            'negations': {},
            'exceptions': {},
            'all': {}
        }

    def endDoc(self):
        if self.currentDoc is None:
            print('Error writing end time. No current document.')
            return

        self.dataStore[self.currentDoc]['performance']['endTime'] = time.time()
        self.currentDoc = None


class AnnotationType(Enum):
    NONE = 0
    DATA_OBJ = 1
    SHARE_VERB = 2
    COLLECT_VERB = 3
    SHARE_AND_COLLECT_VERB = 4
    ENTITY = 5

    @property
    def isShareOrCollect(self):
        return self in [AnnotationType.SHARE_VERB, AnnotationType.COLLECT_VERB, AnnotationType.SHARE_AND_COLLECT_VERB]

    @property
    def isCollect(self):
        return self == AnnotationType.COLLECT_VERB

    @property
    def isData(self):
        return self == AnnotationType.DATA_OBJ

    @property
    def isEntity(self):
        return self == AnnotationType.ENTITY

    @property
    def isNotNone(self):
        return self != AnnotationType.NONE

    @property
    def isNone(self):
        return self == AnnotationType.NONE


# TODO add pass
class KeyphraseTagger:
    # "use" is a special case... We may use your information in cojunction with advertisers to blah blah blah.
    DEFAULT_SHARE = ['share', 'sell', 'provide', 'trade', 'transfer',
                     'give', 'distribute', 'disclose', 'send', 'rent',
                     'exchange', 'report', 'transmit',
                     'pass', 'express', 'supply', 'display', 'publish',
                     'release', 'deliver', 'download', 'lease', 'release',
                     'view', 'show', 'forward', 'hold', 'swap',
                     'tell']
    DEFAULT_COLLECT = ['collect', 'check', 'know', 'use', 'obtain',
                       'access', 'receive', 'gather', 'store', 'save',
                       'know', 'keep', 'add', 'combine', 'record',
                       'retain', 'get', 'perform', 'analyze', 'recognize',
                       'process', 'track', 'log', 'remember', 'relate',
                       'create', 'ask', 'conduct', 'monitor', 'request',
                       'link', 'associate', 'solicit', 'read', 'preserve',
                       'contain', 'include']
    DEFAULT_ENTITIES = ['we', 'i', 'us', 'me', 'you']

    def __init__(self, dataobj_ner_path, share_verbs=None, collect_verbs=None,
                 entity_list=None, use_spacy_ner_entities=True,
                 data_label_types=None, entity_label_types=None,
                 also_tag_ner_entities=False, verb_blocklist=None,
                 multi_word_share=None, multi_word_collect=None):
        self.shareVerbs = [v.lower() for v in (share_verbs if share_verbs is not None else self.DEFAULT_SHARE)]
        self.collectVerbs = [v.lower() for v in (collect_verbs if collect_verbs is not None else self.DEFAULT_COLLECT)]
        self.shareVerbSet = set(self.shareVerbs)
        self.collectVerbSet = set(self.collectVerbs)
        self.verbBlocklist = {v.lower() for v in (verb_blocklist or [])}
        self.multiWordShare = [re.sub(r'\s+', ' ', p.strip().lower()) for p in (multi_word_share or []) if p.strip()]
        self.multiWordCollect = [re.sub(r'\s+', ' ', p.strip().lower()) for p in (multi_word_collect or []) if p.strip()]
        # merged multi-word lemmas are stored as the full phrase
        self.shareVerbSet.update(self.multiWordShare)
        self.collectVerbSet.update(self.multiWordCollect)
        for p in self.multiWordShare:
            if p not in self.shareVerbs:
                self.shareVerbs.append(p)
        for p in self.multiWordCollect:
            if p not in self.collectVerbs:
                self.collectVerbs.append(p)
        # gazetteer: exact token text/lemma plus multi-word phrases
        ents = list(entity_list) if entity_list is not None else list(self.DEFAULT_ENTITIES)
        self.entityTokens = set()
        self.entityPhrases = []  # list of tuple(tokens...)
        for e in ents:
            e = re.sub(r'\s+', ' ', str(e).strip().lower())
            if not e:
                continue
            parts = tuple(e.split(' '))
            if len(parts) == 1:
                self.entityTokens.add(e)
            else:
                self.entityPhrases.append(parts)
                # After NP merge the whole phrase is ONE token ("Our servers").
                self.entityTokens.add(e)
        self.entityPhrases.sort(key=lambda p: -len(p))
        self.use_spacy_ner_entities = use_spacy_ner_entities
        self.also_tag_ner_entities = also_tag_ner_entities
        self.dataobj_ner = DataObjNERHelper(
            dataobj_ner_path,
            data_label_types=data_label_types,
            entity_label_types=entity_label_types,
        ) if dataobj_ner_path else None

    def add_entities(self, extra):
        """Add per-document first-party aliases without rebuilding the tagger."""
        for e in extra or []:
            e = re.sub(r'\s+', ' ', str(e).strip().lower())
            if not e:
                continue
            parts = tuple(e.split(' '))
            self.entityTokens.add(e)
            if len(parts) > 1 and parts not in self.entityPhrases:
                self.entityPhrases.append(parts)
        self.entityPhrases.sort(key=lambda p: -len(p))

    def _token_in_entity_gazetteer(self, token):
        txt = token.text.lower()
        lem = token.lemma_.lower() if token.lemma_ != '-PRON-' else txt
        return txt in self.entityTokens or lem in self.entityTokens

    def _verb_surface(self, token):
        txt = token.text.lower()
        lem = token.lemma_.lower() if token.lemma_ != '-PRON-' else txt
        return txt, lem

    def _is_blocked_verb(self, token):
        txt, lem = self._verb_surface(token)
        return txt in self.verbBlocklist or lem in self.verbBlocklist

    def _accept_as_verb(self, token, lem):
        """Stage-B VerbMatcher: SOC lemmas need not be POS=VERB in clause-head deps."""
        if token.pos == spacy.symbols.VERB:
            return True
        # "is not sold / is collected" often tagged ADJ/VBN, not VERB
        if getattr(token, 'tag_', '') in {'VBN', 'VBD', 'VBZ', 'VBP', 'VB', 'VBG'}:
            return True
        if token.head is not token and token.head.lemma_.lower() == 'be' and token.dep_ in {
            'acomp', 'attr', 'oprd', 'xcomp', 'conj'
        }:
            return True
        if lem in {'view', 'report'} and token.pos != spacy.symbols.VERB:
            return False
        if lem in {'view', 'report'} and token.dep_ in {
            'compound', 'amod', 'nsubj', 'dobj', 'npadvmod', 'pobj'
        }:
            return False
        if lem in {'log', 'access'} and token.dep_ in {'amod', 'acl'}:
            return False
        if token.dep_ in {'ROOT', 'ccomp', 'xcomp', 'relcl', 'advcl', 'conj', 'nsubj'}:
            return True
        return False

    def _skip_use_siteish(self, token, lem):
        if lem != 'use':
            return False
        dobjs = [c for c in token.children if c.dep_ in {'dobj', 'pobj'}]
        return bool(dobjs) and dobjs[0].lemma_.lower() in SITEISH

    def _skip_use_for_purpose(self, token, lem):
        """'use DATA to authenticate' is purpose-of-use, not collect.

        Keep 'is used to collect/measure/track/...' (USED_TO_XCOMP / SOC).
        """
        if lem != 'use':
            return False

        def _has_neg(tok):
            return any(
                getattr(ch, 'dep_', '') == 'neg'
                or ch.lemma_.lower() in {'not', "n't", 'never', 'no'}
                for ch in tok.children
            )

        if _has_neg(token):
            return False
        for ch in token.children:
            if ch.dep_ in {'aux', 'auxpass'} and _has_neg(ch):
                return False
        if token.head is not token and token.head.lemma_.lower() in {'do', 'does', 'did', 'be'}:
            if _has_neg(token.head):
                return False
        soc = self.shareVerbSet | self.collectVerbSet | set(USED_TO_XCOMP)
        xcomps = [ch for ch in token.children if ch.dep_ in {'xcomp', 'advcl'}]
        if not xcomps:
            return False
        return not any(ch.lemma_.lower() in soc for ch in xcomps)

    def _skip_enable_without_soc(self, token, lem):
        if lem not in {'allow', 'enable', 'permit'}:
            return False
        soc = self.shareVerbSet | self.collectVerbSet | set(USED_TO_XCOMP)
        return not any(
            ch.dep_ in {'xcomp', 'ccomp'} and ch.lemma_.lower() in soc
            for ch in token.children
        )

    def _skip_receive_without_data(self, token, lem):
        if lem != 'receive':
            return False
        dobjs = [c for c in token.children if c.dep_ == 'dobj']
        if not dobjs:
            return True
        return any(
            d.lemma_.lower() in {'protection', 'level', 'treatment', 'safeguard'}
            for d in dobjs
        )

    def getTag(self, token):
        def isShareVerb(self, token):
            if self._is_blocked_verb(token):
                return False
            txt, lem = self._verb_surface(token)
            if lem not in self.shareVerbSet and txt not in self.shareVerbSet:
                return False
            if not self._accept_as_verb(token, lem):
                return False
            if self._skip_enable_without_soc(token, lem):
                return False
            return True

        def isCollectVerb(self, token):
            if self._is_blocked_verb(token):
                return False
            txt, lem = self._verb_surface(token)
            if lem not in self.collectVerbSet and txt not in self.collectVerbSet:
                return False
            if not self._accept_as_verb(token, lem):
                return False
            if self._skip_use_siteish(token, lem):
                return False
            if self._skip_use_for_purpose(token, lem):
                return False
            if self._skip_enable_without_soc(token, lem):
                return False
            if self._skip_receive_without_data(token, lem):
                return False
            return True

        def isEntity(self, token):
            if is_blocked_entity(token.text) or is_blocked_entity(token.lemma_):
                return False
            if self._token_in_entity_gazetteer(token):
                return True
            if self.use_spacy_ner_entities and token.ent_type_ in ['PERSON', 'ORG']:
                if is_blocked_entity(token.text):
                    return False
                if token.ent_type_ == 'ORG' and len(token.text) < 3:
                    return False
                return True
            return False

        def isDataObject(self, token):
            if token.pos == spacy.symbols.VERB:
                return False
            if token.ent_type_ == 'DATA':
                return True
            lem = token.lemma_.lower() if token.lemma_ != '-PRON-' else token.text.lower()
            txt = token.text.lower()
            if lem in DATA_HINTS or txt in DATA_HINTS:
                return True
            if 'cookie' in lem or 'cookie' in txt:
                return True
            words = set(re.findall(r"[a-z]+", lem + " " + txt))
            return bool(words & DATA_HINTS)

        #############################
        if isShareVerb(self, token):
            return AnnotationType.SHARE_VERB
        elif isCollectVerb(self, token):
            return AnnotationType.COLLECT_VERB
        elif isDataObject(self, token):
            return AnnotationType.DATA_OBJ
        elif isEntity(self, token):
            return AnnotationType.ENTITY
        return AnnotationType.NONE

    def _mark_phrase_spans(self, sentence, res, tag):
        """Longest-match gazetteer phrases onto spaCy tokens (PPAudit only did unigrams)."""
        toks = list(sentence)
        n = len(toks)
        i = 0
        while i < n:
            matched = None
            for phrase in self.entityPhrases:
                L = len(phrase)
                if i + L > n:
                    continue
                window = []
                for t in toks[i:i + L]:
                    window.append(t.text.lower())
                if tuple(window) == phrase:
                    matched = L
                    break
                window = []
                for t in toks[i:i + L]:
                    window.append(t.lemma_.lower() if t.lemma_ != '-PRON-' else t.text.lower())
                if tuple(window) == phrase:
                    matched = L
                    break
            if matched:
                for t in toks[i:i + matched]:
                    key = (t.i, t)
                    # do not overwrite an already-tagged SOC verb or data object
                    if key not in res or res[key].isNone:
                        res[key] = tag
                i += matched
            else:
                i += 1

    def tagSentence(self, sentence):
        res = {}
        dataobj_ner_results = []
        entity_ner_results = []
        if self.dataobj_ner is not None:
            dataobj_ner_results = self.dataobj_ner.infer_dataobj(sentence.text)
            if self.also_tag_ner_entities:
                entity_ner_results = self.dataobj_ner.infer_entities(sentence.text)
        for token in sentence:
            tag = self.getTag(token)
            for dataobj_ner_result in dataobj_ner_results:  # YZ TODO: check token idx
                if dataobj_ner_result[0] <= token.idx < dataobj_ner_result[1]:
                    tag = AnnotationType.DATA_OBJ
                    break
            if tag.isNotNone and tag != AnnotationType.DATA_OBJ:
                for ent_span in entity_ner_results:
                    if ent_span[0] <= token.idx < ent_span[1]:
                        # verbs always win; otherwise promote to ENTITY
                        if not tag.isShareOrCollect:
                            tag = AnnotationType.ENTITY
                        break
            if tag.isNotNone:
                res[(token.i, token)] = tag
        # multi-word entity phrases from your list
        self._mark_phrase_spans(sentence, res, AnnotationType.ENTITY)
        return res


# TODO Refactor -- these should all be instance methods, so you don't need to keep passing common objects (graph, sentence, tokenTags)
class DependencyGraphConstructor:
    @staticmethod
    def getConjugatedVerbs(sentence, targetTok=None):
        def isComma(token):
            return token.pos_ == 'PUNCT' and token.text == ','

        def isCConj(token):
            return token.pos == spacy.symbols.CCONJ and token.lemma_ in ['and', 'or', 'nor']

        def isNegation(token):
            return token.dep == spacy.symbols.neg

        def getConjugatedVerbsInternal(results, token):
            def _verbish(tok):
                return tok.pos == spacy.symbols.VERB or getattr(tok, 'tag_', '') in {
                    'VBN', 'VBD', 'VBZ', 'VBP', 'VB', 'VBG'
                }

            if _verbish(token):
                results.append(token)
            for tok in token.children:
                if tok.i < token.i:  # Ensure we only look at children that appear AFTER the token in the sentence
                    continue
                if tok.dep == spacy.symbols.conj and _verbish(tok):
                    if not getConjugatedVerbsInternal(results, tok):
                        return False
                elif not (isComma(tok) or isCConj(tok) or isNegation(tok)):
                    return False
            return True

        def isTokenContainedIn(token, conjugatedVerbs):
            for vbuffer in conjugatedVerbs:
                if token in vbuffer:
                    return True
            return False

        conjugatedVerbs = []
        vbuffer = []
        for token in sentence:
            if token.pos == spacy.symbols.VERB:
                # Make sure we didn't already cover the verb...
                if isTokenContainedIn(token, conjugatedVerbs):
                    continue

                vbuffer = []
                getConjugatedVerbsInternal(vbuffer, token)
                if len(vbuffer) > 1:
                    conjugatedVerbs.append(vbuffer)

        if targetTok is not None:
            for vbuffer in conjugatedVerbs:
                if targetTok in vbuffer:
                    return vbuffer
            return []
        return conjugatedVerbs

    @staticmethod
    def getRootNodes(graph):
        def hasNoInEdges(graph, node):
            return len([n for n in graph.in_edges(node)]) == 0
        root = [n for n in graph.nodes if hasNoInEdges(graph, n)]
        return root  # Could be multiple trees...

    @staticmethod
    def collapseConjugatedEntities(graph, sentence, tokenTags):
        def traverseDownward(graph, node):
            outEdges = [dst for src, dst in graph.out_edges(node)]  # Treat it as a stack instead...
            while len(outEdges) > 0:
                n = outEdges.pop()
                if graph[node][n]['label'] == 'conj' and node[2] == n[2] and node[2] in [AnnotationType.DATA_OBJ, AnnotationType.ENTITY]:
                    # Do not fuse "authkeys" with "no personal data"
                    t1 = (node[1].text or '').lower()
                    t2 = (n[1].text or '').lower()
                    n1 = bool(re.search(r'\b(no|none|neither)\b', t1))
                    n2 = bool(re.search(r'\b(no|none|neither)\b', t2))
                    if n1 != n2:
                        traverseDownward(graph, n)
                        continue
                    # Remove link from X --> Y
                    graph.remove_edge(node, n)
                    # Replace node...
                    graph.nodes[node]['lemma'] = '{},{}'.format(graph.nodes[node]['lemma'], graph.nodes[n]['lemma'])
                    graph.nodes[node]['lemmaList'].extend(graph.nodes[n]['lemmaList'])
                    graph.nodes[node]['label'] = '{}({}) - {}'.format(node[2], graph.nodes[node]['lemma'], node[1].i)
                    outEdges2 = [e for e in graph.out_edges(n)]
                    # Add all out links from Y --> Z to X --> Z (return all nodes, so we can add to outEdges...)
                    for src, dst in outEdges2:
                        graph.add_edge(node, dst, label=graph[src][dst]['label'])
                        graph.remove_edge(src, dst)
                        outEdges.append(dst)
                    graph.remove_node(n)
                    continue
                traverseDownward(graph, n)
        ##################
        # Get root node...
        roots = DependencyGraphConstructor.getRootNodes(graph)
        for r in roots:
            traverseDownward(graph, r)

    @staticmethod
    def getNodeAnnotationTag(node):
        return node[2]

    @staticmethod
    def isVerb(graph, node):
        pos = graph.nodes[node].get('pos')
        if pos == 'VERB' or pos == spacy.symbols.VERB:
            return True
        tok = node[1]
        return getattr(tok, 'tag_', '') in {'VBN', 'VBD', 'VBZ', 'VBP', 'VB', 'VBG'}

    @staticmethod
    def areAnnotationTagsEqual(node1, node2):
        t1 = DependencyGraphConstructor.getNodeAnnotationTag(node1)
        t2 = DependencyGraphConstructor.getNodeAnnotationTag(node2)
        return t1 == t2 or t1.isShareOrCollect and t2.isShareOrCollect

    @staticmethod
    def collapseConjugatedVerbs(graph, sentence, tokenTags):
        def getNewTag(n1, n2):
            if n2[2] != AnnotationType.SHARE_AND_COLLECT_VERB and n1[2].isNotNone and n1[2] != n2[2]:
                if n2[2].isNone:
                    return n1[2]
                elif (n1[2] == AnnotationType.SHARE_VERB and n2[2] == AnnotationType.COLLECT_VERB) or (n1[2] == AnnotationType.COLLECT_VERB and n2[2] == AnnotationType.SHARE_VERB) or n1[2] == AnnotationType.SHARE_AND_COLLECT_VERB:
                    return AnnotationType.SHARE_AND_COLLECT_VERB
            return n2[2]

        def addNewVerbNode(graph, node1, node2, docStart, docEnd):
            newTag = getNewTag(node1, node2)  # Get new annotation tag
            newKey = (node2[0], node2[1], newTag)  # FIXME this doesn't really represent the updated tag...
            negation = bool(graph.nodes[node1].get('neg') or graph.nodes[node2].get('neg'))
            newLemma = '{},{}'.format(graph.nodes[node1]['lemma'], graph.nodes[node2]['lemma'])
            newNodeLabel = '{}({}{})'.format(newTag, newLemma, ' - NOT' if negation else '')
            newLemmaList = []
            newLemmaList.extend(graph.nodes[node1]['lemmaList'])
            newLemmaList.extend(graph.nodes[node2]['lemmaList'])
            if newKey != node2:
                graph.add_node(newKey, label=newNodeLabel, lemma=newLemma, lemmaList=newLemmaList, tag=newTag, dep=node2[1].dep_, pos=node2[1].pos_, neg=negation, docStart=docStart, docEnd=docEnd)
                return (newKey, True)
            graph.nodes[node2]['lemma'] = newLemma
            graph.nodes[node2]['label'] = newNodeLabel
            graph.nodes[node2]['neg'] = negation
            graph.nodes[node2]['lemmaList'] = newLemmaList
            graph.nodes[node2]['tag'] = newTag
            graph.nodes[node2]['startDoc'] = docStart
            graph.nodes[node2]['endDoc'] = docEnd
            return (node2, False)

        ######################################
        # Let's just walk the damn graph...
        def traverseDownward(graph, node):
            outEdges = [dst for src, dst in graph.out_edges(node)]  # Treat it as a stack instead...
            while len(outEdges) > 0:
                n = outEdges.pop()
                if graph[node][n]['label'] == 'conj' and DependencyGraphConstructor.areAnnotationTagsEqual(node, n) and DependencyGraphConstructor.isVerb(graph, node) and DependencyGraphConstructor.isVerb(graph, n):
                    # TODO the key changes due to the annotation tag potentially changing...
                    # TODO ensure separation
                    nodeTok = node[1]
                    nodeChildTok = n[1]
                    if nodeChildTok in DependencyGraphConstructor.getConjugatedVerbs(sentence, targetTok=nodeTok):
                        # Remove link from X --> Y
                        graph.remove_edge(node, n)
                        # Get new Tag
                        newTag = getNewTag(node, n)
                        if newTag == node:
                            graph.nodes[node]['lemma'] = '{},{}'.format(graph.nodes[node]['lemma'], graph.nodes[n]['lemma'])
                            graph.nodes[node]['lemmaList'].extend(graph.nodes[n]['lemmaList'])
                            graph.nodes[node]['label'] = '{}({}) - {}'.format(node[2], graph.nodes[node]['lemma'], node[1].i)
                            # Add all out links from Y --> Z to X --> Z (return all nodes, so we can add to outEdges...)
                            for src, dst in graph.out_edges(n):
                                graph.add_edge(node, dst, label=graph[src][dst]['label'])
                                graph.remove_edge(src, dst)
                                outEdges.append(dst)
                            graph.remove_node(n)
                        else:
                            # Add new tag...
                            startDoc = nodeTok.i if nodeTok.i < nodeChildTok.i else nodeChildTok.i
                            endDoc = nodeTok.i if nodeTok.i > nodeChildTok.i else nodeChildTok.i
                            newNode, addedNewNode = addNewVerbNode(graph, n, node, startDoc, endDoc)

                            if addedNewNode:
                                # Add in edges
                                for s, t in list(graph.in_edges(node)):
                                    graph.add_edge(s, newNode, label=graph[s][t]['label'])
                                    graph.remove_edge(s, t)

                                # Add out edges
                                for s, t in list(graph.out_edges(node)):
                                    graph.add_edge(newNode, t, label=graph[s][t]['label'])
                                    graph.remove_edge(s, t)

                            if not addedNewNode:
                                newNode = node

                            # Add all out links from Y --> Z to X --> Z (return all nodes, so we can add to outEdges...)
                            for src, dst in list(graph.out_edges(n)):
                                graph.add_edge(newNode, dst, label=graph[src][dst]['label'])
                                graph.remove_edge(src, dst)
                                outEdges.append(dst)

                            # Remove node from graph
                            if addedNewNode:
                                graph.remove_node(node)
                            node = newNode
                            graph.remove_node(n)
                        continue
                traverseDownward(graph, n)
        roots = DependencyGraphConstructor.getRootNodes(graph)
        for r in roots:
            traverseDownward(graph, r)

    @staticmethod
    def isVerbNegated(token, sentence):
        NEG_LEX = {'not', "n't", 'never', 'no', 'none', 'neither', 'nor'}

        def _has_neg_surface(tok):
            if tok is None:
                return False
            if tok.dep == spacy.symbols.neg or tok.lemma_.lower() in NEG_LEX or tok.text.lower() in NEG_LEX:
                return True
            return bool(re.search(r'\b(no|none|neither|never|not|n\'t)\b', tok.text.lower()))

        def isVerbNegatedInternal(token):
            if token is None:
                return False
            for t in token.children:
                if _has_neg_surface(t):
                    return True
            if token.head is not token and token.head.lemma_.lower() == 'be':
                for t in token.head.children:
                    if _has_neg_surface(t):
                        return True
            for t in token.children:
                if t.dep_ in {'nsubj', 'nsubjpass'}:
                    if _has_neg_surface(t):
                        return True
                    for s in t.children:
                        if s.dep_ in {'det', 'predet', 'amod', 'neg', 'nummod'} and _has_neg_surface(s):
                            return True
            return False

        if isVerbNegatedInternal(token):
            return True

        # Check if verb is part of conjugated verb phrase, if so, check if any of those are negated
        conjugatedVerbs = DependencyGraphConstructor.getConjugatedVerbs(sentence, token)
        for tok in conjugatedVerbs:
            if isVerbNegatedInternal(tok):
                return True

        # Check if verb is xcomp, if so check if prior verb is negated?
        # TODO should also do advcl
        if token.dep == spacy.symbols.xcomp or token.dep == spacy.symbols.advcl:
            return DependencyGraphConstructor.isVerbNegated(token.head, sentence)
        return False

    @staticmethod
    def pruneUnattachedNodes(graph):
        def pruneChildren(graph, node):
            for s, t in list(graph.out_edges(node)):
                pruneChildren(graph, t)
            if node in graph.nodes:
                graph.remove_node(node)

        def removeNodes(graph, nodesToRemove):
            for node in nodesToRemove:
                pruneChildren(graph, node)

        def hasNoOutEdges(graph, node):
            return len([n for n in graph.out_edges(node)]) == 0

        def hasNoInEdges(graph, node):
            return len([n for n in graph.in_edges(node)]) == 0

        def doesGraphContainVerb(graph, root):
            if root[2].isShareOrCollect:
                return True
            for _, n in graph.out_edges(root):
                if doesGraphContainVerb(graph, n):
                    return True
            return False

        nodesToRemove = [node for node in graph.nodes if hasNoOutEdges(graph, node) and hasNoInEdges(graph, node)]
        removeNodes(graph, nodesToRemove)

        # Let's prune graphs that have no verbs...
        potentialRoots = [node for node in graph.nodes if hasNoInEdges(graph, node)]
        if len(potentialRoots) > 1:
            subGraphsToPrune = [r for r in potentialRoots if not doesGraphContainVerb(graph, r)]
            if len(subGraphsToPrune) < len(potentialRoots) and len(subGraphsToPrune) > 0:
                removeNodes(graph, subGraphsToPrune)

    @staticmethod
    def pruneNonSharingVerbs(graph):
        def getHead(graph, node):
            parents = [src for src, _ in graph.in_edges(node)]
            return parents[0] if len(parents) > 0 else node

        def subTreeContainsLabeledTags(graph, node, checkMatch=False):
            if checkMatch and node[2].isNotNone:
                return True
            for _, dst in graph.out_edges(node):
                if subTreeContainsLabeledTags(graph, dst, True):
                    return True
            return False

        def childrenContainDataPractice(node):  # One of the descendents need to contain a share verb...
            if (node[1].pos == spacy.symbols.VERB and node[2].isShareOrCollect and subTreeContainsLabeledTags(graph, node)):
                return True
#           elif(node[1].pos == spacy.symbols.VERB and node[1].dep_ == 'relcl'):# Only IF THE HEAD IS A DATA OBJECT OR ENTITY...
#               n2 = getHead(graph, node)[2]
#               if n2.isData or n2.isEntity:
#                   return True
            for s, child in graph.out_edges(node):
                if childrenContainDataPractice(child):
                    return True
            return False

        def pruneChildren(graph, node):
            for s, t in list(graph.out_edges(node)):
                pruneChildren(graph, t)
            if node in graph.nodes:
                graph.remove_node(node)

        def removeNodes(graph, nodesToRemove):
            for node in nodesToRemove:
                pruneChildren(graph, node)

        def hasNoOutEdges(graph, node):
            return len([n for n in graph.out_edges(node)]) == 0

        def hasNoInEdges(graph, node):
            return len([n for n in graph.in_edges(node)]) == 0

        #############################
        nodesToRemove = [node for node in graph.nodes if node[1].pos == spacy.symbols.VERB and not childrenContainDataPractice(node)]
        removeNodes(graph, nodesToRemove)

        nodesToRemove = [node for node in graph.nodes if hasNoOutEdges(graph, node) and hasNoInEdges(graph, node)]
        removeNodes(graph, nodesToRemove)

        nodesToRemove = [node for node in graph.nodes if hasNoOutEdges(graph, node) and node[2].isNone and node[1].dep not in [spacy.symbols.nsubj, spacy.symbols.dobj, spacy.symbols.nsubjpass]]
        while len(nodesToRemove) > 0:
            removeNodes(graph, nodesToRemove)
            nodesToRemove = [node for node in graph.nodes if hasNoOutEdges(graph, node) and node[2].isNone and node[1].dep not in [spacy.symbols.nsubj, spacy.symbols.dobj, spacy.symbols.nsubjpass]]

        ########################

    @staticmethod
    def convertDTreeToNxGraph(sentence, tokenTags):
        def addNode(key, node, graph, sentence):
            if key not in graph:
                negation = False
                if key[2].isShareOrCollect:
                    negation = DependencyGraphConstructor.isVerbNegated(node, sentence)
                    graph.add_node(key, label='{}({}{}) - {}'.format(key[2], node.lemma_, ' - NOT' if negation else '', node.i), tag=key[2], lemma=node.lemma_, lemmaList=[node.lemma_ if node.lemma_ != '-PRON-' else node.text.lower()], dep=node.dep_, pos=node.pos_, neg=negation, docStart=node.i, docEnd=node.i)
                else:
                    graph.add_node(key, label='{}({}) - {}'.format(key[2], node.lemma_, node.i), tag=key[2], lemma=node.lemma_, lemmaList=[node.lemma_ if node.lemma_ != '-PRON-' else node.text.lower()], dep=node.dep_, pos=node.pos_, neg=negation, docStart=node.i, docEnd=node.i)

        def convertDTreeToNxGraphInternal(root, graph, tokenTags, sentence):
            rkey = DependencyGraphConstructor.getKey(root, tokenTags)

            if rkey not in graph:
                addNode(rkey, root, graph, sentence)

            for c in root.children:
                ckey = DependencyGraphConstructor.getKey(c, tokenTags)
                if ckey not in graph:
                    addNode(ckey, c, graph, sentence)

                graph.add_edge(rkey, ckey, label=c.dep_)
                convertDTreeToNxGraphInternal(c, graph, tokenTags, sentence)
        ##############
        dgraph = nx.DiGraph()
        convertDTreeToNxGraphInternal(sentence.root, dgraph, tokenTags, sentence)
        return dgraph

    @staticmethod
    def drawGraph(g, filename):
        # try:
        #     A = nx.drawing.nx_agraph.to_agraph(g)
        #     A.draw(filename, prog='dot', args='-Granksep=2.0')
        # except:# FIXME unicode error here for some reason...
        #     pass
        return

    @staticmethod
    def getKey(root, tokenTags):
        tKey = (root.i, root)
        tag = AnnotationType.NONE if tKey not in tokenTags else tokenTags[tKey]
        return (root.i, root, tag)

    @staticmethod
    def getSimplifiedDependencyGraph(sentence, tokenTags):
        def getPathBetweenNodes(g, itok, jtok, tokenTags):
            pathNodes = nx.shortest_path(g.to_undirected(), DependencyGraphConstructor.getKey(itok, tokenTags), DependencyGraphConstructor.getKey(jtok, tokenTags))
            return g.subgraph(pathNodes).copy()

        ##############################
        if len(tokenTags) <= 1:  # Need two or more tokens...
            return None

        g = DependencyGraphConstructor.convertDTreeToNxGraph(sentence, tokenTags)
        graphs = []
        taggedTokens = [(token, tokenTags[(token.i, token)]) for i, token in tokenTags]
        for i, (itok, itag) in enumerate(taggedTokens):
            for j, (jtok, jtag) in enumerate(taggedTokens[i+1:]):
                graphs.append(getPathBetweenNodes(g, itok, jtok, tokenTags))

        # Do not prune subjects and objects...
        # TODO is it just share verbs or all?
        for i, (itok, itag) in enumerate(taggedTokens):
            if itag.isShareOrCollect:
                for _, dst in g.out_edges(DependencyGraphConstructor.getKey(itok, tokenTags)):
                    if dst[1].dep in [spacy.symbols.dobj, spacy.symbols.nsubj, spacy.symbols.nsubjpass] and dst[2].isNone:
                        graphs.append(getPathBetweenNodes(g, itok, dst[1], tokenTags))

        #################################

        g = nx.compose_all(graphs)
        DependencyGraphConstructor.collapseConjugatedVerbs(g, sentence, tokenTags)
        # Prune non-attached nodes...
        DependencyGraphConstructor.pruneUnattachedNodes(g)
        DependencyGraphConstructor.collapseConjugatedEntities(g, sentence, tokenTags)
        DependencyGraphConstructor.pruneNonSharingVerbs(g)
        #DependencyGraphConstructor.drawGraph(g, 'simplified_graph.png')
        return g


class PolicyTransformer:
    # TODO refactor so that these are instance methods
    # implicit everyone rule
    @staticmethod
    def applyPolicyTransformationRules(policyStatements, analyticsObj):
        def addPolicies(entity, collect, dataObjects, original_sentence, simplifiedStatements, actionLemma):
            # FIXME should not get a token at this point...
            if (type(entity) == str and entity == 'you') or (type(entity) == spacy.tokens.token.Token and entity.text == 'you'):
                return
            for d in dataObjects:

                simplifiedStatements.append((cleanupUnicodeErrors(entity), cleanupUnicodeErrors(collect), cleanupUnicodeErrors(d), cleanupUnicodeErrors(original_sentence), cleanupUnicodeErrors(actionLemma)))

        def addPoliciesByEntities(entities, collect, dataObjects, original_sentence, simplifiedStatements, actionLemma):
            if entities is not None and len(entities) > 0:
                if type(entities) == list:
                    for e in entities:
                        addPolicies(e, collect, dataObjects, original_sentence, simplifiedStatements, actionLemma)
                else:
                    addPolicies(entities, collect, dataObjects, original_sentence, simplifiedStatements, actionLemma)
            else:
                addPolicies('third_party_implicit', collect, dataObjects, original_sentence, simplifiedStatements, actionLemma)

        def getAgentText(agent):
            if agent is None:  # TODO CHECKME: Should we really have an implicit first party
                return 'we_implicit'
            if type(agent) == str:
                return agent
            if type(agent[1]) == str:
                return agent[1]
            return agent[1].lemma_ if agent[1].lemma_ != '-PRON-' else agent[1].text.lower()
            # return agent[1] if type(agent[1]) == unicode else agent[1].text.lower() #This needs to be the lemma unless -PRON-

        def handleShareVerb(pstatement, actionLemma, simplifiedStatements):
            agents = [getAgentText(a) for a in pstatement['agent']]
            original_sentence = pstatement['original_sentence']
            practice = 'not_share' if pstatement['is_negated'] else 'share'

            # If it is "you share/not share" and entities is nil, do not assume third-party
            if len(agents) == 1 and type(agents[0]) == str and agents[0] == 'you':
                if pstatement['entities'] is None or len(pstatement['entities']) == 0:
                    pstatement['entities'] = ['we_implicit']

            # Actor of a share/sell/disclose verb is sharing, not collecting.
            # "with/to X" is the recipient, not a second actor.
            addPoliciesByEntities(agents, practice, pstatement['data_objects'], original_sentence, simplifiedStatements, actionLemma)

        def handleCollectVerb(pstatement, actionLemma, simplifiedStatements):
            agents = [getAgentText(a) for a in pstatement['agent']]
            collect = 'not_collect' if pstatement['is_negated'] else 'collect'

            if pstatement['is_negated'] and actionLemma == 'use':
                # Drop "do not use X to address/authenticate".
                # Keep "do not use cookies / personal data / tokens".
                blobs = ' '.join(str(d).lower() for d in (pstatement.get('data_objects') or []))
                if not re.search(
                    r'\b(cookie|cooky|data|information|token|email|ip|address|identifier|history)\b',
                    blobs,
                ):
                    return

            # PPAudit used to force store/save → collect even when negated
            # ("do not store PII"). Keep the negation.

            original_sentence = pstatement['original_sentence']
            addPoliciesByEntities(agents, collect, pstatement['data_objects'], original_sentence, simplifiedStatements, actionLemma)

        simplifiedStatements = []
        # Array of statements
        for pstatement in policyStatements:
            # TODO analytics... exceptImpact, negation_distance
            analyticsObj.recordPolicyStatementAnalytics(pstatement)

            # Get the lemmas and do it this way instead...
            share_verbs = pstatement.get('_share_verbs') or KeyphraseTagger.DEFAULT_SHARE
            collect_verbs = pstatement.get('_collect_verbs') or KeyphraseTagger.DEFAULT_COLLECT
            for actionLemma in pstatement['action_lemmas']:
                lemma = actionLemma.lower() if isinstance(actionLemma, str) else actionLemma
                if lemma in share_verbs:
                    handleShareVerb(pstatement, actionLemma, simplifiedStatements)
                elif lemma in collect_verbs:
                    handleCollectVerb(pstatement, actionLemma, simplifiedStatements)

            # if pstatement['action'][2] in [AnnotationType.SHARE_VERB, AnnotationType.SHARE_AND_COLLECT_VERB]:
            #     handleShareVerb(pstatement, simplifiedStatements)
            # if pstatement['action'][2] in [AnnotationType.COLLECT_VERB, AnnotationType.SHARE_AND_COLLECT_VERB]:
            #     handleCollectVerb(pstatement, simplifiedStatements)
        return list(set(simplifiedStatements))

    @staticmethod
    def handleExceptions(policyStatements, tags):
        def clonePolicyStatement(pol):
            return {'data_objects': pol['data_objects'], 'entities': pol['entities'], 'agent': pol['agent'], 'action': pol['action'], 'action_lemmas': pol['action_lemmas'], 'is_negated': pol['is_negated'], 'negation_distance': pol['negation_distance'], 'original_sentence': pol['original_sentence'], 'exceptions': pol['exceptions'], '_share_verbs': pol.get('_share_verbs'), '_collect_verbs': pol.get('_collect_verbs')}

        def lemmatize(tokens):
            return ' '.join(t.lemma_ for t in tokens)

        def getRelevantTags(e, tags):
            return {(term.i, term): tags[(term.i, term)] for term in e if (term.i, term) in tags}

        def isAllData(tags):
            return all(tags[k].isData for k in tags)

        def isAllEntity(tags):
            return all(tags[k].isEntity for k in tags)

        newStatements = []
        removePolicyStatements = []
        for pol in policyStatements:
            if pol['exceptions'] is not None and len(pol['exceptions']) > 0:

                # Get all exceptions at first that can be resolved with keywords or all data and entity
                excepts = [(v, e) for v, e in pol['exceptions']]
                for v, e in pol['exceptions']:

                    # Record how often exceptions affect policy statements...
                    relTags = getRelevantTags(e, tags)
                    elemma = lemmatize(e)
                    if re.search(r'^.*\b(consent|you\sagree|your\s(express\s)?permission|you\sprovide|opt([\s\-](out|in))?|respond\sto\syou(r)?|disclose\sin\sprivacy\spolicy|follow(ing)?\scircumstance|permit\sby\schildren\'s\sonline\sprivacy\sprotection\sact)\b.*$', elemma):
                        # Only do the exceptions in negative cases...
                        # For example, we do not want to reverse: "We collect your personal information without your consent"
                        if not pol['is_negated']:
                            continue
                        newPol = clonePolicyStatement(pol)
                        newPol['is_negated'] = not newPol['is_negated']
                        newPol['exceptions'] = None  # TODO do we ever really need this again?
                        newPol['exceptImpact'] = True
                        newStatements.append(newPol)
                        excepts.remove((v, e))
                        removePolicyStatements.append(pol)
                    elif elemma in ['require by law', 'we receive subpoena', 'law'] or re.search(r'\b(law|legal|subpoena|court order|government)\b', elemma or ''):
                        # Only do the exceptions in negative cases...
                        # For example, we do not want to reverse: "We collect your personal information without your consent"
                        if not pol['is_negated']:
                            continue
                        newPol = clonePolicyStatement(pol)
                        newPol['entities'] = ['government agency']
                        newPol['is_negated'] = not newPol['is_negated']
                        newPol['exceptions'] = None  # TODO do we ever really need this again?
                        newPol['exceptImpact'] = True
                        newStatements.append(newPol)
                        excepts.remove((v, e))
                        # Keep the original not_collect; law is an extra tuple, not a replacement.
                    elif len(relTags) == len(e):
                        newPol = clonePolicyStatement(pol)
                        newPol['is_negated'] = not newPol['is_negated']
                        newPol['exceptions'] = None  # TODO do we ever really need this again?
                        newPol['exceptImpact'] = True
                        # If ALL data items
                        if isAllData(relTags):
                            newPol['data_objects'] = [data.lemma_ for index, data in relTags]
                            newStatements.append(newPol)
                            excepts.remove((v, e))
                            # removePolicyStatements.append(pol)
                        # If ALL entities
                        elif isAllEntity(relTags):
                            if newPol['action'][2].isCollect:
                                newPol['agent'] = [data.lemma_ for index, data in relTags]
                            else:
                                newPol['entities'] = [data.lemma_ for index, data in relTags]
                            newStatements.append(newPol)
                            excepts.remove((v, e))
                            # removePolicyStatements.append(pol)
                    else:  # Not sure what it is, let's flip it anyway...
                        if not pol['is_negated']:
                            continue
                        newPol = clonePolicyStatement(pol)
                        newPol['is_negated'] = not newPol['is_negated']
                        newPol['exceptImpact'] = True
                        newPol['exceptions'] = None  # TODO do we ever really need this again?
                        newStatements.append(newPol)
                        excepts.remove((v, e))
                        removePolicyStatements.append(pol)
        for pol in newStatements:
            policyStatements.append(pol)
        for pol in removePolicyStatements:
            if pol in policyStatements:
                policyStatements.remove(pol)
        return policyStatements



def merge_multiword_verbs(doc, phrases):
    """Collapse each multi-word SOC phrase into one VERB token (spaCy 2 merge or spaCy 3 retokenize)."""
    if not phrases:
        return
    phrases = sorted({re.sub(r'\s+', ' ', p.strip().lower()) for p in phrases if p and p.strip()},
                     key=lambda s: (-len(s.split()), -len(s)))
    i = 0
    while i < len(doc):
        matched = None
        for phrase in phrases:
            parts = phrase.split(' ')
            L = len(parts)
            if L < 2 or i + L > len(doc):
                continue
            window = [doc[i + k].text.lower() for k in range(L)]
            window_lem = [
                (doc[i + k].lemma_.lower() if doc[i + k].lemma_ != '-PRON-' else doc[i + k].text.lower())
                for k in range(L)
            ]
            if window == parts or window_lem == parts:
                matched = (L, phrase)
                break
        if matched is None:
            i += 1
            continue
        L, phrase = matched
        span = doc[i:i + L]
        try:
            span.merge(pos=spacy.symbols.VERB, lemma=phrase)
        except Exception:
            try:
                with doc.retokenize() as retok:
                    retok.merge(span, attrs={'LEMMA': phrase, 'POS': 'VERB'})
            except Exception:
                pass
        i += 1


class MixExtractor:
    def __init__(self, nlpModel, analyticsObj, dataobj_ner_path,
                 share_verbs=None, collect_verbs=None, entity_list=None,
                 use_spacy_ner_entities=True, data_label_types=None,
                 entity_label_types=None, also_tag_ner_entities=False,
                 verb_blocklist=None, multi_word_share=None, multi_word_collect=None):
        self.tagger = KeyphraseTagger(
            dataobj_ner_path,
            share_verbs=share_verbs,
            collect_verbs=collect_verbs,
            entity_list=entity_list,
            use_spacy_ner_entities=use_spacy_ner_entities,
            data_label_types=data_label_types,
            entity_label_types=entity_label_types,
            also_tag_ner_entities=also_tag_ner_entities,
            verb_blocklist=verb_blocklist,
            multi_word_share=multi_word_share,
            multi_word_collect=multi_word_collect,
        )
        # spaCy 2 English object OR a model name / path
        try:
            is_english = isinstance(nlpModel, spacy.lang.en.English)
        except Exception:
            is_english = False
        self.parser = nlpModel if is_english or hasattr(nlpModel, '__call__') and hasattr(nlpModel, 'vocab') else spacy.load(nlpModel)
        self.analyticsObj = analyticsObj        
        
    def parseText(self, paragraph):
        paragraph = re.sub(r'\bid\b', 'identifier', paragraph)  # Spacy parses "id" as "i would", so fix here...
        doc = self.parser(paragraph)
        epm.mergeExcludePhrases(doc, self.parser.vocab)
        # Merge selective multi-word SOC phrases into one VERB token so the
        # rest of the PPAudit graph walk sees a single action node.
        merge_multiword_verbs(
            doc,
            self.tagger.multiWordShare + self.tagger.multiWordCollect,
        )
        npm.mergeNounPhrasesDoc(doc, self.parser.vocab)
        return doc

    def extractData(self, depGraph, sentence, verbose=False):
        def isVerbNearestAncestor(targetVerb, exceptVerb):
            if targetVerb == exceptVerb:
                return True
            if exceptVerb.pos == spacy.symbols.VERB and self.tagger.getTag(exceptVerb).isShareOrCollect:
                return False
            if exceptVerb.head == exceptVerb:  # Hit the root
                return False
            return isVerbNearestAncestor(targetVerb, exceptVerb.head)

        def getReleventExceptions(verb, exceptions):
            if exceptions is None or len(exceptions) == 0:
                return exceptions
            return [(v, e) for v, e in exceptions if isVerbNearestAncestor(verb, v)]

        def getNearestAnnotVerb(depGraph, node):
            for s, _ in depGraph.in_edges(node):
                if s[2].isShareOrCollect:
                    return s

            for s, _ in depGraph.in_edges(node):
                res = getNearestAnnotVerb(depGraph, s)
                if res is not None:
                    return res
            return None

        def hasSubjectAndDobj(depGraph, node):
            hasSubject = any(n for _, n in depGraph.out_edges(node) if n[1].dep in [spacy.symbols.nsubj, spacy.symbols.nsubjpass])
            hasObject = any(n for _, n in depGraph.out_edges(node) if n[1].dep in [spacy.symbols.dobj])
            return hasSubject and hasObject

        def extractDataObjects(depGraph, baseNode):
            def extractDataObjectsInternal(results, depGraph, baseNode, force_neg=False):
                verb_neg = force_neg or bool(depGraph.nodes.get(baseNode, {}).get('neg'))
                for _, node in depGraph.out_edges(baseNode):
                    if node[2].isData:
                        tok = node[1]
                        bits = [tok.text or '', tok.lemma_ or '']
                        if hasattr(tok, 'lefts'):
                            bits.extend(t.text for t in tok.lefts)
                        surface = ' '.join(bits).lower()
                        denied_np = bool(re.search(r'\b(no|none|neither)\b', surface))
                        rel = depGraph[baseNode][node].get('label')
                        if (not verb_neg) and denied_np:
                            continue
                        if verb_neg and not denied_np and rel not in ('nsubj', 'nsubjpass', 'dobj'):
                            continue
                        results.append(node)
                    elif node[2].isShareOrCollect:  # Extract from NEAREST verb only
                        continue
                    elif node[1].pos == spacy.symbols.ADP and node[1].lemma_ in ['except when', 'except where', 'unless when', 'unless where', 'except for', 'except in', 'except under', 'unless for', 'unless in', 'unless under', 'apart from', 'aside from', 'with the exception of', 'other than', 'except to', 'unless to', 'unless as', 'except as']:
                        continue
                    extractDataObjectsInternal(results, depGraph, node, force_neg=verb_neg)
            ##########################
            dataObjects = []
            extractDataObjectsInternal(dataObjects, depGraph, baseNode)

            verb_neg = bool(depGraph.nodes.get(baseNode, {}).get('neg'))
            # Negated verb with no local object: take DATA off the conj-head
            # (asked → no personal data), never off a parent collect verb
            # (uses → authkeys).
            if len(dataObjects) == 0 and verb_neg:
                for src, _ in list(depGraph.in_edges(baseNode)):
                    if src[2].isShareOrCollect:
                        continue
                    extractDataObjectsInternal(dataObjects, depGraph, src, force_neg=True)
            # "IP addresses are required to be logged": subject sits on
            # the parent (required), practice verb is the xcomp (logged).
            if len(dataObjects) == 0:
                for src, _ in list(depGraph.in_edges(baseNode)):
                    if src[2].isShareOrCollect:
                        continue
                    for _, child in depGraph.out_edges(src):
                        rel = depGraph[src][child].get('label')
                        if not child[2].isData or rel not in ('nsubj', 'nsubjpass'):
                            continue
                        tok = child[1]
                        bits = [tok.text or '', tok.lemma_ or '']
                        if hasattr(tok, 'lefts'):
                            bits.extend(t.text for t in tok.lefts)
                        denied_np = bool(re.search(r'\b(no|none|neither)\b', ' '.join(bits).lower()))
                        if verb_neg and not denied_np:
                            continue
                        if (not verb_neg) and denied_np:
                            continue
                        dataObjects.append(child)
            if len(dataObjects) == 0 and not verb_neg and not hasSubjectAndDobj(depGraph, baseNode):
                v = getNearestAnnotVerb(depGraph, baseNode)
                if v is not None and v != baseNode:
                    extractDataObjectsInternal(dataObjects, depGraph, v)
            return dataObjects

        def getAgent(depGraph, baseNode):
            def getEntityConjunctions(depGraph, node):
                def getEntityConjunctionsInternal(depGraph, node, res):
                    for _, target in depGraph.out_edges(node):
                        if depGraph[node][target]['label'] == 'conj':
                            res.append(target)
                            getEntityConjunctionsInternal(depGraph, target, res)
                    return res
                res = [node]
                res = getEntityConjunctionsInternal(depGraph, node, res)
                return res

            def getAgentInternal(depGraph, baseNode, skipTraverseUpwards=False, isXcomp=False):
                nsubj = None
                nsubjpass = None
                agentPobj = None
                dobj = None
                # Check children for the subject or agent if subject is passive...
                for _, node in depGraph.out_edges(baseNode):
                    if depGraph[baseNode][node]['label'] == 'nsubj':
                        nsubj = node
                    elif depGraph[baseNode][node]['label'] == 'nsubjpass':
                        nsubjpass = node
                    elif depGraph[baseNode][node]['label'] == 'dobj' or depGraph[baseNode][node]['label'] == 'dative':
                        dobj = node
                    elif depGraph[baseNode][node]['label'] == 'agent':  # "Agent" dependency tag
                        for _, node2 in depGraph.out_edges(node):
                            if node2[2].isEntity:
                                agentPobj = node2

                if nsubj is None:
                    nsubj = nsubjpass

                parent_lemma = baseNode[1].lemma_.lower() if baseNode and hasattr(baseNode[1], 'lemma_') else ''
                # "use Flash to store data": Flash is the instrument (dobj of use),
                # games/we is the agent. Do not promote dobj when the parent is "use".
                use_instrument = parent_lemma == 'use' or (
                    isXcomp and parent_lemma in {'use', 'used'}
                )
                if isXcomp and not use_instrument:
                    # If xcomp prefer dobj over nsubj...
                    if dobj is not None and dobj[2].isEntity:
                        return getEntityConjunctions(depGraph, dobj)
                    if nsubj is not None and nsubj[2].isEntity:
                        return getEntityConjunctions(depGraph, nsubj)
                    if nsubjpass is not None and nsubjpass[2].isEntity:
                        return getEntityConjunctions(depGraph, nsubjpass)
                    if agentPobj is not None and agentPobj[2].isEntity:
                        return getEntityConjunctions(depGraph, agentPobj)
                elif isXcomp and use_instrument:
                    if nsubj is not None and nsubj[2].isEntity:
                        return getEntityConjunctions(depGraph, nsubj)
                    if nsubjpass is not None and nsubjpass[2].isEntity:
                        return getEntityConjunctions(depGraph, nsubjpass)
                    if agentPobj is not None and agentPobj[2].isEntity:
                        return getEntityConjunctions(depGraph, agentPobj)
                else:
                    if nsubj is not None and nsubj[2].isEntity:
                        return getEntityConjunctions(depGraph, nsubj)
                    if nsubjpass is not None and nsubjpass[2].isEntity:
                        return getEntityConjunctions(depGraph, nsubjpass)
                    if agentPobj is not None and agentPobj[2].isEntity:
                        return getEntityConjunctions(depGraph, agentPobj)
                    if dobj is not None and dobj[2].isEntity:
                        return getEntityConjunctions(depGraph, dobj)

                # Untagged subject ("These advertisers", "Some games"): still the actor
                if nsubj is not None and not nsubj[2].isData:
                    return getEntityConjunctions(depGraph, nsubj)
                if nsubjpass is not None and not nsubjpass[2].isData:
                    return getEntityConjunctions(depGraph, nsubjpass)

                if not skipTraverseUpwards:
                    # If we don't find anything, get the parent verb if exists and search there
                    for node, _ in depGraph.in_edges(baseNode):
                        parent_is_use = node[1].lemma_.lower() == 'use'
                        res = getAgentInternal(
                            depGraph, node,
                            skipTraverseUpwards=True,
                            isXcomp=(baseNode[1].dep in [spacy.symbols.xcomp, spacy.symbols.advcl]) or parent_is_use,
                        )
                        if res is not None:
                            return res
                return None
            ##################
            agent = getAgentInternal(depGraph, baseNode)
            if agent is None or len(agent) == 0:  # If we haven't found anything return the default (i.e., "we") -- Rationale: "Personal information may be collected." means implicit "we"
                return ['we_implicit']  # Implicit first party
            return agent

        def ignoreActionObjectPair(verb, dobjects):
            if verb.lemma_ == 'send':  # Ignore send email or message
                for d in dobjects:
                    if re.search(r'.*\b(email|message)\b.*', d):
                        return True
            return False

        def getVerbNegationDistance(token, sentence):
            def isVerbNegatedInternal(token):
                NEG_LEX = {'not', "n't", 'never', 'no', 'none', 'neither', 'nor'}
                def _neg_idx(tok):
                    if tok.dep == spacy.symbols.neg or tok.lemma_.lower() in NEG_LEX or tok.text.lower() in NEG_LEX:
                        return tok.i
                    if re.search(r'\b(no|none|neither|never|not|n\'t)\b', tok.text.lower()):
                        return tok.i
                    return -1
                for t in token.children:
                    idx = _neg_idx(t)
                    if idx >= 0:
                        return idx
                if token.head is not token and token.head.lemma_.lower() == 'be':
                    for t in token.head.children:
                        idx = _neg_idx(t)
                        if idx >= 0:
                            return idx
                for t in token.children:
                    if t.dep_ in {'nsubj', 'nsubjpass'}:
                        idx = _neg_idx(t)
                        if idx >= 0:
                            return idx
                        for s in t.children:
                            if s.dep_ not in {'det', 'predet', 'amod', 'neg', 'nummod'}:
                                continue
                            idx = _neg_idx(s)
                            if idx >= 0:
                                return idx
                return -1
                # return any(t.dep == spacy.symbols.neg for t in token.children)

            dist = isVerbNegatedInternal(token)
            if dist >= 0:
                return dist

            # Check if verb is part of conjugated verb phrase, if so, check if any of those are negated
            conjugatedVerbs = DependencyGraphConstructor.getConjugatedVerbs(sentence, token)
            for tok in conjugatedVerbs:
                dist = isVerbNegatedInternal(tok)
                if dist >= 0:
                    return dist

            # Check if verb is xcomp, if so check if prior verb is negated? adjks
            if token.dep == spacy.symbols.xcomp:
                return getVerbNegationDistance(token.head, sentence)
            return -1

        def extractEntities(depGraph, baseNode):
            def extractEntitiesInternal(results, depGraph, baseNode):
                agent = getAgent(depGraph, baseNode)
                for _, node in depGraph.out_edges(baseNode):
                    if node[2].isEntity and node not in agent:
                        results.append(node)
                    elif node[2].isShareOrCollect:  # Extract from NEAREST annotated verb only
                        continue
                    elif node[1].pos == spacy.symbols.ADP and node[1].lemma_ in ['except when', 'except where', 'unless when', 'unless where', 'except for', 'except in', 'except under', 'unless for', 'unless in', 'unless under', 'apart from', 'aside from', 'with the exception of', 'other than', 'except to', 'unless to', 'unless as', 'except as']:
                        continue
                    extractEntitiesInternal(results, depGraph, node)
            ##########################
            entities = []
            extractEntitiesInternal(entities, depGraph, baseNode)
            return entities
        #########################

        def convertAgentToText(depGraph, agent):
            if agent is None:
                return agent
            result = []
            for a in agent:
                if type(a) == str:
                    result.append(a)
                    continue
                result.extend(depGraph.nodes[a]['lemmaList'])
            return result

        results = []
        if verbose:
            print('Found match.\n\t', sentence)

        # Start at the verbs...
        exceptions = eh.checkException(sentence)  # TODO should probably check the verb match here instead of doing it below...

        for n in depGraph:
            if n[2].isShareOrCollect:  # YZ: should we keep this condition?
                # dataObjects = [ d[1].lemma_ for d in extractDataObjects(depGraph, n) ]
                dataObjects = []
                for d in extractDataObjects(depGraph, n):
                    dataObjects.extend(depGraph.nodes[d]['lemmaList'])
                if depGraph.nodes[n].get('neg') and not dataObjects:
                    m = re.search(
                        r'\bno\s+((?:personal\s+)?(?:data|information|e-?mail)(?:\s+\w+){0,4})',
                        sentence.text,
                        re.I,
                    )
                    if m:
                        dataObjects.append(m.group(1).strip().lower())

                # entities = [ e[1].lemma_ for e in extractEntities(depGraph, n) ]
                entities = []
                for e in extractEntities(depGraph, n):
                    entities.extend(depGraph.nodes[e]['lemmaList'])

                agent = getAgent(depGraph, n)
                # Agent to text
                agent = convertAgentToText(depGraph, agent)

                if len(dataObjects) == 0 or ignoreActionObjectPair(n[1], dataObjects):  # skip <VERB, send>, <email>
                    continue

                actionLemmas = depGraph.nodes[n]['lemmaList']

                # Get related exceptions rooted under the specific share/collect verb...
                relExcepts = getReleventExceptions(n[1], exceptions)
                if verbose:
                    print(n, ('NOT', n[1].i, getVerbNegationDistance(n[1], sentence)) if depGraph.nodes[n]['neg'] else '')
                    print('\tDATA: ', dataObjects)
                    print('\tAGENT: ', agent)
                    print('\tENTITIES: ', entities)
                    # print '\tTYPE: ', ptype
                    print('\tEXCEPTIONS: ', exceptions)

                negDist = getVerbNegationDistance(n[1], sentence) if depGraph.nodes[n]['neg'] else -1
                results.append({'data_objects': dataObjects, 'entities': entities, 'agent': agent, 'action': n, 'action_lemmas': actionLemmas, 'is_negated': depGraph.nodes[n]['neg'], 'negation_distance': negDist, 'original_sentence': sentence.text, 'exceptions': relExcepts, '_share_verbs': self.tagger.shareVerbs, '_collect_verbs': self.tagger.collectVerbs})
        return results

    def mix_extract(self, paragraph):

        results = []
        doc = self.parseText(paragraph)
        for sentence in doc.sents:
            tags = self.tagger.tagSentence(sentence)

            if len(tags) <= 0:
                continue

            # Skip interrogative sentences (Hao Cui)
            if sentence[-1].lemma_ == '?':
                continue

            # Prune the tree..
            depGraph = DependencyGraphConstructor.getSimplifiedDependencyGraph(sentence, tags)
            if len(tags) <= 0 or depGraph is None:
                continue

            res = self.extractData(depGraph, sentence)
            res = PolicyTransformer.handleExceptions(res, tags)
            res = PolicyTransformer.applyPolicyTransformationRules(res, self.analyticsObj)
            for r in res:
                results.append(r)

        return results if len(results) > 0 else None


def aggregateBySentence(policies):
    results = {}
    if policies is not None:  # We can just do extend instead of append if we're not going to be verbose here...
        for actor, collect, data, orig_sentence, actionLemma in policies:
            if orig_sentence not in results:
                results[orig_sentence] = set()
            results[orig_sentence].add((actor, collect, data, actionLemma))
    return results


def prettyPrintResults(policies):
    res = aggregateBySentence(policies)
    for sen in res:
        print(sen)
        for pol in res[sen]:
            print('\t', pol)

def val(v):
    res = v if type(v) == str else v.lemma_
    return res

def valTxt(v):
    res = v if type(v) == str else v.text
    return res

def dumpTree(tok, tab=''):
    print(tab, tok.lemma_, tok.pos_, tok.dep_, tok.i, tok.ent_type_)
    for child in tok.children:
        dumpTree(child, tab + '\t')

def build_extractor(ner_model, soc_verbs_path, entity_list_path,
                    spacy_model='en_core_web_sm',
                    use_spacy_ner_entities=True,
                    also_tag_ner_entities=False,
                    data_label_types=None,
                    entity_label_types=None):
    """Wire the original MixExtractor to your three artifacts."""
    share_verbs, collect_verbs = load_soc_verbs(soc_verbs_path)
    entity_list = load_entity_list(entity_list_path)
    try:
        nlp = spacy.load(spacy_model)
    except Exception as exc:
        raise RuntimeError(
            'Failed to load spaCy model {!r}. VPVet used a custom PoliCheck model; '
            'en_core_web_sm is a reasonable substitute if you do not have that model. '
            'Original code also requires spaCy 2 APIs (span.merge, Matcher callback). '
            'Original comment: conda install spacy=2.0.18. ({})'.format(spacy_model, exc)
        )
    analytics = Analytics()
    extractor = MixExtractor(
        nlp, analytics, ner_model,
        share_verbs=share_verbs,
        collect_verbs=collect_verbs,
        entity_list=entity_list,
        use_spacy_ner_entities=use_spacy_ner_entities,
        data_label_types=data_label_types,
        entity_label_types=entity_label_types,
        also_tag_ner_entities=also_tag_ner_entities,
    )
    return extractor, analytics


def extract_cus_tuples_from_sentences(extractor, sentences, doc_id='doc'):
    """Run the exact mix_extract loop. Returns the same 5-field records as PPAudit."""
    extractor.analyticsObj.startDoc(doc_id)
    records = []
    for line in sentences:
        if not line or not str(line).strip():
            continue
        res = extractor.mix_extract(line)
        if res is None:
            continue
        res = [(val(ent), val(col), val(dat), valTxt(sen), val(actionLemma)) for ent, col, dat, sen, actionLemma in res]
        for ent, col, dat, sen, actionLemma in res:
            records.append({
                'sentence': sen,
                'entity_phrase': ent,
                'cus_or_not': col,       # 'collect' | 'not_collect'
                'dataobj_phrase': dat,
                'cus_verb': actionLemma,
            })
        prettyPrintResults(res)
    extractor.analyticsObj.endDoc()
    return records


def cus_phrse_tuple_extraction(pp_components, cus_phrase_tuples, analytics_dir,
                               ner_model, soc_verbs_path, entity_list_path,
                               spacy_model='en_core_web_sm'):
    """Original batch driver: JSON files of {hash: {'CUS': [sentences...]}}."""
    pp_components = Path(pp_components)
    cus_phrase_tuples = Path(cus_phrase_tuples)
    analytics_dir = Path(analytics_dir)
    cus_phrase_tuples.mkdir(parents=True, exist_ok=True)
    analytics_dir.mkdir(parents=True, exist_ok=True)

    mix_extractor, analytics = build_extractor(ner_model, soc_verbs_path, entity_list_path, spacy_model=spacy_model)

    for pp_component_file in pp_components.glob('*.json'):
        pp_url_hash = pp_component_file.stem
        if (analytics_dir / '{}.pickle'.format(pp_url_hash)).is_file() and \
           (cus_phrase_tuples / '{}.json'.format(pp_url_hash)).is_file():
            print('[ ] {}'.format(pp_url_hash))
            continue

        print('[+] {}'.format(pp_url_hash))
        with open(pp_component_file, 'r') as rf:
            file_content = json.loads(rf.read())
        cus_sentences_candidates = file_content[pp_url_hash]['CUS']

        records = extract_cus_tuples_from_sentences(mix_extractor, cus_sentences_candidates, doc_id=pp_url_hash)
        payload = {pp_url_hash: records}
        with open(cus_phrase_tuples / '{}.json'.format(pp_url_hash), 'w') as wf:
            wf.write(json.dumps(payload, indent=4))
        pickle.dump(analytics.dataStore[pp_url_hash],
                    open(analytics_dir / '{}.pickle'.format(pp_url_hash), 'wb'))


def _parse_args(argv):
    import argparse
    p = argparse.ArgumentParser(description='PPAudit CUS-tuple extractor with your NER / SOC verbs / entities')
    p.add_argument('--ner-model', required=True, help='HF token-classification model dir or hub id')
    p.add_argument('--soc-verbs', required=True, help='JSON {share:[], collect:[]} or text list')
    p.add_argument('--entities', required=True, help='JSON list or one-entity-per-line file')
    p.add_argument('--spacy-model', default='en_core_web_sm',
                   help='spaCy model. VPVet used other_data/NlpFinalModel_Policheck')
    p.add_argument('--sentences', help='Plain-text file, one candidate CUS sentence per line')
    p.add_argument('--pp-components', help='Directory of PPAudit-style {hash:{CUS:[...]}} JSON files')
    p.add_argument('--out', default='cus_phrase_tuples.json', help='Output JSON path (sentence-file mode)')
    p.add_argument('--out-dir', default='cus_phrase_tuples', help='Output dir (batch mode)')
    p.add_argument('--analytics-dir', default='analytics')
    p.add_argument('--also-tag-ner-entities', action='store_true',
                   help='If your NER also emits ENTITY/ORG/PERSON spans, tag those as ENTITY')
    p.add_argument('--no-spacy-entities', action='store_true',
                   help='Do not treat spaCy PERSON/ORG as entities (gazetteer + NER only)')
    return p.parse_args(argv)


if __name__ == '__main__':
    args = _parse_args(sys.argv[1:])
    start_time = time.time()
    if args.pp_components:
        cus_phrse_tuple_extraction(
            args.pp_components, args.out_dir, args.analytics_dir,
            args.ner_model, args.soc_verbs, args.entities,
            spacy_model=args.spacy_model,
        )
    elif args.sentences:
        extractor, _ = build_extractor(
            args.ner_model, args.soc_verbs, args.entities,
            spacy_model=args.spacy_model,
            use_spacy_ner_entities=not args.no_spacy_entities,
            also_tag_ner_entities=args.also_tag_ner_entities,
        )
        sentences = Path(args.sentences).read_text(encoding='utf-8').splitlines()
        records = extract_cus_tuples_from_sentences(extractor, sentences, doc_id=Path(args.sentences).stem)
        Path(args.out).write_text(json.dumps(records, indent=2), encoding='utf-8')
        print('[LOG]: wrote {} tuples to {}'.format(len(records), args.out))
    else:
        sys.exit('Provide --sentences FILE or --pp-components DIR')
    end_time = time.time()
    print('[LOG]: run cus phrase tuple extraction cost {} s'.format(end_time - start_time))