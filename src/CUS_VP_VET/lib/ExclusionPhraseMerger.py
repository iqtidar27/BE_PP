import spacy
from spacy.matcher import Matcher
from .spacy_compat import merge_span, matcher_add, merge_spans_right_to_left


def mergeExcludePhrases(doc, vocab):
    def getLemma(ephrase):
        return ' '.join([ tok.lemma_ for tok in ephrase if tok.lemma_ != ':' ])

    results = []
    ##########################################
    def callback(matcher, doc, i, matches):
        for match_id, start, end in matches:
            span = doc[start:end]
            results.append(span)
    ##########################################
    matcher = Matcher(vocab)
    matcher_add(matcher,1, callback,
        [   {spacy.attrs.LOWER: "other"},
            {spacy.attrs.LOWER: "than"},
        ])
    matcher_add(matcher,2,callback,
        [   {spacy.attrs.LOWER: "aside"},
            {spacy.attrs.LOWER: "from"},
        ])
    matcher_add(matcher,3,callback,
        [   {spacy.attrs.LOWER: "apart"},
            {spacy.attrs.LOWER: "from"},
        ])
    matcher_add(matcher,4,callback,
        [   {spacy.attrs.LOWER: "with"},
            {spacy.attrs.LOWER: "the"},
            {spacy.attrs.LOWER: "exception"},
            {spacy.attrs.LOWER: "of"}
        ])
    matcher_add(matcher,5,callback,
        [   {spacy.attrs.LOWER: "except"},
            {spacy.attrs.ORTH: ":", 'OP' : '?'},
            {spacy.attrs.LOWER: "for"},
        ])
    matcher_add(matcher,6,callback,
        [   {spacy.attrs.LOWER: "except"},
            {spacy.attrs.ORTH: ":", 'OP' : '?'},
            {spacy.attrs.LOWER: "when"},
        ])
    matcher_add(matcher,7,callback,
        [   {spacy.attrs.LOWER: "except"},
            {spacy.attrs.ORTH: ":", 'OP' : '?'},
            {spacy.attrs.LOWER: "in"},
        ])
    matcher_add(matcher,8,callback,
        [   {spacy.attrs.LOWER: "except"},
            {spacy.attrs.ORTH: ":", 'OP' : '?'},
            {spacy.attrs.LOWER: "to"},
        ])
    matcher_add(matcher,9,callback,
        [   {spacy.attrs.LOWER: "except"},
            {spacy.attrs.ORTH: ":", 'OP' : '?'},
            {spacy.attrs.LOWER: "as"},
        ])
    matcher_add(matcher,10,callback,
        [   {spacy.attrs.LOWER: "except"},
            {spacy.attrs.ORTH: ":", 'OP' : '?'},
            {spacy.attrs.LOWER: "under"},
        ])
    matcher_add(matcher,11,callback,
        [   {spacy.attrs.LOWER: "except"},
            {spacy.attrs.ORTH: ":", 'OP' : '?'},
            {spacy.attrs.LOWER: "where"},
        ])
    matcher_add(matcher,12,callback,
        [   {spacy.attrs.LOWER: "unless"},
            {spacy.attrs.ORTH: ":", 'OP' : '?'},
            {spacy.attrs.LOWER: "when"},
        ])
    matcher_add(matcher,13,callback,
        [   {spacy.attrs.LOWER: "unless"},
            {spacy.attrs.ORTH: ":", 'OP' : '?'},
            {spacy.attrs.LOWER: "in"},
        ])
    matcher_add(matcher,14,callback,
        [   {spacy.attrs.LOWER: "unless"},
            {spacy.attrs.ORTH: ":", 'OP' : '?'},
            {spacy.attrs.LOWER: "to"},
        ])
    matcher_add(matcher,15,callback,
        [   {spacy.attrs.LOWER: "unless"},
            {spacy.attrs.ORTH: ":", 'OP' : '?'},
            {spacy.attrs.LOWER: "as"},
        ])
    matcher_add(matcher,16,callback,
        [   {spacy.attrs.LOWER: "unless"},
            {spacy.attrs.ORTH: ":", 'OP' : '?'},
            {spacy.attrs.LOWER: "under"},
        ])
    matcher_add(matcher,17,callback,
        [   {spacy.attrs.LOWER: "unless"},
            {spacy.attrs.ORTH: ":", 'OP' : '?'},
            {spacy.attrs.LOWER: "where"},
        ])
    matcher_add(matcher,18,callback,
        [   {spacy.attrs.LOWER: "unless"},
            {spacy.attrs.ORTH: ":", 'OP' : '?'},
            {spacy.attrs.LOWER: "for"},
        ])
    matcher_add(matcher,19,callback,
        [   {spacy.attrs.LOWER: "outside"},
            {spacy.attrs.LOWER: "of"},
        ])



    matcher(doc)
    merge_spans_right_to_left(doc, results, lambda ephrase: dict(lemma=getLemma(ephrase)))