"""spaCy 2 / 3 shims for PPAudit phrase mergers."""


def merge_span(span, lemma=None, pos=None, dep=None, ent_type=None):
    if span is None or len(span) == 0:
        return
    if hasattr(span, 'merge'):
        kwargs = {}
        if lemma is not None:
            kwargs['lemma'] = lemma
        if pos is not None:
            kwargs['pos'] = pos
        if dep is not None:
            kwargs['dep'] = dep
        if ent_type is not None:
            kwargs['ent_type'] = ent_type
        span.merge(**kwargs)
        return

    def _as_str(val, default=''):
        if val is None:
            return default
        if isinstance(val, str):
            return val
        try:
            return span.doc.vocab.strings[val]
        except Exception:
            return default

    attrs = {}
    if lemma is not None:
        attrs['LEMMA'] = lemma
    if pos is not None:
        attrs['POS'] = _as_str(pos, 'NOUN')
    if dep is not None:
        attrs['DEP'] = _as_str(dep, 'dep')
    if ent_type is not None:
        attrs['ENT_TYPE'] = ent_type if isinstance(ent_type, str) else _as_str(ent_type, '')
    with span.doc.retokenize() as retok:
        retok.merge(span, attrs=attrs)


def live_span(doc, start_char, end_char):
    if end_char <= start_char:
        return None
    for mode in ('strict', 'contract', 'expand'):
        try:
            span = doc.char_span(start_char, end_char, alignment_mode=mode)
        except Exception:
            span = None
        if span is not None and len(span) > 0:
            return span
    return None


def merge_spans_right_to_left(doc, spans, attr_builder):
    """Merge many spans collected before any retokenize.

    spaCy 3 invalidates token indexes on older Span objects after a merge.
    Snapshot character offsets, merge from right to left.
    attr_builder(span) -> kwargs for merge_span.
    """
    jobs = []
    for sp in spans:
        if sp is None:
            continue
        try:
            if len(sp) == 0:
                continue
            jobs.append((int(sp.start_char), int(sp.end_char)))
        except Exception:
            continue
    seen = set()
    uniq = []
    for job in jobs:
        if job in seen:
            continue
        seen.add(job)
        uniq.append(job)
    uniq.sort(key=lambda x: (x[0], x[1]), reverse=True)
    for start_char, end_char in uniq:
        span = live_span(doc, start_char, end_char)
        if span is None:
            continue
        try:
            kwargs = attr_builder(span) or {}
        except Exception:
            continue
        try:
            merge_span(span, **kwargs)
        except Exception:
            continue


def matcher_add(matcher, key, callback, *patterns):
    """spaCy 2: add(id, callback, pat)  |  spaCy 3: add(name, [pat], on_match=callback)."""
    name = str(key)
    try:
        matcher.add(name, list(patterns), on_match=callback)
    except TypeError:
        matcher.add(key, callback, *patterns)