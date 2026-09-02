# ============================================================
# SOC Verbs for CUS Tuple Extraction
# ============================================================
# Source legend:
#   [VPVet]     = original verbs from PPAudit / VPVet
#   [Mined]     = single-word verbs added from our dataset mining
#   [Multi]     = selective multi-word phrases we are adding
# ============================================================

# ------------------------------------------------------------
# 1. Single-word COLLECT verbs
# ------------------------------------------------------------
COLLECT_VERBS = {
    # --- Original VPVet ---
    "collect", "check", "know", "use", "obtain",
    "access", "receive", "gather", "store", "save",
    "keep", "add", "combine", "record", "retain",
    "get", "perform", "analyze", "recognize",
    "process", "track", "log", "remember", "relate",
    "create", "ask", "conduct", "monitor", "request",
    "link", "associate", "solicit", "read", "preserve",
    "contain", "include",

    # --- Added from our mining [Mined] ---
    "capture", "acquire", "retrieve", "extract",
    "submit", "upload", "fetch", "sync", "restore",
    "aggregate", "hold", "maintain", "synchronize",
    # add to COLLECT_VERBS
    "set",
    "place",
    "log",
    "record",
    "track", "access", "receive", "record", "store", "collect", "process", "log", "set", "place"
}

# ------------------------------------------------------------
# 2. Single-word SHARE verbs
# ------------------------------------------------------------
SHARE_VERBS = {
    # --- Original VPVet ---
    "share", "sell", "provide", "trade", "transfer",
    "give", "distribute", "disclose", "send", "rent",
    "exchange", "report", "transmit",
    "pass", "express", "supply", "display", "publish",
    "release", "deliver", "download", "lease",
    "view", "show", "forward", "hold", "swap",
    "tell",

    # --- Added from our mining [Mined] ---
    "allow", "enable", "offer", "communicate",
    "grant", "permit",

    # add to SHARE_VERBS
    "send",
    "sent",   # only if you store surface forms; otherwise lemma "send" is enough

    # ensure present
    "disclose", "exchange", "transmit", "send", "transfer", "sell", "share", "provide","enable",
    "permit",
}

# ------------------------------------------------------------
# 3. Multi-word phrases (very selective)
# ------------------------------------------------------------
# These are matched with spaCy PhraseMatcher (or equivalent)
# and treated as a single SOC verb unit.

MULTI_WORD_COLLECT = [
    # [Multi]
    "make use of",
    "gain access to",
    "obtain access to",
    "take possession of",
]

MULTI_WORD_SHARE = [
    # [Multi]
    "make available",
    "pass on",
    "hand over",
    "give access",
    "provide access",
    "grant access",
    "share with",
    "disclose to",
    "transfer to",
    "send to",
    "transmit to",
]

# soc_verbs.py
SOC_VERB_BLOCKLIST = {
    "ask", "request", "include", "includes", "including",
    "retain", "retained", "link", "linked",
    "show", "shows", "showing",
    "get", "gets",  # rights: "get a copy"
    "object",  # "object to processing"
    "check", "checks", "checking",
    "relate", "related", "relates",
    "enable", "enables", "enabling",
    "create", "creates", "created", "creating",
    "read", "reads", "reading",
    "check", "checks", "checking",
    "relate", "related", "relates",
    "enable", "enables", "enabling",
}