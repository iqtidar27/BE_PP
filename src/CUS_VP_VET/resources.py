# ============================================================
# Entity lists + SOC verbs for CUS / VPVet-style extraction
# ============================================================

# ------------------------------------------------------------
# 1. First-party + common generics
# ------------------------------------------------------------
FIRST_PARTY = {
    "we", "us", "our", "i", "me", "you", "your",
    "the company", "the app", "the application", "the service",
    "the extension", "the platform", "the website", "the site",
    "our company", "our app", "our service", "our extension",
    "the developer", "the developers",
    "the service provider",
    "the data controller",
    "the data processor",
    "our service provider",
    "the platform provider",
    "the add-on",
    "the addon",
    "this extension",
    "this service",
    "our servers",
}

# Extra first-party surfaces from src/CUS/cus_extractor.py
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

# Noun lemmas treated as DATA if the NER span is missing (Stage-B fallback).
DATA_HINTS = {
    "token", "information", "data", "datum",
    "identifier", "email", "address", "cookie", "cookies", "log",
    "report", "crash", "traffic",
    "authkey", "authkeys", "teamid", "teamids",
    "password", "credential", "credentials",
    "ip", "ips",
}


def is_blocked_entity(text: str) -> bool:
    n = " ".join(str(text).lower().split())
    if n in ENTITY_BLOCKLIST or n in ORG_NOISE:
        return True
    if any(s in n for s in ORG_REJECT_SUBSTR):
        return True
    if len(n.split()) > 6:
        return True
    return False

# ------------------------------------------------------------
# 2. Third-party entities
# ------------------------------------------------------------
THIRD_PARTY = {
    "ISP", "your ISP", "Internet Service Provider",
    "carrier", "carriers", "midway carriers",
    "these services", "those services", "their services",
    "the services", "third-party partners", "our third-party partners",
    "third-party partner",
    "PostHog", "AppsFlyer", "Firebase Crashlytics", "Crashlytics",
    "New Relic", "Sentry", "MixPanel", "Mixpanel",
    "Microsoft Clarity", "Vercel Analytics",
    "Google Analytics", "Google Tag Manager",
    "Meta Pixel", "Facebook Pixel",
    "Paypal", "PayPal", "Stripe", "Chargebee", "Lemon Squeezy",
    "Razorpay", "Dodo Payments", "Polar",
    "Google Cloud", "Firestore", "Google Cloud SQL Postgres",
    "Cloudflare", "Cloudflare R2", "BigQuery",
    "Supabase", "Supabase Auth", "Railway", "Upstash Redis",
    "Vercel", "Alibaba Cloud", "Firebase",
    "OpenAI", "Anthropic", "Anthropic AI", "Claude.ai", "Claude",
    "Google Vertex AI", "Gemini API", "Gemini", "Fireworks AI",
    "ElevenLabs", "Deepgram", "Recall.ai", "fal.ai",
    "Qwen 3-ASR", "Qwen 3-ASR Flash", "XAI", "XAI speech-to-text",
    "Replicate WhisperX", "Cohere Transcribe", "Grok Whisper",
    "Moonshot Kimi", "China Kimi", "Hugging Face",
    "Metalayers server", "XRO AI Services",
    "Google Ads", "Google Adsense", "Google AdSense",
    "LinkedIn Ads", "Meta", "Instagram", "Adapty",
    "Flutter", "FCM", "Google Calendar API", "Microsoft Graph",
    "Supadata", "Crisp", "Cygnet", "AC3", "Overbridge", "Siloq",
    "Crossref", "Semantic Scholar", "CamaraUX servers",
    "Wizper", "EachLabs",
    "Google", "Amazon", "Apple", "YouTube", "Gmail", "LinkedIn", "GitHub",
    "JobLog", "MongoDB Atlas", "MongoDB",
    "F-Secure", "F-Secure Security Cloud",
    "Better Auth", "Intercom", "Segment", "Twilio",
    "these services",
    "advertiser", "advertisers", "these advertisers",
    "third party", "third parties", "third-party",
}

def first_party_aliases(extension_name: str) -> set:
    if not extension_name:
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


def all_entities(extension_name: str = None):
    ents = set(FIRST_PARTY) | set(FIRST_PARTY_ALWAYS) | {e.lower() for e in THIRD_PARTY}
    ents |= first_party_aliases(extension_name)
    ents = {e for e in ents if not is_blocked_entity(e)}
    return sorted(ents, key=lambda s: (-len(s.split()), -len(s), s))


# ------------------------------------------------------------
# SOC verbs
# ------------------------------------------------------------
COLLECT_VERBS = {
    "collect", "check", "know", "use", "obtain",
    "access", "receive", "gather", "store", "save",
    "keep", "add", "combine", "record", "retain",
    "get", "perform", "analyze", "recognize",
    "process", "track", "log", "remember", "relate",
    "create", "ask", "conduct", "monitor", "request",
    "link", "associate", "solicit", "read", "preserve",
    "contain", "include",
    "capture", "acquire", "retrieve", "extract",
    "submit", "upload", "fetch", "sync", "restore",
    "aggregate", "hold", "maintain", "synchronize",
    "set", "place",
}

SHARE_VERBS = {
    "share", "sell", "provide", "trade", "transfer",
    "give", "distribute", "disclose", "send", "rent",
    "exchange", "report", "transmit",
    "pass", "express", "supply", "display", "publish",
    "release", "deliver", "download", "lease",
    "view", "show", "forward", "hold", "swap",
    "tell",
    "allow", "enable", "offer", "communicate",
    "grant", "permit",
    "sent",
}

# Merge these into a single VERB token before the PPAudit graph walk.
# Do NOT merge "share with" / "disclose to" / ... — the preposition must stay
# so recipient entities remain attached (that is how PPAudit finds them).
MULTI_WORD_COLLECT = [
    "make use of",
    "gain access to",
    "obtain access to",
    "take possession of",
]

MULTI_WORD_SHARE = [
    "make available",
    "pass on",
    "hand over",
    "give access",
    "provide access",
    "grant access",
]

# Already covered by single-word SHARE_VERBS; kept only as documentation.
MULTI_WORD_SHARE_PREP = [
    "share with",
    "disclose to",
    "transfer to",
    "send to",
    "transmit to",
]

SOC_VERB_BLOCKLIST = {
    "ask", "request", "include", "includes", "including",
    "retain", "retained", "link", "linked",
    "show", "shows", "showing",
    "get", "gets",
    "object",
    "check", "checks", "checking",
    "relate", "related", "relates",
    "enable", "enables", "enabling",
    "create", "creates", "created", "creating",
    "read", "reads", "reading",
    "associate", "associated", "associates", "associating",
    "need", "needed", "needs",
    "require", "required", "requires", "requiered",
}

DEFAULT_NER_MODEL = (
    "src/NER_detection/saved_model/"
    "dataobj_ner_v5_focal1.5_len256_unf6_dataset_version_v6_augmented_final"
)
