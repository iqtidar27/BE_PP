# ============================================================
# Entity lists for CUS / VPVet-style extraction
# ============================================================

# ------------------------------------------------------------
# 1. First-party + Common generics
# ------------------------------------------------------------
FIRST_PARTY = {
    # Pronouns
    "we", "us", "our", "i", "me", "you", "your",

    # Common first-party generics
    "the company", "the app", "the application", "the service",
    "the extension", "the platform", "the website", "the site",
    "our company", "our app", "our service", "our extension",
    "the developer", "the developers",

    # longer forms (important)
    "the service provider",
    "the data controller",
    "the data processor",
    "our service provider",
    "the platform provider",
    
}

# ------------------------------------------------------------
# 2. Third-party entities
# ------------------------------------------------------------
THIRD_PARTY = {
    # ---------- Analytics / Monitoring / Crash reporting ----------
    "PostHog", "AppsFlyer", "Firebase Crashlytics", "Crashlytics",
    "New Relic", "Sentry", "MixPanel", "Mixpanel",          # both spellings
    "Microsoft Clarity", "Vercel Analytics",
    "Google Analytics", "Google Tag Manager",
    "Meta Pixel", "Facebook Pixel",

    # ---------- Payment ----------
    "Paypal", "PayPal", "Stripe", "Chargebee", "Lemon Squeezy",
    "Razorpay", "Dodo Payments", "Polar",

    # ---------- Cloud / Infrastructure ----------
    "Google Cloud", "Firestore", "Google Cloud SQL Postgres",
    "Cloudflare", "Cloudflare R2", "BigQuery",
    "Supabase", "Supabase Auth", "Railway", "Upstash Redis",
    "Vercel", "Alibaba Cloud", "Firebase",

    # ---------- AI / LLM / Speech ----------
    "OpenAI", "Anthropic", "Anthropic AI", "Claude.ai", "Claude",
    "Google Vertex AI", "Gemini API", "Gemini", "Fireworks AI",
    "ElevenLabs", "Deepgram", "Recall.ai", "fal.ai",
    "Qwen 3-ASR", "Qwen 3-ASR Flash", "XAI", "XAI speech-to-text",
    "Replicate WhisperX", "Cohere Transcribe", "Grok Whisper",
    "Moonshot Kimi", "China Kimi", "Hugging Face",
    "Metalayers server", "XRO AI Services",

    # ---------- Ads / Marketing ----------
    "Google Ads", "Google Adsense", "Google AdSense",
    "LinkedIn Ads", "Meta", "Instagram", "Adapty",

    # ---------- Other services (from handwritten list) ----------
    "Flutter", "FCM", "Google Calendar API", "Microsoft Graph",
    "Supadata", "Crisp", "Cygnet", "AC3", "Overbridge", "Siloq",
    "Crossref", "Semantic Scholar", "CamaraUX servers",
    "Wizper", "EachLabs",

    # ---------- Added from VPVet-style detection (high value) ----------
    "Google",
    "Amazon",
    "Apple",
    "YouTube",
    "Gmail",
    "LinkedIn",
    "GitHub",
}

# Optional: combine both for convenience
ALL_ENTITIES = FIRST_PARTY | {e.lower() for e in THIRD_PARTY}