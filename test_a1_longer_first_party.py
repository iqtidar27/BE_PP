#!/usr/bin/env python3
#!/usr/bin/env python3
"""A1 test: longer first-party spans + regression check."""

from pathlib import Path

from CUS.cus_extractor import CUSExtractor

REPO_ROOT = Path(__file__).resolve().parent
NER_MODEL_PATH = str(
    REPO_ROOT
    / "src"
    / "NER_detection"
    / "saved_model"
    / "dataobj_ner_v5_focal1.5_len256_unf6"
)

NER_MODEL_PATH = (
    "src/NER_detection/saved_model/dataobj_ner_v5_focal1.5_len256_unf6"
)

# --- Focused A1 cases (expect improvement) ---
A1_CASES = [
    "The service provider may access your account information.",
    "The service provider may access your account information and preferences.",
    "Our service provider processes payment information on our behalf.",
    "The data controller collects personal information from users.",
]

# --- Regression: must not break ---
REGRESSION = [
    "We collect your IP address and email address when you visit our website.",
    "The company may share your name, phone number, and device identifiers with third parties.",
    "Google Analytics collects anonymized usage data from our users.",
    "Stripe processes your payment information on our behalf.",
    "The extension does not transmit any browsing history to external servers.",
    "PostHog and Mixpanel collect analytics data from users.",
    "No personal information is collected through this service.",
    "We do not sell or share your personal information with advertisers.",
    "The app does not send any personal data to third parties.",
    "Your email address will not be sold to third parties.",
]

def show(title, sentences, extractor):
    print("=" * 80)
    print(title)
    print("=" * 80)
    for i, s in enumerate(sentences, 1):
        print(f"\n[{i}] {s}")
        print("-" * 80)
        # Entity view via public extract + also raw entities if needed
        doc = extractor.nlp(s)
        ents = extractor.entity_recognizer.find_entities(doc)
        print("  Entities:", [(e.text, e.entity_type) for e in ents])
        tuples = extractor.extract(s)
        if not tuples:
            print("  Tuples  : (none)")
        else:
            for t in tuples:
                print(
                    f"  Tuple   : {t.entity} | {t.action} | {t.data_object} | neg={t.negated}"
                )

def main():
    extractor = CUSExtractor(ner_model_path=NER_MODEL_PATH)
    show("A1 FOCUS (longer first-party)", A1_CASES, extractor)
    show("REGRESSION (must stay healthy)", REGRESSION, extractor)

if __name__ == "__main__":
    main()