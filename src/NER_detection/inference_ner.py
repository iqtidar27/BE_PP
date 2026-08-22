#!/usr/bin/env python3
"""
Inference script for Data Object NER model (PrivBERT fine-tuned).

Usage example:
    from infer_ner import DataObjectNER

    ner = DataObjectNER(model_path="models/dataobj_ner_v5_focal1.5_len256_unf6")
    result = ner.predict("We collect your IP address and email address.")
    print(result)
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


class DataObjectNER:
    """
    Named Entity Recognition model for extracting data objects
    from privacy policy text.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[str] = None,
        max_length: int = 256,
    ):
        """
        Initialize the NER model.

        Args:
            model_path: Path to the fine-tuned model directory
                        (must contain config.json, model weights, and tokenizer files).
            device: Device to run inference on ("cuda", "cpu", or None for auto).
            max_length: Maximum sequence length for tokenization.
        """
        self.model_path = Path(model_path)
        self.max_length = max_length

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

        # Device selection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

        # Label mappings
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id

        print(f"[DataObjectNER] Model loaded from: {self.model_path}")
        print(f"[DataObjectNER] Running on device: {self.device}")
        print(f"[DataObjectNER] Labels: {list(self.id2label.values())}")

    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize a single sentence."""
        return self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            padding=False,
        )

    def _align_predictions(
        self,
        text: str,
        offset_mapping: List[tuple],
        predicted_ids: List[int],
    ) -> List[Dict[str, Any]]:
        """
        Align subword predictions back to character spans in the original text
        and extract entity spans.
        """
        entities = []
        current_entity = None

        for idx, (start, end) in enumerate(offset_mapping):
            # Skip special tokens ([CLS], [SEP], padding)
            if start == end:
                continue

            label = self.id2label[predicted_ids[idx]]

            if label == "B":
                # Close previous entity if exists
                if current_entity is not None:
                    entities.append(current_entity)

                current_entity = {
                    "start": start,
                    "end": end,
                    "label": "DATA_OBJECT",
                    "text": text[start:end],
                }

            elif label == "I":
                if current_entity is not None:
                    # Continue the current entity
                    current_entity["end"] = end
                    current_entity["text"] = text[current_entity["start"]:end]
                else:
                    # Orphan I → treat as beginning of a new entity
                    current_entity = {
                        "start": start,
                        "end": end,
                        "label": "DATA_OBJECT",
                        "text": text[start:end],
                    }

            else:  # "O"
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None

        # Append last entity if sentence ends with an entity
        if current_entity is not None:
            entities.append(current_entity)

        return entities

    @torch.no_grad()
    def predict(self, sentence: str) -> Dict[str, Any]:
        """
        Run NER prediction on a single sentence.

        Args:
            sentence: Input text (string).

        Returns:
            Dictionary containing:
                - text: original sentence
                - entities: list of detected data objects
                  each with keys: start, end, label, text
                - tokens: list of (token, predicted_label) for debugging
        """
        if not isinstance(sentence, str) or not sentence.strip():
            return {
                "text": sentence,
                "entities": [],
                "tokens": [],
            }

        # Tokenize
        encoded = self._tokenize(sentence)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        offset_mapping = encoded["offset_mapping"][0].tolist()

        # Forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()

        # Align predictions to original text and extract entities
        entities = self._align_predictions(sentence, offset_mapping, predictions)

        # Optional: token-level view (useful for debugging)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        token_preds = [
            {
                "token": tok,
                "label": self.id2label[pred_id],
            }
            for tok, pred_id in zip(tokens, predictions)
            if tok not in self.tokenizer.all_special_tokens
        ]

        return {
            "text": sentence,
            "entities": entities,
            "tokens": token_preds,
        }

    def predict_batch(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """
        Run prediction on a list of sentences.
        (Simple sequential implementation for now.)
        """
        return [self.predict(sent) for sent in sentences]


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    # Path to your best model
    MODEL_DIR = "/home/ayon/git/BE_PP/src/NER_detection/saved_model/dataobj_ner_v5_focal1.5_len256_unf6"

    ner = DataObjectNER(model_path=MODEL_DIR)

    test_sentences = [
        "We collect your IP address and email address when you visit our website.",
        "The company may share your name, phone number, and device identifiers with third parties.",
        "No personal information is collected through this service.",
        "We use cookies to store preferences and settings.",
    ]

    print("\n" + "=" * 70)
    print("Inference Results")
    print("=" * 70)

    for sent in test_sentences:
        result = ner.predict(sent)

        print(f"\nText: {result['text']}")
        if result["entities"]:
            print("Detected data objects:")
            for ent in result["entities"]:
                print(f"  • '{ent['text']}'  (span: {ent['start']}-{ent['end']})")
        else:
            print("  (no data objects detected)")