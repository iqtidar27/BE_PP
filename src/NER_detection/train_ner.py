#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from seqeval.metrics import precision_score, recall_score, f1_score

# ====================== CONFIG ======================
Dataset_version = 'v6_augmented_final'
DATA = Path("NER_dataset/bio_augmented_rare50_balanced_v6.jsonl")
MODEL_NAME = "mukund/privbert"

MAX_LENGTH = 256                 # sweet spot for 4070 12GB
BATCH_SIZE = 32

EARLY_STOPPING_PATIENCE = 30
EARLY_STOPPING_THRESHOLD = 0.001

FREEZE_LAYERS = False
NUM_UNFROZEN_LAYERS = 6          # increased from 4

# Loss strategy: choose ONE
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 1.5                # milder than 2.0
USE_CLASS_WEIGHTS = False        # ← turn OFF when using focal

#LR
LEARNING_RATE = 1e-5

EPOCHS = 200
# ====================================================

label_names = ["O", "B", "I"]
label2id = {l: i for i, l in enumerate(label_names)}
id2label = {i: l for l, i in label2id.items()}

OUT = Path(
    f"saved_model/dataobj_ner_v5_focal{FOCAL_GAMMA}"
    f"_len{MAX_LENGTH}_unf{NUM_UNFROZEN_LAYERS}_dataset_version_{Dataset_version}"
)
print(f"Output directory: {OUT}")

# -------------------- Load & split --------------------
rows = [json.loads(l) for l in DATA.open(encoding="utf-8")]
print(f"Total examples loaded: {len(rows)}")

def get_stratify_key(ner_tags):
    has_b = any(t == "B" for t in ner_tags)
    has_i = any(t == "I" for t in ner_tags)
    return f"{int(has_b)}_{int(has_i)}"

stratify_keys = [get_stratify_key(r["ner_tags"]) for r in rows]

train_rows, temp_rows, _, temp_keys = train_test_split(
    rows, stratify_keys, test_size=0.30, random_state=12, stratify=stratify_keys
)
val_rows, test_rows = train_test_split(
    temp_rows, test_size=0.50, random_state=12, stratify=temp_keys
)

print(f"Train={len(train_rows)}  Val={len(val_rows)}  Test={len(test_rows)}")

def count_labels(recs):
    c = Counter()
    for r in recs:
        c.update(r["ner_tags"])
    return dict(c)

print("Label distribution (Train):", count_labels(train_rows))

# -------------------- Class weights (optional) --------------------
label_counts = Counter()
for r in train_rows:
    label_counts.update(r["ner_tags"])

total_tokens = sum(label_counts.values())
class_weights = []
for lab in label_names:
    cnt = max(label_counts.get(lab, 1), 1)
    # milder weight: sqrt of inverse frequency
    w = (total_tokens / (len(label_names) * cnt)) ** 0.5
    class_weights.append(w)

class_weights = torch.tensor(class_weights, dtype=torch.float)
print("Class weights (O, B, I):", [round(w, 3) for w in class_weights.tolist()])

# -------------------- Tokenization --------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(examples):
    tok = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    labels = []
    for i, tags in enumerate(examples["ner_tags"]):
        word_ids = tok.word_ids(batch_index=i)
        lab, prev = [], None
        for wid in word_ids:
            if wid is None:
                lab.append(-100)
            elif wid != prev:
                lab.append(label2id.get(tags[wid], 0))
            else:
                t = tags[wid]
                lab.append(label2id["I"] if t == "B" else label2id.get(t, 0))
            prev = wid
        labels.append(lab)
    tok["labels"] = labels
    return tok

def make(recs):
    return Dataset.from_dict({
        "tokens": [r["tokens"] for r in recs],
        "ner_tags": [r["ner_tags"] for r in recs],
    }).map(tokenize, batched=True, remove_columns=["tokens", "ner_tags"])

ds_train = make(train_rows)
ds_val   = make(val_rows)
ds_test  = make(test_rows)

# -------------------- Model --------------------
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
)

if FREEZE_LAYERS:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    encoder_layers = None
    if hasattr(model, "roberta"):
        encoder_layers = model.roberta.encoder.layer
    elif hasattr(model, "bert"):
        encoder_layers = model.bert.encoder.layer

    if encoder_layers is not None and NUM_UNFROZEN_LAYERS > 0:
        for layer in encoder_layers[-NUM_UNFROZEN_LAYERS:]:
            for param in layer.parameters():
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# -------------------- Focal Loss --------------------
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=1.5, ignore_index=-100):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        ce = F.cross_entropy(
            logits, labels,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none"
        )
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()

class FocalTrainer(Trainer):
    def __init__(self, class_weights=None, gamma=1.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        device = next(self.model.parameters()).device
        w = class_weights.to(device) if class_weights is not None else None
        self.loss_fct = FocalLoss(weight=w, gamma=gamma)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.loss_fct(
            outputs.logits.view(-1, model.config.num_labels),
            labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss

# -------------------- Metrics --------------------
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=2)
    true_p, true_l = [], []
    for pred, lab in zip(preds, p.label_ids):
        pr, lr = [], []
        for a, b in zip(pred, lab):
            if b == -100:
                continue
            pr.append(id2label[a])
            lr.append(id2label[b])
        true_p.append(pr)
        true_l.append(lr)
    return {
        "precision": precision_score(true_l, true_p),
        "recall":    recall_score(true_l, true_p),
        "f1":        f1_score(true_l, true_p),
    }

def detailed_report(trainer, dataset, split_name=""):
    predictions = trainer.predict(dataset)
    preds = np.argmax(predictions.predictions, axis=2)
    labels = predictions.label_ids

    y_true, y_pred = [], []
    for pred_seq, lab_seq in zip(preds, labels):
        for p, l in zip(pred_seq, lab_seq):
            if l == -100:
                continue
            y_true.append(id2label[l])
            y_pred.append(id2label[p])

    print(f"\n{'='*60}")
    print(f"  Detailed Token-level Report — {split_name}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, labels=label_names, digits=4, zero_division=0))
    print(f"Overall Accuracy: {accuracy_score(y_true, y_pred):.4f}")

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_names, zero_division=0
    )
    print("\nPer-class summary:")
    for i, lab in enumerate(label_names):
        print(f"  {lab}:  P={prec[i]:.4f}  R={rec[i]:.4f}  F1={f1[i]:.4f}  Support={support[i]}")

# -------------------- Training --------------------
args = TrainingArguments(
    output_dir=str(OUT / "checkpoints"),
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.05,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=50,
    report_to="none",
    dataloader_pin_memory=False,
    save_total_limit=1,
    fp16=True,
)

weight_for_loss = class_weights if (USE_FOCAL_LOSS and USE_CLASS_WEIGHTS) else None

trainer = FocalTrainer(
    class_weights=weight_for_loss,
    gamma=FOCAL_GAMMA,
    model=model,
    args=args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    processing_class=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
        )
    ],
)

trainer.train()

# -------------------- Final Evaluation --------------------
print("\n" + "="*60)
print("FINAL EVALUATION WITH BEST VALIDATION MODEL")
print("="*60)

print("\n=== VAL ===")
print(trainer.evaluate(ds_val))
detailed_report(trainer, ds_val, "VALIDATION")

print("\n=== TEST ===")
print(trainer.evaluate(ds_test))
detailed_report(trainer, ds_test, "TEST")

OUT.mkdir(parents=True, exist_ok=True)
trainer.save_model(str(OUT))
tokenizer.save_pretrained(str(OUT))
print(f"\nBest model saved → {OUT}")