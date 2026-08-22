# Named Entity Recognition for Data Objects in Privacy Policies

## 1. Overview

We present a fine-tuned Named Entity Recognition (NER) system for identifying **data objects** in privacy policy text. The task uses a BIO tagging scheme with three labels:

- `O` — Outside any entity
- `B` — Beginning of a data object span
- `I` — Inside / continuation of a data object span

The final model achieves strong performance on both token-level and entity-level metrics, reaching a **test set seqeval F1 of 0.933**.

---

## 2. Dataset

### 2.1 Original Dataset

- Source: Manually annotated privacy policy sentences using Label Studio
- File: `bio_from_labelstudio_dataobj_5559.jsonl`
- Total examples: **5,559**
- Format: Tokenized sentences with corresponding BIO tags

**Original label distribution (approximate):**

| Label | Count   | Notes                  |
|-------|---------|------------------------|
| O     | ~135k   | Heavily dominant       |
| B     | ~4.7k   | Entity starts          |
| I     | ~5.3k   | Entity continuations   |

The severe class imbalance (O ≫ B/I) motivated a targeted data augmentation strategy.

### 2.2 Data Split

We used a **stratified 70/15/15** train/validation/test split based on the presence of `B` and `I` tags in each sentence (`has_B` × `has_I` stratification key). This ensured that entity-containing sentences were proportionally distributed across splits.

---

## 3. Data Augmentation

To mitigate the extreme imbalance of the `O` class and to improve coverage of rare data objects, we applied an **entity-swapping augmentation** procedure.

### 3.1 Methodology

1. All unique data object spans were extracted from the training data.
2. Entities were ranked by frequency.
3. Only the **rarest 50%** of entities were retained as replacement candidates.
4. For every original sentence that contained at least one entity:
   - Multiple augmented versions were generated (5 per original sentence).
   - One or more existing entity spans were replaced by randomly sampled rare entities.
   - With probability 0.35, two entities in the same sentence were replaced (multi-replace).
5. BIO tags were correctly regenerated after each replacement to account for differing entity lengths.

### 3.2 Augmentation Statistics

| Statistic                        | Value     |
|----------------------------------|-----------|
| Original examples                | 5,559     |
| Sentences containing entities    | 3,011     |
| New augmented examples           | 15,055    |
| **Total after augmentation**     | **20,614**|

**Label distribution after augmentation:**

| Label | Count   | Percentage |
|-------|---------|------------|
| O     | 668,808 | 87.7%      |
| B     | 40,260  | 5.3%       |
| I     | 53,870  | 7.1%       |

This augmentation substantially increased the absolute number of `B` and `I` tokens (approximately 8–10×) while preserving natural sentence contexts.

---

## 4. Model

- **Base model**: [`mukund/privbert`](https://huggingface.co/mukund/privbert)
- Architecture: RoBERTa-based
- Pre-training: Approximately 1 million privacy policy documents (PrivaSeer corpus)
- Task head: Linear classification head for token classification (3 labels)

PrivBERT was selected because of its domain-specific pre-training on privacy policy language, which provides a strong inductive bias for the target task.

---

## 5. Training Setup

### 5.1 Hyperparameters

| Parameter                    | Value                          |
|-----------------------------|--------------------------------|
| Maximum sequence length     | 256                            |
| Batch size                  | 32                             |
| Learning rate               | 1×10⁻⁵                         |
| Weight decay                | 0.05                           |
| Number of epochs            | up to 200 (with early stopping)|
| Early stopping patience     | 30 epochs                      |
| Early stopping threshold    | 0.001                          |
| Mixed precision             | FP16                           |
| Optimizer                   | AdamW (via Hugging Face Trainer)|
| Evaluation metric           | seqeval F1 (entity-level)      |
| Best model selection        | Highest validation seqeval F1  |

### 5.2 Fine-tuning Strategy

- **Full fine-tuning**: All transformer layers were unfrozen (`FREEZE_LAYERS = False`).
- Earlier experiments with partial freezing (last 4 or 6 layers only) yielded lower performance.

### 5.3 Loss Function

We used **Focal Loss** to further address residual class imbalance:

$$
\mathcal{L}_{\text{focal}} = -(1 - p_t)^\gamma \log(p_t)
$$

- Focusing parameter: $\gamma = 1.5$
- Class weights were **not** used in the final configuration (Focal Loss alone proved sufficient).

---

## 6. Evaluation Metrics

We report two complementary sets of metrics:

1. **Token-level** metrics (Precision, Recall, F1, Accuracy) for each class (`O`, `B`, `I`).
2. **Entity-level** metrics using the `seqeval` library (exact span matching), which is the primary metric for model selection.

---

## 7. Results

### 7.1 Final Model Performance

**Validation Set**

| Metric              | Value   |
|---------------------|---------|
| Seqeval Precision   | 0.9301  |
| Seqeval Recall      | 0.9521  |
| **Seqeval F1**      | **0.9410** |
| Token Accuracy      | 0.9883  |

**Token-level (Validation)**

| Class | Precision | Recall | F1     | Support |
|-------|-----------|--------|--------|---------|
| O     | 0.9946    | 0.9925 | 0.9935 | 112,460 |
| B     | 0.9476    | 0.9594 | 0.9535 | 6,055   |
| I     | 0.9530    | 0.9649 | 0.9589 | 12,833  |
| Macro | 0.9651    | 0.9722 | 0.9686 | —       |

**Test Set**

| Metric              | Value   |
|---------------------|---------|
| Seqeval Precision   | 0.9263  |
| Seqeval Recall      | 0.9397  |
| **Seqeval F1**      | **0.9330** |
| Token Accuracy      | 0.9869  |

**Token-level (Test)**

| Class | Precision | Recall | F1     | Support |
|-------|-----------|--------|--------|---------|
| O     | 0.9932    | 0.9922 | 0.9927 | 112,407 |
| B     | 0.9468    | 0.9501 | 0.9484 | 6,008   |
| I     | 0.9509    | 0.9584 | 0.9546 | 13,012  |
| Macro | 0.9636    | 0.9669 | 0.9653 | —       |

### 7.2 Ablation Insights

| Configuration                        | Test Seqeval F1 |
|--------------------------------------|-----------------|
| Original data + partial freeze (4–6 layers) | ~0.66          |
| Augmented data + partial freeze (6 layers)  | 0.922          |
| **Augmented data + full fine-tuning**       | **0.933**      |

Key observations:
- Data augmentation provided the largest performance gain (+25+ F1 points).
- Full fine-tuning of PrivBERT yielded an additional ~1.1 F1 improvement over partial layer freezing.
- Focal Loss helped maintain balanced precision and recall for the minority classes `B` and `I`.

---

## 8. Implementation Details

- Framework: Hugging Face Transformers + PyTorch
- Hardware: NVIDIA RTX 4070 (12 GB)
- Training time: Approximately 100–110 epochs until early stopping
- Model checkpoint: Best validation seqeval F1 model was retained

---

## 9. Conclusion

By combining domain-specific pre-training (PrivBERT), aggressive rare-entity data augmentation, full model fine-tuning, and Focal Loss, we obtained a high-performing NER system for data object extraction from privacy policies. The final model achieves a **test seqeval F1 of 0.933**, with strong and balanced performance on both beginning (`B`) and inside (`I`) tokens.

The approach demonstrates that targeted entity-level augmentation is particularly effective for highly imbalanced sequence labeling tasks in specialized domains.