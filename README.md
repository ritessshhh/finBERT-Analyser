---
language:
- en
license: mit
tags:
- text-classification
- multi-label-classification
- financial-nlp
- finance
- event-detection
datasets:
- sentivent
metrics:
- f1
- precision
- recall
pipeline_tag: text-classification
---

# FinDeBERTa: Multi-Label Financial Event Classifier

FinDeBERTa is a fine-tuned DeBERTa-v3-Large model for multi-label financial event classification. It predicts one or more event types from financial news headlines with state-of-the-art performance.

## Model Details

- **Base Model**: microsoft/deberta-v3-large
- **Task**: Multi-label text classification
- **Training**: Fine-tuned with Focal Loss and per-class threshold optimization

## Event Labels

The model classifies text into 18 financial event types:

```python
["CSR/Brand", "Deal", "Dividend", "Employment", "Expense", "Facility",
 "FinancialReport", "Financing", "Investment", "Legal", "Macroeconomics",
 "Merger/Acquisition", "Product/Service", "Profit/Loss", "Rating", "Revenue",
 "SalesVolume", "SecurityValue"]
```

## Usage

### Optimized Usage (with per-class thresholds) (Recommended)

For best performance, use the per-class optimized thresholds:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from huggingface_hub import hf_hub_download

model_name = "ritessshhh/FinDeBERTa"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Download per-class thresholds
thresholds_path = hf_hub_download(repo_id=model_name, filename="thresholds.npy")
thresholds = np.load(thresholds_path)

text = "Tesla to acquire a battery startup in a 400 million dollar deal."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()

# Apply per-class thresholds
predictions = [
    {"label": model.config.id2label[i], "score": float(prob)}
    for i, prob in enumerate(probs) if prob >= thresholds[i]
]

# Sort by score
predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)
print(predictions)
# Output: [{"label": "Merger/Acquisition", "score": 0.98}, ...]
```

### Basic Usage (with default threshold)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

model_name = "ritessshhh/FinDeBERTa"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "Tesla to acquire a battery startup in a 400 million dollar deal."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()

# Using default threshold of 0.5
threshold = 0.5
predictions = [
    {"label": model.config.id2label[i], "score": float(prob)}
    for i, prob in enumerate(probs) if prob >= threshold
]

print(predictions)
```

## Training Details

- **Loss Function**: Focal Loss (gamma=2.0) with dampened class weights
- **Optimizer**: AdamW with cosine learning rate scheduling
- **Batch Size**: 8 (with gradient accumulation steps=2)
- **Epochs**: 10
- **Learning Rate**: 2e-5
- **Weight Decay**: 0.02

## Performance

| Metric | Score |
|--------|-------|
| Macro F1 | 0.692 |
| Micro F1 | 0.691 |
| Precision (Macro) | 0.738 |
| Recall (Macro) | 0.691 |
| Exact Match Ratio | 0.532 |

### Per-Class Performance

| Label | F1 Score | Precision | Recall |
|-------|----------|-----------|--------|
| Dividend | 1.000 | 1.000 | 1.000 |
| Employment | 0.923 | 0.857 | 1.000 |
| Merger/Acquisition | 0.892 | 0.967 | 0.829 |
| Profit/Loss | 0.833 | 0.824 | 0.843 |
| SecurityValue | 0.829 | 0.843 | 0.815 |
| Rating | 0.790 | 0.800 | 0.780 |
| Revenue | 0.780 | 0.800 | 0.762 |
| SalesVolume | 0.748 | 0.833 | 0.678 |
| Financing | 0.714 | 0.714 | 0.714 |
| Deal | 0.696 | 0.889 | 0.571 |

## Limitations

- Optimized for financial news headlines (short text)
- May not generalize well to other domains
- Performance varies by event type (rare events like "Facility" have lower F1)

## Citation

If you use this model, please cite:

```bibtex
@misc{findeberta2024,
  author = {ritessshhh},
  title = {FinDeBERTa: Multi-Label Financial Event Classifier},
  year = {2024},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/ritessshhh/FinDeBERTa}}
}
```
