# BERT Sentiment Analyzer

Fine-tuned `bert-base-uncased` achieves **86% accuracy** on IMDb sentiment classification — deployable in 1 click.

🚀 **[Live Demo](https://bert-sentiment-app-011.streamlit.app/)** &nbsp;|&nbsp; 🤗 **[Model on HuggingFace](https://huggingface.co/Xtern0723/bert-imdb-sentiment)**

---

## What it does

Input any text → get **Positive / Negative** prediction + confidence score instantly.

Built to demonstrate fine-tuning a pre-trained BERT model on a downstream classification task — the core skill behind production AI support bots and review analysis pipelines.

---

## Results

| Metric | Value |
|---|---|
| Eval Accuracy | **86%** |
| Eval Dataset | 1,000 IMDb reviews |
| Training Dataset | 5,000 IMDb reviews |
| Base Model | `bert-base-uncased` |
| Epochs | 3 |
| Learning Rate | 2e-5 |
| Batch Size | 16 |
| Max Token Length | 128 |
| Mixed Precision | fp16 (GTX 1650 4GB) |

> Training was intentionally constrained to 5,000 samples (vs. the full 25,000) to fit a 4GB VRAM GPU. Accuracy scales further with more data and epochs.

---

## Stack

- 🤗 HuggingFace Transformers — `Trainer` API, `AutoModelForSequenceClassification`
- `DataCollatorWithPadding` — dynamic per-batch padding (saves memory vs. global padding)
- PyTorch — inference with `torch.no_grad()` + `F.softmax`
- Streamlit — UI with `@st.cache_resource` for single model load
- HuggingFace Hub — model hosting (SafeTensors format, 438MB)

---

## Run locally

```bash
git clone https://github.com/codewith-krishh/bert-sentiment-app.git
cd bert-sentiment-app
pip install -r requirements.txt
streamlit run app.py
```

Requirements:
```
streamlit>=1.28.0
transformers>=4.35.0
torch>=2.1.0
```

---

## Repo structure

```
bert-sentiment-app/
├── app.py               # Streamlit UI + inference logic
├── bert_finetune.ipynb  # Full fine-tuning notebook (Day 11)
├── requirements.txt     # Streamlit Cloud dependencies
└── README.md
```

---

## How it works

```
Input text
    ↓
WordPiece tokenizer (max 128 tokens)
    ↓
bert-base-uncased (12 layers, 110M params)
    ↓
[CLS] token embedding → 2-class linear head
    ↓
Softmax → Positive / Negative + confidence %
```

**Key design decisions:**
- `@st.cache_resource` keeps the model in memory across reruns (vs. reloading on every click)
- `model.eval()` disables dropout for deterministic inference
- `torch.no_grad()` skips autograd tape — saves memory, speeds up inference

---

## Model

Hosted on HuggingFace Hub → [`Xtern0723/bert-imdb-sentiment`](https://huggingface.co/Xtern0723/bert-imdb-sentiment)

Load it directly:
```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="Xtern0723/bert-imdb-sentiment")
classifier("This product completely changed how our team handles support.")
# [{'label': 'POSITIVE', 'score': 0.97}]
```

---

## Author

Built by **Krish** — B.Tech AI student building an LLM freelance practice targeting US SaaS founders.

[![X](https://img.shields.io/badge/X-%40Born__TechK-black)](https://twitter.com/Born_TechK)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-krish--manji011-blue)](https://linkedin.com/in/krish-manji011)
[![GitHub](https://img.shields.io/badge/GitHub-codewith--krishh-lightgrey)](https://github.com/codewith-krishh)