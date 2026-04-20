# 🤖 Domain Adaptation for Sentiment Analysis (BERT)

An NLP project that transfers sentiment knowledge from IMDB movie reviews to Amazon and Yelp reviews — without using any target domain labels — using BERT, Domain-Adaptive Pre-Training (DAPT), and pseudo-labeling.

---

## 📌 Project Overview

Sentiment models trained on one domain often fail when applied to another. This project builds a fully unsupervised cross-domain adaptation pipeline that achieves competitive accuracy on unseen target domains without any manual annotation.

|                    |                                                 |
| ------------------ | ----------------------------------------------- |
| **Source Domain**  | IMDB Movie Reviews (50,000 labeled samples)     |
| **Target Domains** | Amazon Product Reviews · Yelp Business Reviews  |
| **Model**          | `bert-base-uncased` (Hugging Face Transformers) |
| **Best Accuracy**  | 93.79% on Amazon · 93.65% on Yelp               |

---

## 🔧 Tech Stack

| Tool                         | Purpose                 |
| ---------------------------- | ----------------------- |
| Python                       | Core implementation     |
| Hugging Face Transformers    | BERT model, Trainer API |
| PyTorch                      | Model training          |
| Google Colab (NVIDIA T4 GPU) | Training environment    |
| VADER                        | Rule-based baseline     |

---

## ⚙️ Pipeline

```
IMDB (labeled source)        Target Domain (unlabeled)
         │                            │
         │                      Pool A (60%)
         │                      DAPT — continue MLM
         │                      pre-training on raw
         │                      target text
         │                            │
         └──────────────┬─────────────┘
                        ▼
              BERT + DAPT + IMDB Fine-Tuning
              (supervised on IMDB labels)
                        │
                   Pool B (20%)
                   Pseudo-Labeling
                   (confidence ≥ 0.9)
                        │
                        ▼
                  Final Model
                        │
                   Pool C (20%)
                   Held-out Evaluation
```

**Stage 1 — DAPT:** Continue BERT's masked language model pre-training on unlabeled target text (Pool A) to align representations with target domain vocabulary and writing style.

**Stage 2 — IMDB Fine-Tuning:** Add a classification head and fine-tune on 50,000 labeled IMDB samples to instill sentiment classification capability.

**Stage 3 — Pseudo-Labeling:** Use the fine-tuned model to generate labels for Pool B. Retain only high-confidence predictions (≥ 0.9) and fine-tune further on these pseudo-labels.

**Stage 4 — Evaluation:** Evaluate all baselines on the fully held-out Pool C (10,000 samples per domain).

---

## 📊 Results

### Amazon Reviews

| Baseline                             | Accuracy   | Macro F1   |
| ------------------------------------ | ---------- | ---------- |
| 1. VADER (rule-based)                | 72.47%     | 0.7128     |
| 2. BERT + IMDB (zero-shot)           | 91.38%     | 0.9136     |
| 3. BERT + IMDB + 1K labeled Amazon   | 94.40%     | 0.9440     |
| 4. BERT + IMDB + Pseudo (no DAPT)    | 93.21%     | 0.9321     |
| 5. BERT + DAPT + IMDB (no Pseudo)    | 93.17%     | 0.9316     |
| **6. Full Pipeline (DAPT + Pseudo)** | **93.79%** | **0.9378** |

### Yelp Reviews

| Baseline                             | Accuracy   | Macro F1   |
| ------------------------------------ | ---------- | ---------- |
| 1. VADER (rule-based)                | 71.99%     | 0.7015     |
| 2. BERT + IMDB (zero-shot)           | 90.38%     | 0.9037     |
| 3. BERT + IMDB + 1K labeled Yelp     | 93.63%     | 0.9363     |
| 4. BERT + IMDB + Pseudo (no DAPT)    | 91.61%     | 0.9160     |
| 5. BERT + DAPT + IMDB (no Pseudo)    | 92.79%     | 0.9278     |
| **6. Full Pipeline (DAPT + Pseudo)** | **93.65%** | **0.9365** |

**Key finding:** On Yelp, the fully unsupervised pipeline matches a model trained on 1,000 labeled target samples — demonstrating that DAPT + pseudo-labeling can substitute for manual annotation in low-resource settings.

---

## 🧪 Experimental Design

### Data Splits (per target domain)

| Pool   | Size          | Purpose                                        |
| ------ | ------------- | ---------------------------------------------- |
| Pool A | 60% (~30,000) | DAPT — labels stripped, raw text only          |
| Pool B | 20% (~10,000) | Pseudo-label generation + 1K oracle baseline   |
| Pool C | 20% (~10,000) | Held-out test set — never seen during training |

### Hyperparameters

| Parameter              | Value               |
| ---------------------- | ------------------- |
| Base model             | `bert-base-uncased` |
| Optimizer              | AdamW               |
| Learning rate          | 2e-5                |
| Batch size             | 16                  |
| Epochs                 | 3 (each stage)      |
| Pseudo-label threshold | 0.9                 |
| Evaluation metric      | Accuracy + Macro F1 |

---

## 💡 Key Findings

- **DAPT and pseudo-labeling are complementary** — combining both consistently outperforms either technique alone
- **Domain proximity matters** — IMDB-to-Amazon transfer is stronger than IMDB-to-Yelp because Amazon reviews share a more similar writing style with IMDB
- **DAPT helps more where the domain gap is larger** — Yelp gains +2.41% from DAPT vs +1.79% for Amazon
- **Unsupervised pipeline matches supervised on Yelp** — achieving 93.65% vs 93.63% with 1,000 labeled samples

---

## 📁 Repository Structure

```
domain-adaptation-sentiment/
│
├── dapt.py                  # Domain-Adaptive Pre-Training script
├── finetune_imdb.py         # IMDB supervised fine-tuning
├── pseudo_label.py          # Pseudo-label generation and fine-tuning
├── evaluate.py              # Evaluation on held-out test set
├── baselines.py             # VADER and zero-shot BERT baselines
├── data_prep.py             # Dataset loading and pool splitting
├── domain_adaptation_report.pdf  # Full project report
└── README.md
```
