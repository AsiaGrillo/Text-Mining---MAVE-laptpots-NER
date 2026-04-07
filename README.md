# Named Entity Recognition on MAVE Laptops Dataset
### (Text Mining and Data Visualization)

A comparative study of NER models for structured product attribute extraction: from a BiLSTM+CRF baseline to a fine-tuned DeBERTa-v3, with a RAG-assisted label cleaning pipeline and a controlled experiment isolating the effect of annotation quality on model evaluation.

By [@AsiaGrillo](https://github.com/AsiaGrillo)

---

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Notebooks](#notebooks)
  - [01 — EDA](#01--eda)
  - [02 — RAG Label Cleaning](#02--rag-label-cleaning)
  - [03 — BiLSTM + CRF](#03--bilstm--crf)
  - [04 — DeBERTa-v3](#04--deberta-v3)
- [Key Results](#key-results)
- [Interactive Dashboard](#interactive-dashboard)
- [Repository Structure](#repository-structure)
- [Reproducibility Instructions](#reproducibility-instructions)
- [References](#references)

---

## Introduction

This project addresses **Named Entity Recognition (NER)** on the MAVE Laptops dataset — a collection of Amazon product titles annotated with five entity classes: `BRAND`, `SCREEN_SIZE`, `PROCESSOR`, `RESOLUTION`, and `BATTERY`.

The central finding is that **high performance on noisy data does not necessarily reflect correct semantic understanding, but rather consistency with annotation artifacts**. The original MAVE annotations contain systematic noise — most critically, Intel processor tokens are frequently mislabelled as `BRAND` rather than `PROCESSOR`. When both training and test sets share the same errors, standard evaluation metrics reward the noise rather than the model's true understanding.

To isolate this effect, the project implements a **RAG-assisted label cleaning pipeline** and runs a controlled experiment across two models and two dataset versions, producing four experimental conditions that together answer two questions: *how much does architecture matter, and how much does label quality matter?*

---

## Dataset

**Source:** [MAVE: A Product Dataset for Multi-source Attribute Value Extraction](https://github.com/google-research-datasets/MAVE) (Yang et al., WSDM 2022)

The dataset consists of Amazon laptop product titles annotated in **BIO format** across five entity classes:

| Class | Description | Train spans |
|---|---|---|
| `BRAND` | Laptop/processor manufacturer | 6,267 |
| `SCREEN_SIZE` | Physical display dimension | 7,398 |
| `PROCESSOR` | CPU family/model | 4,085 |
| `RESOLUTION` | Display resolution | 934 |
| `BATTERY` | Battery capacity/life | 79 |

**Splits:** 7,479 train / 935 val / 935 test (80/10/10)

The dataset is **pre-tokenized**: each record in the JSONL files contains a `tokens` list and a parallel `labels` list in BIO format. No additional tokenization is required for the EDA or BiLSTM notebooks.

---

## Project Pipeline

```
01_EDA.ipynb
    └── Annotation noise quantified → motivates cleaning

02_RAG_Label_Cleaning.ipynb
    └── Rule-first + RAG + LLM pipeline → cleaned JSONL files

03_BiLSTM.ipynb  (local, VS Code)
    ├── Experiment 1: train on original, evaluate on original
    └── Experiment 2: train on cleaned, evaluate on original + cleaned

04_DeBERTa.ipynb  (Google Colab, T4 GPU)
    ├── Experiment 1: train on original, evaluate on original
    └── Experiment 2: train on cleaned, evaluate on original + cleaned

05_Dashboard.py
    └── Interactive visualization of all results
```

---

## Notebooks

### 01 — EDA

**File:** `01_EDA.ipynb` | **Environment:** local

Exploratory analysis of the original MAVE Laptops dataset. Key analyses include:

- Sequence length distribution and split consistency
- Label distribution and class imbalance (79.7% O tokens)
- Entity coverage and span count per class
- Entity value distribution — first visual evidence of BRAND annotation noise
- Entity co-occurrence matrix
- Lexical ambiguity analysis (1,495 ambiguous tokens)
- Token frequency per class (Okabe-Ito palette, stopwords excluded)
- Annotation noise quantification: 426 tokens with conflicting B- labels
- BRAND ambiguity deep-dive: 20% of sequences have 2+ BRAND spans
- RESOLUTION lexical variability: 474 unique surface forms for 934 spans
- BATTERY data scarcity: 79 train spans, 10 test spans
- Class imbalance summary and model selection rationale

### 02 — RAG Label Cleaning

**File:** `02_RAG_Label_Cleaning.ipynb` | **Environment:** local

A two-stage label cleaning pipeline applied to all three dataset splits:

**Stage 1 — Deterministic rules** (zero API cost): 20 rules derived from EDA findings correct the most frequent unambiguous patterns (e.g. any span headed by `intel`, `core`, `celeron` → `PROCESSOR`; any span headed by `ram`, `ssd`, `laptop` → `O`).

**Stage 2 — RAG + LLM fallback**: for genuinely ambiguous spans, a TF-IDF retriever (one index per entity class, ngram_range=(1,2), sublinear_tf=True) retrieves top-5 support examples from a trusted corpus built from the training split. The retrieved examples ground a few-shot Chain-of-Thought prompt sent to `llama-3.1-8b-instant` via the Groq API.

**Results on training split:**
- 7,273 spans corrected (47.7% of ambiguous spans)
- 4,208 corrections are BRAND → PROCESSOR (dominant pattern)
- 1 API failure across the full run

**Output files** (saved to `DATA_DIR`):
- `laptops_train_rag_cleaned.jsonl`
- `laptops_val_rag_cleaned.jsonl`
- `laptops_test_rag_cleaned.jsonl`

> **Note:** The Groq API key is entered at runtime via `getpass` and is never stored. The full cleaning run takes approximately 10 hours. Pre-cleaned files are available in the Google Drive folder linked below.

### 03 — BiLSTM + CRF

**File:** `03_BiLSTM.ipynb` | **Environment:** local (VS Code, CPU)

Controlled experiment using the **BiLSTM Enhanced** model (GloVe 300d + Char-CNN + 2-layer BiLSTM + CRF). All hyperparameters, seeds, and preprocessing steps are identical across both experiments.

**Architecture:**
- Word embeddings: GloVe 6B 300d (15.5% vocabulary coverage on MAVE)
- Character embeddings: Char-CNN (kernel sizes 2,3; 50 filters each → 100-dim output)
- Encoder: 2-layer BiLSTM, hidden dim 256 per direction
- Output: CRF layer with learned transition scores, Viterbi decoding

**Hyperparameters:** batch=32, LR=5e-4, Adam (wd=1e-4), epochs=30, patience=5, StepLR (step=10, γ=0.5), grad clip=5.0

**Results:**

| Train | Test | Micro F1 | BRAND | SCREEN_SIZE | PROCESSOR | RESOLUTION | BATTERY |
|---|---|---|---|---|---|---|---|
| Original | Original | 0.7346 | 0.6589 | 0.8252 | 0.6979 | 0.5561 | 0.8889 |
| Cleaned | Original | 0.5293 | 0.0026 | 0.8185 | 0.4453 | 0.5622 | 0.7368 |
| Cleaned | Cleaned | **0.7518** | 0.0073 | 0.8419 | 0.7847 | 0.6974 | 0.7778 |

### 04 — DeBERTa-v3

**File:** `04_DeBERTa.ipynb` | **Environment:** Google Colab (T4 GPU)

Same controlled experiment using **DeBERTa-v3-base** (`microsoft/deberta-v3-base`). The model uses disentangled attention (content and position encoded separately) and ELECTRA-style pre-training.

**Key implementation notes:**
- `.float()` cast required after loading to prevent float16 NaN gradients on T4
- `torch.cuda.empty_cache()` between experiments to avoid OOM
- `ignore_mismatched_sizes=True` to replace the pre-trained classification head
- Classifier head re-initialized with Xavier uniform weights before each experiment
- Subword alignment: first subword of each MAVE token receives the gold BIO label; subsequent subwords receive -100 (ignored in loss and evaluation)

**Hyperparameters:** batch=16, LR=2e-5, AdamW (wd=0.01), epochs=15, patience=5, warmup=10%, grad clip=1.0

**Results:**

| Train | Test | Micro F1 | BRAND | SCREEN_SIZE | PROCESSOR | RESOLUTION | BATTERY |
|---|---|---|---|---|---|---|---|
| Original | Original | 0.7456 | 0.6718 | 0.8282 | 0.7234 | 0.5963 | 0.8889 |
| Cleaned | Original | 0.5346 | 0.0357 | 0.8246 | 0.4502 | 0.6075 | 0.8421 |
| Cleaned | Cleaned | **0.7689** | 0.1937 | 0.8479 | 0.8095 | 0.7293 | 0.8889 |

---

## Key Results

The drop from Cleaned→Original to Cleaned→Cleaned is not a failure of the cleaning pipeline — it is evidence that the original test set rewards annotation artifacts. The model trained on cleaned labels predicts `PROCESSOR` for *Intel* and *Core*, which is semantically correct, but the original test set labels these tokens as `BRAND` and penalises the predictions as false positives.

**The cleaned test evaluation (Setting 2) is the semantically valid benchmark.** Under this benchmark:
- DeBERTa achieves 0.7689 micro F1 — the best result across all conditions
- `PROCESSOR` F1 improves from 0.7234 to 0.8095 (+0.0861)
- `RESOLUTION` F1 improves from 0.5963 to 0.7293 (+0.1330)
- `BRAND` F1 remains near zero in both models due to a structural dataset limitation: real manufacturer names (*Dell*, *HP*, *Lenovo*) were labelled `O` in the original annotations and cannot be recovered by span-level correction alone

---

## Interactive Dashboard

**File:** `05_Dashboard.py`

A Plotly Dash dashboard with five tabs and dark theme:

- **📊 Dataset** — KPI cards, entity span counts, co-occurrence matrix, annotation noise analysis (BRAND ambiguity, RESOLUTION variability)
- **🧹 Label Cleaning** — correction matrix, before/after BRAND and PROCESSOR distributions
- **🧠 BiLSTM + CRF** — architecture overview, training curves, per-class F1, span-level confusion matrices
- **🤖 DeBERTa-v3** — same structure + comparison with BiLSTM
- **🔍 Live Demo** — enter any laptop product title and see predictions from all 4 models (BiLSTM Original, BiLSTM Cleaned, DeBERTa Original, DeBERTa Cleaned) with color-coded entity chips

---

## Repository Structure

```
.
├── 01_EDA.ipynb                        # Exploratory Data Analysis
├── 02_RAG_Label_Cleaning.ipynb         # RAG-assisted label cleaning pipeline
├── 03_BiLSTM.ipynb                     # BiLSTM+CRF: Experiment 1 + 2 (local)
├── 04_DeBERTa.ipynb                    # DeBERTa-v3: Experiment 1 + 2 (Colab)
├── 05_Dashboard.py                     # Interactive Plotly/Dash dashboard
├── .gitignore
└── README.md
```

All required data, model weights, and embeddings are available in the following Google Drive folder:
[Google Drive](https://drive.google.com/drive/folders/1qUhvnhMPszQt-5pF-3THV2QPkHMQXsXj?usp=sharing)

| File | Description |
|---|---|
| `laptops_train.jsonl` | Original training set |
| `laptops_val.jsonl` | Original validation set |
| `laptops_test.jsonl` | Original test set |
| `laptops_train_rag_cleaned.jsonl` | Cleaned training set |
| `laptops_val_rag_cleaned.jsonl` | Cleaned validation set |
| `laptops_test_rag_cleaned.jsonl` | Cleaned test set |
| `bilstm_enhanced_original.pt` | BiLSTM weights (Experiment 1) |
| `bilstm_enhanced_cleaned.pt` | BiLSTM weights (Experiment 2) |
| `deberta_v2_original.pt` | DeBERTa weights (Experiment 1) |
| `deberta_v2_cleaned.pt` | DeBERTa weights (Experiment 2) |
| `glove.6B.300d.txt` | GloVe pre-trained embeddings |

Files are organized into subfolders (data_original, data_cleaned, models, embeddings) inside the Drive directory.

---

## Reproducibility Instructions

### 1. Clone the repository

```bash
git clone https://github.com/AsiaGrillo/Text-Mining---MAVE-laptops-NER.git
cd Text-Mining---MAVE-laptops-NER
```

### 2. Download data and weights

Download all files from the [Google Drive folder](https://drive.google.com/drive/folders/1qUhvnhMPszQt-5pF-3THV2QPkHMQXsXj?usp=sharing) and place them in a single local directory (e.g. `MAVE_data/`).

The folder must contain:

- original datasets (`laptops_train.jsonl`, `laptops_val.jsonl`, `laptops_test.jsonl`)
- cleaned datasets (`*_rag_cleaned.jsonl`)
- model weights (`*.pt`)
- GloVe embeddings (`glove.6B.300d.txt`)

Then update the `DATA_DIR` path at the top of each notebook and in `05_Dashboard.py`:

```python
DATA_DIR = '/absolute/path/to/MAVE_data/'
```

No additional code modifications are required.

For notebooks 01, 02, 03, and the dashboard, this is the only path change needed.

For **notebook 04** (DeBERTa on Google Colab), mount your Google Drive and set:

```python
DATA_DIR = '/content/drive/MyDrive/MAVE/'
```

Then place all data and weight files in that Google Drive folder before running.

### 3. Create the environment

```bash
conda create -n mave-ner python=3.10 -y
conda activate mave-ner
```

### 4. Install dependencies

```bash
pip install torch torchvision torchaudio
pip install transformers sentencepiece
pip install scikit-learn
pip install dash plotly
pip install matplotlib seaborn
pip install tqdm requests
```

For the RAG cleaning notebook, you also need a **Groq API key**. The notebook will prompt you to enter it securely at runtime via `getpass` — no key is stored in the code.

### 5. Run the notebooks

**Notebook 01 — EDA** (local):
```bash
jupyter notebook 01_EDA.ipynb
```

**Notebook 02 — RAG Label Cleaning** (local):
```bash
jupyter notebook 02_RAG_Label_Cleaning.ipynb
```
> Skip this step if you download the pre-cleaned JSONL files from Google Drive. The full cleaning run takes approximately 10 hours.

**Notebook 03 — BiLSTM** (local, CPU sufficient):
```bash
jupyter notebook 03_BiLSTM.ipynb
```
> Training takes approximately 30–60 minutes on CPU. Skip training and load pre-trained weights directly using the reload cell provided in the notebook.

**Notebook 04 — DeBERTa** (Google Colab, T4 GPU required):

Upload `04_DeBERTa.ipynb` to Google Colab, mount your Google Drive, and run all cells. Training takes approximately 2–3 hours per experiment on a T4 GPU.

### 6. Run the dashboard

```bash
cd /path/to/your/local/folder/
python 05_Dashboard.py
```

Open your browser at: [http://127.0.0.1:8050](http://127.0.0.1:8050)

> The dashboard loads all four model weights at startup. The first inference call for DeBERTa models will take a few seconds while the tokenizer and model are loaded into memory.

---

## References

- Yang, L. et al. (2022). *MAVE: A Product Dataset for Multi-source Attribute Value Extraction*. WSDM '22.
- Ni, J. et al. (2019). *Justifying Recommendations using Distantly-Labeled Reviews and Fine-grained Aspects*. EMNLP 2019.
- He, P. et al. (2021). *DeBERTa: Decoding-enhanced BERT with Disentangled Attention*. ICLR 2021.
- Lample, G. et al. (2016). *Neural Architectures for Named Entity Recognition*. NAACL 2016.
- Pennington, J. et al. (2014). *GloVe: Global Vectors for Word Representation*. EMNLP 2014.
- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020.
- Northcutt, C. et al. (2021). *Confident Learning: Estimating Uncertainty in Dataset Labels*. JAIR 2021.
- Ratner, A. et al. (2017). *Snorkel: Rapid Training Data Creation with Weak Supervision*. VLDB 2017.
- Gilardi, F. et al. (2023). *ChatGPT Outperforms Crowd Workers for Text Annotation Tasks*. PNAS 2023.
- Ng, A. (2021). *A Chat with Andrew on MLOps: From Model-centric to Data-centric AI*. DeepLearning.AI.
- Frenay, B. & Verleysen, M. (2014). *Classification in the Presence of Label Noise: A Survey*. IEEE TNNLS 2014.
- Brown, T. et al. (2020). *Language Models are Few-Shot Learners*. NeurIPS 2020.
- Ratinov, L. & Roth, D. (2009). *Design Challenges and Misconceptions in Named Entity Recognition*. CoNLL 2009.
- Manning, C. et al. (2014). *The Stanford CoreNLP Natural Language Processing Toolkit*. ACL 2014.
