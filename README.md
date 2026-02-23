# 🧠 PyPI Text Classification — TOPSIS Model Selection

> Benchmark **6 real PyPI sentiment libraries** on the same dataset, then use **TOPSIS** (multi-criteria decision analysis) to find the best model based on your priorities — accuracy, speed, memory, or a balanced mix.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [What is TOPSIS?](#-what-is-topsis)
- [PyPI Models Used](#-pypi-models-used)
- [Environment Setup](#-environment-setup)
- [How to Run](#-how-to-run)
- [How the Code Works](#-how-the-code-works)
- [Weight Configurations](#-weight-configurations)
- [Results](#-results)
- [Key Findings](#-key-findings)
- [Output Files](#-output-files)
- [Customising the Analysis](#-customising-the-analysis)

---

## 📌 Overview

This project compares **six sentiment classification libraries from PyPI** — each with its own pre-trained model, no training from scratch required. All models are evaluated on the same 25-sentence dataset covering **Positive / Negative / Neutral** sentiment.

After evaluation, **TOPSIS** ranks the models across multiple criteria simultaneously (accuracy, F1, precision, recall, speed, and memory), rather than picking a winner on a single metric alone.

**Three weight scenarios** are run to show how the winner changes depending on your priorities:

| Scenario | Focus |
|---|---|
| `Balanced` | Equal importance across performance + efficiency |
| `Accuracy-Priority` | Maximise accuracy and F1 |
| `Speed-Priority` | Minimise inference time and memory usage |

Results are exported to a colour-coded **Excel workbook** with one sheet per scenario.

---

## 🔢 What is TOPSIS?

**TOPSIS** (Technique for Order of Preference by Similarity to Ideal Solution) is a multi-criteria decision analysis (MCDA) method. Instead of ranking on one metric, it finds the model that is:

- **Closest** to the ideal best solution (A⁺)
- **Furthest** from the ideal worst solution (A⁻)

across all weighted criteria at once.

### Algorithm Steps

1. Build the **decision matrix** — rows = models, columns = criteria
2. **Normalize** the matrix using vector normalization (same scale for all criteria)
3. Apply **weights** to produce the weighted normalized matrix
4. Identify **A⁺ (Ideal Best)** — highest value for benefit criteria, lowest for cost criteria
5. Identify **A⁻ (Ideal Worst)** — opposite of A⁺
6. Calculate **Euclidean distance** of each model from A⁺ (S⁺) and A⁻ (S⁻)
7. Compute **TOPSIS Score** = `S⁻ / (S⁺ + S⁻)` — a score of `1.0` means the model IS the ideal solution
8. **Rank** models by TOPSIS Score (highest = best)

---

## 📦 PyPI Models Used

Each model comes from a **completely separate PyPI package** with its own pre-trained weights.

| # | Model | PyPI Package | Type | Install |
|---|---|---|---|---|
| 1 | TextBlob | `textblob` | Rule-based polarity | `pip install textblob` |
| 2 | VADER | `vaderSentiment` | Lexicon + rule-based | `pip install vaderSentiment` |
| 3 | Flair | `flair` | Neural LSTM (IMDB) | `pip install flair` |
| 4 | Zero-Shot | `transformers` | NLI zero-shot (HF) | `pip install transformers torch` |
| 5 | DistilBERT | `transformers` | Fine-tuned SST-2 | `pip install transformers torch` |
| 6 | NLTK | `nltk` | VADER wrapper | `pip install nltk` |

### Model Descriptions

**TextBlob** — Uses `PatternAnalyzer` to compute a polarity score from −1 to +1. Scores above 0.1 = positive, below −0.1 = negative, else neutral. Extremely lightweight with no GPU needed.

**VADER** — Valence Aware Dictionary and sEntiment Reasoner. Built for social media text. Handles punctuation, capitalization, and emoticons via a compound score. Very fast with no model download required.

**Flair** — Neural LSTM using character-level embeddings trained on IMDB reviews. Downloads the `en-sentiment` model on first run (~100MB). Binary classifier — neutral is inferred from confidence scores.

**Zero-Shot (HF)** — Uses `cross-encoder/nli-MiniLM2-L6-H768` from Hugging Face, a Natural Language Inference model repurposed for 3-class classification without any task-specific training.

**DistilBERT (HF)** — `distilbert-base-uncased-finetuned-sst-2-english`, a BERT model distilled to 40% smaller and fine-tuned on the Stanford Sentiment Treebank. Binary output; scores near 0.5 are mapped to neutral.

**NLTK** — NLTK's built-in `SentimentIntensityAnalyzer`, which wraps the same VADER lexicon through NLTK's own API. Separate package from `vaderSentiment` but nearly identical behaviour.

---

## 🛠 Environment Setup

> **Why a virtual environment?** This project installs PyTorch, Flair, and Hugging Face Transformers — all heavy packages with strict version dependencies. A `venv` keeps everything isolated from your system Python.

### Step 1 — Create the virtual environment

```bash
# Windows
python -m venv topsis_env

# macOS / Linux
python3 -m venv topsis_env
```

### Step 2 — Activate it

```bash
# Windows
topsis_env\Scripts\activate

# macOS / Linux
source topsis_env/bin/activate

# You should now see (topsis_env) in your terminal prompt
```

### Step 3 — Install all packages

```bash
pip install textblob vaderSentiment flair transformers torch nltk numpy pandas scikit-learn openpyxl
```

### Step 4 — Download additional model data

```bash
python -m textblob.download_corpora
python -m nltk.downloader vader_lexicon
```

### Step 5 — Save dependencies (optional but recommended)

```bash
pip freeze > requirements.txt
```

To recreate this environment on another machine:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
python ranker2.py
```

The script will:
- Run all 6 models sequentially (Flair + HF models download weights on first run)
- Print accuracy, F1, time, and memory for each model in the terminal
- Print TOPSIS rankings for all 3 weight scenarios
- Save a timestamped Excel report: `topsis_results_YYYYMMDD_HHMMSS.xlsx`

### Deactivate when done

```bash
deactivate
```

> **Note:** Models that are missing their package are **gracefully skipped** with an install hint — so you can run with a subset of packages installed.

---

## ⚙️ How the Code Works

The script runs in five sequential stages:

```
Stage 1 — Dataset
    25 labelled sentences (positive / negative / neutral) defined inline.
    No external download needed.
         ↓
Stage 2 — Model Evaluation
    Each run_*() function calls that library's own API.
    Inference time and peak memory are measured via tracemalloc.
         ↓
Stage 3 — Metrics Computation
    scikit-learn computes Accuracy, weighted F1, Precision, and Recall
    by comparing predictions to ground-truth labels.
         ↓
Stage 4 — TOPSIS Ranking
    topsis() normalizes the matrix, applies weights, and computes
    the closeness coefficient for each model × each weight scenario.
         ↓
Stage 5 — Excel Export
    openpyxl writes a colour-coded workbook: Summary sheet + one
    detailed sheet per weight configuration.
```

### Sentiment Mapping

All models output different raw formats. A `normalize_sentiment()` helper maps everything to a common label space:

| Raw Output | Mapped Label |
|---|---|
| `POSITIVE`, `pos`, `1` | `1` (Positive) |
| `NEGATIVE`, `neg`, `0` | `0` (Negative) |
| Anything else / low confidence | `2` (Neutral) |

---

## ⚖️ Weight Configurations

Three TOPSIS weight scenarios are evaluated out of the box:

| Criterion | Balanced | Accuracy-Priority | Speed-Priority | Direction |
|---|---|---|---|---|
| Accuracy | 30% | 45% | 5% | ↑ maximize |
| F1-Score (weighted) | 25% | 35% | 5% | ↑ maximize |
| Precision | 15% | 7% | 5% | ↑ maximize |
| Recall | 15% | 7% | 5% | ↑ maximize |
| Inference Time (s) | 8% | 3% | 45% | ↓ minimize |
| Peak Memory (MB) | 7% | 3% | 35% | ↓ minimize |

Weights are automatically normalized to sum to 1.0 internally.

---

## 📊 Results

### TOPSIS Sensitivity Analysis — All Weight Scenarios

| Model | Accuracy | F1-Score | Precision | Recall | Time | Memory | Balanced | Accuracy-Priority | Speed-Priority | Best Rank |
|---|---|---|---|---|---|---|---|---|---|---|
| **Zero-Shot (HF)** | 92.00% | 91.87% | 93.60% | 92.00% | 8.228s | 17.77MB | 🥇 1 (0.8951) | 🥇 1 (0.9505) | 5 (0.8603) | **1** |
| **VADER** | 76.00% | 73.63% | 83.14% | 76.00% | 0.098s | 2.70MB | 🥈 2 (0.7370) | 🥈 2 (0.5298) | 🥇 1 (0.9848) | **1** |
| **NLTK** | 76.00% | 73.63% | 83.14% | 76.00% | 0.460s | 2.44MB | 🥉 3 (0.7363) | 🥉 3 (0.5294) | 🥈 2 (0.9837) | **2** |
| **TextBlob** | 68.00% | 67.32% | 70.15% | 68.00% | 2.243s | 25.74MB | 4 (0.6259) | 4 (0.3749) | 4 (0.9256) | **4** |
| **DistilBERT (HF)** | 68.00% | 55.05% | 46.26% | 68.00% | 3.850s | 10.13MB | 5 (0.5515) | 5 (0.2880) | 🥉 3 (0.9267) | **3** |
| **Flair** | 68.00% | 56.37% | 49.47% | 68.00% | 47.494s | 234.68MB | 6 (0.0248) | 6 (0.0277) | 6 (0.0017) | **6** |

> TOPSIS Score shown in brackets — closer to 1.0 = better. Sorted by Balanced rank.

---

## 🔍 Key Findings

### 🥇 Zero-Shot (HF) — Best Overall Accuracy
With **92% accuracy** and **91.87% weighted F1**, the Hugging Face zero-shot model dominates any scenario that prioritises accuracy. It ranks **1st** in both the Balanced and Accuracy-Priority scenarios, achieving a near-perfect TOPSIS score of **0.9505** in the accuracy-focused run. This is remarkable given it requires **zero task-specific training** — it classifies by understanding the meaning of the label words directly.

### ⚡ VADER — Best Speed & Memory
At just **0.098 seconds** inference time and **2.70MB** peak memory, VADER is the most efficient model by a large margin. It ranks **1st** in the Speed-Priority scenario with a TOPSIS score of **0.9848**. When latency or resource constraints matter — such as in real-time apps or edge deployments — VADER is the clear winner and requires no GPU.

### 🔄 NLTK — Consistent Runner-Up for Speed
NLTK wraps the same VADER lexicon and achieves nearly identical accuracy (76%) and F1 (73.63%). It is slightly slower at 0.460s but uses marginally less memory (2.44MB). It ranks **2nd** in the Speed-Priority scenario and **3rd** in both other scenarios — a reliable, lightweight choice.

### ❌ Flair — Worst Performer
Flair ranks **last (6th) in every scenario**. Its LSTM model takes **47 seconds** to run and uses **234MB** of memory, yet achieves only 68% accuracy and a low F1 of 56.37%. This is largely because Flair's `en-sentiment` model is **binary** (Positive / Negative only) — it has no native neutral class, making it a poor fit for 3-class tasks.

### ⚠️ DistilBERT — Underperforms Despite Size
DistilBERT is a fine-tuned BERT model — theoretically one of the strongest models here — but ranks **5th** in the Balanced and Accuracy-Priority scenarios. Like Flair, it is **binary**, and the neutral-class heuristic (mapping low-confidence scores to neutral) severely hurts its weighted F1 on a 3-class task.

### 💡 Which Model Should You Use?

| Your Priority | Recommended Model | Reason |
|---|---|---|
| Best accuracy on 3-class sentiment | Zero-Shot (HF) | 92% Acc, 91.87% F1 — highest across all accuracy metrics |
| Fastest inference / real-time use | VADER | 0.098s inference, 2.70MB memory — no GPU needed |
| Lightweight + no internet required | TextBlob or NLTK | Pure rule-based, installs in seconds, works fully offline |
| Best balance of accuracy + speed | Zero-Shot (HF) | TOPSIS Score 0.8951 balanced — best overall |
| Lowest memory footprint | NLTK | 2.44MB peak memory — lowest of all models |

---

## 📁 Output Files

After running the script, an Excel file is generated:

```
topsis_results_YYYYMMDD_HHMMSS.xlsx
```

| Sheet | Contents |
|---|---|
| `Summary` | All models × all raw metrics + TOPSIS Rank for every scenario side by side, with 🥇🥈🥉 medal colours |
| `Balanced` | Full TOPSIS detail: raw scores, S⁺, S⁻, TOPSIS Score, and Rank for the balanced scenario |
| `Accuracy-Priority` | Full TOPSIS detail for the accuracy-focused scenario |
| `Speed-Priority` | Full TOPSIS detail for the speed/memory-focused scenario |

---

## 🔧 Customising the Analysis

### Add a new weight scenario

Add a new entry to `WEIGHT_CONFIGS` at the top of the script:

```python
WEIGHT_CONFIGS["My Scenario"] = {
    "weights": {
        "Accuracy":            0.20,
        "F1-Score (weighted)": 0.20,
        "Precision":           0.20,
        "Recall":              0.20,
        "Inference Time (s)":  0.10,
        "Peak Memory (MB)":    0.10,
    },
    "description": "My custom scenario description",
}
```

A new sheet will automatically appear in the Excel output.

### Add a new model

1. Write a `run_yourmodel(texts)` function that returns a list of `0`, `1`, or `2` labels
2. Add it to the `MODELS` dictionary with its required packages:

```python
MODELS["My Model"] = (run_yourmodel, ["your-package"])
```

That's it — evaluation, TOPSIS ranking, and Excel export are handled automatically.

---

## 📦 Full Requirements

```
textblob
vaderSentiment
flair
transformers
torch
nltk
numpy
pandas
scikit-learn
openpyxl
```

Install all at once:

```bash
pip install textblob vaderSentiment flair transformers torch nltk numpy pandas scikit-learn openpyxl
```

---

## 📄 License

MIT License — free to use, modify, and distribute.
