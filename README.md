# NP-Calibrated Pairwise Similarity

A research-grade framework for **pairwise semantic similarity / duplicate detection**
under a **Neyman–Pearson (NP) constraint**, with explicit control over
**false positive rate (FPR)**.

The framework is designed for **high-dimensional embedding features**
and evaluates classical ML models alongside custom
**multi-dimensional thresholding rules**, using a clean
**train / calibration / evaluation** protocol.

---

## Motivation

In pairwise semantic tasks (e.g. duplicate question detection, semantic cache reuse),
**false positives are costly**:
- wrong cache hits
- incorrect reuse
- semantic drift

Instead of maximizing accuracy or AUC, this project enforces:

> **FPR ≤ α**, then maximizes **TPR**

This matches real-world decision systems far better than unconstrained classifiers.

---

## Problem Setup

Given:
- Paired inputs `(x₁, x₂)`
- Binary label:
  - `H0` – non-duplicate
  - `H1` – duplicate
- Target false-positive rate `α`

We evaluate decision rules that:
1. Learn parameters from **training data**
2. Calibrate an NP threshold using **negative calibration data only**
3. Are evaluated on a **held-out test set**

Metrics:
- TPR (recall on H1)
- FPR (measured on eval, not assumed)
- Inference time (ms)

---

## Feature Representation

Each input pair contains embeddings `(U, V)`.

Processing pipeline:
1. L2-normalize embeddings
2. Construct **Hadamard features**:
```

X = U ⊙ V

```
3. Property:
```

sum(X) = cosine(U, V)

````

This makes cosine similarity a strict baseline within the same feature space.

---

## Dataset Format

Input is a `.pkl` file containing either:
- A list of samples, or
- A dict with a split key (default: `train`)

Each sample is a dict:
```python
{
"q1_emb": np.ndarray,
"q2_emb": np.ndarray,
"is_duplicate": int  # {0,1}
}
````

---

## Dataset Download

The dataset (`quora_question_pairs_with_embeddings.pkl`) is hosted on HuggingFace.
It is **~10 GB** and is not tracked by git.

### Option 1: Python (`huggingface_hub`)

```bash
pip install huggingface_hub
```

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="malmatal/quora_question_pairs_with_embeddings",
    filename="quora_question_pairs_with_embeddings.pkl",
    repo_type="dataset",
    local_dir="np_bench/data",
)
```

### Option 2: HuggingFace CLI

```bash
pip install huggingface_hub
huggingface-cli download \
    malmatal/quora_question_pairs_with_embeddings \
    quora_question_pairs_with_embeddings.pkl \
    --repo-type dataset \
    --local-dir np_bench/data
```

### Option 3: `datasets` library

```bash
pip install datasets
```

```python
from datasets import load_dataset

ds = load_dataset("malmatal/quora_question_pairs_with_embeddings")
```

### Option 4: `wget` / direct URL

```bash
wget -P np_bench/data/ \
    https://huggingface.co/datasets/malmatal/quora_question_pairs_with_embeddings/resolve/main/quora_question_pairs_with_embeddings.pkl
```

### Option 5: Git LFS

```bash
git lfs install
git clone https://huggingface.co/datasets/malmatal/quora_question_pairs_with_embeddings np_bench/data/hf_repo
mv np_bench/data/hf_repo/quora_question_pairs_with_embeddings.pkl np_bench/data/
rm -rf np_bench/data/hf_repo
```

After downloading, the experiments expect the file at:
```
np_bench/data/quora_question_pairs_with_embeddings.pkl
```

---

## Repository Structure

```
np_bench/
│
├── data/
│   └── quora_embeddings.py      # loading + feature construction
│
├── utils/
│   ├── split.py                 # train / calib / eval splitting
│   ├── metrics.py               # NP metrics
│   ├── timing.py                # inference timing
│   ├── fisher.py                # Fisher score feature ranking
│   ├── plotting.py
│   └── io.py
│
├── methods/
│   ├── cosine.py
│   ├── weighted_vector.py
│   ├── naive_bayes.py
│   ├── logistic_regression.py
│   ├── lda.py
│   ├── xgboost.py
│   ├── tiny_mlp.py
│   ├── andbox.py                # AND-box rules
│   └── base.py                  # BaseMethod + NP calibration
│
└── experiments/
    ├── dims_sweep/              # sweep feature dimension
    └── n_sweep/                 # sweep sample size
```

---

## Evaluation Protocol (Important)

Each trial uses **three disjoint sets per class**:

* **Train**

  * Fit model parameters
  * Learn weights / rules
* **Calibration**

  * Contains **H0 only** for NP threshold calibration
* **Evaluation**

  * Final measurement of TPR / FPR / time

⚠️ **No threshold is ever tuned on evaluation data.**
⚠️ **NP calibration uses negatives only.**

This avoids data leakage and makes FPR claims meaningful.

---

## Methods Evaluated

### Baselines

* **Cosine** – sum of Hadamard features
* **Vector-Weighted** – linear score using Fisher weights

### Classical ML

* **Naive Bayes** (Gaussian, LLR scoring)
* **Logistic Regression**
* **LDA** (shrinkage, high-dimensional safe)
* **XGBoost (Light)**
* **Tiny MLP**

### Custom Decision Rules

* **AndBox-HC** – sparse AND-box via hill climbing
* **AndBox-Wgt** – weighted dimension selection

All methods are calibrated to the same target FPR.

---

## Experiments

### 1️⃣ Dimension Sweep (`dims_sweep`)

**Goal:**
Measure robustness and scaling as feature dimension increases.

**Setup:**

* Fixed samples per class
* Sweep `d ∈ {8,16,32,64,128,256,512,1024}`

**Run:**

```bash
python -m np_bench.experiments.dims_sweep.run \
  --pkl np_bench/data/quora_question_pairs_with_embeddings.pkl \
  --n_train 400 \
  --n_calib 400 \
  --n_eval 2000 \
  --alpha 0.05 \
  --n_trials 3
```

**Optional (sweep one split while sweeping d):**

```bash
python -m np_bench.experiments.dims_sweep.run \
  --pkl np_bench/data/quora_question_pairs_with_embeddings.pkl \
  --sweep train \
  --n_list 100,200,500,1000 \
  --n_calib 400 \
  --n_eval 2000 \
  --alpha 0.05 \
  --n_trials 3
```

Outputs:

* `benchmark_tpr_final.png`
* `benchmark_fpr_final.png`
* `benchmark_train_tpr.png`
* `benchmark_train_fpr.png`
* `benchmark_time_final.png`
* `results.csv`
* `summary.json`

---

### 2️⃣ Sample Size Sweep (`n_sweep`)

**Goal:**
Measure performance vs. available data at fixed dimension.

**Setup:**

* Fixed `d` (default: 1024)
* Sweep number of samples per class

**Run (no sweep, fixed sizes):**

```bash
python -m np_bench.experiments.n_sweep.run \
  --pkl np_bench/data/quora_question_pairs_with_embeddings.pkl \
  --d 1024 \
  --sweep none \
  --n_train 400 \
  --n_calib 400 \
  --n_eval 2000 \
  --alpha 0.05 \
  --n_trials 3
```

**Run (sweep one split):**

```bash
python -m np_bench.experiments.n_sweep.run \
  --pkl np_bench/data/quora_question_pairs_with_embeddings.pkl \
  --d 1024 \
  --sweep train \
  --n_list 100,200,500,1000,2000 \
  --n_calib 400 \
  --n_eval 2000 \
  --alpha 0.05 \
  --n_trials 3
```

**Run (sweep total per class and split by fractions):**

```bash
python -m np_bench.experiments.n_sweep.run \
  --pkl np_bench/data/quora_question_pairs_with_embeddings.pkl \
  --d 1024 \
  --sweep total \
  --n_list 200,500,1000,2000 \
  --train_frac 0.2 \
  --calib_frac 0.2 \
  --alpha 0.05 \
  --n_trials 3
```

Parameters:
* `--sweep` – which quantity `n_list` controls: `none|train|calib|eval|total`
* `--n_train` – samples per class in training set (used when not swept)
* `--n_calib` – samples per class in calibration set (H0 only; used when not swept)
* `--n_eval` – samples per class in evaluation set (used when not swept)
* `--train_frac` – fraction of total for training (only with `--sweep total`)
* `--calib_frac` – fraction of total for calibration (only with `--sweep total`)

---

## Output Format

### `results.csv`

One row per:

* trial
* method
* configuration

Includes:

* `tpr`
* `fpr`
* `train_tpr`
* `train_fpr`
* `time_ms`

### `summary.json`

Stores:

* experiment configuration
* dataset reference
* NP parameters

---

## Design Principles

* Explicit NP calibration
* No evaluation leakage
* High-dimensional safe methods
* Inference-time measured separately
* Modular, reproducible experiments

---

## Intended Use

* Pairwise semantic similarity
* Duplicate detection
* Semantic cache admission
* Risk-controlled reuse in RAG / LLM systems

---

## Status

✔ Modularized
✔ NP-correct protocol
✔ Reproducible
✔ Ready for research / GitHub / extension

