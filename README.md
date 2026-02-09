
# Multi-Dimensional Threshold Benchmark (NP-Style)

A framework for benchmarking **classification and decision rules under a Neyman–Pearson (NP) constraint** (fixing $FPR \approx \alpha$ to maximize $TPR$) in **high-dimensional embedding spaces**.

The project compares classical ML models against **vector-threshold**, **M-of-N**, and **AND-box** rules, with explicit control over false-positive rates and inference latency.

---

## 🎯 Problem Setting

**Objective:** Distinguish between two classes under a strict False Positive Rate constraint.

* **Hypotheses:**
    * $H_0$: Negative / Non-duplicate
    * $H_1$: Positive / Duplicate
* **Constraint:** Target False-Positive Rate (FPR) $\alpha$.

**The Evaluation Protocol:**
1.  **Calibration:** Thresholds are learned using **$H_0$ data only**.
2.  **Constraint:** Enforce $FPR \le \alpha$ on the validation set.
3.  **Measurement:** Evaluate **TPR**, **FPR**, and **Inference Time (ms)** on the test set.

---

## 📐 Feature Representation

The input consists of paired embeddings $(U, V)$ where $U, V \in \mathbb{R}^d$.

**Preprocessing pipeline:**
1.  **L2 Normalization:** Normalize each embedding to unit length.
2.  **Hadamard Features:** Construct the element-wise product feature vector $X$:
    $$X = U \odot V$$

**Key Property:**
The sum of the Hadamard features equals the cosine similarity, providing a strict baseline within the same feature space:
$$\sum_{i=1}^{d} X_i = \text{cosine}(U, V)$$

---

## 📂 Repository Structure

```text
Multi_Dim_Threshold/
│
├── np_bench/
│   ├── data/          # Data loading, Hadamard construction, & balancing
│   ├── utils/         # Metrics, timing decorators, plotting, IO
│   ├── methods/       # Decision rules & model implementations
│   └── experiments/   # Reproducible experiment logic
│
├── outputs/           # Auto-generated artifacts (Plots, CSVs, Logs)
└── README.md

```

### Key Modules

* **`np_bench/methods`**: Contains isolated implementations of each algorithm. All methods adhere to a unified interface:
* `fit(H0, H1, alpha, ...)`  `predict`
* Returns: `(TPR, FPR, time_ms)`


* **`np_bench/experiments`**: Self-contained experiment scripts (no method-specific logic). Handles sweeping, logging, and visualization.

---

## 🚀 Supported Methods

### Baselines

* **Cosine**: Sum of Hadamard features (standard similarity).
* **Vec (Wgt)**: Linear score utilizing Fisher weights.

### Classical ML

* **Naive Bayes**: Gaussian NB with Log-Likelihood Ratio (LLR) scoring.
* **Logistic Regression**: Standard linear classification.
* **LDA**: Linear Discriminant Analysis with shrinkage (optimized for high-dim).
* **XGBoost (Light)**: Shallow trees, single-thread forced (for low-latency simulation).
* **Tiny MLP**: Lightweight fully connected neural network.

### Custom Decision Rules

* **M-of-N (Weighted)**: Optimizes per-dimension thresholds for weighted voting logic.
* **AND-Box (HC)**: Sparse AND-box optimization via Hill Climbing.
* **AND-Box (Weighted)**: Weighted dimension selection variant of the AND-box.

---

## 🧪 Experiments

To reproduce results, use the module syntax `python -m np_bench...`.

### 1. Dimension Sweep (`dims_sweep`)

Evaluates how performance scales with feature dimension .

* **Setup:** Fixed sample size (), sweep .
* **Execution:**
```bash
python -m np_bench.experiments.dims_sweep.run \
  --pkl np_bench/data/quora_question_pairs_with_embeddings.pkl \
  --n_samples 1000 \
  --alpha 0.05 \
  --n_trials 3

```



### 2. Sample Size Sweep (`n_sweep`)

Evaluates robustness relative to the number of training samples.

* **Setup:** Fixed dimension (), sweep .
* **Execution:**
```bash
python -m np_bench.experiments.n_sweep.run \
  --pkl np_bench/data/quora_question_pairs_with_embeddings.pkl \
  --d 1024 \
  --n_list 100,200,500,1000,2000 \
  --alpha 0.05 \
  --n_trials 3

```



---

## 📊 Output Artifacts

Experiments automatically generate the following in the `outputs/` directory:

| File | Description |
| --- | --- |
| `results.csv` | Raw metrics for every trial (see schema below). |
| `summary.json` | Metadata, full config, and aggregate stats. |
| `*_tpr_final.png` | Visualization of TPR vs. Variable (Dim/N). |
| `*_time_final.png` | Visualization of Inference Time vs. Variable. |

**CSV Schema:**
`experiment`, `trial`, `seed`, `d`, `n_samples_per_class`, `alpha`, `method`, `tpr`, `fpr`, `time_ms`

---

## ✅ Design Principles & Status

* **Strict NP Calibration:** Thresholds are derived strictly from null-hypothesis () data.
* **No Data Leakage:** rigorous train/test splits.
* **Latency Aware:** Explicit inference-time measurement included in benchmarks.
* **Reproducibility:** Seed control and configuration serialization.

