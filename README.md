# NP-Calibrated Pairwise Similarity

A research-grade framework for **pairwise semantic similarity / duplicate detection**
under a **Neyman–Pearson (NP) constraint**, with explicit control over
**false positive rate (FPR)**.

The framework is designed for **high-dimensional embedding features** and
evaluates classical ML models alongside custom multi-dimensional thresholding
rules using a clean **train / calibration / evaluation** protocol.

---

## Motivation

In pairwise semantic tasks such as duplicate question detection and semantic
cache reuse, **false positives are costly**:

- wrong cache hits
- incorrect response reuse
- semantic drift
- unsafe cache admission

Instead of maximizing accuracy or AUC, this project enforces:

> **FPR ≤ α**, then maximizes **TPR**.

This matches real-world decision systems better than unconstrained classifiers,
especially when false positives correspond to unsafe reuse.

---

## Problem Setup

Given:

- paired inputs `(x₁, x₂)`
- binary label:
  - `H0`: non-duplicate / non-reusable pair
  - `H1`: duplicate / reusable pair
- target false-positive rate `α`

we evaluate decision rules that:

1. learn parameters from **training data**
2. calibrate an NP threshold using **negative calibration data**
3. are evaluated on a **held-out test set**

Primary metrics:

- **TPR**: recall on `H1`
- **FPR**: false-positive rate on held-out `H0`
- **TrainN**: number of train/cache samples used
- **Inference time**: measured separately

The main constrained objective is:

```text
maximize TPR subject to FPR <= α
```

---

## Feature Representation

Each input pair contains embeddings `(U, V)`.

The default processing pipeline is:

1. L2-normalize embeddings
2. construct Hadamard features:

```text
X = U ⊙ V
```

3. use the identity:

```text
sum(X) = cosine(U, V)
```

This makes cosine similarity a strict baseline within the same feature space.

---

## Dataset Format

For the original `np_bench` experiments, the expected dataset is a `.pkl` file
containing either:

- a list of samples, or
- a dict with a split key, defaulting to `train`

Each sample is a dict:

```python
{
    "q1_emb": np.ndarray,
    "q2_emb": np.ndarray,
    "is_duplicate": int  # {0, 1}
}
```

For the `NeighborCache.region_local_threshold` experiments, the expected dataset
is an `.npz` file containing embeddings, labels, and region keys.

Example:

```bash
NeighborCache/data/h1h0_final.npz
```

---

## Dataset Download

The Quora embedding dataset used by the original benchmark
(`quora_question_pairs_with_embeddings.pkl`) is hosted on HuggingFace.

It is approximately **10 GB** and is not tracked by git.

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

```text
np_bench/data/quora_question_pairs_with_embeddings.pkl
```

---

## Repository Structure

```text
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

The region-local threshold experiments are located under:

```text
NeighborCache/region_local_threshold/
```

---

## Evaluation Protocol

Each trial uses disjoint data splits.

### Train

Used to fit method parameters:

- linear weights
- whitening transforms
- LDA directions
- MLP/XGBoost models
- ensemble components
- thresholding rules

### Calibration

Used to select an NP threshold.

Threshold calibration is performed using negative calibration samples, i.e. `H0`,
so that the target FPR constraint is controlled independently from training.

### Evaluation

Used only for final measurement.

Reported evaluation metrics are measured on held-out data:

- eval TPR
- eval FPR
- macro TPR/FPR over regions
- inference time
- selected train/cache size

Important:

```text
No threshold is tuned on evaluation data.
Evaluation data is not used for train-dosage selection.
Monitor data is used only for stopping/dosage selection.
```

---

## Methods Evaluated

### Baselines

- **Cosine**: sum of Hadamard features
- **HadamardCosine**: cosine-equivalent Hadamard scorer
- **Vector-Weighted**: linear score using weighted dimensions

### Classical ML

- **Naive Bayes**
- **Logistic Regression**
- **LDA**
- **XGBoost**
- **Tiny MLP**

### Custom Decision Rules

- **AndBox-HC**
- **AndBox-Wgt**
- **WhitenedCosine**
- **StabilizedWhitenedCosine**
- **WeightedEnsemble**

### Ablations

The framework also includes several ablation methods, including:

- raw Hadamard linear
- center-only Hadamard linear
- PCA-whitened Hadamard linear
- diagonal-whitened Hadamard linear
- H0-only PCA whitening
- H1-only PCA whitening
- within-class PCA whitening
- no-rank-truncation PCA whitening
- shuffled-label control

All methods are calibrated to the same target FPR.

---

## Original `np_bench` Experiments

### Dimension Sweep

Goal:

Measure robustness and scaling as feature dimension increases.

Setup:

```text
d ∈ {8,16,32,64,128,256,512,1024}
```

Run:

```bash
python -m np_bench.experiments.dims_sweep.run \
  --pkl np_bench/data/quora_question_pairs_with_embeddings.pkl \
  --n_list 500,1000,2000 \
  --train_frac 0.5 \
  --eval_frac 0.5 \
  --alpha 0.05 \
  --n_trials 3
```

Outputs:

- `benchmark_tpr_final.png`
- `benchmark_fpr_final.png`
- `benchmark_train_tpr.png`
- `benchmark_train_fpr.png`
- `benchmark_time_final.png`
- `results.csv`
- `summary.json`

---

### Sample Size Sweep

Goal:

Measure performance as a function of available data at fixed dimension.

Run:

```bash
python -m np_bench.experiments.n_sweep.run \
  --pkl np_bench/data/quora_question_pairs_with_embeddings.pkl \
  --d 1024 \
  --n_list 200,500,1000,2000 \
  --train_frac 0.5 \
  --eval_frac 0.5 \
  --alpha 0.05 \
  --n_trials 3
```

---

## Region-Local Threshold Benchmark

Basic example:

```bash
python -m NeighborCache.region_local_threshold.cli \
  --data NeighborCache/data/h1h0_final.npz \
  --region_key global_cluster \
  --alpha 0.05 \
  --tau_mode global \
  --n_trials 5 \
  --seed 42 \
  --n_train 1270 \
  --n_calib 1240 \
  --n_eval 1270 \
  --hadamard_preprocess
```

This benchmark evaluates methods under a fixed-FPR constraint using
region-aware train/calibration/evaluation splits.

---

## Output Format

### `trial_summary.csv`

One row per:

- trial
- method
- configuration

Important columns:

- `method`
- `trial`
- `seed`
- `train_n`
- `tpr`
- `fpr`
- `train_tpr`
- `train_fpr`
- `macro_tpr`
- `macro_fpr`
- `time_ms`
- `ok`

### `ranking.json`

Stores final ranking information, including:

- constrained ranking
- unconstrained ranking
- safety-first ranking
- cache-efficiency ranking
- selected method metadata

### `train_dosage_search.csv`

One row per:

- trial
- method
- dosage candidate

Important columns:

- `trial`
- `seed`
- `method`
- `train_dosage_total`
- `train_h0`
- `train_h1`
- `monitor_tpr`
- `monitor_fpr`
- `alpha`
- `train_dosage_fpr_margin`
- `fpr_feasible`
- `selected`
- `selection_reason`
- `best_feasible_monitor_tpr`
- `tpr_gap_from_best_feasible`
- `train_dosage_tpr_tolerance`
- `train_dosage_max_train`
- `train_dosage_no_auto_full`

---

## Design Principles

- Explicit NP-style calibration
- No evaluation leakage
- Train/calibration/evaluation separation
- High-dimensional safe scoring methods
- Cache-size-aware train-dosage selection
- FPR-first reporting
- Inference-time measured separately
- Reproducible CLI-driven experiments

---

## Intended Use

- Pairwise semantic similarity
- Duplicate detection
- Semantic cache admission
- Risk-controlled LLM response reuse
- RAG reuse filtering
- Low-FPR cache-hit validation

---

## Status

```text
Modularized
NP-style protocol
Region-local benchmark supported
Hadamard preprocessing supported
Train-dosage calibration supported
Online stopping supported
Optuna stopping tuner supported
Ready for research / GitHub / extension
```

---

# Final Cache-Capped Train-Dosage Calibration

This section describes the final cache-size calibration protocol used for the
NeighborCache region-local threshold benchmark.

The goal is to select the smallest practical training/cache dosage that gives
strong reuse performance while keeping the empirical false-positive rate close
to the target Neyman–Pearson constraint:

```text
alpha = 0.05
```

The final result should be interpreted as an empirical cache-size calibration
under a target FPR constraint, not as a formal per-seed FPR guarantee.

---

## Final Calibration Command

```bash
python -m NeighborCache.region_local_threshold.cli \
  --data NeighborCache/data/h1h0_final.npz \
  --region_key global_cluster \
  --alpha 0.05 \
  --tau_mode global \
  --n_trials 10 \
  --seed 49 \
  --n_train 5270 \
  --n_calib 5240 \
  --n_eval 5270 \
  --hadamard_preprocess \
  --enable_train_dosage_search \
  --train_dosage_grid 800,1200,1600,2000,2540,4000 \
  --train_dosage_no_auto_full \
  --train_dosage_max_train 4000 \
  --train_dosage_tpr_tolerance 0.01 \
  --train_dosage_fpr_margin 0.0075 \
  --n_monitor_h0 2000 \
  --n_monitor_h1 300 \
  --run_name final_train_dosage_cap4000_margin0075_monitor2000
```

---

## Why These Settings Are Final

### `--train_dosage_grid 800,1200,1600,2000,2540,4000`

Defines the candidate cache/training sizes.

Very small dosages such as `400` were removed from the final protocol because
they were too unstable for the main calibration claim.

### `--train_dosage_no_auto_full`

Prevents the CLI from automatically appending the full training set to the
dosage grid.

This is required because the experiment is about cache-size calibration. If the
full train size is automatically added, strong methods may select the full
cache, which defeats the purpose of the capped-dosage experiment.

### `--train_dosage_max_train 4000`

Hard-caps the selected train/cache dosage.

No method may select more than:

```text
4000 training examples
```

### `--train_dosage_tpr_tolerance 0.01`

Allows the selector to choose the smallest dosage whose monitor TPR is within
one percentage point of the best feasible monitor TPR.

This trades a small amount of TPR for a smaller cache.

### `--train_dosage_fpr_margin 0.0075`

Uses a stricter monitor-side FPR requirement:

```text
monitor_fpr <= alpha - margin
monitor_fpr <= 0.05 - 0.0075 = 0.0425
```

This margin is needed because monitor FPR close to `0.05` may drift above the
target on held-out evaluation.

### `--n_monitor_h0 2000`

Uses a large H0 monitor set because FPR stability is the main safety concern.

### `--n_monitor_h1 300`

Uses a smaller H1 monitor set for TPR-based dosage selection.

---

## Train-Dosage Selection Rule

For each method and trial, the selector evaluates all candidate train dosages.

First, it identifies FPR-feasible candidates:

```text
monitor_fpr <= alpha - train_dosage_fpr_margin
```

Then it computes:

```text
best_feasible_monitor_tpr = max(monitor_tpr over feasible candidates)
```

Finally, it selects the smallest train dosage satisfying:

```text
monitor_tpr >= best_feasible_monitor_tpr - train_dosage_tpr_tolerance
```

If no dosage satisfies the FPR-feasibility condition, the selector falls back to:

```text
fallback_lowest_monitor_fpr_no_fpr_safe_candidate
```

Fallback rows are useful for diagnostics, but they should not be interpreted as
strictly FPR-safe calibrated selections.

---

## Final Practical Result

The final practical result is:

```text
WeightedEnsemble is approximately calibrated around alpha = 0.05.
```

WeightedEnsemble provides the strongest practical TPR/cache-size trade-off among
the evaluated cache-capped configurations.

The correct interpretation is:

```text
WeightedEnsemble is approximately calibrated around α = 0.05 and provides the
best practical cache-capped reuse performance.
```

Do **not** state:

```text
WeightedEnsemble guarantees FPR <= 0.05 for every seed.
```

The method is near-target calibrated in the empirical sense, but it is not a
strict per-seed FPR guarantee.

---

## Recommended Reporting Language

Use this wording in papers, reports, or experiment summaries:

> Under a cache-capped train-dosage calibration protocol, WeightedEnsemble
> achieved the strongest practical TPR/cache-size trade-off while remaining
> approximately calibrated around the target FPR level α = 0.05.

A shorter version:

> WeightedEnsemble is approximately calibrated around α = 0.05 and provides the
> best practical cache-capped reuse performance.

---

## Final Decision Criterion

For strict safety analysis, rank methods by:

```text
1. valid_rate
2. mean_eval_fpr
3. max_eval_fpr
4. mean_eval_tpr
5. mean_train_n
```

For the practical cache-efficiency result, rank methods by:

```text
1. empirical FPR close to alpha = 0.05
2. highest mean_eval_tpr
3. lowest mean_train_n
```

Using the practical criterion, WeightedEnsemble is the selected final method.

---

## Notes on Interpretation

This protocol is an empirical calibration procedure.

It is intended to answer:

```text
How much cache/training data is enough to obtain strong reuse performance while
remaining approximately calibrated around the target FPR?
```

It is not intended to prove a formal Neyman–Pearson guarantee.

The final calibrated setup is:

```text
alpha = 0.05
max_train = 4000
train_dosage_grid = 800,1200,1600,2000,2540,4000
train_dosage_tpr_tolerance = 0.01
train_dosage_fpr_margin = 0.0075
n_monitor_h0 = 2000
n_monitor_h1 = 300
selected practical method = WeightedEnsemble
```

---

# Online Stopping CLI

The online stopping path is separate from the train-dosage calibration protocol.

Use online stopping when the goal is to decide when an online stream has provided
enough data for a model update.

Use train-dosage calibration when the goal is to choose a cache-size budget for
each method.

Example online stopping command:

```bash
python -m NeighborCache.region_local_threshold.cli \
  --data NeighborCache/data/h1h0_final.npz \
  --region_key global_cluster \
  --alpha 0.05 \
  --tau_mode global \
  --seed 42 \
  --n_train 1200 \
  --n_calib 1200 \
  --n_eval 1200 \
  --hadamard_preprocess \
  --enable_online_stopping \
  --online_init_h0 200 \
  --online_init_h1 200 \
  --online_batch_size 25 \
  --online_mem_cap 1200 \
  --online_update_mode reservoir \
  --online_hill_lr 0.05 \
  --n_monitor_h0 300 \
  --n_monitor_h1 300 \
  --stop_check_every 1 \
  --stop_window 3 \
  --stop_patience 2 \
  --stop_eps_tpr 0.005 \
  --stop_eps_fpr 0.003 \
  --stop_eps_tau 0.01 \
  --stop_fpr_margin 0.005
```

---

# Optuna Stopping Tuner

`NeighborCache.region_local_threshold.optuna_stopping` is an offline tuning
utility for the existing `OnlineStopper` / `StopConfig` rule.

It should be used as an offline validation or refresh tool, not in the live cache
request path.

The tuner samples stopping and online-update hyperparameters, runs the existing
online stopping simulation, freezes the resulting model and threshold, and then
evaluates on held-out eval samples after monitor samples have been removed.

Example:

```bash
python -m NeighborCache.region_local_threshold.optuna_stopping \
  --data NeighborCache/data/h1h0_final.npz \
  --region_key global_cluster \
  --alpha 0.05 \
  --tau_mode global \
  --n_trials 100 \
  --seed 42 \
  --n_train 1270 \
  --n_calib 1240 \
  --n_eval 1270 \
  --hadamard_preprocess \
  --selection_mode strict_np \
  --out_dir NeighborCache/outputs/optuna_stopping \
  --study_name np_stopping_alpha_005 \
  --storage sqlite:///NeighborCache/outputs/optuna_stopping/study.db
```

Smoke check:

```bash
python -m NeighborCache.region_local_threshold.optuna_stopping \
  --data NeighborCache/data/h1h0_final.npz \
  --region_key global_cluster \
  --alpha 0.05 \
  --tau_mode global \
  --n_trials 2 \
  --eval_seeds 0 \
  --seed 42 \
  --n_train 80 \
  --n_calib 40 \
  --n_eval 80 \
  --hadamard_preprocess \
  --selection_mode strict_np \
  --out_dir NeighborCache/outputs/optuna_stopping_smoke
```

---

# vCache Online Comparison

The vCache comparison lives under `experiments/` and evaluates every method as
an online semantic cache policy:

```text
stream of prompts -> retrieve nearest cached prompt -> decide hit/exploit or
miss/explore -> update cache metadata/cache entries online
```

vCache is evaluated as an online cache, not as a pairwise H0/H1 classifier.
The official vCache policy controls marginal stream-level error, reported here
as `FP / n`, while maximizing cache hit rate.

NeighborCache methods are trained and calibrated as pairwise reuse scorers over
H0/H1 examples. They primarily control reuse-decision false positives and
precision over retrieved candidates. Because these are different error-control
families, the comparison reports both metric families:

- vCache primary metric: stream-level error `error_rate_stream = FP / n`
- NeighborCache primary metrics: reuse-decision `false_positive_rate` and
  `precision`
- fair primary comparison: highest hit rate at matched empirical stream-level
  error `FP / n`
- secondary analyses: TPR/FPR and precision/hit-rate tradeoffs over reuse
  decisions

The evaluator fixes the stream order, seed, cache capacity, embedding column,
LLM response source, and equivalence function across all requested methods.
It writes one raw row per request per method/parameter setting, plus one summary
row per full online run.

Example:

```bash
python -m experiments.compare_vcache \
  --dataset path/to/dataset.jsonl \
  --output_dir results/vcache_comparison \
  --methods vcache ours cosine \
  --delta_values 0.01 0.02 0.03 0.05 0.08 \
  --thresholds 0.80 0.85 0.90 0.93 0.95 0.97 0.98 0.99 0.995 0.999 \
  --seed 42 \
  --cache_size 4096 \
  --ours_method WeightedEnsemble \
  --ours_pairwise_data NeighborCache/data/h1h0_final_hadamard.npz \
  --ours_pairwise_alt_feature_key cosine_to_anchor
```

Matched-budget postprocessing:

```bash
python -m experiments.summarize_vcache_comparison \
  --input_dir results/vcache_comparison \
  --budgets 0.01 0.02 0.03 0.05
```

Outputs:

```text
results/vcache_comparison/
  raw_decisions.csv
  summary_metrics.csv
  summary_metrics.json
  matched_budget_summary.csv
  config.json
  plots/
```

The vCache adapter first imports the official `vcache` package at runtime and
uses the official `VCache` and `VerifiedDecisionPolicy`/benchmark policy
classes. When `--vcache_repo_path` is provided and direct import is blocked by
optional top-level backends that are not used in the offline experiment, it loads
only the needed official vCache core/policy source files from that repository.
If the active Python environment cannot parse or load those official source
files, the run stops with the exact missing dependency/API error. It does not
replace vCache with a hand-written imitation.

For deterministic offline experiments with precomputed responses, the adapter
injects precomputed embeddings/responses through benchmark-style engines and
uses a thin metadata-storage wrapper to log nearest prompt ids/text. By default
it also runs vCache background updates synchronously so per-request metadata is
complete and reproducible; pass `--vcache_async_updates` to keep the official
asynchronous update behavior.

---

# vCache Real-Dataset Comparison

Banking77 is only a sanity-check dataset in this repository. It is not used as a
main semantic-cache benchmark because intent labels make it too easy for static
cosine similarity.

The real comparison entry point is:

```bash
python -m experiments.compare_vcache_real_datasets \
  --datasets SemCacheLMArena SemCacheSearchQueries \
  --output_dir results/vcache_real_datasets_comparison \
  --methods vcache cosine ours_whitened_hadamard ours_weighted_ensemble \
  --delta_values 0.01 0.02 0.03 0.05 0.08 \
  --thresholds 0.80 0.85 0.90 0.93 0.95 0.97 0.98 0.99 0.995 0.999 \
  --target_budgets 0.01 0.02 0.03 0.05 0.08 \
  --hard_neighbor_thresholds 0.80 0.85 0.90 \
  --embedding_model GTE \
  --seed 42 \
  --cache_size 4096
```

Matched-budget postprocessing:

```bash
python -m experiments.summarize_vcache_real_datasets \
  --input_dir results/vcache_real_datasets_comparison \
  --budgets 0.01 0.02 0.03 0.05
```

This runner targets the official vCache semantic-cache workloads:

- `SemCacheLMArena`, backed by `vCache/SemBenchmarkLmArena`
- `SemCacheSearchQueries`, backed by `vCache/SemBenchmarkSearchQueries`

It evaluates each workload as:

- the full deterministic stream
- hard-neighbor subsets with dry-run nearest-neighbor cosine `>= 0.80`,
  `>= 0.85`, and `>= 0.90`

vCache is still evaluated as an online cache, not as a pairwise classifier. It
controls marginal stream-level error `FP / n`. The learned methods control
pairwise reuse false positives and precision over retrieved candidates, using
Hadamard features `E(query) * E(candidate)`. The paper should therefore report
both metric families. The primary fair comparison is hit rate at matched
empirical stream-level error `FP / n`; secondary analysis reports
precision/hit-rate and TPR/FPR on hard nearest-neighbor subsets.

The runner writes:

```text
results/vcache_real_datasets_comparison/
  config.json
  dataset_stats.csv
  hard_subset_stats.csv
  raw_decisions.csv
  summary_metrics.csv
  summary_metrics.json
  matched_budget_summary.csv
  pairwise_cache/
  plots/
```

Embedding-model matching is enforced. For example, do not use BGE pairwise data
with official GTE stream embeddings. If `--embedding_model BAAI/bge-large-en-v1.5`
is requested but the official stream does not provide a BGE embedding column,
the runner stops instead of silently remapping it to GTE.

## Online-Mined Calibration For Learned Methods

The initial learned-pairwise vCache comparison calibrated
`ours_whitened_hadamard` and `ours_weighted_ensemble` on a static pairwise NPZ.
That offline calibration could satisfy the nominal H0 false-positive constraint
on the NPZ but still fail badly online, because those H0 pairs were too easy and
did not match the nearest-neighbor candidates seen by the cache-admission
policy.

The corrected protocol mines candidate pairs with the same online retrieval
process used at deployment:

```text
stream prompt -> retrieve nearest cached prompt by cosine -> label reuse -> save pair
```

The mining pass is force-explore: it always adds the current prompt to the cache
after logging the nearest-neighbor candidate. It is not evaluating a cache
policy. It creates chronological `online_train`, `online_calib`, and
`online_eval` splits so the learned scorer is trained on retrieved candidates,
thresholds are calibrated on future-but-held-out retrieved candidates, and final
evaluation uses a later held-out stream without leakage.

Mine online candidates:

```bash
python -m experiments.mine_online_candidate_pairs \
  --dataset SemCacheLMArena \
  --embedding_model GTE \
  --output_dir results/online_candidate_pairs/lmarena_gte_seed42 \
  --seed 42 \
  --cache_size 4096 \
  --split_ratios 0.4 0.2 0.4 \
  --equivalence_mode cluster
```

Verify calibration before running vCache comparisons:

```bash
python -m experiments.debug_ours_online_mined_calibration \
  --pairs_dir results/online_candidate_pairs/lmarena_gte_seed42 \
  --methods ours_whitened_hadamard ours_weighted_ensemble \
  --target_fprs 0.01 0.02 0.03 0.05 0.08 \
  --seed 42 \
  --output_dir results/debug_ours_online_mined_calibration/lmarena_gte_seed42
```

Then run the comparison with the online-mined calibration source:

```bash
python -m experiments.compare_vcache_real_datasets \
  --datasets SemCacheLMArena \
  --methods vcache cosine ours_whitened_hadamard ours_weighted_ensemble \
  --target_budgets 0.01 0.02 0.03 0.05 \
  --delta_values 0.01 0.02 0.03 0.05 \
  --thresholds 0.80 0.85 0.90 0.93 0.95 0.97 0.98 0.99 0.995 0.999 \
  --hard_neighbor_thresholds 0.80 0.85 0.90 \
  --embedding_model GTE \
  --seed 42 \
  --cache_size 4096 \
  --ours_calibration_source online_mined \
  --online_pairs_dir results/online_candidate_pairs/lmarena_gte_seed42
```

The learned scorer is therefore not evaluated on arbitrary prompt pairs. It is
evaluated on retrieved nearest-neighbor candidates, which is the distribution
that actually determines online cache hits and false hits.

For `SemCacheSearchQueries`, this repository currently has the official stream
locally but not a separate Hadamard pairwise train/calibration file. By default
the runner stops for learned methods in that case. Passing
`--allow_pairwise_from_stream_annotations` explicitly builds a pairwise training
cache from official `id_set` annotations and records a fairness warning in
`config.json`; use a separate matching pairwise file for strict no-future
experimental claims.
