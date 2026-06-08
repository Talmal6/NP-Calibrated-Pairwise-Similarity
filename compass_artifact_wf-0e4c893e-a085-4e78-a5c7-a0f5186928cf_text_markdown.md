# Incremental & Online Whitening of High-Dimensional Embeddings: A Literature Review for Streaming Semantic-Cache Admission

## TL;DR
- For a d=1024 embedding stream with day-to-week drift, the strongest fit is an **exponentially-weighted (forgetting-factor) streaming covariance** maintained with Welford-style updates, paired with **periodic eigendecomposition** to refresh the PCA/ZCA whitening transform — this keeps per-request scoring at O(d·k) (low single-digit ms) and amortizes the O(d³) eigendecomposition over minutes-to-hours, which is acceptable given slow drift.
- The strongest alternatives that may dominate the naive "Welford + periodic full eigendecomposition" baseline in specific regimes are **scikit-learn IncrementalPCA (Brand-style incremental SVD with the Ross et al. forgetting factor)** when you want a continuously-maintained low-rank basis without a separate eigendecomposition step, and **Frequent Directions** when memory or robustness of the low-rank covariance sketch matters; subspace trackers (GROUSE/PAST/OPAST) and Oja-type SGD are faster per step but converge to *directions*, not a calibrated whitening transform, and add tuning and robustness risk.
- The specific combination you need — an *online/incrementally maintained whitening transform for sentence/query embeddings in a retrieval or semantic-cache setting* — appears to be a genuine gap in the published literature; all embedding-whitening work (Su et al. 2021; WhiteningBERT) fits the transform once offline, so your streaming extension is novel and you should expect to rely on empirical evaluation rather than a drop-in published recipe.

## Key Findings
1. **Streaming covariance + periodic eigendecomposition is the most defensible default.** Welford's algorithm extends naturally to the full covariance matrix at O(d²) per sample / O(b·d²) per minibatch, with documented numerical-stability advantages over the naive Gram (sum-of-outer-products) accumulator. The recent unified treatment by Reichel ("2B or Not 2B," arXiv:2605.00247, 2026 preprint) reports that "Welford is the uniquely stable choice under large data shifts, maintaining double-precision accuracy where Gram loses up to 9 decimal digits," while Gram is only ~1.6× faster than `numpy.cov` for batch computation with tuned BLAS. Both Gram and Welford are O(p²) per update. Adding a forgetting factor (EWMA covariance) handles drift continuously at no extra asymptotic cost. The whitening transform is obtained by eigendecomposing the maintained covariance — an O(d³) operation that for d=1024 runs in a fraction of a second to a few seconds and need only run periodically given day-to-week drift.
2. **Incremental PCA/SVD (Brand 2002/2006; Ross et al. 2008) directly maintains a low-rank basis and is available off-the-shelf** in scikit-learn's `IncrementalPCA` (with a `whiten=True` flag and `partial_fit`). Ross, Lim, Lin & Yang (IJCV 2008) explicitly designed their incremental PCA for nonstationary streams, with an update that "includes two important features: a method for correctly updating the sample mean, and a forgetting factor to ensure less modeling power is expended fitting older observations," and in practice updated the eigenbasis/mean "every fifth frame." It is batch-incremental, not single-sample streaming.
3. **Frequent Directions (Liberty 2013; Ghashami et al. 2016) gives deterministic, provably-bounded low-rank covariance sketches** at O(d·ℓ) per row (ℓ = sketch size). Ghashami, Liberty, Phillips & Woodruff (SIAM J. Computing 45(5), 2016) prove that for any k<ℓ, ‖AᵀA − BᵀB‖₂ ≤ ‖A − A_k‖²_F/(ℓ−k), and state "both of these bounds are the best possible for the space allowed." It is mergeable/parallelizable, making it attractive for a maintained truncated whitening transform.
4. **Subspace trackers (PAST/OPAST, GROUSE) and Oja-type stochastic PCA are the cheapest per step** (O(dk)) but they track *principal directions/subspaces*, not a ready whitening transform, and require a separate scaling step plus careful step-size/forgetting tuning. They also assume a fixed target rank and (for convergence theory) i.i.d. data — assumptions your nonstationary stream violates.
5. **Online whitening as a distinct construct exists mainly in deep learning (Decorrelated Batch Normalization, IterNorm) and in neuroscience/signal processing**, not in the embedding/retrieval literature. DBN/IterNorm maintain running whitening statistics for inference and established that **ZCA whitening avoids the "stochastic axis swapping" instability that afflicts PCA whitening**. Huang et al. (CVPR 2018) describe how "PCA whitening works by performing rotation followed by scaling, but the rotation can cause a problem we call stochastic axis swapping, which... in effect randomly permutes the neurons of a layer for each batch... to the extent that training never converges," whereas ZCA "rotates the PCA-whitened activations back such that the distortion of the original activations is minimal." This is a direct warning for a truncated PCA-whitening approach under a drifting basis.
6. **Drift handling is well-developed for covariance/mean tracking** (fixed and adaptive forgetting factors; Bodenham & Adams 2016) and for **change-point detection on covariance structure** (Subspace-CUSUM; spectral/random-matrix monitors), giving you principled triggers for full recomputation.

## Method Taxonomy

### Family A — Streaming covariance estimators (then decompose)
Maintain a running estimate of the d×d covariance (or its sufficient statistics) and derive the whitening transform on demand. **Welford's algorithm** updates a running mean and an outer-product correction matrix M_t = M_{t-1} + (x_t − m_{t-1})(x_t − m_t)ᵀ, achieving O(d²) per sample with strong numerical stability. The naive **Gram accumulator** (running Σxxᵀ and Σx) is algebraically equivalent and slightly faster with tuned BLAS but loses precision under large shifts. **EWMA / forgetting-factor covariance** replaces uniform averaging with geometrically-decaying weights (half-life H = −log2/log β), giving continuous adaptation to drift. **Sketching methods (Frequent Directions)** maintain a rank-ℓ sketch B such that BᵀB ≈ AᵀA with deterministic error bounds, trading a controlled bias for O(d·ℓ) memory and update cost. All of Family A require a separate eigendecomposition/Cholesky step to produce the whitening matrix, but that step is infrequent in your slow-drift regime.

### Family B — Incremental PCA / SVD
Directly maintain a truncated SVD/eigenbasis as data arrives. **Brand's incremental SVD (2002, 2006)** performs rank-1/low-rank updates of the thin SVD with truncation, and **Ross et al. (2008)** added correct mean updating and a forgetting factor for nonstationary streams — this is exactly the algorithm scikit-learn's `IncrementalPCA` implements. The output (singular vectors + values) yields a whitening transform almost directly (with the `whiten` flag). These methods are batch-incremental and accumulate orthogonality drift over millions of updates (a point Brand explicitly notes).

### Family C — Subspace tracking & stochastic PCA
Track a k-dimensional subspace with cheap per-step updates. **Oja's rule** is the canonical stochastic-gradient eigenvector update; modern analyses (Jain et al. 2016; Allen-Zhu & Li 2017) give near-optimal finite-sample rates but assume i.i.d. data. **PAST/PASTd (Yang 1995)** use recursive-least-squares projection approximation at O(dk) per update; **OPAST** adds per-step orthonormalization. **GROUSE (Balzano et al. 2010)** does incremental gradient descent on the Grassmannian, O(dk) per step, robust to missing data; **GRASTA** adds outlier robustness. These return a subspace/basis, not a scaled whitening transform, and need an explicit variance-normalization step.

### Family D — Online whitening transforms (deep learning / signal processing)
Methods built to whiten, not just find directions. **Decorrelated Batch Normalization (Huang et al. 2018)** whitens activations per batch and maintains running statistics for inference; it established the PCA-vs-ZCA axis-swapping result above. **IterNorm (Huang et al. 2019)** "employs Newton's iterations for much more efficient whitening, while simultaneously avoiding the eigen-decomposition" — computing Σ^(−1/2) iteratively (T=5 iterations recommended), directly relevant if periodic O(d³) eigendecomposition becomes a bottleneck. In signal processing, **PASTd-based whitening** and neuroscience-inspired adaptive whitening (Duong et al. 2023) maintain whitening online but in non-text domains.

### Family E — Embedding-specific whitening (offline today)
**Su et al. (2021) "Whitening Sentence Representations"** show that "the whitening operation in traditional machine learning can similarly enhance the isotropy of sentence representations and achieve competitive results... also capable of reducing the dimensionality... significantly reduce the storage cost and accelerate the model retrieval speed" — a one-shot post-processing fit (mean→0, covariance→I). **WhiteningBERT (Huang et al. 2021)** confirms a <10-line whitening normalization "consistently boosts performance." **WhitenedCSE (Zhuo et al. 2023)** uses per-batch group whitening during contrastive training. Recent retrieval work (Isotropic Representation for Dense Retrieval, PAKDD 2023; Soft-ZCA whitening for code search, ESANN 2025) confirms whitening helps cosine-based retrieval but all fit the transform offline. **No published work maintains the embedding whitening transform online/incrementally** — this is the gap your project fills.

## Detailed Comparison

### Family A: Streaming Covariance

**Welford (covariance form)**
- Citation: Welford (1962), Technometrics; covariance extension via Chan, Golub & LeVeque (1983). Recent unified treatment: Reichel, "2B or Not 2B: A Tale of Three Algorithms for Streaming Covariance Estimation after Welford and Chan–Golub–LeVeque," arXiv:2605.00247 (2026, preprint).
- Update cost: O(d²) per sample, O(b·d²) per minibatch (~10⁶ flops/sample at d=1024).
- Memory: O(d²) ≈ 1024² × 8 bytes ≈ 8 MB for covariance + mean. Well within budget.
- Whitening: requires separate eigendecomposition (O(d³)) or Cholesky.
- Failure modes: O(d²) memory and update cost grow quadratically — at d=4096 covariance is ~134 MB and updates 16× costlier. Catastrophic cancellation in the naive Gram variant under large shifts (Reichel: Gram "loses up to 9 decimal digits"; conformal interval widths "inflate catastrophically under large shifts while Welford's remain tight").
- Implementations: `welford-torch` (PyPI), `river.covariance.EmpiricalCovariance` (online empirical covariance with `update`/`update_many`/`revert`), Carsten Schelp's online-covariance gist.

**EWMA / forgetting-factor covariance**
- Citation: standard in finance (RiskMetrics); EWMA estimator with half-life parameterization in Reichel arXiv:2605.00247 and Boyd et al. ("A Simple Method for Predicting Covariance Matrices of Financial Returns"); adaptive variant Bodenham & Adams, "Continuous monitoring for changepoints… using adaptive estimation," Statistics and Computing 27 (2016), doi:10.1007/s11222-016-9684-8.
- Update cost: O(d²) per sample (same as Welford).
- Memory: O(d²).
- Whitening: separate eigendecomposition.
- Failure modes: forgetting factor β is a critical hyperparameter (adaptivity vs. variance); too-aggressive forgetting destabilizes the estimate / can make it rank-deficient.
- Implementations: `ffstream` (R, CRAN) for adaptive forgetting factors; trivially added on top of any covariance accumulator.

**Frequent Directions**
- Citation: Liberty, KDD 2013; Ghashami, Liberty, Phillips, Woodruff, "Frequent Directions: Simple and Deterministic Matrix Sketching," SIAM J. Computing 45(5) (2016), arXiv:1501.01711. Simplified covariance-sketch proof: Liberty, "Even Simpler Deterministic Matrix Sketching," arXiv:2202.01780 (2022).
- Update cost: O(d·ℓ) per row (ℓ = sketch size).
- Memory: O(d·ℓ) — for d=1024, ℓ=256 → ~2 MB.
- Whitening: produces a rank-ℓ sketch B; whitening transform from SVD of B. Naturally yields *truncated* whitening, matching your truncated approach.
- Error guarantee: ‖AᵀA − BᵀB‖₂ ≤ ‖A − A_k‖²_F/(ℓ−k), provably space-optimal.
- Failure modes: deterministic downward bias on small singular values (the shrinkage is the mechanism); not designed for drift (needs windowing/forgetting variant); choice of ℓ.
- Implementations: `edoliberty/frequent-directions` (Python reference).

### Family B: Incremental PCA / SVD

**Brand incremental SVD / Ross et al. IPCA (scikit-learn IncrementalPCA)**
- Citation: Brand, "Incremental SVD of uncertain data with missing values," ECCV 2002; Brand, "Fast low-rank modifications of the thin SVD," Linear Algebra Appl. 415 (2006), MERL TR2006-059; Ross, Lim, Lin, Yang, "Incremental Learning for Robust Visual Tracking," IJCV 77(1–3):125–141 (2008) — adds mean update + forgetting factor, doi:10.1007/s11263-007-0075-7.
- Update cost: per minibatch, sklearn notes SVD overhead O(batch_size · n_features²) with constant memory, vs. one large O(n_samples · n_features²) SVD for batch PCA.
- Memory: O(d·k) for retained components + batch; sklearn IncrementalPCA has constant memory on the order of batch_size · n_features.
- Whitening: direct via `whiten=True` (divides components by singular values to give unit-variance, uncorrelated outputs).
- Failure modes: orthogonality of U erodes slowly over millions of updates (Brand notes this); truncation loses information vs. full eigendecomposition; batch-incremental, not pure streaming; a reported `partial_fit` memory blow-up exists for very large feature counts (sklearn issue #7109).
- Implementations: `sklearn.decomposition.IncrementalPCA`; `RichieHakim/incremental_pca` (PyTorch/GPU, Welford-stabilized); `bchaoss/incremental-SVD`; `JuliaLinearAlgebra/IncrementalSVD.jl`.

### Family C: Subspace Tracking & Stochastic PCA

**Oja's rule / streaming k-PCA**
- Citation: Oja (1982), J. Math. Biology; finite-sample rates: Jain et al., "Streaming PCA: Matching Matrix Bernstein…," COLT 2016, arXiv:1602.06929; Allen-Zhu & Li, "First Efficient Convergence for Streaming k-PCA," NeurIPS 2017.
- Update cost: O(dk) per sample.
- Memory: O(dk).
- Whitening: returns eigenvectors only; needs separate eigenvalue/variance estimation for scaling.
- Failure modes: step-size sensitivity; convergence theory assumes i.i.d. (violated by drift); no native forgetting unless added; slow initial convergence.
- Implementations: trivial to implement; no canonical maintained library.

**PAST / OPAST**
- Citation: Yang, "Projection Approximation Subspace Tracking," IEEE Trans. Signal Processing 43 (1995); convergence analysis Yang, Signal Processing 50 (1996); OPAST: Abed-Meraim et al. (2000).
- Update cost: PAST ~3dk + O(k²) flops/update; OPAST O(dk).
- Memory: O(dk).
- Whitening: tracks signal subspace basis; the PASTd variant tracks individual eigencomponents (eigenvalues), enabling whitening, but a dedicated whitening adaptation is needed (e.g., the N-PASTD prewhitening line of work).
- Failure modes: projection-approximation bias; exponential window forgetting must be tuned; designed for slowly-varying subspaces.
- Implementations: research MATLAB; no mainstream Python package.

**GROUSE / GRASTA**
- Citation: Balzano, Nowak, Recht, "Online Identification and Tracking of Subspaces from Highly Incomplete Information," Allerton 2010, arXiv:1006.4046; GRASTA (He, Balzano, Szlam), CVPR 2012.
- Update cost: O(dk) per step (linear in subspace dimension).
- Memory: O(dk).
- Whitening: tracks subspace only; no scaling.
- Failure modes: l2 GROUSE is sensitive to outliers (GRASTA fixes this via an l1/augmented-Lagrangian loss); step-size schedule matters; fixed-rank assumption.
- Implementations: MATLAB (grouse.m, grouse2.m); GRASTA C++/mex.

### Family D: Online Whitening (DL/SP)

**Decorrelated Batch Normalization (DBN)**
- Citation: Huang, Yang, Lang, Deng, "Decorrelated Batch Normalization," CVPR 2018, arXiv:1804.08450.
- Cost: per-batch eigendecomposition O(d³) (or group-wise to reduce).
- Whitening: produces a ZCA whitening matrix; maintains running statistics for inference.
- Key lesson: **PCA whitening causes stochastic axis swapping (detrimental, can prevent convergence); ZCA whitening does not, because it rotates back to minimize distortion of the original activations** — strongly suggests ZCA over PCA whitening when the basis drifts.
- Implementations: `choltz95/keras-decorrelated-batch-norm`.

**IterNorm**
- Citation: Huang et al., "Iterative Normalization: Beyond Standardization towards Efficient Whitening," CVPR 2019, arXiv:1904.03441.
- Cost: Newton-Schulz iterations, O(T·d²) per whitening (T=5 recommended), avoids O(d³) eigendecomposition.
- Whitening: computes Σ^(−1/2) iteratively — directly a ZCA-style whitening transform.
- Failure modes: needs trace normalization for stability; approximate; T is a tuning knob.
- Relevance: if periodic eigendecomposition cost becomes limiting, IterNorm offers an eigendecomposition-free refresh of the whitening matrix.

### Family E: Embedding Whitening (offline)
- Su, Cao, Liu, Ou (2021), arXiv:2103.15316; Huang et al. WhiteningBERT, Findings of EMNLP 2021, arXiv:2104.01767, aclanthology 2021.findings-emnlp.23; Zhuo et al. WhitenedCSE, ACL 2023; Kessy, Lewin, Strimmer, "Optimal Whitening and Decorrelation," The American Statistician 72(4) (2018), arXiv:1512.00809 — the canonical reference distinguishing PCA/ZCA/Cholesky/PCA-cor/ZCA-cor whitening families, recommending **ZCA-cor** (maximally similar to original variables) and **PCA-cor** (maximal compression). All offline; provide the theoretical grounding. R reference implementation: CRAN `whitening` package (strimmerlab).

## Drift Handling by Family
- **Family A (covariance):** Best-equipped. EWMA/forgetting factors give continuous adaptation; adaptive forgetting (Bodenham & Adams 2016) tunes β online without supervision. Pair with change-point detection (Subspace-CUSUM, Xie et al. 2018/2020, arXiv:1806.10760 / 1811.03936; spectral/random-matrix monitors, arXiv:2601.22602) to trigger full recomputation on abrupt shifts.
- **Family B (IPCA):** Ross et al.'s forgetting factor handles gradual drift; abrupt drift requires re-initialization. Orthogonality erosion is a slow-burn risk over many updates.
- **Family C (subspace tracking):** Built for tracking slowly-varying subspaces (exponential windows), but abrupt changes need restart; outliers can derail GROUSE (use GRASTA). Convergence guarantees assume stationarity.
- **Family D (online whitening):** DBN/IterNorm use running averages (momentum) that adapt on the momentum timescale; no explicit change-point logic.
- **Family E (offline embedding whitening):** No drift handling — precisely why a streaming extension is needed.

## Positioning "Welford + Periodic Full Eigendecomposition"
This baseline is **strong and hard to dominate in your specific regime** (d=1024, day-to-week drift, few-hundred-MB memory budget, ms scoring, tens-of-ms updates):
- **Scoring cost is unaffected** by the choice of estimator — once you have W_k, scoring is O(d·k) regardless. So the estimator choice is purely about update cost, memory, and accuracy.
- **The O(d³) eigendecomposition is amortizable.** With day-to-week drift you can recompute the transform every few minutes to hours; a single d=1024 eigendecomposition runs well under your latency tolerance off the request path. This neutralizes the main theoretical objection to "compute covariance, then decompose." (Note: the original Ross et al. IPCA likewise refreshed only "every fifth frame" rather than every sample.)
- **Where alternatives may dominate:**
  - *IncrementalPCA* dominates when you want to avoid even periodic O(d³) work and prefer a continuously-maintained low-rank basis with a built-in whiten flag, accepting orthogonality-erosion risk and batch-incremental (not pure streaming) updates.
  - *Frequent Directions* dominates on **memory** if d grows (≥4096) where O(d²) covariance becomes heavy, and gives deterministic error bounds on the truncated transform you already use.
  - *IterNorm* dominates when the eigendecomposition specifically (not the covariance) is the bottleneck, by computing Σ^(−1/2) iteratively.
  - *Subspace trackers / Oja* dominate only when per-update cost must be O(dk) and you can tolerate looser accuracy and extra tuning — unlikely to be necessary at your cadence (tens of ms per minibatch is generous for O(b·d²)).
- **Where the baseline wins:** simplicity, numerical stability (Welford), exactness (full covariance, no truncation bias), trivial addition of a forgetting factor, and full inspectability/composability of the transform — all of which you explicitly want.

## Recommendations
**Implement and compare these 2–3 candidates empirically:**

1. **EWMA (forgetting-factor) Welford covariance + periodic ZCA-cor eigendecomposition (primary recommendation).** Your baseline made drift-aware and switched from PCA to ZCA-cor whitening. Reasons: matches all constraints; Welford is numerically stable under shifts; the forgetting factor handles day-to-week drift continuously; ZCA-cor (per Kessy et al. and the DBN axis-swapping result) avoids the instability of PCA whitening under a rotating basis while staying maximally similar to the original embeddings (important since you then take cosine). Tune β so the half-life is on the order of your drift timescale. Trigger off-cycle recomputation with a covariance change-point monitor. *Benchmark/threshold to beat:* reproduce your offline result (e.g., vCache LmArena TPR ≈ 0.605 at FPR ≤ 0.05 vs. raw-cosine 0.014) in the streaming setting and verify TPR does not degrade beyond a small tolerance as the transform adapts.

2. **scikit-learn IncrementalPCA with `whiten=True` and an added forgetting factor (Ross et al. 2008) (primary alternative).** Reasons: off-the-shelf, maintained, batch-incremental with `partial_fit`, directly emits a whitening transform, and avoids explicit periodic eigendecomposition. Use it to test whether a continuously-maintained truncated basis beats periodic full decomposition on your FPR-budgeted TPR metric. Watch orthogonality erosion over long runs (periodically re-orthonormalize or rebuild).

3. **Frequent Directions sketch → truncated whitening (secondary / future-proofing).** Reasons: deterministic error bounds, low memory that scales gracefully if d rises to 4096+, and it natively produces the truncated transform you use. Worth implementing if memory pressure or higher dimensions materialize, or as a robustness baseline.

**Decision thresholds that would change the recommendation:**
- If periodic eigendecomposition ever lands on the request path or exceeds the update budget → switch the decomposition step to **IterNorm** (Newton-Schulz Σ^(−1/2), T≈5).
- If d grows to ≥4096 and O(d²) memory/update becomes uncomfortable → move to **Frequent Directions** or a subspace tracker.
- If drift turns out to be much faster (hours, not days) → lean harder on adaptive forgetting (Bodenham & Adams) and consider **GROUSE/GRASTA** for cheap continuous tracking, accepting the loss of an exact transform.
- If outliers/poisoned queries are a concern → robustify (GRASTA-style l1 loss, or a robust covariance estimator) before trusting the transform.

## Caveats
- **No published online embedding-whitening method exists.** The combination of streaming whitening + sentence/query embeddings + retrieval/semantic-cache admission is unaddressed; your design is novel, and only empirical evaluation on the vCache benchmarks (and your other datasets) can validate it. The building blocks exist separately (offline embedding whitening; online whitening in neuroscience/DL; streaming covariance), but no source ties them together.
- **PCA vs. ZCA whitening under drift for cosine scoring is unresolved empirically for text embeddings.** The DBN axis-swapping result is from deep-net *training*, not from a drifting inference-time transform feeding cosine; this needs direct measurement in your pipeline. Note also that ZCA and PCA whitening produce *different* cosine geometries — if your offline gains were specifically measured with PCA whitening, re-validate before switching to ZCA-cor.
- **Interaction of truncation rank k with forgetting factor β is unexplored.** Both control the bias/variance of the transform; their joint setting under nonstationarity is an empirical question.
- **Change-point triggers vs. continuous forgetting:** whether a hybrid (continuous EWMA + change-point-triggered full rebuild) beats either alone for your FPR-budgeted admission metric is open.
- **Threshold-calibration coupling (systems risk specific to your design):** since your downstream FPR threshold is calibrated on whitened-cosine scores, any transform update shifts the score distribution; how often re-calibration must accompany a transform refresh is unstudied and should be instrumented from day one.
- **Numerical stability at scale:** behavior of forgetting-factor covariance over hundreds of millions of updates (rank deficiency, conditioning) is under-documented; monitor the condition number and floor small eigenvalues (ridge/shrinkage) to keep the inverse-square-root well-defined.
- **Tooling gap:** `river` provides online empirical covariance/precision (`EmpiricalCovariance`, `EmpiricalPrecision`) but **no online PCA or whitening transformer**; `sklearn.IncrementalPCA` is batch-incremental, not single-sample streaming. You will be writing the whitening/eigendecomposition layer yourself in either case.
- **Source quality:** Reichel arXiv:2605.00247 (the streaming-covariance comparison) is a 2026 preprint, not yet peer-reviewed — treat its specific numerical-stability figures (9-digit loss, 1.6× speed) as indicative, and note the 1.6× figure is for *batch* BLAS computation, not per-update streaming. WhiteningBERT (EMNLP 2021 Findings), DBN (CVPR 2018), IterNorm (CVPR 2019), GROUSE (Allerton 2010), Yang's PAST (IEEE TSP 1995), Ross et al. (IJCV 2008), Frequent Directions (SIAM J. Computing 2016), and Kessy et al. (The American Statistician 2018) are peer-reviewed. Su et al. (2021) "Whitening Sentence Representations" is a widely-cited arXiv preprint.