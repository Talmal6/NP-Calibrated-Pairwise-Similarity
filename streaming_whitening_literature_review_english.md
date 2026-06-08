# Literature Review: Incremental and Online Whitening Transforms for Streaming Embedding Vectors

## 1. Executive summary

For the target setting—embedding dimension \(d=1024\), drift over days to weeks, low single-digit millisecond scoring latency, minibatch updates in the tens of milliseconds, and a need for an explicit inspectable transform—the most promising families are: **exact streaming covariance with periodic truncated eigendecomposition**, using Welford/Chan/Pébay-style numerically stable moment updates plus shrinkage or eigenvalue flooring; **exponentially weighted or sliding-window covariance with periodic eigendecomposition**, when recency is more important than global historical fidelity; **Frequent Directions / DS-FD matrix sketching plus periodic SVD**, when memory, mergeability, or future higher-dimensional embeddings dominate; and **fast top-\(k\) subspace trackers such as PAST/OPAST**, when continuous adaptation is more important than matching the offline PCA whitening transform exactly. Standard `IncrementalPCA` is practical and easy to test, but at \(d=1024\) it is not necessarily a decisive computational improvement over exact covariance plus periodic decomposition. The largest gap in the literature is that almost no work directly studies **online whitening for frozen text/query embeddings in dense retrieval, RAG, or semantic caching**; most sentence-embedding whitening work is offline or training-time.

---

## 2. Method taxonomy

### 2.1 Exact streaming second-moment estimators

This family maintains the sample mean and covariance, or unnormalized second central moment, explicitly. Welford-style algorithms are the scalar foundation; Chan–Golub–LeVeque and Pébay generalize stable mergeable updates to multivariate and parallel settings. In this design, the online part maintains \(\mu_t\) and \(\Sigma_t\), while the whitening transform is rebuilt periodically via eigendecomposition or SVD. Given the top \(k\) eigenpairs \((U_k,\Lambda_k)\), truncated PCA whitening is

\[
W_k=\operatorname{diag}\left((\lambda_i+\varepsilon)^{-1/2}\right)U_k^\top.
\]

This is the closest online analogue of the existing offline PCA-Whitened Cosine method. For \(d=1024\), full covariance storage is modest: \(1024^2\) entries, about 4 MiB in `float32` or 8 MiB in `float64`. Even at \(d=3072\), full covariance is roughly 36 MiB in `float32` or 72 MiB in `float64`. The main cost is not memory but \(O(d^2)\) update work and periodic \(O(d^3)\) decomposition.

Key sources: [Chan, Golub & LeVeque 1983](https://www.tandfonline.com/doi/abs/10.1080/00031305.1983.10483115), [Pébay et al. 2016](https://www.osti.gov/servlets/purl/1427275), [Schubert & Gertz 2018](https://ds.ifi.uni-heidelberg.de/files/Team/eschubert/publications/SSDBM18-covariance-authorcopy.pdf).

### 2.2 Forgetting-factor and sliding-window covariance

For non-stationary streams, exact cumulative covariance can become too inert: old traffic dominates the geometry. Forgetting-factor covariance replaces the cumulative estimator with an exponentially weighted moving estimate, while sliding-window covariance retains only the last \(W\) samples or last time interval. Both still require a decomposition step to produce whitening, but they make the transform responsive to drift. Forgetting is smoother and cheaper in state; sliding windows provide clearer “recent data only” semantics but require storing enough raw samples or sufficient delete-aware statistics to remove old contributions exactly.

Key sources: [Pébay et al. 2016](https://www.osti.gov/servlets/purl/1427275), [Ross et al. 2008](https://www.cs.toronto.edu/~dross/ivt/RossLimLinYang_ijcv.pdf), [Jeng 2010](https://www.sciencedirect.com/science/article/abs/pii/S1876107010000532), [Datar et al. 2002](https://epubs.siam.org/doi/10.1137/S0097539701398363).

### 2.3 Incremental PCA and incremental SVD

Incremental PCA/SVD methods update the low-rank PCA representation directly rather than maintaining the full covariance matrix. Foundational examples include Hall–Marshall–Martin incremental eigenanalysis, Levy–Lindenbaum sequential Karhunen–Loève, Ross et al. incremental learning for visual tracking, and Brand’s incremental SVD. This family is attractive because it returns principal components and variances directly, from which a whitening transform can be constructed. It is especially useful when one wants to avoid storing all past samples and when a fixed low-rank \(k\) representation is acceptable.

However, implementations differ. `scikit-learn`’s `IncrementalPCA` performs an SVD per minibatch and documents \(O(Bd^2)\) complexity for a minibatch of size \(B\), with memory \(O(Bd)\). That is convenient and memory-efficient, but it does not eliminate the quadratic dependence on \(d\). Brand-style rank updates can be closer to \(O(dk+k^3)\) per vector, but require more custom engineering and numerical maintenance.

Key sources: [Hall et al. 1999](https://www.bmva-archive.org.uk/bmvc/1999/papers/45.pdf), [Levy & Lindenbaum 2000](https://scispace.com/pdf/incremental-eigenanalysis-for-classification-2sx0wgor3e.pdf), [Ross et al. 2008](https://www.cs.toronto.edu/~dross/ivt/RossLimLinYang_ijcv.pdf), [Brand 2002](https://www.merl.com/publications/docs/TR2002-24.pdf), [Brand 2006](https://www.merl.com/publications/docs/TR2006-059.pdf), [`scikit-learn` IncrementalPCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html).

### 2.4 Stochastic PCA and covariance-free PCA

Oja’s rule, Sanger’s Generalized Hebbian Algorithm (GHA), CCIPCA, and modern stochastic PCA variants update the dominant eigenspace using online stochastic updates. Their typical per-sample cost is \(O(dk)\), which is attractive when \(k \ll d\). Recent theory gives strong convergence results for stochastic PCA in stationary or near-stationary settings, often under i.i.d. or sub-Gaussian assumptions. For a streaming semantic-cache workload, these methods are useful as fast top-\(k\) trackers, but they do not directly return a full whitening transform unless eigenvalue estimates and regularization are also maintained.

Key sources: [Oja 1982](https://link.springer.com/article/10.1007/BF00275687), [Sanger 1989](https://www.researchgate.net/profile/Terence-Sanger/publication/222464584_Optimal_Unsupervised_Learning_in_a_Single-Layer_Linear_Feedforward_Neural_Network/links/5a6a8a73a6fdcc2aedee0daf/Optimal-Unsupervised-Learning-in-a-Single-Layer-Linear-Feedforward-Neural-Network.pdf), [Weng, Zhang & Hwang 2003](https://www.cse.msu.edu/~weng/research/CCIPCApami.pdf), [Arora et al. 2012](https://home.ttic.edu/~klivescu/papers/arora_etal_allerton2012.pdf), [Allen-Zhu & Li 2017](https://arxiv.org/abs/1607.07837), [Huang et al. 2021](https://proceedings.mlr.press/v134/huang21a/huang21a.pdf).

### 2.5 Subspace tracking: PAST, OPAST, GROUSE

Subspace tracking methods originate largely in signal processing. PAST and OPAST maintain an orthonormal basis for a dominant low-rank subspace and often include a forgetting factor naturally. Their per-sample cost is usually \(O(dk)\) or \(O(dk+k^2)\). GROUSE and related Grassmannian methods are especially relevant when data are partially observed or have missing entries, which is not the main issue for fully observed embedding vectors. For the semantic-cache use case, PAST/OPAST are more directly relevant than GROUSE because they target low-cost tracking under drift and can be adapted to provide a PCA-like basis.

Key sources: [Yang 1995](https://www.semanticscholar.org/paper/Projection-approximation-subspace-tracking-Yang/2aaaff998488e3f08edd50f37970c248c6f0a972), [Abed-Meraim et al. 2000](https://www.researchgate.net/publication/3342532_Fast_orthonormal_PAST_algorithm), [Badeau et al. 2003](https://perso.telecom-paristech.fr/grichard/Publications/ICASSP-03_2.pdf), [Balzano et al. 2018](https://arxiv.org/abs/1806.04609).

### 2.6 Matrix sketching: Frequent Directions and DS-FD

Frequent Directions maintains a compact sketch \(B\in\mathbb{R}^{\ell\times d}\) that approximates the covariance of the stream with deterministic error guarantees. It is mergeable, memory-efficient, and naturally suited to distributed ingestion. Its update cost is typically stated as \(O(d\ell)\) amortized per row and memory is \(O(d\ell)\). To obtain a whitening transform, one still performs an SVD/EVD on the sketch periodically. DS-FD extends sketching to sliding-window streams and is directly relevant when the desired transform should reflect only recent traffic.

This family is attractive if the embedding dimension increases substantially, if multiple workers maintain local summaries, or if the stream volume makes exact covariance updates too expensive. It is less attractive if exact covariance at \(d=1024\) is already cheap enough and the primary requirement is fidelity to the offline PCA whitening transform.

Key sources: [Liberty 2013](https://edoliberty.github.io/papers/simpleMatrixSketching.pdf), [Ghashami et al. 2016](https://epubs.siam.org/doi/10.1137/15M1009718), [Yin et al. 2024, DS-FD, preprint](https://arxiv.org/abs/2405.07792).

### 2.7 Whitening layers and deep-learning normalization

Decorrelated Batch Normalization and IterNorm whiten neural activations during training. They are relevant because they study practical whitening and matrix inverse square roots, but they are not a direct fit for a semantic-cache admission system with frozen embeddings and a persistent versioned transform. These methods operate inside model training and depend on minibatch activations, not on a long-lived unsupervised transform applied to cached vectors over time.

Key sources: [Decorrelated Batch Normalization, Huang et al. 2018](https://arxiv.org/abs/1804.08450), [IterNorm, Huang et al. 2019](https://arxiv.org/abs/1904.03441).

### 2.8 Sentence-embedding whitening and dense retrieval isotropy

The sentence-embedding literature repeatedly finds that anisotropy harms cosine similarity and that whitening or isotropy-improving post-processing can improve semantic similarity and retrieval. Examples include BERT-flow, Whitening Sentence Representations, WhiteningBERT, isotropic dense retrieval, and WhitenedCSE. These papers support the motivation for whitening embeddings, but they are almost entirely offline post-processing or training-time methods. They do not provide a mature recipe for maintaining a whitening transform online as traffic drifts.

Key sources: [BERT-flow, Li et al. 2020](https://aclanthology.org/2020.emnlp-main.733/), [Whitening Sentence Representations, Su et al. 2021](https://arxiv.org/abs/2103.15316), [WhiteningBERT, Huang et al. 2021](https://arxiv.org/abs/2104.01767), [Isotropic Representation Can Improve Dense Retrieval, Jung et al. 2022](https://arxiv.org/abs/2209.00218), [WhitenedCSE, ACL 2023](https://aclanthology.org/2023.acl-long.677.pdf).

### 2.9 Semantic caching literature

Recent semantic-cache work focuses mainly on admission thresholds, false-positive constraints, and online calibration of decision rules. In the report’s source set, vCache is the most relevant example: it uses online learning to update thresholds under a budget, not online geometric correction of the embedding space. This means the streaming whitening component is likely a novel design axis rather than a solved semantic-cache subroutine.

Key source: [vCache, OpenReview](https://openreview.net/forum?id=zF0A0xw3HZ).

---

## 3. Detailed comparison

For every method that yields \(U_k,\Lambda_k\), whitening construction is straightforward:

\[
W_k=\operatorname{diag}\left((\lambda_i+\varepsilon)^{-1/2}\right)U_k^\top.
\]

For ZCA whitening:

\[
W_{\mathrm{ZCA},k}=U_k\operatorname{diag}\left((\lambda_i+\varepsilon)^{-1/2}\right)U_k^\top.
\]

PCA whitening is usually the better operational fit for this setting because it produces a compact \(k\)-dimensional vector for cosine-style comparison. ZCA preserves the original coordinate system but generally keeps a denser \(d\)-dimensional transform.

| Method / family | Representative citation | Update cost | Memory | Whitening output? | Main failure modes | Implementations |
|---|---|---:|---:|---|---|---|
| Welford / Chan / Pébay + periodic EVD | [Chan et al. 1983](https://www.tandfonline.com/doi/abs/10.1080/00031305.1983.10483115); [Pébay et al. 2016](https://www.osti.gov/servlets/purl/1427275); [Schubert & Gertz 2018](https://ds.ifi.uni-heidelberg.de/files/Team/eschubert/publications/SSDBM18-covariance-authorcopy.pdf) | \(O(d^2)\) per sample; \(O(Bd^2)\) per minibatch; periodic EVD \(O(d^3)\) | \(O(d^2)\) | Not directly; EVD/SVD needed | Cumulative estimator adapts slowly to drift; EVD can dominate if too frequent; small eigenvalues amplify noise | Mostly custom; R `onlinePCA` has covariance helpers |
| EWMA / forgetting-factor covariance | [Pébay et al. 2016](https://www.osti.gov/servlets/purl/1427275); [Ross et al. 2008](https://www.cs.toronto.edu/~dross/ivt/RossLimLinYang_ijcv.pdf) | \(O(d^2)\) per sample; \(O(Bd^2)\) per minibatch | \(O(d^2)\) | Not directly; decomposition needed | Forgetting factor is hard to tune; too slow = stale, too fast = noisy | Custom implementation is straightforward |
| Sliding-window covariance | [Jeng 2010](https://www.sciencedirect.com/science/article/abs/pii/S1876107010000532); [Datar et al. 2002](https://epubs.siam.org/doi/10.1137/S0097539701398363) | \(O(d^2)\) add/drop update; periodic EVD | \(O(d^2+Wd)\) for exact raw window | Not directly; decomposition needed | Window size controls bias/variance; hard boundaries can create jumps | Mostly custom |
| `IncrementalPCA` / minibatch IPCA | [Hall et al. 1999](https://www.bmva-archive.org.uk/bmvc/1999/papers/45.pdf); [`scikit-learn` docs](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html) | `sklearn`: \(O(Bd^2)\) per SVD minibatch | \(O(Bd)\) plus components | Almost; components and variances are returned | Not necessarily faster than exact covariance at \(d=1024\); weak drift handling unless reset/forgetting added | `scikit-learn`, `cuML` |
| Brand-style incremental SVD | [Brand 2002](https://www.merl.com/publications/docs/TR2002-24.pdf); [Brand 2006](https://www.merl.com/publications/docs/TR2006-059.pdf) | Approx. \(O(dk+k^3)\) per vector or small update | \(O(dk)\) | Almost; singular vectors/values give PCA-like whitening | More custom engineering; orthogonality and rank-management issues | Research/reference implementations; Julia `IncrementalSVD.jl` |
| CCIPCA | [Weng et al. 2003](https://www.cse.msu.edu/~weng/research/CCIPCApami.pdf) | Typically \(O(dk)\) per sample | \(O(dk)\) | Not directly; needs eigenvalue scaling and mean tracking | Sensitive to sample order and mean update; less exact in tail spectrum | R `onlinePCA`; Python research code exists |
| Oja / GHA / stochastic PCA | [Oja 1982](https://link.springer.com/article/10.1007/BF00275687); [Sanger 1989](https://www.researchgate.net/profile/Terence-Sanger/publication/222464584_Optimal_Unsupervised_Learning_in_a_Single-Layer_Linear_Feedforward_Neural_Network/links/5a6a8a73a6fdcc2aedee0daf/Optimal-Unsupervised-Learning-in-a-Single-Layer-Linear-Feedforward-Neural-Network.pdf); [Arora et al. 2012](https://home.ttic.edu/~klivescu/papers/arora_etal_allerton2012.pdf) | \(O(dk)\) per sample; \(O(Bdk)\) for block updates | \(O(dk)\) | Not directly; requires variance estimates | Learning-rate tuning; stationary/i.i.d. assumptions in theory; noisy updates | R `onlinePCA`; custom Python |
| PAST / OPAST | [Yang 1995](https://www.semanticscholar.org/paper/Projection-approximation-subspace-tracking-Yang/2aaaff998488e3f08edd50f37970c248c6f0a972); [Abed-Meraim et al. 2000](https://www.researchgate.net/publication/3342532_Fast_orthonormal_PAST_algorithm); [Badeau et al. 2003](https://perso.telecom-paristech.fr/grichard/Publications/ICASSP-03_2.pdf) | \(O(dk)\) or \(O(dk+k^2)\) per sample | \(O(dk+k^2)\) | Not directly; basis plus scale must be converted to whitening | Requires fixed \(k\); forgetting factor tuning; may chase noise under abrupt drift | Mostly signal-processing/research code |
| GROUSE / Grassmannian tracking | [Balzano et al. 2018](https://arxiv.org/abs/1806.04609) | Often \(O(dk)\), especially useful for missing data | \(O(dk)\) | No; subspace only | Best advantage is missing data, which this setting does not need | Research code |
| Frequent Directions | [Liberty 2013](https://edoliberty.github.io/papers/simpleMatrixSketching.pdf); [Ghashami et al. 2016](https://epubs.siam.org/doi/10.1137/15M1009718) | \(O(d\ell)\) amortized per row | \(O(d\ell)\) | No; SVD on sketch needed | Approximation error depends on \(\ell\); low-rank bias; needs centering strategy | [Edo Liberty repo](https://github.com/edoliberty/frequent-directions), R package |
| DS-FD sliding-window sketch | [Yin et al. 2024, preprint](https://arxiv.org/abs/2405.07792) | Sketch update near \(O(d\ell)\), window-management overhead depends on implementation | \(O(d\ell)\) plus window/sketch metadata | No; SVD on sketch needed | Preprint; assumptions may not map exactly to production time windows; complexity of implementation | [DS-FD repo](https://github.com/yinhanyan/DS-FD) |
| DBN / IterNorm whitening layers | [Decorrelated BN](https://arxiv.org/abs/1804.08450); [IterNorm](https://arxiv.org/abs/1904.03441) | Training-time minibatch whitening; often matrix inverse square root or iterative approximations | Layer running stats | Not in the required persistent streaming-transform sense | Batch-size dependence; modifies model/training pipeline; versioning mismatch | Official repos: [DBN](https://github.com/princeton-vl/DecorrelatedBN), [IterNorm](https://github.com/huangleiBuaa/IterNorm) |
| Sentence-embedding whitening | [BERT-flow](https://aclanthology.org/2020.emnlp-main.733/); [Whitening Sentence Representations](https://arxiv.org/abs/2103.15316); [WhiteningBERT](https://arxiv.org/abs/2104.01767) | Usually offline fit once; not online | Depends on fitted transform | Yes offline, usually not streaming | No mature online update recipe; often task/model-specific | Research code varies |

---

## 4. Drift handling

### 4.1 Cumulative estimators

Cumulative Welford/Chan/Pébay covariance handles numerical stability well, but it does not handle distribution shift by itself. The effective memory of the estimator grows with time, so new traffic has decreasing influence. This is acceptable only if the embedding distribution is effectively stationary or if the transform is periodically reset/recomputed from a recent pool.

### 4.2 Forgetting factors

Exponential forgetting gives the estimator an effective time horizon. If the forgetting factor is \(\alpha\), recent data receives more weight while old data decays geometrically. This is a good fit for days-to-weeks drift because it avoids abrupt transform changes. The hard part is tuning: too much forgetting makes covariance noisy; too little makes the transform stale.

Practical design: implement EWMA covariance with a half-life measured in requests or wall-clock time, not an arbitrary \(\alpha\). For example, a one-week half-life is easier to reason about operationally than a raw decay coefficient.

### 4.3 Sliding windows

Sliding-window covariance gives clear recency semantics: the transform reflects the last \(W\) samples or the last \(T\) hours/days. It is preferable when older traffic should be treated as irrelevant rather than merely downweighted. Its disadvantages are memory for the raw window and discontinuities when samples expire.

### 4.4 Sketch-based windows

Frequent Directions handles unbounded streams well; DS-FD is specifically designed for sliding-window matrix sketching. This is attractive when an exact raw window is too expensive or when workers need mergeable summaries. The tradeoff is approximation: the sketch must be large enough to preserve the directions that matter for whitening and admission performance.

### 4.5 Subspace tracking with forgetting

PAST/OPAST and similar methods naturally include forgetting and are good at continuously tracking a changing top-\(k\) subspace. They are appropriate if the dominant subspace evolves slowly and the cache admission score only needs a stable top-\(k\) whitening transform. They are less appropriate if tail eigenvalues matter, if \(k\) is hard to set, or if exact correspondence to the offline covariance-based transform is essential.

### 4.6 Drift detectors and refresh triggers

Drift detectors such as ADWIN and Page-Hinkley can be used as triggers for recomputing or resetting the transform. They should not be applied to every covariance entry. Instead, apply them to scalar diagnostics, such as:

- whitened norm distribution;
- reconstruction residual / PCA \(Q\)-statistic;
- explained variance ratio of the current top-\(k\);
- stability of calibrated admission threshold;
- false-positive estimate if labels or delayed feedback exist;
- score distribution shift between query/candidate pairs.

Key sources: [ADWIN, Bifet & Gavaldà 2007](https://epubs.siam.org/doi/10.1137/1.9781611972771.42), [River ADWIN implementation](https://riverml.xyz/dev/api/drift/ADWIN/), [River Page-Hinkley implementation](https://riverml.xyz/dev/api/drift/PageHinkley/).

### 4.7 Regularization under drift

Regardless of update method, whitening requires regularization. Since \(1/\sqrt{\lambda}\) amplifies small eigenvalue noise, practical systems should use at least one of:

- eigenvalue floor: \(\lambda_i \leftarrow \max(\lambda_i,\varepsilon)\);
- shrinkage covariance: \(\hat{\Sigma}_{\mathrm{shrunk}}=(1-\rho)\hat{\Sigma}+\rho \tau I\);
- spectral tempering: \(\lambda_i^{-\gamma}\) with \(0<\gamma<1/2\);
- truncation to \(k\) with a robust selection rule;
- delayed transform activation until enough samples accumulate.

Relevant shrinkage sources: [Ledoit & Wolf 2004](https://perso.ens-lyon.fr/patrick.flandrin/LedoitWolf_JMA2004.pdf), [Chen et al. 2010 OAS](https://www.cs.huji.ac.il/~amiw/chen_tsp_2010.pdf), [`scikit-learn` LedoitWolf](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html).

---

## 5. Positioning “Welford + periodic full eigendecomposition”

“Welford + periodic eigendecomposition” is a stronger baseline than it may initially appear. At \(d=1024\), full covariance is small relative to the memory budget: only a few MiB. The method is also easy to validate because it closely matches the successful offline PCA-Whitened Cosine pipeline. It is exact, inspectable, mergeable with Chan/Pébay-style updates, and simple to reason about.

Its main weakness is not memory; it is **adaptivity and decomposition scheduling**. If the estimator is cumulative, it adapts slowly. If the eigendecomposition is frequent, decomposition cost can dominate. If the eigendecomposition is rare, the transform can become stale. Therefore, the realistic baseline should not be “plain cumulative Welford forever,” but:

1. numerically stable minibatch covariance updates;
2. optional EWMA or sliding-window mode;
3. periodic or drift-triggered truncated EVD;
4. eigenvalue floor or shrinkage;
5. transform versioning for cached embeddings.

### When alternatives may dominate

**EWMA/windowed covariance dominates cumulative Welford** when the workload is explicitly non-stationary and old requests should matter less. This is likely relevant for semantic-cache workloads where topics, users, or model traffic change over days and weeks.

**Frequent Directions dominates exact covariance** when \(d\) grows substantially, workers are distributed, memory is constrained, or approximate mergeable summaries are operationally valuable. At \(d=1024\), the case is weaker because full covariance is already cheap.

**PAST/OPAST dominates exact covariance updates** when \(k \ll d\), update latency is more important than exact covariance fidelity, and the system needs continuous adaptation. It is less attractive if the admission score is sensitive to accurate eigenvalue scaling or if exact reproducibility of the offline PCA-whitening behavior is important.

**`IncrementalPCA` dominates only in convenience**, not necessarily in asymptotic update cost at this dimension. It is a good engineering baseline because it returns components and variances directly, but the documented `scikit-learn` complexity remains \(O(Bd^2)\) per minibatch SVD.

---

## 6. Recommendations

### Candidate 1: exact covariance + periodic truncated EVD

Implement this first as the reference streaming version of the offline method.

Recommended state:

- sample count or effective weight;
- running mean \(\mu\);
- covariance or second central moment \(M_2\);
- optional EWMA covariance state;
- current transform version: \((\mu, U_k, \lambda_k, \varepsilon, \gamma, \text{timestamp})\).

Recommended transform:

\[
z = \operatorname{normalize}\left(\operatorname{diag}((\lambda_i+\varepsilon)^{-\gamma})U_k^\top(x-\mu)\right)
\]

where \(\gamma=1/2\) is full whitening and \(\gamma<1/2\) is tempered whitening.

Why this is first: it preserves fidelity to the offline PCA-Whitened Cosine method that already produced a large TPR gain under the FPR budget. It also provides a trustworthy oracle for evaluating approximate methods.

### Candidate 2: EWMA or sliding-window covariance + periodic EVD

Implement this as the drift-aware exact variant.

Use EWMA if you want smooth adaptation and low state. Use sliding-window covariance if you want explicit “last \(N\) requests” or “last \(T\) days” semantics. Recompute \(U_k,\lambda_k\) periodically or when scalar drift diagnostics fire.

Why this is second: the target drift timescale is days to weeks, so a recency-aware exact method may capture most benefits without moving to more complex subspace trackers.

### Candidate 3: Frequent Directions / DS-FD + periodic SVD

Implement this as the approximate scalable contender.

Start with standard Frequent Directions unless exact sliding-window semantics are important. Choose sketch size \(\ell\) larger than \(k\), e.g. \(\ell=2k\), \(4k\), or tune empirically. Periodically compute SVD on the sketch and build the whitening transform from the resulting singular values/vectors.

Why this is third: it has strong deterministic covariance-approximation guarantees, compact state, and a path to higher dimensions or distributed ingestion. It is also a clean research comparison against exact covariance.

### Optional candidate: OPAST / PAST

If implementation time allows, evaluate OPAST/PAST as a fast adaptive top-\(k\) tracker. This is the best candidate if update cost must be reduced from \(O(d^2)\) to \(O(dk)\), but it is a more complex fit because it returns a tracked subspace rather than a complete covariance estimate. The whitening transform still needs stable eigenvalue or scale estimates.

### Not recommended as primary methods

- Deep-learning whitening layers such as DBN/IterNorm: useful conceptually, but not a drop-in persistent transform for frozen embeddings.
- GROUSE: mostly useful for missing-data subspace tracking, which is not the main deployment constraint.
- Pure offline sentence-embedding whitening papers: useful motivation, not direct streaming algorithms.

---

## 7. Open questions and gaps

The literature is thin at the exact intersection of **online whitening**, **frozen dense embeddings**, **semantic-cache admission**, and **FPR-constrained calibrated thresholds**. Existing work supports individual components—streaming covariance, online PCA, subspace tracking, matrix sketching, and offline embedding whitening—but does not settle the end-to-end design.

The main empirical questions are:

1. Is cache admission performance more sensitive to exact covariance fidelity or to recency adaptation?
2. How large should \(k\) be for whitened cosine in streaming embeddings?
3. Does full whitening \(\gamma=1/2\) over-amplify noisy tail directions compared with tempered whitening \(0<\gamma<1/2\)?
4. How should cached items be handled when the transform version changes?
5. Does the threshold calibration need to be version-aware?
6. Does an approximate sketch preserve the directions that matter for FPR-constrained admission, not merely covariance in spectral/Frobenius norm?
7. Are query embeddings and candidate/cache embeddings drawn from the same distribution, or should they have separate means/covariances?
8. How often can the system recompute the transform without causing score discontinuities?
9. Can scalar drift diagnostics reliably predict degradation in admission quality before labels are available?

A likely publishable contribution is to show that online geometric correction of embedding space—via exact, windowed, or sketched streaming whitening—materially improves semantic-cache admission under a false-positive-rate budget, because existing semantic-cache papers appear to focus more on thresholding than on maintaining the embedding geometry itself.

---

## 8. Practical implementation notes

### 8.1 Transform versioning

Every cached embedding should either store:

1. its raw embedding and be transformed lazily under the current \(W_k\); or
2. its transformed vector plus a transform version ID.

The first is simpler and avoids stale transformed vectors, but costs more at scoring time unless transformed vectors are cached. The second is faster but requires migration or mixed-version scoring logic.

### 8.2 Recompute schedule

A practical schedule is:

- update covariance every minibatch;
- recompute \(W_k\) every fixed interval, e.g. every \(M\) minibatches or every \(T\) minutes/hours;
- recompute immediately on drift alarm;
- keep the previous transform active until the new transform passes sanity checks.

### 8.3 Sanity checks before activating a transform

Before switching versions, check:

- minimum eigenvalue after flooring;
- explained variance ratio stability;
- distribution of whitened norms;
- cosine score distribution on a held-out recent sample;
- admission threshold shift relative to previous version;
- TPR/FPR on delayed labels if available.

### 8.4 Numerical recommendations

Use `float64` for covariance accumulation even if embeddings are `float32`. The memory cost is still acceptable at \(d=1024\). Store the final transform in `float32` if latency requires it. Always use eigenvalue flooring or shrinkage before inverse square-root scaling.

---

## 9. Source index

### Streaming covariance and moments

- Chan, Golub & LeVeque. “Algorithms for Computing the Sample Variance: Analysis and Recommendations.” 1983. <https://www.tandfonline.com/doi/abs/10.1080/00031305.1983.10483115>
- Pébay et al. “Numerically Stable, Scalable Formulas for Parallel and Online Computation of Higher-Order Multivariate Central Moments with Arbitrary Weights.” 2016. <https://www.osti.gov/servlets/purl/1427275>
- Schubert & Gertz. “Numerically Stable Parallel Computation of (Co-)Variance.” 2018. <https://ds.ifi.uni-heidelberg.de/files/Team/eschubert/publications/SSDBM18-covariance-authorcopy.pdf>

### Incremental PCA and SVD

- Hall, Marshall & Martin. “Incremental Eigenanalysis for Classification.” 1999. <https://www.bmva-archive.org.uk/bmvc/1999/papers/45.pdf>
- Levy & Lindenbaum. “Sequential Karhunen-Loeve Basis Extraction and its Application to Images.” 2000. <https://scispace.com/pdf/incremental-eigenanalysis-for-classification-2sx0wgor3e.pdf>
- Ross et al. “Incremental Learning for Robust Visual Tracking.” 2008. <https://www.cs.toronto.edu/~dross/ivt/RossLimLinYang_ijcv.pdf>
- Brand. “Incremental Singular Value Decomposition of Uncertain Data with Missing Values.” 2002. <https://www.merl.com/publications/docs/TR2002-24.pdf>
- Brand. “Fast Low-Rank Modifications of the Thin Singular Value Decomposition.” 2006. <https://www.merl.com/publications/docs/TR2006-059.pdf>
- `scikit-learn` IncrementalPCA documentation. <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html>
- RAPIDS cuML IncrementalPCA documentation. <https://docs.rapids.ai/api/cuml/nightly/api/generated/cuml.decomposition.incrementalpca/>
- Julia `IncrementalSVD.jl`. <https://github.com/JuliaLinearAlgebra/IncrementalSVD.jl>

### Online PCA, stochastic PCA, and subspace tracking

- Oja. “Simplified neuron model as a principal component analyzer.” 1982. <https://link.springer.com/article/10.1007/BF00275687>
- Sanger. “Optimal Unsupervised Learning in a Single-Layer Linear Feedforward Neural Network.” 1989. <https://www.researchgate.net/profile/Terence-Sanger/publication/222464584_Optimal_Unsupervised_Learning_in_a_Single-Layer_Linear_Feedforward_Neural_Network/links/5a6a8a73a6fdcc2aedee0daf/Optimal-Unsupervised-Learning-in-a-Single-Layer-Linear-Feedforward-Neural-Network.pdf>
- Weng, Zhang & Hwang. “Candid Covariance-free Incremental Principal Component Analysis.” 2003. <https://www.cse.msu.edu/~weng/research/CCIPCApami.pdf>
- Arora et al. “Stochastic Optimization for PCA and PLS.” 2012. <https://home.ttic.edu/~klivescu/papers/arora_etal_allerton2012.pdf>
- Allen-Zhu & Li. “First Efficient Convergence for Streaming k-PCA: a Global, Gap-Free, and Near-Optimal Rate.” 2017. <https://arxiv.org/abs/1607.07837>
- Huang et al. “Streaming PCA from Incomplete Data.” 2021. <https://proceedings.mlr.press/v134/huang21a/huang21a.pdf>
- Yang. “Projection Approximation Subspace Tracking.” 1995. <https://www.semanticscholar.org/paper/Projection-approximation-subspace-tracking-Yang/2aaaff998488e3f08edd50f37970c248c6f0a972>
- Abed-Meraim et al. “Fast Orthonormal PAST Algorithm.” 2000. <https://www.researchgate.net/publication/3342532_Fast_orthonormal_PAST_algorithm>
- Badeau et al. “Sliding window adaptive SVD algorithms.” 2003. <https://perso.telecom-paristech.fr/grichard/Publications/ICASSP-03_2.pdf>
- Balzano et al. “Online Identification and Tracking of Subspaces from Highly Incomplete Information.” 2018. <https://arxiv.org/abs/1806.04609>
- R `onlinePCA` package. <https://rdrr.io/cran/onlinePCA/>
- `online_psp` repository. <https://github.com/flatironinstitute/online_psp>

### Matrix sketching

- Liberty. “Simple and Deterministic Matrix Sketching.” 2013. <https://edoliberty.github.io/papers/simpleMatrixSketching.pdf>
- Ghashami et al. “Frequent Directions: Simple and Deterministic Matrix Sketching.” 2016. <https://epubs.siam.org/doi/10.1137/15M1009718>
- Frequent Directions repository. <https://github.com/edoliberty/frequent-directions>
- R Frequent Directions package. <https://cran.r-project.org/web/packages/frequentdirections/frequentdirections.pdf>
- Yin et al. “Deterministic Matrix Sketching for Sliding Window Model.” 2024, preprint. <https://arxiv.org/abs/2405.07792>
- DS-FD repository. <https://github.com/yinhanyan/DS-FD>

### Whitening and embedding isotropy

- Kessy, Lewin & Strimmer. “Optimal Whitening and Decorrelation.” 2018. <https://arxiv.org/abs/1512.00809>
- Huang et al. “Decorrelated Batch Normalization.” 2018. <https://arxiv.org/abs/1804.08450>
- Huang et al. “Iterative Normalization: Beyond Standardization towards Efficient Whitening.” 2019. <https://arxiv.org/abs/1904.03441>
- Decorrelated BN repository. <https://github.com/princeton-vl/DecorrelatedBN>
- IterNorm repository. <https://github.com/huangleiBuaa/IterNorm>
- Li et al. “On the Sentence Embeddings from Pre-trained Language Models.” EMNLP 2020. <https://aclanthology.org/2020.emnlp-main.733/>
- Su et al. “Whitening Sentence Representations for Better Semantics and Faster Retrieval.” 2021. <https://arxiv.org/abs/2103.15316>
- Huang et al. “WhiteningBERT.” 2021. <https://arxiv.org/abs/2104.01767>
- Jung et al. “A Simple yet Effective Approach for Improving Dense Retrieval via Isotropic Representation.” 2022. <https://arxiv.org/abs/2209.00218>
- WhitenedCSE. ACL 2023. <https://aclanthology.org/2023.acl-long.677.pdf>

### Drift detection, shrinkage, and adjacent anomaly detection

- Bifet & Gavaldà. “Learning from Time-Changing Data with Adaptive Windowing.” 2007. <https://epubs.siam.org/doi/10.1137/1.9781611972771.42>
- River ADWIN documentation. <https://riverml.xyz/dev/api/drift/ADWIN/>
- River Page-Hinkley documentation. <https://riverml.xyz/dev/api/drift/PageHinkley/>
- Ledoit & Wolf. “A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices.” 2004. <https://perso.ens-lyon.fr/patrick.flandrin/LedoitWolf_JMA2004.pdf>
- Chen et al. “Shrinkage Algorithms for MMSE Covariance Estimation.” 2010. <https://www.cs.huji.ac.il/~amiw/chen_tsp_2010.pdf>
- `scikit-learn` LedoitWolf documentation. <https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html>
- Efficient anomaly detection via matrix sketching. <https://papers.neurips.cc/paper/8030-efficient-anomaly-detection-via-matrix-sketching.pdf>
- Matrix sketching for anomaly detection source. <https://arxiv.org/abs/1804.03065>

### Semantic caching

- vCache, OpenReview. <https://openreview.net/forum?id=zF0A0xw3HZ>
