# TAME: Memorization-Free Diffusion Models for Tabular Data Synthesis with Correlation Preservation

<p align="center">
  <strong>ICDE Submission • Anonymous Code Repository</strong>
</p>

This repository contains the implementation of **TAME**, a framework that reduces verbatim memorization in tabular diffusion models while preserving inter-attribute correlations. 

<p align="center">
  📄 <a href="icde27_table_synthesis.pdf">Paper PDF (ICDE submission)</a>
</p>

---

## Abstract

Diffusion models have become powerful tools for generating high-quality tabular data, yet they are prone to **memorization**: synthetic records can closely resemble training rows, leaking sensitive information. Traditional differential-privacy (DP) defenses mitigate memorization but often destroy the correlations between heterogeneous attributes (e.g., a young patient incorrectly assigned a disease that only occurs in older patients).

**TAME** addresses both problems through two complementary mechanisms:

1. **Structured Anti-Memory Denoising** injects attribute-specific noise calibrated by a structured probability matrix and a stepwise memorization feedback signal.
2. **Tabular Attribute Alignment** guides the denoising process with gradients from a pretrained attribute predictor so that inter-attribute correlations are preserved.

Experiments on **TabSyn** and **TabDDPM** across four datasets (Adult, Default, Shoppers, Cardio) show that TAME consistently achieves top-tier trade-offs between **machine-learning efficiency (MLE) / memorization** and **data quality (DQ) / memorization**, and provides an **11.04% relative reduction** in membership-inference attack AUC over DP baselines.

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
- [Theoretical Insights](#theoretical-insights)
- [Experimental Setup](#experimental-setup)
- [Main Results](#main-results)
- [Additional Results from the Appendix](#additional-results-from-the-appendix)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)

---

## Overview

<p align="center">
  <img src="classifier guidance1.png" alt="TAME pipeline" width="800"/>
  <br>
  <em>Figure 1: Overview of the TAME pipeline. During sampling, TAME adds structured anti-memory noise and applies attribute-alignment guidance on top of the standard diffusion denoising step.</em>
</p>

### Why Existing Defenses Fall Short

DP-based methods inject Laplace noise uniformly across heterogeneous features. This causes two issues:

- **Heterogeneous Noise Disruption:** numerical and categorical attributes have different distributional structures, so uniform noise breaks inter-attribute dependencies.
- **Lack of Correlation Constraints:** without explicit guidance, synthetic rows can violate real-world correlations (e.g., `Disease` vs. `Age` and `Fee`).

TAME solves these by coupling memorization-aware noise with correlation-preserving alignment.

### Key Contributions

- A **memorization-aware sampling framework** for tabular diffusion models that does not require retraining the base diffusion model.
- **Structured anti-memory denoising** that calibrates noise separately for numerical and categorical attributes and adapts its strength via stepwise memorization feedback.
- **Tabular attribute alignment** that uses an auxiliary attribute predictor to preserve inter-attribute correlations during generation.
- Extensive experiments showing state-of-the-art privacy–utility trade-offs and strong resilience against membership inference attacks.

---

## Quick Start

### Requirements

- Python 3.10
- Linux (experiments run on Ubuntu 22.04; other platforms may need minor adjustments)
- Four Tesla V100-SXM2 GPUs (32 GB) were used for the paper experiments

### Supported Datasets

The paper experiments focus on four mixed-type datasets: `adult`, `default`, `shoppers`, and `cardio_train`. The codebase additionally supports `Churn_Modelling`, `MiniBooNE`, `magic`, `beijing`, `news`, and `wilt` from the TabSyn suite. Use lowercase names with `--dataname`.

### Environment Setup

Create and activate the main environment:

```bash
conda create -n tame python=3.10
conda activate tame
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

Create a separate environment for the `synthcity`-based quality metrics:

```bash
conda create -n synthcity python=3.10
conda activate synthcity
pip install synthcity
pip install category_encoders
```

### Training

**TabDDPM**

```bash
python main.py --dataname shoppers --method tabddpm --mode train --gpu 0
```

**TabSyn** (two-stage: train VAE first, then diffusion)

```bash
python main.py --dataname shoppers --method vae --mode train --gpu 0
python main.py --dataname shoppers --method tabsyn --mode train --gpu 0
```

**DP-SGD training for TabDDPM**

```bash
python main.py --dataname shoppers --method tabddpm --mode train_sgd --dp_mode_num 4 --gpu 0
```

Valid `--dp_mode_num` choices: `1, 4, 8, 16, 32`.

> **Implementation notes:** diffusion models are trained with batch size 4096 for 100,000 steps; numerical columns are quantile-normalized and categorical columns are one-hot encoded. See the paper PDF (Appendix B) for full details.

### Sampling

```bash
python main.py --dataname shoppers --method tabddpm --mode sample \
    --save_path sample_end_csv/output.csv --task_name mytask --gpu 0
```

For DP-SGD models, pass the same `--dp_mode_num`:

```bash
python main.py --dataname shoppers --method tabddpm --mode sample \
    --dp_mode_num 4 --save_path sample_end_csv/output.csv
```

To skip automatic evaluation after sampling (useful during development):

```bash
python main.py --dataname shoppers --method tabddpm --mode sample \
    --save_path sample_end_csv/output.csv --eval_flag False
```

Default save path: `synthetic/{dataname}/{method}.csv`.

### Checkpoints

Trained models are saved under:

- TabDDPM: `baselines/tabddpm/ckpt/{dataname}/`
- TabDDPM DP-SGD: `baselines/tabddpm/ckpt_sgd/{dataname}/`
- TabSyn VAE: `tabsyn/vae/ckpt/{dataname}/`
- TabSyn diffusion: `tabsyn/ckpt/{dataname}/`

### Evaluation

Full evaluation (memorization + MLE + density + DCR + detection):

```bash
python -m eval.eval_all
```

Memorization ratio only:

```bash
python cal_memorization.py
```

Alpha Precision / Beta Recall (run inside the `synthcity` environment):

```bash
python eval/eval_quality.py
```

> **Tip:** use `--eval_flag False` during development to skip automatic evaluation after sampling.

---

## Repository Structure

```
.
├── main.py                    # Single entry point for training/sampling
├── baselines/tabddpm/         # TabDDPM baseline and DP-SGD variants
├── tabsyn/                    # TabSyn VAE + latent diffusion code
├── src/                       # Data preprocessing, config, and metrics
├── eval/                      # Evaluation scripts (MLE, DQ, DCR, detection, MIA)
├── data/                      # Preprocessed datasets
├── sample_end_csv/            # Default output directory for custom sampling paths
├── synthetic/                 # Default output directory for generated data
├── assets/appendix_figures/   # Appendix figure PDFs
└── icde27_table_synthesis.pdf # ICDE submission paper
```

See `CLAUDE.md` for a detailed developer guide and `requirements.txt` for dependencies.

---

## Methodology

TAME modifies only the **sampling (generation)** phase; the diffusion model itself is trained normally.

### Tabular Attribute Alignment

To preserve correlations, TAME trains an attribute predictor $p_{\phi}(y \mid x_t)$ on noisy states $x_t$ with the target attribute $y$ (by default the dataset label column) as supervision. By Bayes' rule:

$$
\nabla_{x_t} \log p_{\theta}(x_t \mid y) = \nabla_{x_t} \log p_{\theta}(x_t) + \nabla_{x_t} \log p_{\phi}(y \mid x_t).
$$

The alignment guidance applied during sampling is:

$$
s \cdot \nabla_{x_t} \log \mathcal{A}(x_t,t) = c_1 M_t \sqrt{1 - \bar{\alpha}_t} \cdot \nabla_{x_t} \log p_{\phi}(y \mid x_t).
$$

Both the diffusion model and the predictor are **frozen** during sampling; no backpropagation is needed. Before sampling, the attribute predictor is trained on noisy versions of the original table; see Algorithm 1 in the paper PDF for the training procedure.

> **Why the label column?** The label aggregates the joint effect of all input attributes, is strongly correlated with other columns, and is uniquely defined across datasets, making the alignment objective reproducible.

---

## Theoretical Insights

A key question is *why* tabular diffusion models memorize. The optimal score function under empirical score matching has a closed form:

$$
s^{*}_{\theta}(x_t, t) = \frac{\sum_{n=1}^N w_n \frac{\sqrt{\bar{\alpha}_t} x_{n0} - x_t}{1 - \bar{\alpha}_t}}{\sum_{n=1}^N w_n}, \quad w_n = \exp\left(- \frac{\|x_t - \sqrt{\bar{\alpha}_t} x_{n0}\|^2}{2(1 - \bar{\alpha}_t)}\right).
$$

The weights $w_n$ are large when $x_t$ is close to a scaled training row $\sqrt{\bar{\alpha}_t}x_{n0}$, so the denoising step naturally pulls generated samples toward the training manifold. Moreover, if two attribute subsets $A$ and $B$ are independent, the score decomposes additively over the subsets; for example, $s^{*}_{\theta}(x_t, t)$ can be written as the concatenation of the scores for $x_{ta}$ and $x_{tb}$.

This means memorization can happen **per attribute subset** even when full rows are not duplicated, motivating attribute-level rather than only row-level interventions.

See the paper PDF (Appendix A) for full propositions and proofs.

---

## Experimental Setup

### Datasets

We evaluate on four real-world mixed-type datasets.

| Dataset  | Rows   | Num | Cat | Train  | Validation | Test   |
|----------|--------|-----|-----|--------|------------|--------|
| Adult    | 48,842 | 6   | 9   | 28,943 | 3,618      | 16,281 |
| Default  | 30,000 | 14  | 11  | 24,000 | 3,000      | 3,000  |
| Shoppers | 12,330 | 10  | 8   | 9,864  | 1,233      | 1,233  |
| Cardio   | 70,000 | 5   | 7   | 44,800 | 11,200     | 14,000 |

### Baselines

- **TabDDPM** — Gaussian-multinomial diffusion model operating directly on tabular features.
- **TabSyn** — VAE + latent-space diffusion model.
- **+RN** — adds random Gaussian noise as a simple memorization baseline.
- **+DP** — adds Laplace noise satisfying differential privacy ($\epsilon=8$, $\Delta=0.8$).
- **+TAME** — our framework on top of TabDDPM or TabSyn.

### Evaluation Metrics

**Memorization**

For a synthetic row $r'$ and its two nearest neighbors $\mathbf{N}_1, \mathbf{N}_2$ in the original table, using mixed-type distance $d_{\text{mix}}(r, r') = \lambda d_{\text{num}} + (1 - \lambda) d_{\text{cat}}$:

$$
C_{r'} = \mathbb{I}\left(d_{\text{mix}}(r', \mathbf{N}_1) < \sigma \, d_{\text{mix}}(r', \mathbf{N}_2)\right), \quad M = \frac{1}{N}\sum_{r' \in \mathcal{T}'} C_{r'}.
$$

Empirically $\lambda = 0.05$ and $\sigma = 1/3$.

**Machine Learning Efficiency (MLE)**

Downstream task performance measured by F1 (%), WE (%), AUR (%), ACC (%), and their average AVG (%). WE quantifies the relative utility gap between models trained on real vs. synthetic data.

**Data Quality (DQ)**

- **$\alpha$-Precision** — fidelity of synthetic samples to real data regions.
- **$\beta$-Recall** — coverage of real data diversity by synthetic data.
- **Shape** — marginal distribution similarity (KS for numerical, TVD for categorical).
- **Trend** — pairwise correlation preservation.

**Privacy Trade-offs**

We report **MLE/Mem** = AVG MLE / Mem(%) and **DQ/Mem** = avg($\alpha$-Pre, $\beta$-Re, Shape, Trend) / Mem(%).

**Membership Inference Attacks (MIA)**

We adopt a repeat-shadow two-sample protocol and report Exact Match, Attack AUC, JS Divergence, Mean Probability Gap, and KS P-Value.

---

## Main Results

### Overall Performance

TAME consistently ranks among the top two methods in MLE/Mem and DQ/Mem across all datasets:

| Dataset | Method | MLE/Mem | DQ/Mem | Mem (%) |
|---------|--------|---------|--------|---------|
| Adult | TabDDPM+TAME | **5.84** | **7.41** | 11.27 |
| Adult | TabSyn+TAME | 5.51 | 6.51 | 10.77 |
| Default | TabDDPM+TAME | 4.91 | 8.86 | 9.13 |
| Default | TabSyn+TAME | 3.90 | **9.18** | 8.43 |
| Shoppers | TabDDPM+TAME | 4.25 | 5.38 | 13.81 |
| Shoppers | TabSyn+TAME | 3.80 | 5.30 | 15.02 |
| Cardio | TabSyn+TAME | **14.57** | 14.58 | 4.95 |
| Cardio | TabDDPM+TAME | 14.28 | 17.18 | 5.01 |

For Adult, TabDDPM+TAME achieves the best MLE/Mem and DQ/Mem. For Default, TabSyn+TAME leads in DQ/Mem and TabDDPM+TAME is second in MLE/Mem. For Shoppers, TabDDPM+TAME ranks second in both metrics (the small dataset size makes stable distribution learning harder). For Cardio, TabSyn+TAME leads in MLE/Mem and TabDDPM+TAME leads in DQ/Mem. The full quantitative table is in the paper PDF (Table 1).

### Membership Inference Attacks

TAME consistently outperforms DP-based methods on all five MIA metrics. Notably, it reduces **Exact Matches** from 0.675 to 0.605 and **Attack AUC** from 0.598 to 0.532, demonstrating lower verbatim leakage and stronger resistance to membership inference.

<p align="center">
  <img src="assets/appendix_figures/MIA.png" alt="Evaluation against membership inference attacks" width="800"/>
  <br>
  <em>Figure 2: Evaluation against membership inference attacks. Normalized scores; lower is better.</em>
</p>

### Ablation Study

We conduct a component-level ablation on the **Shoppers** dataset with **TabDDPM** to validate the contribution of each TAME module. The results show that all three components are necessary to achieve the best privacy–utility trade-off.

| Variant | Mem ↓ | Avg Binary F1 ↑ | Shape ↑ | Trend ↑ | α-Precision ↑ | β-Recall ↑ |
|---------|-------|-----------------|---------|---------|---------------|------------|
| **TAME (full)** | **0.1381** | 0.5863 | 0.8957 | 0.8760 | 0.8633 | 0.3365 |
| w/o Stepwise Feedback | 0.1517 | **0.6572** | 0.8827 | 0.8590 | 0.8497 | 0.3053 |
| w/o Structured Denoising | 0.1658 | 0.5548 | **0.9670** | **0.9219** | 0.9047 | 0.4445 |
| w/o Attribute Alignment | 0.1542 | 0.6170 | 0.8408 | 0.9131 | **0.9759** | 0.4230 |
| Baseline (TabDDPM) | 0.1561 | 0.6407 | **0.9715** | **0.9283** | 0.9159 | **0.5493** |

Key observations:

- **Structured denoising** is the most critical component for reducing memorization; removing it increases Mem from 0.1381 to 0.1658.
- **Stepwise feedback** balances privacy and utility: its removal improves Avg Binary F1 but also raises memorization.
- **Attribute alignment** preserves distributional coverage and correlations; without it, Shape drops substantially while α-Precision peaks, indicating synthetic samples concentrate in high-density real-data regions.

### Correlation and Distribution Preservation

<p align="center">
  <img src="heatmap.png" alt="correlation heatmaps" width="800"/>
  <br>
  <em>Figure 3: Pairwise column correlation divergence between synthetic and real data (lighter is better).</em>
</p>

<p align="center">
  <img src="assets/distribution.png" alt="distribution density" width="800"/>
  <br>
  <em>Figure 4: Marginal distribution density of synthetic vs. real data.</em>
</p>

---

## Additional Results from the Appendix

Selected visualizations and analyses from the paper appendix are highlighted below.

### Stepwise Memorization

The memorization ratio grows as sampling proceeds (especially between steps 20–40) and stabilizes toward the end. Activating tabular attribute alignment reduces the stepwise memorization ratio on Default/TabDDPM from above 0.10 to about 0.09.

<p align="center">
  <img src="assets/appendix_figures/RQ4111.png" alt="Stepwise memorization ratio" width="800"/>
  <br>
  <em>Stepwise memorization ratio.</em>
</p>

### Resilience to Membership Inference Attacks

Compared with specialized counter-attack defenses (MIST, MIAShield) on the Shoppers dataset, TAME achieves far fewer exact matches (121 vs. ~1,800–2,000) and the lowest Attack AUC (0.532).

<p align="center">
  <img src="assets/appendix_figures/appendix_mia_analysis.png" alt="MIA comparison with counter-attack defenses" width="800"/>
  <br>
  <em>MIA comparison with counter-attack defenses.</em>
</p>

### High-Dimensional Dependencies

On varying attribute combinations in Shoppers (3 to 6 attributes), TAME consistently reduces the Correlation Difference compared with TabDDPM while keeping memorization stable.

<p align="center">
  <img src="assets/appendix_figures/appendix_high_dim_analysis.png" alt="High-dimensional dependency analysis" width="800"/>
  <br>
  <em>High-dimensional dependency analysis.</em>
</p>

### Correlation vs. Memorization Dynamics

There is a strong positive correlation between the recovered correlation ratio and the memorization score across sampling steps: the more structure the model recovers, the higher the memorization risk. TAME explicitly decouples the two through attribute alignment.

<p align="center">
  <img src="assets/appendix_figures/appendix_correlation_mem.png" alt="Correlation-memorization dynamics" width="800"/>
  <br>
  <em>Correlation-memorization dynamics.</em>
</p>

### Ablation: Why Structured Denoising Is Necessary

Applying attribute alignment on top of a standard DP baseline (DP + Attr. Align) actually **increases** memorization (14.05% → 15.70%) and degrades MLE AVG (64.55% → 57.14%). Only when alignment is combined with TAME's structured noise does memorization drop to 13.81% while maintaining strong trend and utility scores.

<p align="center">
  <img src="assets/appendix_figures/dp_ablation_analysis.png" alt="DP + alignment ablation" width="800"/>
  <br>
  <em>DP + alignment ablation.</em>
</p>

### Comprehensive Distribution Analysis

The appendix includes detailed visual analyses across all four datasets:

| Analysis | Adult | Cardio | Default | Shoppers |
|----------|-------|--------|---------|----------|
| PCA | <a href="assets/appendix_figures/adult_pca.pdf"><img src="assets/appendix_figures/adult_pca.png" width="180"/></a> | <a href="assets/appendix_figures/cardio_pca.pdf"><img src="assets/appendix_figures/cardio_pca.png" width="180"/></a> | <a href="assets/appendix_figures/default_pca.pdf"><img src="assets/appendix_figures/default_pca.png" width="180"/></a> | <a href="assets/appendix_figures/shoppers_pca.pdf"><img src="assets/appendix_figures/shoppers_pca.png" width="180"/></a> |
| t-SNE | <a href="assets/appendix_figures/adult_tsne.pdf"><img src="assets/appendix_figures/adult_tsne.png" width="180"/></a> | <a href="assets/appendix_figures/cardio_tsne.pdf"><img src="assets/appendix_figures/cardio_tsne.png" width="180"/></a> | <a href="assets/appendix_figures/default_tsne.pdf"><img src="assets/appendix_figures/default_tsne.png" width="180"/></a> | <a href="assets/appendix_figures/shoppers_tsne.pdf"><img src="assets/appendix_figures/shoppers_tsne.png" width="180"/></a> |
| CDF | <a href="assets/appendix_figures/adult_cdf.pdf"><img src="assets/appendix_figures/adult_cdf.png" width="180"/></a> | <a href="assets/appendix_figures/cardio_cdf.pdf"><img src="assets/appendix_figures/cardio_cdf.png" width="180"/></a> | <a href="assets/appendix_figures/default_cdf.pdf"><img src="assets/appendix_figures/default_cdf.png" width="180"/></a> | <a href="assets/appendix_figures/shoppers_cdf.pdf"><img src="assets/appendix_figures/shoppers_cdf.png" width="180"/></a> |
| Categorical | <a href="assets/appendix_figures/adult_categorical.pdf"><img src="assets/appendix_figures/adult_categorical.png" width="180"/></a> | <a href="assets/appendix_figures/cardio_categorical.pdf"><img src="assets/appendix_figures/cardio_categorical.png" width="180"/></a> | <a href="assets/appendix_figures/default_categorical.pdf"><img src="assets/appendix_figures/default_categorical.png" width="180"/></a> | <a href="assets/appendix_figures/shoppers_categorical.pdf"><img src="assets/appendix_figures/shoppers_categorical.png" width="180"/></a> |
| Bivariate | <a href="assets/appendix_figures/adult_bivariate.pdf"><img src="assets/appendix_figures/adult_bivariate.png" width="180"/></a> | <a href="assets/appendix_figures/cardio_bivariate.pdf"><img src="assets/appendix_figures/cardio_bivariate.png" width="180"/></a> | <a href="assets/appendix_figures/default_bivariate.pdf"><img src="assets/appendix_figures/default_bivariate.png" width="180"/></a> | <a href="assets/appendix_figures/shoppers_bivariate.pdf"><img src="assets/appendix_figures/shoppers_bivariate.png" width="180"/></a> |
| Boxplots | <a href="assets/appendix_figures/adult_boxplots.pdf"><img src="assets/appendix_figures/adult_boxplots.png" width="180"/></a> | <a href="assets/appendix_figures/cardio_boxplots.pdf"><img src="assets/appendix_figures/cardio_boxplots.png" width="180"/></a> | <a href="assets/appendix_figures/default_boxplots.pdf"><img src="assets/appendix_figures/default_boxplots.png" width="180"/></a> | <a href="assets/appendix_figures/shoppers_boxplots.pdf"><img src="assets/appendix_figures/shoppers_boxplots.png" width="180"/></a> |

Key observations:
- **PCA/t-SNE:** TAME overlaps with real-data manifolds and avoids the mode collapse seen in TabDDPM.
- **CDF:** TAME accurately captures irregular distributions (e.g., stepwise `age`, zero-inflated capital features, heavy-tailed financial features).
- **Categorical:** rare categories are preserved without mode collapse.
- **Boxplots:** TAME matches the real-data IQR while generating fewer extreme outliers, reducing privacy risk from verbatim memorization of anomalous records.


## Future Work

- **High-order dependency modeling:** extend alignment beyond a single target attribute to multi-attribute and graph-based dependencies.
- **Adaptive privacy–utility control:** dynamically schedule noise and alignment strengths based on estimated privacy risk or downstream task loss.
- **Broader threat evaluation:** evaluate attribute inference, record reconstruction, and linkage attacks in addition to MIA.
- **Scalability:** develop memory-efficient strategies for very large or dynamically evolving tables.
- **Regulated domains:** adapt TAME for healthcare, finance, and other domains with fairness and compliance constraints.

---

## Acknowledgements

This project was built upon code from [TabSyn](https://github.com/amazon-science/tabsyn). We are deeply grateful for their open-source contributions, which have significantly helped shape the development of this project.

Many model components and the preprocessing pipeline are based on the foundation provided by [TabSyn](https://github.com/amazon-science/tabsyn).
