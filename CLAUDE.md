# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TAME is a research codebase for tabular data synthesis using diffusion models (TabSyn and TabDDPM). It focuses on generating memorization-free synthetic data while preserving inter-attribute correlations. The project is built on top of the TabSyn codebase from Amazon Science.

## Environment Setup

Python 3.10 is required. Two conda environments are used:

- **`tame`** — Main environment for training and sampling. Install with `pip install -r requirements.txt` (includes torch, opacus, sdmetrics, etc.).
- **`synthcity`** — Separate environment for quality evaluation (Alpha Precision / Beta Recall). Install with `pip install synthcity category_encoders`.

The `st.sh` script demonstrates the two-environment workflow: sample in `tame`, then switch to `synthcity` to run `eval/bash_quality.py`.

## Common Commands

### Training

TabDDPM:
```bash
python main.py --dataname shoppers --method tabddpm --mode train --gpu 0
```

TabSyn (two-stage: VAE first, then diffusion):
```bash
python main.py --dataname shoppers --method vae --mode train --gpu 0
python main.py --dataname shoppers --method tabsyn --mode train --gpu 0
```

DP-SGD training for TabDDPM:
```bash
python main.py --dataname shoppers --method tabddpm --mode train_sgd --dp_mode_num 4 --gpu 0
```
Valid `--dp_mode_num` choices: `1, 4, 8, 16, 32`.

### Sampling

```bash
python main.py --dataname shoppers --method tabddpm --mode sample \
    --save_path sample_end_csv/output.csv --task_name mytask --gpu 0
```

Default save path is `synthetic/{dataname}/{method}.csv`.

For DP-SGD models, pass the same `--dp_mode_num`:
```bash
python main.py --dataname shoppers --method tabddpm --mode sample \
    --dp_mode_num 4 --save_path sample_end_csv/output.csv
```

### Evaluation

Full evaluation (memory + MLE + density + DCR + detection):
```bash
python -m eval.eval_all
```

Alpha Precision / Beta Recall (quality metrics, requires `synthcity` environment):
```bash
python eval/eval_quality.py
```

Memorization ratio:
```bash
python cal_memorization.py
```

### Datasets

Supported dataset names (lowercase): `adult`, `default`, `shoppers`, `cardio_train`, plus `Churn_Modelling`, `MiniBooNE`, `magic`, `beijing`, `news`, `wilt`.

## Architecture

### Entry Point Routing

`main.py` is the single entry point. It parses CLI args with `utils.get_args()`, selects GPU, then calls `utils.execute_function(method, mode)` which dynamically imports the correct module:

- `tabsyn` → `tabsyn.main` (train) or `tabsyn.sample` (sample)
- `vae` → `tabsyn.vae.main`
- `tabddpm` → `baselines.tabddpm.main_train` or `baselines.tabddpm.main_sample`
- `tabddpm` + `train_sgd` mode → `baselines.tabddpm.main_train_sgd`

### TabSyn Pipeline (`tabsyn/`)

TabSyn uses a two-stage architecture:
1. **VAE** (`tabsyn/vae/`) — Encodes tabular data into latent token embeddings saved as `train_z.npy`.
2. **Diffusion** (`tabsyn/main.py`) — Trains an MLP diffusion model (`MLPDiffusion` in `tabsyn/model.py`) in the latent space. Sampling uses `tabsyn/diffusion_utils.py`.

The `tabsyn/latent_utils.py` module bridges the two stages: `get_input_train()` loads VAE embeddings for training the diffusion model; `get_input_generate()` loads embeddings plus decoder/inverse transforms for recovering tabular data from latent samples.

### TabDDPM Pipeline (`baselines/tabddpm/`)

TabDDPM operates directly on preprocessed tabular data (not latent space). Each dataset has a TOML config in `baselines/tabddpm/configs/`. The `train.py` module handles the diffusion training loop; `sample.py` handles generation. DP-SGD variants live in `train_sgd.py` and `main_train_sgd.py`, using Opacus for privacy.

### Data Preprocessing (`src/`)

`src/data.py` defines the `Dataset` dataclass and the full preprocessing pipeline:
- `Dataset.from_dir()` loads `.npy` arrays from `data/{dataname}/`
- `transform_dataset()` applies normalization, categorical encoding (one-hot / counter / ordinal), NaN handling, and rare category dropping
- `build_dataset()` is the high-level entry point used by TabDDPM
- `prepare_tensors()` converts numpy arrays to torch tensors

`src/util.py` provides config loading (`load_config` for TOML), task type enums (`binclass`, `multiclass`, `regression`), and JSON/pickle I/O helpers.

`src/metrics.py` and `src/deep.py` contain ML evaluation utilities.

### Evaluation (`eval/`)

- `eval_all.py` — Orchestrates all metrics: memorization, MLE, density, DCR, detection
- `eval_quality.py` — Alpha Precision and Beta Recall via `synthcity` (requires synthcity env)
- `eval_mle.py`, `eval_density.py`, `eval_dcr.py`, `eval_detection.py` — Individual metric modules
- `cal_memorization.py` — Computes categorical and numerical memorization ratios separately, plus a weighted combination

### Memorization Metrics

`cal_memorization.py` defines per-dataset column indices (`column_indices`) mapping numerical and categorical columns. The key metrics are:
- `cal_cat_ori()` — Categorical memorization using exact-match distance ratios
- `cal_num_ori()` — Numerical memorization using normalized Euclidean distance ratios
- `cal_mem_weight()` — Weighted combination (95% categorical, 5% numerical)

### Checkpoints and Outputs

- TabDDPM checkpoints: `baselines/tabddpm/ckpt/{dataname}/`
- TabDDPM DP-SGD checkpoints: `baselines/tabddpm/ckpt_sgd/{dataname}/`
- TabSyn VAE checkpoints: `tabsyn/vae/ckpt/{dataname}/`
- TabSyn diffusion checkpoints: `tabsyn/ckpt/{dataname}/`
- Generated data: `synthetic/{dataname}/` (default) or `sample_end_csv/` (custom paths)
- Evaluation results: `eval/result/{task_name}.txt`

## Important Code Patterns

- Dataset-specific metadata lives in `data/{dataname}/info.json` (column indices, task type, sizes).
- TabDDPM configs are TOML files in `baselines/tabddpm/configs/`.
- The `args` object from `utils.get_args()` is passed through the entire pipeline; it contains both generic and method-specific flags.
- When modifying sampling or generation logic, test with `--eval_flag False` to skip automatic evaluation.
- There is no dedicated test suite. Validate changes by running the relevant `main.py` command and inspecting outputs in `sample_end_csv/` or `eval/result/`.
