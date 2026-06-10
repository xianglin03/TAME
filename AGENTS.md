# Repository Guidelines

## Project Structure & Module Organization
`main.py` is the top-level entry point for training and sampling. Core shared logic lives in `utils.py`, `src/`, `tabsyn/`, and `baselines/tabddpm/`. Dataset preparation scripts are in `process_data/`. Evaluation code is in `eval/`, plotting helpers are in `draw_utils/`, and privacy or rebuttal analyses live in `rebuttle/`.

Large tracked artifacts are part of this repo: raw and processed datasets in `data/`, model checkpoints in `model_*`, `tabsyn/ckpt/`, and `baselines/tabddpm/ckpt*`, plus generated outputs in `synthetic/`, `sample_end_csv/`, `memorization/`, and `eval/result/`. Treat these paths carefully and avoid noisy diffs.

## Build, Test, and Development Commands
Use Python 3.10.

```bash
# sample
python main.py --dataname shoppers --method tabddpm --mode sample --save_path sample_end_csv/somename.csv --task_name somename
# 测试
python -m eval.eval_all
python eval/eval_quality
```

`st.sh` shows the expected two-environment workflow: train/sample in the main environment, then run quality evaluation in a separate `synthcity` environment.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, snake_case for functions, files, and CLI flags, and lowercase dataset names such as `adult` or `shoppers`. Keep new scripts focused and runnable from the repo root. There is no configured formatter or linter in the repository, so keep imports tidy and stay close to PEP 8 manually.

## Testing Guidelines
There is no dedicated `tests/` suite or pytest configuration. Validate changes with targeted smoke runs using the relevant `main.py` command and the matching evaluation script in `eval/`. When modifying data processing or metrics, record the exact command used and check the generated CSV or TXT output under `synthetic/`, `sample_end_csv/`, or `eval/result/`.

## Commit & Pull Request Guidelines
Recent commits use short, imperative subjects such as `feat new train_sgd`, `feat sample reject`, and `sgd training`. Prefer concise messages, optionally with a `feat` prefix when adding functionality. PRs should state the dataset and method affected, list the commands run, and attach key metrics or plots when outputs change. Call out any newly added checkpoints, generated CSVs, or large binary files explicitly.
