# TabSyn + DP-SGD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add DP-SGD training support for TabSyn's diffusion stage, keeping VAE unchanged and reusing existing sampling.

**Architecture:** Add a simplified DP-SGD trainer (3000 fixed steps, hooks-only, no EMA/early-stopping) that trains `MLPDiffusion` in latent space using Opacus. Reuse `tabsyn/sample.py` for generation via a `--sgd` flag that switches checkpoint paths.

**Tech Stack:** PyTorch, Opacus (PrivacyEngine + BatchMemoryManager)

---

### File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `tabsyn/train_sgd.py` | Create | `SimpleDPSGDTrainer` — Opacus-wrapped training loop |
| `tabsyn/main_train_sgd.py` | Create | Entry point — load VAE embeddings, build model, launch trainer |
| `utils.py` | Modify | Add `train_sgd` routing for `tabsyn` + `--sgd` flag in `get_args()` |
| `tabsyn/latent_utils.py` | Modify | `get_input_generate` switches to `ckpt_sgd` when `args.sgd=True` |
| `tabsyn/sample.py` | Modify | Add `--sgd` to standalone parser for independent execution |

---

### Task 1: Create `tabsyn/train_sgd.py`

**Files:**
- Create: `tabsyn/train_sgd.py`

- [ ] **Step 1: Write the trainer module**

```python
import json
import os

import numpy as np
import torch
import torch.nn as nn


class SimpleDPSGDTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        model_save_path: str,
        device: torch.device | str,
        args,
    ):
        self.device = device
        self.model_save_path = model_save_path
        self.train_loader = train_loader

        self.dp_steps = int(getattr(args, "dp_steps", 3000))
        self.dp_noise_multiplier = float(getattr(args, "dp_noise_multiplier", 1.0))
        self.dp_max_grad_norm = float(getattr(args, "dp_max_grad_norm", 1.0))
        self.dp_delta = float(getattr(args, "dp_delta", 1e-5))
        self.dp_physical_batch_size = int(getattr(args, "dp_physical_batch_size", 1024))
        self.log_every = int(getattr(args, "dp_log_every", 50))

        try:
            from opacus import PrivacyEngine
            from opacus.utils.batch_memory_manager import BatchMemoryManager
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Opacus is required for DP-SGD training. Install it with: pip install opacus"
            ) from exc

        self.batch_memory_manager_cls = BatchMemoryManager

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3, weight_decay=0
        )

        self.privacy_engine = PrivacyEngine()
        private_result = self.privacy_engine.make_private(
            module=model,
            optimizer=self.optimizer,
            data_loader=train_loader,
            noise_multiplier=self.dp_noise_multiplier,
            max_grad_norm=self.dp_max_grad_norm,
            loss_reduction="mean",
        )
        if len(private_result) == 4:
            self.model, self.optimizer, _, self.train_loader = private_result
        else:
            self.model, self.optimizer, self.train_loader = private_result

    def run_loop(self):
        step = 0
        best_loss = float("inf")
        loss_records = []

        print(f"DP-SGD training: {self.dp_steps} steps")
        print(
            f"DP params -> noise_multiplier={self.dp_noise_multiplier}, "
            f"max_grad_norm={self.dp_max_grad_norm}, delta={self.dp_delta}"
        )
        print(
            f"DP batch -> logical={self.train_loader.batch_size}, "
            f"physical={self.dp_physical_batch_size}"
        )

        with self.batch_memory_manager_cls(
            data_loader=self.train_loader,
            max_physical_batch_size=self.dp_physical_batch_size,
            optimizer=self.optimizer,
        ) as memory_safe_loader:
            train_iter = iter(memory_safe_loader)
            while step < self.dp_steps:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(memory_safe_loader)
                    batch = next(train_iter)

                inputs = batch.float().to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                loss = self.model(inputs)
                loss.backward()
                self.optimizer.step()

                if bool(getattr(self.optimizer, "_is_last_step_skipped", False)):
                    continue

                if (step + 1) % self.log_every == 0:
                    loss_val = float(loss.item())
                    if np.isnan(loss_val):
                        print("Found NaN loss. Stopping training.")
                        break

                    epsilon = None
                    try:
                        epsilon = float(self.privacy_engine.get_epsilon(self.dp_delta))
                    except Exception:
                        pass

                    eps_str = f"{epsilon:.4f}" if epsilon is not None else "N/A"
                    print(
                        f"Step {step + 1}/{self.dp_steps} "
                        f"Loss: {loss_val:.6f} Eps: {eps_str}"
                    )

                    loss_records.append(
                        {
                            "step": step + 1,
                            "loss": loss_val,
                            "epsilon": epsilon if epsilon is not None else np.nan,
                        }
                    )

                    if loss_val < best_loss:
                        best_loss = loss_val
                        torch.save(
                            self.model.state_dict(),
                            os.path.join(self.model_save_path, "model.pt"),
                        )

                step += 1

        # Save final metrics
        final_epsilon = None
        try:
            final_epsilon = float(self.privacy_engine.get_epsilon(self.dp_delta))
        except Exception:
            pass

        summary = {
            "dp_noise_multiplier": self.dp_noise_multiplier,
            "dp_max_grad_norm": self.dp_max_grad_norm,
            "dp_delta": self.dp_delta,
            "dp_steps": self.dp_steps,
            "completed_steps": step,
            "best_loss": float(best_loss),
            "final_epsilon": final_epsilon,
        }
        import pandas as pd

        pd.DataFrame(loss_records).to_csv(
            os.path.join(self.model_save_path, "loss.csv"), index=False
        )
        with open(
            os.path.join(self.model_save_path, "dp_metrics.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(summary, f, indent=2)

        print(f"Training complete. Best loss: {best_loss:.6f}")
        if final_epsilon is not None:
            print(f"Final epsilon: {final_epsilon:.4f}")
```

- [ ] **Step 2: Verify the file exists**

Run: `ls -la tabsyn/train_sgd.py`
Expected: File exists with non-zero size

---

### Task 2: Create `tabsyn/main_train_sgd.py`

**Files:**
- Create: `tabsyn/main_train_sgd.py`

- [ ] **Step 1: Write the entry point**

```python
import argparse
import os

import torch
from torch.utils.data import DataLoader

from tabsyn.latent_utils import get_input_train
from tabsyn.model import MLPDiffusion, Model
from tabsyn.train_sgd import SimpleDPSGDTrainer


def main(args):
    device = args.device
    dataname = args.dataname

    # Load VAE embeddings (same as normal training)
    train_z, curr_dir, dataset_dir, ckpt_path, info = get_input_train(args)
    in_dim = train_z.shape[1]

    # Normalize (same as tabsyn/main.py)
    mean, std = train_z.mean(0), train_z.std(0)
    train_z = (train_z - mean) / 2

    # DataLoader
    train_loader = DataLoader(
        train_z,
        batch_size=4096,
        shuffle=True,
        num_workers=4,
    )

    # Model
    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    print(denoise_fn)
    num_params = sum(p.numel() for p in denoise_fn.parameters())
    print("Number of parameters:", num_params)

    model = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)
    model.train()

    # DP-SGD training
    model_save_path = f"{curr_dir}/ckpt_sgd/{dataname}"
    os.makedirs(model_save_path, exist_ok=True)
    print(f"Saving checkpoints to {model_save_path}")

    trainer = SimpleDPSGDTrainer(
        model=model,
        train_loader=train_loader,
        model_save_path=model_save_path,
        device=device,
        args=args,
    )
    trainer.run_loop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TabSyn DP-SGD Training")
    parser.add_argument("--dataname", type=str, default="adult", help="Dataset name.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index.")
    parser.add_argument(
        "--dp_noise_multiplier", type=float, default=1.0, help="Noise multiplier for DP-SGD."
    )
    parser.add_argument(
        "--dp_max_grad_norm", type=float, default=1.0, help="Per-sample gradient clipping norm."
    )
    parser.add_argument(
        "--dp_delta", type=float, default=1e-5, help="Target delta for privacy accountant."
    )
    parser.add_argument(
        "--dp_steps", type=int, default=3000, help="Total training steps."
    )
    parser.add_argument(
        "--dp_physical_batch_size",
        type=int,
        default=1024,
        help="Physical batch size for BatchMemoryManager.",
    )

    args = parser.parse_args()
    args.device = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"
    main(args)
```

- [ ] **Step 2: Verify the file exists**

Run: `ls -la tabsyn/main_train_sgd.py`
Expected: File exists with non-zero size

---

### Task 3: Modify `utils.py` — Add routing and `--sgd` flag

**Files:**
- Modify: `utils.py`

- [ ] **Step 1: Add `train_sgd` routing for `tabsyn`**

In `execute_function()`, change the `tabsyn` branch from:

```python
elif method == 'tabsyn':
    module_name = f"tabsyn.{mode_name}"
```

To:

```python
elif method == 'tabsyn':
    if requested_mode == 'train_sgd':
        module_name = "tabsyn.main_train_sgd"
    else:
        module_name = f"tabsyn.{mode_name}"
```

- [ ] **Step 2: Add `--sgd` flag to `get_args()`**

Add this argument near the end of `get_args()`, after the existing DP-SGD arguments (around line 184, before `args = parser.parse_args()`):

```python
    # configs for TabSyn DP-SGD sampling
    parser.add_argument('--sgd', action='store_true', default=False, help='Load DP-SGD checkpoint (ckpt_sgd) instead of normal checkpoint for TabSyn sampling.')
```

- [ ] **Step 3: Verify the changes**

Run: `grep -n "main_train_sgd" utils.py && grep -n "sgd" utils.py`
Expected: Both patterns found in the file

---

### Task 4: Modify `tabsyn/latent_utils.py` — CKPT path switch

**Files:**
- Modify: `tabsyn/latent_utils.py`

- [ ] **Step 1: Modify `get_input_generate` to support sgd flag**

In `get_input_generate()`, change:

```python
    ckpt_dir = f'{curr_dir}/ckpt/{dataname}'
```

To:

```python
    ckpt_dir = f'{curr_dir}/ckpt/{dataname}'
    if getattr(args, 'sgd', False):
        ckpt_dir = f'{curr_dir}/ckpt_sgd/{dataname}'
```

- [ ] **Step 2: Verify the change**

Run: `grep -n "ckpt_sgd" tabsyn/latent_utils.py`
Expected: Pattern found

---

### Task 5: Modify `tabsyn/sample.py` — Add `--sgd` to standalone parser

**Files:**
- Modify: `tabsyn/sample.py`

- [ ] **Step 1: Add `--sgd` argument to the standalone parser**

In the `if __name__ == '__main__':` block, add after the existing arguments:

```python
    parser.add_argument('--sgd', action='store_true', default=False, help='Load DP-SGD checkpoint for sampling.')
```

Insert after:
```python
    parser.add_argument('--steps', type=int, default=None, help='Number of function evaluations.')
```

- [ ] **Step 2: Verify the change**

Run: `grep -n "sgd" tabsyn/sample.py`
Expected: Pattern found

---

### Task 6: Integration test — Verify imports and routing

**Files:**
- Test: Smoke test only (no automated test suite in this repo)

- [ ] **Step 1: Verify module imports work**

Run:
```bash
python -c "from tabsyn.train_sgd import SimpleDPSGDTrainer; print('train_sgd import OK')"
python -c "from tabsyn.main_train_sgd import main; print('main_train_sgd import OK')"
```

Expected: Both print OK

- [ ] **Step 2: Verify routing works**

Run:
```bash
python -c "
import utils
args = utils.get_args()
fn = utils.execute_function('tabsyn', 'train_sgd')
print('Routing OK:', fn.__module__)
"
```

Expected: `Routing OK: tabsyn.main_train_sgd`

Note: This will show argparse help and exit; the key is that the module name is correct.

- [ ] **Step 3: Verify `get_input_generate` with sgd flag**

Run:
```bash
python -c "
import argparse
from tabsyn.latent_utils import get_input_generate
args = argparse.Namespace(dataname='adult', sgd=True)
try:
    result = get_input_generate(args)
    print('sgd path OK')
except Exception as e:
    print('Expected error (VAE not trained):', type(e).__name__)
"
```

Expected: Either "sgd path OK" or a FileNotFoundError (which is expected if VAE hasn't been trained).

---

### Task 7: End-to-end smoke test (requires VAE pre-trained)

**Files:**
- Test: Manual smoke test

- [ ] **Step 1: Check if VAE embeddings exist**

Run:
```bash
ls tabsyn/vae/ckpt/shoppers/train_z.npy 2>/dev/null && echo "VAE ready" || echo "Need to train VAE first"
```

- [ ] **Step 2: Train VAE if needed**

If VAE embeddings don't exist:
```bash
python main.py --dataname shoppers --method vae --mode train --gpu 0
```

Expected: VAE training completes, `tabsyn/vae/ckpt/shoppers/train_z.npy` created

- [ ] **Step 3: Run DP-SGD training**

```bash
python main.py --dataname shoppers --method tabsyn --mode train_sgd --gpu 0 --dp_steps 100
```

Use `--dp_steps 100` for a quick smoke test (full training uses 3000).

Expected:
- `DP-SGD training: 100 steps`
- Training progresses with loss values
- `tabsyn/ckpt_sgd/shoppers/model.pt` created
- `tabsyn/ckpt_sgd/shoppers/dp_metrics.json` created
- `tabsyn/ckpt_sgd/shoppers/loss.csv` created

- [ ] **Step 4: Run sampling from DP-SGD checkpoint**

```bash
python main.py --dataname shoppers --method tabsyn --mode sample --sgd --save_path sample_end_csv/tabsyn_sgd_test.csv --gpu 0
```

Expected:
- Model loads from `tabsyn/ckpt_sgd/shoppers/model.pt`
- Synthetic CSV generated at `sample_end_csv/tabsyn_sgd_test.csv`

- [ ] **Step 5: Verify outputs**

Run:
```bash
ls -la tabsyn/ckpt_sgd/shoppers/
head -5 sample_end_csv/tabsyn_sgd_test.csv
```

Expected: `model.pt`, `dp_metrics.json`, `loss.csv` exist; CSV has data rows

- [ ] **Step 6: Verify dp_metrics.json**

Run:
```bash
cat tabsyn/ckpt_sgd/shoppers/dp_metrics.json
```

Expected: JSON with keys `dp_noise_multiplier`, `dp_max_grad_norm`, `dp_delta`, `dp_steps`, `completed_steps`, `best_loss`, `final_epsilon`

---

### Task 8: Commit

- [ ] **Step 1: Stage and commit**

```bash
git add tabsyn/train_sgd.py tabsyn/main_train_sgd.py utils.py tabsyn/latent_utils.py tabsyn/sample.py docs/superpowers/specs/2026-05-27-tabsyn-dp-sgd-design.md docs/superpowers/plans/2026-05-27-tabsyn-dp-sgd.md
git commit -m "feat: add TabSyn DP-SGD training support

- Add SimpleDPSGDTrainer with Opacus (hooks mode, 3000 fixed steps)
- Add tabsyn/main_train_sgd.py entry point
- Route --method tabsyn --mode train_sgd via utils.py
- Add --sgd flag to switch between ckpt and ckpt_sgd for sampling
- Support BatchMemoryManager for physical batch splitting"
```

---

## Self-Review

**Spec coverage check:**
- [x] Simplified DP-SGD trainer (3000 fixed steps) → Task 1
- [x] Entry point loading VAE embeddings → Task 2
- [x] Routing in utils.py → Task 3
- [x] CKPT path switching for sampling → Task 4 + Task 5
- [x] Only hooks mode (no ghost) → Task 1 uses standard `make_private`
- [x] No EMA / early stopping / adaptive LR → Task 1 design
- [x] 4 DP CLI parameters → Task 2 parser
- [x] End-to-end test → Task 6 + Task 7

**Placeholder scan:** No TBD/TODO/"implement later"/"similar to" found.

**Type consistency:**
- `SimpleDPSGDTrainer.__init__` takes `model`, `train_loader`, `model_save_path`, `device`, `args` — used consistently in Task 1 and Task 2
- `args.sgd` boolean flag used in Task 3 (`get_args`), Task 4 (`get_input_generate`), Task 5 (`sample.py`)
- `dp_steps` default 3000 consistent across Task 1 and Task 2
