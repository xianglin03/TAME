import json
import os

import numpy as np
import torch


class SimpleDPSGDTrainer:
    def __init__(self, model, train_loader, model_save_path, device, args):
        self.model = model
        self.train_loader = train_loader
        self.model_save_path = model_save_path
        self.device = device
        self.dp_steps = int(getattr(args, "dp_steps", 3000))
        self.dp_noise_multiplier = float(getattr(args, "dp_noise_multiplier", 1.0))
        self.dp_max_grad_norm = float(getattr(args, "dp_max_grad_norm", 1.0))
        self.dp_delta = float(getattr(args, "dp_delta", 1e-5))
        self.dp_physical_batch_size = int(getattr(args, "dp_physical_batch_size", 1024))
        self.log_every = int(getattr(args, "dp_log_every", 50))
        self.dp_grad_sample_mode = str(getattr(args, "dp_grad_sample_mode", "ghost"))

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)

        try:
            from opacus import PrivacyEngine
            from opacus.utils.batch_memory_manager import BatchMemoryManager
            from opacus.utils.fast_gradient_clipping_utils import DPTensorFastGradientClipping
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Opacus is required for DP-SGD training. "
                "Please install it with: pip install opacus"
            ) from exc

        self.batch_memory_manager_cls = BatchMemoryManager
        self.dp_tensor_cls = DPTensorFastGradientClipping
        self.privacy_engine = PrivacyEngine()

        make_private_kwargs = dict(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=self.dp_noise_multiplier,
            max_grad_norm=self.dp_max_grad_norm,
            loss_reduction="mean",
        )

        if self.dp_grad_sample_mode == "ghost":
            private_result = self.privacy_engine.make_private(
                grad_sample_mode="ghost", **make_private_kwargs
            )
        else:
            private_result = self.privacy_engine.make_private(**make_private_kwargs)

        if len(private_result) == 4:
            self.model, self.optimizer, _, self.train_loader = private_result
        else:
            self.model, self.optimizer, self.train_loader = private_result

        self.loss_records = []

    def _compute_loss(self, inputs):
        if self.dp_grad_sample_mode == "ghost":
            # Get per-sample loss from model internals
            # model.loss_fn returns (batch_size, dim), mean over dim for per-sample
            loss_per_sample = self.model._module.loss_fn(
                self.model._module.denoise_fn_D, inputs
            ).mean(-1)
            loss_val = loss_per_sample.mean()
            loss = self.dp_tensor_cls(
                module=self.model,
                optimizer=self.optimizer,
                loss_per_sample=loss_per_sample,
                loss_reduction="mean",
            )
            return loss_val, loss
        else:
            loss = self.model(inputs)
            return loss, loss

    def run_loop(self):
        step = 0
        best_loss = float("inf")
        completed_steps = 0

        os.makedirs(self.model_save_path, exist_ok=True)

        with self.batch_memory_manager_cls(
            data_loader=self.train_loader,
            max_physical_batch_size=self.dp_physical_batch_size,
            optimizer=self.optimizer,
        ) as memory_safe_data_loader:
            train_iter = iter(memory_safe_data_loader)
            while step < self.dp_steps:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(memory_safe_data_loader)
                    batch = next(train_iter)

                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch

                inputs = inputs.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                loss_val, loss = self._compute_loss(inputs)
                loss.backward()
                self.optimizer.step()

                if getattr(self.optimizer, "_is_last_step_skipped", False):
                    continue

                loss_val = loss_val.item()
                completed_steps += 1

                if np.isnan(loss_val):
                    print(f"Step {step + 1}: NaN loss detected. Stopping training early.")
                    break

                if (step + 1) % self.log_every == 0:
                    try:
                        epsilon = self.privacy_engine.get_epsilon(self.dp_delta)
                    except Exception:
                        epsilon = None
                    eps_str = f"{epsilon:.4f}" if epsilon is not None else "N/A"
                    print(
                        f"Step {step + 1}/{self.dp_steps} "
                        f"Loss: {loss_val:.6f} Epsilon: {eps_str}"
                    )
                    self.loss_records.append(
                        {
                            "step": step + 1,
                            "loss": loss_val,
                            "epsilon": epsilon if epsilon is not None else float("nan"),
                        }
                    )

                    if loss_val < best_loss:
                        best_loss = loss_val
                        torch.save(
                            self.model.state_dict(),
                            os.path.join(self.model_save_path, "model.pt"),
                        )

                step += 1

        try:
            final_epsilon = self.privacy_engine.get_epsilon(self.dp_delta)
        except Exception:
            final_epsilon = None

        import csv

        loss_csv_path = os.path.join(self.model_save_path, "loss.csv")
        with open(loss_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "loss", "epsilon"])
            writer.writeheader()
            writer.writerows(self.loss_records)

        dp_metrics = {
            "dp_noise_multiplier": self.dp_noise_multiplier,
            "dp_max_grad_norm": self.dp_max_grad_norm,
            "dp_delta": self.dp_delta,
            "dp_steps": self.dp_steps,
            "completed_steps": completed_steps,
            "best_loss": best_loss if best_loss != float("inf") else None,
            "final_epsilon": final_epsilon,
            "grad_sample_mode": self.dp_grad_sample_mode,
        }
        metrics_path = os.path.join(self.model_save_path, "dp_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(dp_metrics, f, indent=2)
