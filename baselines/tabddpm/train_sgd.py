import json
import math
import os
from copy import deepcopy
from typing import Optional

import numpy as np
import pandas as pd
import torch

import src
from baselines.tabddpm.models.gaussian_multinomial_distribution import GaussianMultinomialDiffusion
from baselines.tabddpm.models.modules import MLPDiffusion
from utils_train import make_dataset


SUPPORTED_DP_MODE_NUMS = (1, 4, 8, 16, 32)
DEFAULT_DP_MODE_NUM = 8


def resolve_dp_mode_num(args) -> int:
    if args is None:
        return DEFAULT_DP_MODE_NUM

    dp_mode_num = getattr(args, "dp_mode_num", None)
    if dp_mode_num is None:
        return DEFAULT_DP_MODE_NUM

    dp_mode_num = int(dp_mode_num)
    if dp_mode_num not in SUPPORTED_DP_MODE_NUMS:
        raise ValueError(
            f"Unsupported dp_mode_num={dp_mode_num}. Expected one of {SUPPORTED_DP_MODE_NUMS}."
        )
    return dp_mode_num


def has_explicit_dp_mode_num(args) -> bool:
    return args is not None and getattr(args, "dp_mode_num", None) is not None


def apply_dp_mode_preset(args) -> tuple[int, float, float]:
    dp_mode_num = resolve_dp_mode_num(args)
    if args is None:
        base_noise_multiplier = 1.0
    else:
        base_noise_multiplier = float(
            getattr(args, "_base_dp_noise_multiplier", getattr(args, "dp_noise_multiplier", 1.0))
        )
        args._base_dp_noise_multiplier = base_noise_multiplier
        args.dp_mode_num = dp_mode_num
        # Match dp_sample.py semantics: smaller epsilon-like mode means stronger noise.
        args.dp_noise_multiplier = base_noise_multiplier * (DEFAULT_DP_MODE_NUM / float(dp_mode_num))

    resolved_noise_multiplier = (
        base_noise_multiplier * (DEFAULT_DP_MODE_NUM / float(dp_mode_num))
    )
    return dp_mode_num, base_noise_multiplier, resolved_noise_multiplier


def resolve_dp_mode_model_save_path(model_save_path: str, dp_mode_num: int) -> str:
    model_save_path = os.path.normpath(model_save_path)
    parent_dir, leaf_name = os.path.split(model_save_path)
    parent_name = os.path.basename(parent_dir)

    if parent_name == "ckpt_sgd" or parent_name.startswith("ckpt_sgd_"):
        mode_parent_dir = os.path.join(os.path.dirname(parent_dir), f"ckpt_sgd_{dp_mode_num}")
        return os.path.join(mode_parent_dir, leaf_name)

    if os.path.basename(model_save_path) == "ckpt_sgd":
        return os.path.join(os.path.dirname(model_save_path), f"ckpt_sgd_{dp_mode_num}")

    return model_save_path


def _require_opacus():
    try:
        from opacus import PrivacyEngine  # type: ignore
        from opacus.utils.batch_memory_manager import BatchMemoryManager  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Opacus is required for --mode train_sgd. Please install it with: pip install opacus"
        ) from exc
    try:
        from opacus.utils.fast_gradient_clipping_utils import DPTensorFastGradientClipping  # type: ignore
    except (ImportError, ModuleNotFoundError):
        DPTensorFastGradientClipping = None
    return PrivacyEngine, BatchMemoryManager, DPTensorFastGradientClipping


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model._module if hasattr(model, "_module") else model


def get_model(model_name, model_params, n_num_features, category_sizes):
    if model_name == "mlp":
        return MLPDiffusion(**model_params)
    raise ValueError(f"Unknown model: {model_name}")


class DPSGDTrainer:
    def __init__(
        self,
        diffusion: GaussianMultinomialDiffusion,
        train_loader: torch.utils.data.DataLoader,
        lr: float,
        weight_decay: float,
        steps: int,
        model_save_path: str,
        device: torch.device | str,
        args=None,
    ):
        self.diffusion = diffusion
        self.model = diffusion._denoise_fn
        self.train_loader = train_loader
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.device = device
        self.model_save_path = model_save_path
        self.args = args
        self.loss_records = []
        self.dp_mode_num = resolve_dp_mode_num(args)
        self.base_dp_noise_multiplier = float(
            getattr(args, "_base_dp_noise_multiplier", getattr(args, "dp_noise_multiplier", 1.0))
        )

        self.log_every = int(getattr(args, "dp_log_every", 50))
        self.print_every = int(getattr(args, "dp_print_every", self.log_every))
        self.epsilon_every = int(getattr(args, "dp_epsilon_every", max(200, self.log_every)))

        self.dp_noise_multiplier = float(getattr(args, "dp_noise_multiplier", 1.0))
        self.dp_max_grad_norm = float(getattr(args, "dp_max_grad_norm", 1.0))
        self.dp_delta = float(getattr(args, "dp_delta", 1e-5))
        self.dp_secure_mode = bool(getattr(args, "dp_secure_mode", False))
        self.dp_grad_sample_mode = str(getattr(args, "dp_grad_sample_mode", "ghost"))
        logical_batch_size = getattr(train_loader, "batch_size", None) or 1
        requested_physical_batch_size = int(getattr(args, "dp_physical_batch_size", 1024))
        if requested_physical_batch_size <= 0:
            requested_physical_batch_size = logical_batch_size
        self.dp_logical_batch_size = int(logical_batch_size)
        self.dp_physical_batch_size = min(requested_physical_batch_size, self.dp_logical_batch_size)
        self.loss_reduction = "mean"
        self.cached_epsilon: Optional[float] = None

        PrivacyEngine, BatchMemoryManager, DPTensorFastGradientClipping = _require_opacus()
        self.batch_memory_manager_cls = BatchMemoryManager
        self.dp_tensor_cls = DPTensorFastGradientClipping
        self.privacy_engine = PrivacyEngine(secure_mode=self.dp_secure_mode)
        make_private_kwargs = dict(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=self.dp_noise_multiplier,
            max_grad_norm=self.dp_max_grad_norm,
            loss_reduction=self.loss_reduction,
        )
        requested_grad_sample_mode = self.dp_grad_sample_mode
        if self.dp_grad_sample_mode == "ghost" and self.dp_tensor_cls is None:
            print("Ghost clipping is unavailable in this Opacus version. Falling back to hooks mode.")
            requested_grad_sample_mode = "hooks"
        try:
            private_result = self.privacy_engine.make_private(
                grad_sample_mode=requested_grad_sample_mode,
                **make_private_kwargs,
            )
        except TypeError:
            if requested_grad_sample_mode != "hooks":
                print("This Opacus version does not support grad_sample_mode. Falling back to hooks mode.")
            requested_grad_sample_mode = "hooks"
            private_result = self.privacy_engine.make_private(**make_private_kwargs)
        self.dp_grad_sample_mode = requested_grad_sample_mode
        if len(private_result) == 4:
            self.model, self.optimizer, _, self.train_loader = private_result
        else:
            self.model, self.optimizer, self.train_loader = private_result
        self.diffusion._denoise_fn = self.model

    def _anneal_lr(self, step: int):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x: torch.Tensor):
        x = x.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        if self.dp_grad_sample_mode == "ghost":
            loss_multi_ps, loss_gauss_ps = self.diffusion.mixed_loss(x, reduce=False)
            loss_multi = loss_multi_ps.mean()
            loss_gauss = loss_gauss_ps.mean()
            loss = self.dp_tensor_cls(
                module=self.model,
                optimizer=self.optimizer,
                loss_per_sample=loss_multi_ps + loss_gauss_ps,
                loss_reduction=self.loss_reduction,
            )
        else:
            loss_multi, loss_gauss = self.diffusion.mixed_loss(x)
            loss = loss_multi + loss_gauss
        loss.backward()
        self.optimizer.step()
        step_applied = not bool(getattr(self.optimizer, "_is_last_step_skipped", False))
        return loss_multi, loss_gauss, step_applied

    def _save_model(self, file_name: str):
        base_model = _unwrap_model(self.model)
        torch.save(base_model.state_dict(), os.path.join(self.model_save_path, file_name))

    def _get_epsilon(self) -> Optional[float]:
        try:
            return float(self.privacy_engine.get_epsilon(self.dp_delta))
        except Exception:
            return None

    def run_loop(self):
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0
        curr_count = 0
        best_loss = np.inf

        dataname = getattr(self.args, "dataname", "")
        if dataname == "default":
            self.steps = 15000
        elif self.steps <= 0:
            self.steps = 100000

        print("DP-SGD steps:", self.steps)
        print(
            f"DP mode -> dp_mode_num={self.dp_mode_num}, "
            f"base_noise_multiplier={self.base_dp_noise_multiplier}, "
            f"resolved_noise_multiplier={self.dp_noise_multiplier}"
        )
        print(
            f"DP params -> noise_multiplier={self.dp_noise_multiplier}, "
            f"max_grad_norm={self.dp_max_grad_norm}, delta={self.dp_delta}"
        )
        print(
            f"DP engine -> grad_sample_mode={self.dp_grad_sample_mode}"
        )
        print(
            f"DP batch sizes -> logical_batch_size={self.dp_logical_batch_size}, "
            f"physical_batch_size={self.dp_physical_batch_size}"
        )

        with self.batch_memory_manager_cls(
            data_loader=self.train_loader,
            max_physical_batch_size=self.dp_physical_batch_size,
            optimizer=self.optimizer,
        ) as memory_safe_data_loader:
            train_iter = iter(memory_safe_data_loader)
            while step < self.steps:
                try:
                    x = next(train_iter)[0]
                except StopIteration:
                    train_iter = iter(memory_safe_data_loader)
                    x = next(train_iter)[0]

                batch_loss_multi, batch_loss_gauss, step_applied = self._run_step(x)

                batch_len = len(x)
                curr_count += batch_len
                curr_loss_multi += batch_loss_multi.item() * batch_len
                curr_loss_gauss += batch_loss_gauss.item() * batch_len

                if not step_applied:
                    continue

                self._anneal_lr(step)

                if (step + 1) % self.log_every == 0:
                    mloss = np.around(curr_loss_multi / curr_count, 4)
                    gloss = np.around(curr_loss_gauss / curr_count, 4)
                    if np.isnan(gloss):
                        print("Found NaN loss. Stop training.")
                        break

                    epsilon = self.cached_epsilon
                    if (step + 1) % self.epsilon_every == 0 or (step + 1) == self.steps:
                        epsilon = self._get_epsilon()
                        self.cached_epsilon = epsilon
                    eps_str = f"{epsilon:.4f}" if epsilon is not None else "N/A"
                    if (step + 1) % self.print_every == 0:
                        print(
                            f"Step {step + 1}/{self.steps} "
                            f"MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss} Eps: {eps_str}"
                        )

                    self.loss_records.append(
                        {
                            "step": step + 1,
                            "mloss": mloss,
                            "gloss": gloss,
                            "loss": mloss + gloss,
                            "epsilon": epsilon if epsilon is not None else np.nan,
                        }
                    )

                    curr_count = 0
                    curr_loss_gauss = 0.0
                    curr_loss_multi = 0.0

                    if mloss + gloss < best_loss:
                        best_loss = mloss + gloss
                        self._save_model("model.pt")

                    if (step + 1) % 10000 == 0:
                        self._save_model(f"model_{step + 1}.pt")

                step += 1

        final_epsilon = self._get_epsilon()
        summary = {
            "dp_mode_num": self.dp_mode_num,
            "base_noise_multiplier": self.base_dp_noise_multiplier,
            "noise_multiplier": self.dp_noise_multiplier,
            "max_grad_norm": self.dp_max_grad_norm,
            "delta": self.dp_delta,
            "steps": int(step),
            "epsilon": final_epsilon,
        }
        with open(os.path.join(self.model_save_path, "dp_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


def _legacy_train_sgd(
    model_save_path,
    real_data_path,
    steps=1000,
    lr=0.002,
    weight_decay=1e-4,
    batch_size=1024,
    task_type="binclass",
    model_type="mlp",
    model_params=None,
    num_timesteps=1000,
    gaussian_loss_type="mse",
    scheduler="cosine",
    T_dict=None,
    num_numerical_features=0,
    device=torch.device("cuda:6"),
    seed=0,
    change_val=False,
    args=None,
):
    dp_mode_num, _, _ = apply_dp_mode_preset(args)
    model_save_path = resolve_dp_mode_model_save_path(model_save_path, dp_mode_num)
    real_data_path = os.path.normpath(real_data_path)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    T = src.Transformations(**T_dict)
    dataset = make_dataset(
        real_data_path,
        T,
        task_type=task_type,
        change_val=False,
    )

    K = np.array(dataset.get_category_sizes("train"))
    if len(K) == 0 or T_dict["cat_encoding"] == "one-hot":
        K = np.array([0])

    num_numerical_features = dataset.X_num["train"].shape[1] if dataset.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features
    model_params["d_in"] = d_in

    model = get_model(
        model_type,
        model_params,
        num_numerical_features,
        category_sizes=dataset.get_category_sizes("train"),
    )
    model.to(device)

    if dataset.X_num is not None and dataset.X_cat is not None:
        x_train_np = np.concatenate([dataset.X_num["train"], dataset.X_cat["train"]], axis=1)
    elif dataset.X_num is not None:
        x_train_np = dataset.X_num["train"]
    elif dataset.X_cat is not None:
        x_train_np = dataset.X_cat["train"]
    else:
        raise ValueError("Both numerical and categorical features are missing.")
    x_train = torch.from_numpy(x_train_np).float()
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    diffusion = GaussianMultinomialDiffusion(
        num_classes=K,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        device=device,
    )

    diffusion.to(device)
    diffusion.train()

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    trainer = DPSGDTrainer(
        diffusion,
        train_loader,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        model_save_path=model_save_path,
        device=device,
        args=args,
    )
    trainer.run_loop()

    final_model = _unwrap_model(trainer.model)
    torch.save(final_model.state_dict(), os.path.join(model_save_path, "model.pt"))
    torch.save(final_model.state_dict(), os.path.join(model_save_path, "model_ema.pt"))
    pd.DataFrame(trainer.loss_records).to_csv(os.path.join(model_save_path, "loss.csv"), index=False)


def _clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _update_ema(ema_model: torch.nn.Module, source_model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_param, src_param in zip(ema_model.parameters(), source_model.parameters()):
            ema_param.mul_(decay).add_(src_param.detach(), alpha=1.0 - decay)
        for ema_buffer, src_buffer in zip(ema_model.buffers(), source_model.buffers()):
            ema_buffer.copy_(src_buffer.detach())


class AdaptiveDPSGDTrainer:
    def __init__(
        self,
        diffusion: GaussianMultinomialDiffusion,
        train_loader: torch.utils.data.DataLoader,
        lr: float,
        weight_decay: float,
        config_steps: int,
        num_train_examples: int,
        model_save_path: str,
        device: torch.device | str,
        args=None,
    ):
        self.diffusion = diffusion
        self.model = diffusion._denoise_fn
        self.train_loader = train_loader
        self.init_lr = float(lr)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        self.device = device
        self.model_save_path = model_save_path
        self.args = args
        self.loss_records: list[dict[str, float]] = []
        self.dp_mode_num = resolve_dp_mode_num(args)
        self.base_dp_noise_multiplier = float(
            getattr(args, "_base_dp_noise_multiplier", getattr(args, "dp_noise_multiplier", 1.0))
        )

        self.num_train_examples = int(num_train_examples)
        self.logical_batch_size = int(getattr(train_loader, "batch_size", 1) or 1)
        self.steps_per_epoch = max(1, math.ceil(self.num_train_examples / self.logical_batch_size))

        self.log_every = int(getattr(args, "dp_log_every", 50))
        self.print_every = int(getattr(args, "dp_print_every", self.log_every))
        self.epsilon_every = int(getattr(args, "dp_epsilon_every", 1000))

        self.dp_noise_multiplier = float(getattr(args, "dp_noise_multiplier", 1.0))
        self.dp_max_grad_norm = float(getattr(args, "dp_max_grad_norm", 1.0))
        self.dp_delta = float(getattr(args, "dp_delta", 1e-5))
        self.dp_secure_mode = bool(getattr(args, "dp_secure_mode", False))
        self.dp_grad_sample_mode = str(getattr(args, "dp_grad_sample_mode", "hooks"))
        requested_physical_batch_size = int(getattr(args, "dp_physical_batch_size", 1024))
        if requested_physical_batch_size <= 0:
            requested_physical_batch_size = self.logical_batch_size
        self.dp_physical_batch_size = min(requested_physical_batch_size, self.logical_batch_size)

        self.dp_target_epochs = int(getattr(args, "dp_target_epochs", 800))
        self.dp_max_updates = int(getattr(args, "dp_max_updates", 3000))
        self.dp_min_updates = int(getattr(args, "dp_min_updates", 1000))
        self.dp_metric_smoothing = float(getattr(args, "dp_metric_smoothing", 0.98))
        self.dp_min_lr_ratio = float(getattr(args, "dp_min_lr_ratio", 0.05))
        self.dp_warmup_ratio = float(getattr(args, "dp_warmup_ratio", 0.05))
        self.dp_ema_decay = float(getattr(args, "dp_ema_decay", 0.999))
        self.dp_early_stop_patience = int(
            getattr(args, "dp_early_stop_patience", max(300, self.steps_per_epoch * 120))
        )
        self.dp_early_stop_min_steps = int(
            getattr(args, "dp_early_stop_min_steps", max(500, self.steps_per_epoch * 200))
        )
        self.dp_improvement_tol = float(getattr(args, "dp_improvement_tol", 1e-4))

        self.total_updates = self._resolve_total_updates(config_steps)
        self.warmup_steps = max(1, int(self.total_updates * self.dp_warmup_ratio))
        self.cached_epsilon: Optional[float] = None

        PrivacyEngine, BatchMemoryManager, DPTensorFastGradientClipping = _require_opacus()
        self.batch_memory_manager_cls = BatchMemoryManager
        self.dp_tensor_cls = DPTensorFastGradientClipping
        self.loss_reduction = "mean"
        self.privacy_engine = PrivacyEngine(secure_mode=self.dp_secure_mode)

        make_private_kwargs = dict(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=self.dp_noise_multiplier,
            max_grad_norm=self.dp_max_grad_norm,
            loss_reduction=self.loss_reduction,
        )

        requested_grad_sample_mode = self.dp_grad_sample_mode
        if requested_grad_sample_mode == "ghost" and self.dp_tensor_cls is None:
            print("Ghost clipping is unavailable in this Opacus version. Falling back to hooks mode.")
            requested_grad_sample_mode = "hooks"

        try:
            private_result = self.privacy_engine.make_private(
                grad_sample_mode=requested_grad_sample_mode,
                **make_private_kwargs,
            )
        except TypeError:
            if requested_grad_sample_mode != "hooks":
                print("This Opacus version does not support grad_sample_mode. Falling back to hooks mode.")
            requested_grad_sample_mode = "hooks"
            private_result = self.privacy_engine.make_private(**make_private_kwargs)

        self.dp_grad_sample_mode = requested_grad_sample_mode
        if len(private_result) == 4:
            self.model, self.optimizer, _, self.train_loader = private_result
        else:
            self.model, self.optimizer, self.train_loader = private_result
        self.diffusion._denoise_fn = self.model

        self.ema_model = deepcopy(_unwrap_model(self.model)).to(self.device)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

        self.best_metric = float("inf")
        self.best_step = 0
        self.best_epsilon: Optional[float] = None
        self.best_raw_state = _clone_state_dict(_unwrap_model(self.model))
        self.best_ema_state = _clone_state_dict(self.ema_model)
        self.stop_reason = "max_updates"

    def _resolve_total_updates(self, config_steps: int) -> int:
        target_epochs = self.dp_target_epochs
        if self.num_train_examples >= 50_000:
            target_epochs = min(target_epochs, 400)
        elif self.num_train_examples >= 20_000:
            target_epochs = min(target_epochs, 600)

        adaptive_updates = self.steps_per_epoch * max(1, target_epochs)
        resolved = adaptive_updates
        if config_steps > 0:
            resolved = min(resolved, int(config_steps))
        resolved = min(resolved, self.dp_max_updates)
        resolved = max(self.dp_min_updates, resolved)
        return int(resolved)

    def _set_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            lr = self.init_lr * float(step + 1) / float(self.warmup_steps)
        else:
            denom = max(1, self.total_updates - self.warmup_steps)
            progress = min(1.0, float(step - self.warmup_steps) / float(denom))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_lr = self.init_lr * self.dp_min_lr_ratio
            lr = min_lr + (self.init_lr - min_lr) * cosine

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def _compute_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.dp_grad_sample_mode == "ghost":
            loss_multi_ps, loss_gauss_ps = self.diffusion.mixed_loss(x, reduce=False)
            loss_multi = loss_multi_ps.mean()
            loss_gauss = loss_gauss_ps.mean()
            loss = self.dp_tensor_cls(
                module=self.model,
                optimizer=self.optimizer,
                loss_per_sample=loss_multi_ps + loss_gauss_ps,
                loss_reduction=self.loss_reduction,
            )
            return loss_multi, loss_gauss, loss

        loss_multi, loss_gauss = self.diffusion.mixed_loss(x)
        return loss_multi, loss_gauss, loss_multi + loss_gauss

    def _run_step(self, x: torch.Tensor):
        x = x.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        loss_multi, loss_gauss, loss = self._compute_loss(x)
        loss.backward()
        self.optimizer.step()
        step_applied = not bool(getattr(self.optimizer, "_is_last_step_skipped", False))
        return loss_multi, loss_gauss, step_applied

    def _get_epsilon(self) -> Optional[float]:
        try:
            return float(self.privacy_engine.get_epsilon(self.dp_delta))
        except Exception:
            return None

    def _save_best_models(self) -> None:
        torch.save(self.best_raw_state, os.path.join(self.model_save_path, "model.pt"))
        torch.save(self.best_ema_state, os.path.join(self.model_save_path, "model_ema.pt"))

    def run_loop(self):
        update = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0
        curr_count = 0
        smoothed_metric: Optional[float] = None

        print(f"DP-SGD updates: {self.total_updates}")
        print(
            f"DP mode -> dp_mode_num={self.dp_mode_num}, "
            f"base_noise_multiplier={self.base_dp_noise_multiplier}, "
            f"resolved_noise_multiplier={self.dp_noise_multiplier}"
        )
        print(
            f"DP params -> noise_multiplier={self.dp_noise_multiplier}, "
            f"max_grad_norm={self.dp_max_grad_norm}, delta={self.dp_delta}"
        )
        print(
            f"DP engine -> grad_sample_mode={self.dp_grad_sample_mode}, "
            f"logical_batch_size={self.logical_batch_size}, physical_batch_size={self.dp_physical_batch_size}"
        )
        print(
            f"DP schedule -> steps_per_epoch={self.steps_per_epoch}, target_epochs={self.dp_target_epochs}, "
            f"warmup_steps={self.warmup_steps}, early_stop_patience={self.dp_early_stop_patience}"
        )

        self._set_lr(0)

        with self.batch_memory_manager_cls(
            data_loader=self.train_loader,
            max_physical_batch_size=self.dp_physical_batch_size,
            optimizer=self.optimizer,
        ) as memory_safe_data_loader:
            train_iter = iter(memory_safe_data_loader)
            while update < self.total_updates:
                try:
                    x = next(train_iter)[0]
                except StopIteration:
                    train_iter = iter(memory_safe_data_loader)
                    x = next(train_iter)[0]

                loss_multi, loss_gauss, step_applied = self._run_step(x)

                batch_len = len(x)
                curr_count += batch_len
                curr_loss_multi += float(loss_multi.item()) * batch_len
                curr_loss_gauss += float(loss_gauss.item()) * batch_len

                if not step_applied:
                    continue

                base_model = _unwrap_model(self.model)
                _update_ema(self.ema_model, base_model, self.dp_ema_decay)
                lr = self._set_lr(update + 1)

                if (update + 1) % self.log_every == 0:
                    mloss = float(np.around(curr_loss_multi / max(1, curr_count), 4))
                    gloss = float(np.around(curr_loss_gauss / max(1, curr_count), 4))
                    total_loss = mloss + gloss
                    if np.isnan(total_loss):
                        self.stop_reason = "nan_loss"
                        print("Found NaN loss. Stop training.")
                        break

                    if smoothed_metric is None:
                        smoothed_metric = total_loss
                    else:
                        smoothed_metric = (
                            self.dp_metric_smoothing * smoothed_metric
                            + (1.0 - self.dp_metric_smoothing) * total_loss
                        )

                    epsilon = self.cached_epsilon
                    if (update + 1) % self.epsilon_every == 0 or (update + 1) == self.total_updates:
                        epsilon = self._get_epsilon()
                        self.cached_epsilon = epsilon

                    if smoothed_metric + self.dp_improvement_tol < self.best_metric:
                        self.best_metric = smoothed_metric
                        self.best_step = update + 1
                        self.best_epsilon = epsilon
                        self.best_raw_state = _clone_state_dict(base_model)
                        self.best_ema_state = _clone_state_dict(self.ema_model)

                    if (update + 1) % self.print_every == 0:
                        eps_str = f"{epsilon:.4f}" if epsilon is not None else "N/A"
                        print(
                            f"Step {update + 1}/{self.total_updates} "
                            f"MLoss: {mloss:.4f} GLoss: {gloss:.4f} "
                            f"Sum: {total_loss:.4f} Smooth: {smoothed_metric:.4f} "
                            f"LR: {lr:.6f} Eps: {eps_str}"
                        )

                    self.loss_records.append(
                        {
                            "step": update + 1,
                            "mloss": mloss,
                            "gloss": gloss,
                            "loss": total_loss,
                            "smooth_loss": float(smoothed_metric),
                            "lr": float(lr),
                            "epsilon": epsilon if epsilon is not None else np.nan,
                        }
                    )

                    curr_count = 0
                    curr_loss_multi = 0.0
                    curr_loss_gauss = 0.0

                    if (
                        (update + 1) >= self.dp_early_stop_min_steps
                        and (update + 1 - self.best_step) >= self.dp_early_stop_patience
                    ):
                        self.stop_reason = "early_stop"
                        print(
                            f"Early stop at step {update + 1}. "
                            f"Best smoothed loss {self.best_metric:.4f} at step {self.best_step}."
                        )
                        break

                update += 1

        self._save_best_models()

        final_epsilon = self._get_epsilon()
        summary = {
            "dp_mode_num": self.dp_mode_num,
            "base_noise_multiplier": self.base_dp_noise_multiplier,
            "noise_multiplier": self.dp_noise_multiplier,
            "max_grad_norm": self.dp_max_grad_norm,
            "delta": self.dp_delta,
            "num_train_examples": self.num_train_examples,
            "logical_batch_size": self.logical_batch_size,
            "physical_batch_size": self.dp_physical_batch_size,
            "steps_per_epoch": self.steps_per_epoch,
            "target_epochs": self.dp_target_epochs,
            "resolved_updates": int(self.total_updates),
            "completed_updates": int(update),
            "best_step": int(self.best_step),
            "best_smooth_loss": float(self.best_metric),
            "best_epsilon": self.best_epsilon,
            "final_epsilon": final_epsilon,
            "stop_reason": self.stop_reason,
            "grad_sample_mode": self.dp_grad_sample_mode,
        }
        with open(os.path.join(self.model_save_path, "dp_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


def train_sgd(
    model_save_path,
    real_data_path,
    steps=1000,
    lr=0.002,
    weight_decay=1e-4,
    batch_size=1024,
    task_type="binclass",
    model_type="mlp",
    model_params=None,
    num_timesteps=1000,
    gaussian_loss_type="mse",
    scheduler="cosine",
    T_dict=None,
    num_numerical_features=0,
    device=torch.device("cuda:6"),
    seed=0,
    change_val=False,
    args=None,
):
    dp_mode_num, base_noise_multiplier, resolved_noise_multiplier = apply_dp_mode_preset(args)
    model_save_path = resolve_dp_mode_model_save_path(model_save_path, dp_mode_num)
    real_data_path = os.path.normpath(real_data_path)
    os.makedirs(model_save_path, exist_ok=True)

    print(
        f"DP mode preset -> dp_mode_num={dp_mode_num}, "
        f"base_noise_multiplier={base_noise_multiplier}, "
        f"resolved_noise_multiplier={resolved_noise_multiplier}"
    )
    print(f"DP checkpoint path -> {model_save_path}")

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    T = src.Transformations(**T_dict)
    dataset = make_dataset(
        real_data_path,
        T,
        task_type=task_type,
        change_val=False,
    )

    K = np.array(dataset.get_category_sizes("train"))
    if len(K) == 0 or T_dict["cat_encoding"] == "one-hot":
        K = np.array([0])

    num_numerical_features = dataset.X_num["train"].shape[1] if dataset.X_num is not None else 0
    d_in = int(np.sum(K) + num_numerical_features)
    model_params["d_in"] = d_in

    model = get_model(
        model_type,
        model_params,
        num_numerical_features,
        category_sizes=dataset.get_category_sizes("train"),
    )
    model.to(device)

    if dataset.X_num is not None and dataset.X_cat is not None:
        x_train_np = np.concatenate([dataset.X_num["train"], dataset.X_cat["train"]], axis=1)
    elif dataset.X_num is not None:
        x_train_np = dataset.X_num["train"]
    elif dataset.X_cat is not None:
        x_train_np = dataset.X_cat["train"]
    else:
        raise ValueError("Both numerical and categorical features are missing.")

    x_train = torch.from_numpy(x_train_np).float()
    num_train_examples = int(x_train.shape[0])
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available() and str(device).startswith("cuda"),
        drop_last=False,
    )

    diffusion = GaussianMultinomialDiffusion(
        num_classes=K,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        device=device,
    )

    diffusion.to(device)
    diffusion.train()

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    trainer = AdaptiveDPSGDTrainer(
        diffusion=diffusion,
        train_loader=train_loader,
        lr=lr,
        weight_decay=weight_decay,
        config_steps=steps,
        num_train_examples=num_train_examples,
        model_save_path=model_save_path,
        device=device,
        args=args,
    )
    trainer.run_loop()

    pd.DataFrame(trainer.loss_records).to_csv(os.path.join(model_save_path, "loss.csv"), index=False)
