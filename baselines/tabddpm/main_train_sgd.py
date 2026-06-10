import argparse
import os

import src
from baselines.tabddpm.train_sgd import (
    resolve_dp_mode_model_save_path,
    resolve_dp_mode_num,
    train_sgd,
)


def main(args):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataname = args.dataname
    device = args.device

    config_path = f"{curr_dir}/configs/{dataname}.toml"
    model_save_path = f"{curr_dir}/ckpt_sgd/{dataname}"
    model_save_path = resolve_dp_mode_model_save_path(model_save_path, resolve_dp_mode_num(args))
    real_data_path = f"data/{dataname}"

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    args.train = True
    raw_config = src.load_config(config_path)

    print("START DP-SGD TRAINING")
    train_sgd(
        **raw_config["train"]["main"],
        **raw_config["diffusion_params"],
        model_save_path=model_save_path,
        real_data_path=real_data_path,
        task_type=raw_config["task_type"],
        model_type=raw_config["model_type"],
        model_params=raw_config["model_params"],
        T_dict=raw_config["train"]["T"],
        num_numerical_features=raw_config["num_numerical_features"],
        device=device,
        args=args,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", metavar="FILE")
    parser.add_argument("--dataname", type=str, default="shoppers")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--dp_mode_num",
        type=int,
        default=None,
        choices=[1, 4, 8, 16, 32],
        help="DP mode preset. 8 keeps the current default train_sgd.py settings.",
    )
    parser.add_argument("--dp_noise_multiplier", type=float, default=1.0)
    parser.add_argument("--dp_max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dp_delta", type=float, default=1e-5)
    parser.add_argument("--dp_secure_mode", action="store_true", default=False)
    parser.add_argument("--dp_physical_batch_size", type=int, default=1024)
    parser.add_argument("--dp_grad_sample_mode", type=str, default="hooks", choices=["ghost", "hooks"])
    parser.add_argument("--dp_log_every", type=int, default=50)
    parser.add_argument("--dp_print_every", type=int, default=50)
    parser.add_argument("--dp_epsilon_every", type=int, default=1000)
    parser.add_argument("--dp_target_epochs", type=int, default=800)
    parser.add_argument("--dp_max_updates", type=int, default=3000)
    parser.add_argument("--dp_min_updates", type=int, default=1000)
    parser.add_argument("--dp_warmup_ratio", type=float, default=0.05)
    parser.add_argument("--dp_min_lr_ratio", type=float, default=0.05)
    parser.add_argument("--dp_ema_decay", type=float, default=0.999)
    parser.add_argument("--dp_metric_smoothing", type=float, default=0.98)
    parser.add_argument("--dp_early_stop_patience", type=int, default=400)
    parser.add_argument("--dp_early_stop_min_steps", type=int, default=800)
    parser.add_argument("--dp_improvement_tol", type=float, default=1e-4)

    args = parser.parse_args()
    args.device = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"
    
    main(args)
