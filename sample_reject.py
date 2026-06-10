import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import src
from baselines.tabddpm.models.gaussian_multinomial_distribution import (
    GaussianMultinomialDiffusion,
)
from baselines.tabddpm.sample import (
    get_model as get_tabddpm_model,
    recover_data as recover_tabddpm_data,
    split_num_cat_target as split_tabddpm_num_cat_target,
)
from cal_memorization import column_indices
from tabsyn.diffusion_utils import sample as sample_tabsyn_latent
from tabsyn.latent_utils import (
    recover_data as recover_tabsyn_data,
    split_num_cat_target as split_tabsyn_num_cat_target,
)
from tabsyn.model import MLPDiffusion as TabsynMLPDiffusion
from tabsyn.model import Model as TabsynModel
from tabsyn.vae.model import Decoder_model
from utils_train import make_dataset, preprocess


REFERENCE_THRESHOLD_PRESETS = {
    "shoppers": {
        "result_path": Path("eval/result/ngradshoppers_-5_600000_1.txt"),
        "weight_cat": 0.95,
        "reject_rule": "weighted_indicator",
        "default_save_path": Path("sample_reject_shoppers.csv"),
    }
}


class MemorizationRejector:
    def __init__(
        self,
        dataname,
        train_df,
        cat_threshold,
        num_threshold,
        weight_cat,
        reject_rule,
        reject_threshold,
    ):
        if dataname not in column_indices:
            raise ValueError(f"Unsupported dataname for memorization filtering: {dataname}")

        self.dataname = dataname
        self.train_df = train_df
        self.cat_threshold = cat_threshold
        self.num_threshold = num_threshold
        self.weight_cat = weight_cat
        self.weight_num = 1.0 - weight_cat
        self.reject_rule = reject_rule
        self.reject_threshold = reject_threshold

        self.num_cols = train_df.columns[column_indices[dataname]["numerical"]].tolist()
        self.cat_cols = train_df.columns[column_indices[dataname]["categorical"]].tolist()

        self.train_num = None
        self.train_cat = None
        if self.num_cols:
            self.train_num = train_df[self.num_cols].astype(float).to_numpy(dtype=np.float64)
        if self.cat_cols:
            self.train_cat = train_df[self.cat_cols].to_numpy(dtype=object)

    @staticmethod
    def _two_smallest(distances):
        if distances.shape[0] < 2:
            return np.inf, np.inf

        smallest = np.argpartition(distances, 1)[:2]
        values = np.sort(distances[smallest].astype(np.float64))
        return float(values[0]), float(values[1])

    @classmethod
    def _ratio_from_distances(cls, distances):
        min_distance, second_min_distance = cls._two_smallest(distances)
        if not np.isfinite(second_min_distance) or second_min_distance <= 0:
            return np.inf
        return min_distance / second_min_distance

    def _cat_ratio(self, row):
        if self.train_cat is None:
            return np.inf

        row_cat = row[self.cat_cols].to_numpy(dtype=object)
        distances = (self.train_cat != row_cat).sum(axis=1).astype(np.float64)
        distances /= len(self.cat_cols)
        return self._ratio_from_distances(distances)

    def _num_ratio(self, row):
        if self.train_num is None:
            return np.inf

        row_num = row[self.num_cols].to_numpy(dtype=np.float64)
        distances = np.sqrt(np.sum((self.train_num - row_num) ** 2, axis=1))

        min_distance = float(np.min(distances))
        max_distance = float(np.max(distances))
        if max_distance > min_distance:
            distances = (distances - min_distance) / (max_distance - min_distance)
        else:
            distances = np.zeros_like(distances, dtype=np.float64)

        return self._ratio_from_distances(distances)

    def _weighted_scores(self, cat_ratio, num_ratio, cat_mem, num_mem):
        weights = []
        ratio_terms = []
        indicator_terms = []

        if self.cat_cols:
            weights.append(self.weight_cat)
            ratio_terms.append(cat_ratio)
            indicator_terms.append(float(cat_mem))
        if self.num_cols:
            weights.append(self.weight_num)
            ratio_terms.append(num_ratio)
            indicator_terms.append(float(num_mem))

        if not weights:
            return np.inf, 0.0

        normalizer = sum(weights)
        weighted_ratio = sum(w * r for w, r in zip(weights, ratio_terms)) / normalizer
        weighted_indicator = sum(w * r for w, r in zip(weights, indicator_terms)) / normalizer
        return weighted_ratio, weighted_indicator

    def evaluate_row(self, row):
        cat_ratio = self._cat_ratio(row)
        num_ratio = self._num_ratio(row)

        cat_mem = bool(cat_ratio < self.cat_threshold)
        num_mem = bool(num_ratio < self.num_threshold)
        weighted_ratio, weighted_indicator = self._weighted_scores(
            cat_ratio,
            num_ratio,
            cat_mem,
            num_mem,
        )

        if self.reject_rule == "weighted_indicator":
            reject = weighted_indicator >= self.reject_threshold
        elif self.reject_rule == "weighted_ratio":
            reject = weighted_ratio < self.reject_threshold
        elif self.reject_rule == "either":
            reject = cat_mem or num_mem
        elif self.reject_rule == "cat_only":
            reject = cat_mem
        elif self.reject_rule == "num_only":
            reject = num_mem
        else:
            raise ValueError(f"Unsupported reject rule: {self.reject_rule}")

        return {
            "cat_ratio": cat_ratio,
            "num_ratio": num_ratio,
            "cat_mem": cat_mem,
            "num_mem": num_mem,
            "weighted_ratio": weighted_ratio,
            "weighted_indicator": weighted_indicator,
            "reject": bool(reject),
        }

    def filter_dataframe(self, generated_df):
        stats = [self.evaluate_row(row) for _, row in generated_df.iterrows()]
        stats_df = pd.DataFrame(stats)
        accepted_mask = ~stats_df["reject"]
        accepted_df = generated_df.loc[accepted_mask].reset_index(drop=True)
        accepted_stats_df = stats_df.loc[accepted_mask].reset_index(drop=True)
        return accepted_df, accepted_stats_df, stats_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rejection-sampling baseline that drops memorized samples."
    )
    parser.add_argument("--dataname", type=str, required=True, help="Dataset name.")
    parser.add_argument(
        "--method",
        type=str,
        default="tabsyn",
        choices=["tabsyn", "tabddpm"],
        help="Backbone sampler to use before rejection.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index. Use -1 for CPU.")
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help=(
            "Path to save the filtered synthetic CSV. Defaults to the dataset preset when "
            "available, otherwise {dataname}_reject.csv."
        ),
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default=None,
        help="Reference training CSV used for memorization filtering.",
    )
    parser.add_argument(
        "--target_samples",
        type=int,
        default=None,
        help="Number of accepted rows to keep. Defaults to the training set size.",
    )
    parser.add_argument(
        "--cat_threshold",
        type=float,
        default=None,
        help="Categorical memorization threshold. Defaults to the reference result when available.",
    )
    parser.add_argument(
        "--num_threshold",
        type=float,
        default=None,
        help="Numerical memorization threshold. Defaults to the reference result when available.",
    )
    parser.add_argument(
        "--weight_cat",
        type=float,
        default=None,
        help="Categorical weight used by the weighted rejection rule.",
    )
    parser.add_argument(
        "--reject_rule",
        type=str,
        default=None,
        choices=["weighted_indicator", "weighted_ratio", "either", "cat_only", "num_only"],
        help="How to map per-row memorization scores to reject/keep decisions.",
    )
    parser.add_argument(
        "--reject_threshold",
        type=float,
        default=None,
        help="Threshold used by the weighted rejection rules.",
    )
    parser.add_argument(
        "--reference_result_path",
        type=str,
        default=None,
        help=(
            "Optional eval/result/*.txt used to auto-fill reject thresholds from "
            "mem_all/cat_ori/num_ori."
        ),
    )
    parser.add_argument(
        "--disable_reference_thresholds",
        action="store_true",
        default=False,
        help="Disable auto-calibration from reference eval results and use fallback defaults.",
    )
    parser.add_argument(
        "--oversample_factor",
        type=float,
        default=1.5,
        help="Target remaining rows are multiplied by this factor before sampling each round.",
    )
    parser.add_argument(
        "--min_round_size",
        type=int,
        default=512,
        help="Minimum number of candidates to generate per round.",
    )
    parser.add_argument(
        "--max_round_size",
        type=int,
        default=4096,
        help="Maximum number of candidates to generate per round.",
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=200,
        help="Maximum rejection-sampling rounds.",
    )
    parser.add_argument(
        "--internal_batch_size",
        type=int,
        default=None,
        help="Internal diffusion batch size for tabddpm sampling.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Sampling steps. Used by tabsyn and tabddpm DDIM.",
    )
    parser.add_argument(
        "--ddim",
        action="store_true",
        default=False,
        help="Use DDIM for tabddpm sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    return parser.parse_args()


def load_reference_thresholds(result_path):
    result_path = Path(result_path)
    if not result_path.exists():
        raise FileNotFoundError(f"Reference result file does not exist: {result_path}")

    pattern = re.compile(
        r"mem_all:\s*([0-9eE+\-.]+),\s*cat_ori:\s*([0-9eE+\-.]+),\s*num_ori:\s*([0-9eE+\-.]+)"
    )
    with open(result_path, "r", encoding="utf-8") as handle:
        for line in reversed(handle.readlines()):
            match = pattern.search(line)
            if match:
                return {
                    "reject_threshold": float(match.group(1)),
                    "cat_threshold": float(match.group(2)),
                    "num_threshold": float(match.group(3)),
                }

    raise ValueError(
        f"Unable to find a 'mem_all/cat_ori/num_ori' summary line in {result_path}"
    )


def resolve_threshold_config(args):
    fallback = {
        "cat_threshold": 1.0 / 3.0,
        "num_threshold": 1.0 / 3.0,
        "weight_cat": 0.95,
        "reject_rule": "weighted_indicator",
        "reject_threshold": 1.0 / 3.0,
        "threshold_source": "fallback defaults",
    }

    if args.disable_reference_thresholds:
        resolved = dict(fallback)
    else:
        preset = REFERENCE_THRESHOLD_PRESETS.get(args.dataname, {})
        reference_result_path = args.reference_result_path or preset.get("result_path")
        if reference_result_path is not None:
            resolved = {
                **fallback,
                **load_reference_thresholds(reference_result_path),
                "weight_cat": preset.get("weight_cat", fallback["weight_cat"]),
                "reject_rule": preset.get("reject_rule", fallback["reject_rule"]),
                "threshold_source": f"reference result: {Path(reference_result_path)}",
            }
        else:
            resolved = dict(fallback)

    cli_overrides = {
        "cat_threshold": args.cat_threshold,
        "num_threshold": args.num_threshold,
        "weight_cat": args.weight_cat,
        "reject_rule": args.reject_rule,
        "reject_threshold": args.reject_threshold,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            resolved[key] = value

    return resolved


def resolve_save_path(args):
    preset = REFERENCE_THRESHOLD_PRESETS.get(args.dataname, {})
    return Path(args.save_path or preset.get("default_save_path") or f"{args.dataname}_reject.csv")


def resolve_device(gpu):
    if gpu >= 0 and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu >= gpu_count:
            print(
                f"Requested --gpu={gpu}, but only {gpu_count} CUDA device(s) are available. "
                "Fallback to --gpu=0."
            )
            gpu = 0
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def align_columns(df, expected_columns):
    missing = [col for col in expected_columns if col not in df.columns]
    extra = [col for col in df.columns if col not in expected_columns]
    if missing or extra:
        raise ValueError(
            f"Generated columns do not match training columns. missing={missing}, extra={extra}"
        )
    return df.loc[:, expected_columns]


def load_tabsyn_components(dataname, device):
    dataset_dir = Path("data") / dataname
    tabsyn_dir = Path("tabsyn")

    with open(dataset_dir / "info.json", "r", encoding="utf-8") as handle:
        info = json.load(handle)

    task_type = info["task_type"]
    _, _, categories, d_numerical, num_inverse, cat_inverse = preprocess(
        str(dataset_dir),
        task_type=task_type,
        inverse=True,
    )

    embedding_path = tabsyn_dir / "vae" / "ckpt" / dataname / "train_z.npy"
    train_z = torch.tensor(np.load(embedding_path), dtype=torch.float32)
    train_z = train_z[:, 1:, :]
    _, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim
    train_z = train_z.view(train_z.size(0), in_dim)
    mean = train_z.mean(0)

    pre_decoder = Decoder_model(2, d_numerical, categories, 4, n_head=1, factor=32)
    decoder_path = tabsyn_dir / "vae" / "ckpt" / dataname / "decoder.pt"
    pre_decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))
    pre_decoder.eval()

    denoise_fn = TabsynMLPDiffusion(in_dim, 1024).to(device)
    model = TabsynModel(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)
    model_path = tabsyn_dir / "ckpt" / dataname / "model.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    info["pre_decoder"] = pre_decoder
    info["token_dim"] = token_dim
    idx_name_mapping = {int(key): value for key, value in info["idx_name_mapping"].items()}

    return {
        "model": model,
        "mean": mean.to(device),
        "info": info,
        "num_inverse": num_inverse,
        "cat_inverse": cat_inverse,
        "idx_name_mapping": idx_name_mapping,
        "sample_dim": in_dim,
    }


def build_tabsyn_sampler(dataname, device, steps):
    components = load_tabsyn_components(dataname, device)

    def sample_fn(num_samples):
        x_next = sample_tabsyn_latent(
            components["model"].denoise_fn_D,
            num_samples,
            components["sample_dim"],
            num_steps=steps,
            device=device,
        )
        x_next = x_next * 2 + components["mean"]
        syn_data = x_next.detach().float().cpu().numpy()
        syn_num, syn_cat, syn_target = split_tabsyn_num_cat_target(
            syn_data,
            components["info"],
            components["num_inverse"],
            components["cat_inverse"],
            str(device),
        )
        syn_df = recover_tabsyn_data(syn_num, syn_cat, syn_target, components["info"])
        syn_df.rename(columns=components["idx_name_mapping"], inplace=True)
        return syn_df

    return sample_fn


def load_tabddpm_components(dataname, device, internal_batch_size, task_name):
    baseline_dir = Path("baselines") / "tabddpm"
    config_path = baseline_dir / "configs" / f"{dataname}.toml"
    model_save_path = baseline_dir / "ckpt" / dataname / "model.pt"
    real_data_path = Path("data") / dataname

    raw_config = src.load_config(str(config_path))
    transformations = src.Transformations(**raw_config["train"]["T"])
    dataset = make_dataset(
        str(real_data_path),
        transformations,
        task_type=raw_config["task_type"],
        change_val=False,
    )

    category_sizes = np.array(dataset.get_category_sizes("train"))
    if len(category_sizes) == 0 or raw_config["train"]["T"]["cat_encoding"] == "one-hot":
        category_sizes = np.array([0])

    num_numerical_features = dataset.X_num["train"].shape[1] if dataset.X_num is not None else 0
    model_params = dict(raw_config["model_params"])
    model_params["d_in"] = int(np.sum(category_sizes) + num_numerical_features)

    model = get_tabddpm_model(
        raw_config["model_type"],
        model_params,
        num_numerical_features,
        dataset.get_category_sizes("train"),
    )
    model.load_state_dict(torch.load(model_save_path, map_location="cpu"))

    diffusion = GaussianMultinomialDiffusion(
        category_sizes,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        num_timesteps=raw_config["diffusion_params"]["num_timesteps"],
        gaussian_loss_type=raw_config["diffusion_params"]["gaussian_loss_type"],
        scheduler=raw_config["diffusion_params"].get("scheduler", "cosine"),
        device=device,
        task_name=task_name,
    )
    diffusion.to(device)
    diffusion.eval()

    with open(real_data_path / "info.json", "r", encoding="utf-8") as handle:
        info = json.load(handle)
    idx_name_mapping = {int(key): value for key, value in info["idx_name_mapping"].items()}

    return {
        "diffusion": diffusion,
        "dataset": dataset,
        "info": info,
        "idx_name_mapping": idx_name_mapping,
        "batch_size": internal_batch_size or raw_config["sample"]["batch_size"],
    }


def build_tabddpm_sampler(dataname, device, internal_batch_size, ddim, steps, task_name):
    components = load_tabddpm_components(dataname, device, internal_batch_size, task_name)
    num_inverse = components["dataset"].num_transform.inverse_transform
    cat_inverse = components["dataset"].cat_transform.inverse_transform

    def sample_fn(num_samples):
        if ddim:
            x_gen = components["diffusion"].sample_all(
                num_samples,
                components["batch_size"],
                ddim=True,
                steps=steps,
            )
        else:
            x_gen = components["diffusion"].sample_all(
                num_samples,
                components["batch_size"],
                ddim=False,
            )

        syn_num, syn_cat, syn_target = split_tabddpm_num_cat_target(
            x_gen,
            components["info"],
            num_inverse,
            cat_inverse,
        )
        syn_df = recover_tabddpm_data(syn_num, syn_cat, syn_target, components["info"])
        syn_df.rename(columns=components["idx_name_mapping"], inplace=True)
        return syn_df

    return sample_fn


def build_sampler(args, device):
    task_name = f"{args.dataname}_reject"
    if args.method == "tabsyn":
        return build_tabsyn_sampler(args.dataname, device, args.steps)
    if args.method == "tabddpm":
        return build_tabddpm_sampler(
            args.dataname,
            device,
            args.internal_batch_size,
            args.ddim,
            args.steps,
            task_name,
        )
    raise ValueError(f"Unsupported method: {args.method}")


def choose_round_size(remaining, oversample_factor, min_round_size, max_round_size):
    target = int(math.ceil(remaining * oversample_factor))
    return max(min_round_size, min(max_round_size, target))


def summarize_stats(stats_df, weight_cat):
    if stats_df.empty:
        return {
            "cat_mem": 0.0,
            "num_mem": 0.0,
            "mem_weight": 0.0,
            "weighted_ratio_mean": float("nan"),
            "weighted_indicator_mean": float("nan"),
        }

    cat_mem = float(stats_df["cat_mem"].mean())
    num_mem = float(stats_df["num_mem"].mean())
    mem_weight = weight_cat * cat_mem + (1.0 - weight_cat) * num_mem
    return {
        "cat_mem": cat_mem,
        "num_mem": num_mem,
        "mem_weight": mem_weight,
        "weighted_ratio_mean": float(stats_df["weighted_ratio"].mean()),
        "weighted_indicator_mean": float(stats_df["weighted_indicator"].mean()),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    threshold_config = resolve_threshold_config(args)
    args.cat_threshold = threshold_config["cat_threshold"]
    args.num_threshold = threshold_config["num_threshold"]
    args.weight_cat = threshold_config["weight_cat"]
    args.reject_rule = threshold_config["reject_rule"]
    args.reject_threshold = threshold_config["reject_threshold"]

    if not 0.0 <= args.weight_cat <= 1.0:
        raise ValueError("--weight_cat must be in [0, 1].")
    if args.min_round_size <= 0 or args.max_round_size <= 0:
        raise ValueError("--min_round_size and --max_round_size must be positive.")
    if args.min_round_size > args.max_round_size:
        raise ValueError("--min_round_size cannot be larger than --max_round_size.")

    device = resolve_device(args.gpu)
    train_path = Path(args.train_path or Path("data") / args.dataname / "train.csv")
    save_path = resolve_save_path(args)

    train_df = pd.read_csv(train_path)
    target_samples = args.target_samples or len(train_df)
    expected_columns = train_df.columns.tolist()

    rejector = MemorizationRejector(
        dataname=args.dataname,
        train_df=train_df,
        cat_threshold=args.cat_threshold,
        num_threshold=args.num_threshold,
        weight_cat=args.weight_cat,
        reject_rule=args.reject_rule,
        reject_threshold=args.reject_threshold,
    )
    sampler = build_sampler(args, device)

    accepted_chunks = []
    accepted_stats_chunks = []
    accepted_total = 0
    total_generated = 0
    total_discarded = 0
    round_idx = 0
    start_time = time.time()

    print(f"Sampling method: {args.method}")
    print(f"Reference train data: {train_path}")
    print(f"Target accepted rows: {target_samples}")
    print(f"Threshold source: {threshold_config['threshold_source']}")
    print(
        "Reject config: "
        f"rule={args.reject_rule}, cat_threshold={args.cat_threshold}, "
        f"num_threshold={args.num_threshold}, reject_threshold={args.reject_threshold}, "
        f"weight_cat={args.weight_cat}"
    )

    while accepted_total < target_samples:
        round_idx += 1
        if round_idx > args.max_rounds:
            raise RuntimeError(
                f"Reached max_rounds={args.max_rounds} before collecting enough accepted samples."
            )

        remaining = target_samples - accepted_total
        round_size = choose_round_size(
            remaining,
            args.oversample_factor,
            args.min_round_size,
            args.max_round_size,
        )

        round_start = time.time()
        generated_df = sampler(round_size)
        generated_df = align_columns(generated_df, expected_columns)

        accepted_df, accepted_stats_df, stats_df = rejector.filter_dataframe(generated_df)
        rule_rejected = int(stats_df["reject"].sum())
        if len(accepted_df) > remaining:
            accepted_df = accepted_df.iloc[:remaining].reset_index(drop=True)
            accepted_stats_df = accepted_stats_df.iloc[:remaining].reset_index(drop=True)

        total_generated += len(generated_df)
        total_discarded += len(generated_df) - len(accepted_df)
        accepted_total += len(accepted_df)
        accepted_chunks.append(accepted_df)
        accepted_stats_chunks.append(accepted_stats_df)

        round_summary = summarize_stats(stats_df, args.weight_cat)
        round_accept_rate = len(accepted_df) / max(len(generated_df), 1)
        print(
            f"[Round {round_idx}] generated={len(generated_df)}, accepted={len(accepted_df)}, "
            f"discarded={len(generated_df) - len(accepted_df)}, rule_rejected={rule_rejected}, "
            f"kept_total={accepted_total}/{target_samples}, "
            f"accept_rate={round_accept_rate:.2%}, mem_weight={round_summary['mem_weight']:.4f}, "
            f"time={time.time() - round_start:.2f}s"
        )

    final_df = pd.concat(accepted_chunks, ignore_index=True).iloc[:target_samples].copy()
    final_stats_df = pd.concat(accepted_stats_chunks, ignore_index=True).iloc[:target_samples].copy()
    final_df = final_df.loc[:, expected_columns]

    save_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(save_path, index=False)

    final_summary = summarize_stats(final_stats_df, args.weight_cat)
    total_time = time.time() - start_time
    print(f"Saved filtered samples to {save_path}")
    print(
        "Final stats: "
        f"generated={total_generated}, accepted={len(final_df)}, discarded={total_discarded}, "
        f"overall_accept_rate={len(final_df) / max(total_generated, 1):.2%}, "
        f"cat_mem={final_summary['cat_mem']:.4f}, num_mem={final_summary['num_mem']:.4f}, "
        f"mem_weight={final_summary['mem_weight']:.4f}, total_time={total_time:.2f}s"
    )


if __name__ == "__main__":
    main()
