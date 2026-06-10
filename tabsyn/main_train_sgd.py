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

    train_z, curr_dir, _, ckpt_path, _ = get_input_train(args)
    in_dim = train_z.shape[1]

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    mean, std = train_z.mean(0), train_z.std(0)
    train_z = (train_z - mean) / 2

    batch_size = int(getattr(args, 'batch_size', 4096))
    train_loader = DataLoader(
        train_z,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    print(denoise_fn)
    num_params = sum(p.numel() for p in denoise_fn.parameters())
    print("the number of parameters", num_params)

    model = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)

    model.train()

    model_save_path = f"{curr_dir}/ckpt_sgd/{dataname}"
    os.makedirs(model_save_path, exist_ok=True)

    trainer = SimpleDPSGDTrainer(
        model, train_loader, model_save_path, device, args
    )
    trainer.run_loop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TabSyn DP-SGD Training")
    parser.add_argument("--dataname", type=str, default="adult")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dp_noise_multiplier", type=float, default=1.0)
    parser.add_argument("--dp_max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dp_delta", type=float, default=1e-5)
    parser.add_argument("--dp_steps", type=int, default=3000)
    parser.add_argument("--dp_physical_batch_size", type=int, default=1024)
    args = parser.parse_args()
    args.device = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"
    main(args)
