"""
Based on https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
and https://github.com/ehoogeboom/multinomial_diffusion
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import pandas as pd
import numpy as np
import json

from baselines.tabddpm.models.utils import *
import src
from cal_memorization import cal_mem_weight, cal_cat_ori, cal_mem_weight_shoppers
from utils_train import make_dataset
from Gcat import model_output # TODO 是否采用原生态Gcat
# from Gcat_mid import model_output
from rebuttle.cal_table_correlation import cal_table_correlation
from calc_cost import profile_block

"""
Based in part on: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L281
"""
eps = 1e-8

def sample_laplace_noise(shape, b, device="cpu"):
    # 使用指数分布构造法更稳定
    exp = torch.distributions.Exponential(1 / b)
    u = torch.rand(shape, device=device)
    sign = torch.where(u < 0.5, -1.0, 1.0)
    noise = sign * exp.sample(sample_shape=shape).to(device)
    return noise


# ---- add data utilities ----
@torch.no_grad()
def split_num_cat_target(syn_data, info, num_inverse, cat_inverse):
    task_type = info['task_type']

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    n_num_feat = len(num_col_idx)
    n_cat_feat = len(cat_col_idx)

    if task_type == 'regression':
        n_num_feat += len(target_col_idx)
    else:
        n_cat_feat += len(target_col_idx)

    syn_num = syn_data[:, :n_num_feat]
    syn_cat = syn_data[:, n_num_feat:]

    syn_num = num_inverse(syn_num).astype(np.float32)
    syn_cat = cat_inverse(syn_cat)


    if info['task_type'] == 'regression':
        syn_target = syn_num[:, :len(target_col_idx)]
        syn_num = syn_num[:, len(target_col_idx):]
    
    else:
        # print(syn_cat.shape)
        syn_target = syn_cat[:, :len(target_col_idx)]
        syn_cat = syn_cat[:, len(target_col_idx):]

    return syn_num, syn_cat, syn_target

def recover_data(syn_num, syn_cat, syn_target, info):

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']


    idx_mapping = info['idx_mapping']
    idx_mapping = {int(key): value for key, value in idx_mapping.items()}

    syn_df = pd.DataFrame()

    if info['task_type'] == 'regression':
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]] 
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]


    else:
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]]
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]

    return syn_df

def predict_x0(x_t, model_out, alpha_bar_t):
    """
    根据 Diffusion Kernel 公式预测 \hat{x}_0

    参数:
    x_t: 当前时间步的样本
    model_out: 模型输出的噪声 \(\hat{\epsilon}\)
    alpha_bar_t: 当前时间步的 \(\bar{\alpha}_t\)

    返回:
    \hat{x}_0: 预测的 \hat{x}_0
    """
    alpha_bar_t_cpu = alpha_bar_t.cpu().numpy()
    
    # 计算 sqrt_alpha_bar_t 和 sqrt_one_minus_alpha_bar_t
    sqrt_alpha_bar_t = torch.tensor(np.sqrt(alpha_bar_t_cpu)).to(x_t.device)
    sqrt_one_minus_alpha_bar_t = torch.tensor(np.sqrt(1 - alpha_bar_t_cpu)).to(x_t.device)
    
    # 计算 x0_hat
    x0_hat = (x_t - sqrt_one_minus_alpha_bar_t * model_out) / sqrt_alpha_bar_t
    return x0_hat
# ---- add data utilities ----

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class GaussianMultinomialDiffusion(torch.nn.Module):
    def __init__(
            self,
            num_classes: np.array,
            num_numerical_features: int,
            denoise_fn,
            num_timesteps=1000,
            gaussian_loss_type='mse',
            gaussian_parametrization='eps',
            multinomial_loss_type='vb_stochastic',
            parametrization='x0',
            scheduler='cosine',
            device=torch.device('cpu'),
            task_name=None # add by lxler
        ):

        super(GaussianMultinomialDiffusion, self).__init__()
        assert multinomial_loss_type in ('vb_stochastic', 'vb_all')
        assert parametrization in ('x0', 'direct')

        if multinomial_loss_type == 'vb_all':
            print('Computing the loss using the bound on _all_ timesteps.'
                  ' This is expensive both in terms of memory and computation.')
        self.task_name = task_name
        self.num_numerical_features = num_numerical_features
        self.num_classes = num_classes # it as a vector [K1, K2, ..., Km]
        self.num_classes_expanded = torch.from_numpy(
            np.concatenate([num_classes[i].repeat(num_classes[i]) for i in range(len(num_classes))])
        ).to(device)

        self.slices_for_classes = [np.arange(self.num_classes[0])]
        offsets = np.cumsum(self.num_classes)
        for i in range(1, len(offsets)):
            self.slices_for_classes.append(np.arange(offsets[i - 1], offsets[i]))
        self.offsets = torch.from_numpy(np.append([0], offsets)).to(device)

        self._denoise_fn = denoise_fn
        self.gaussian_loss_type = gaussian_loss_type
        self.gaussian_parametrization = gaussian_parametrization
        self.multinomial_loss_type = multinomial_loss_type
        self.num_timesteps = num_timesteps
        self.parametrization = parametrization
        self.scheduler = scheduler

        alphas = 1. - get_named_beta_schedule(scheduler, num_timesteps)
        alphas = torch.tensor(alphas.astype('float64')) # alpha2_t
        betas = 1. - alphas    # beta2_t

        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)

        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)

        alphas_cumprod = np.cumprod(alphas, axis=0) # tilde_alpha2_t
        alphas_cumprod_prev = torch.tensor(np.append(1.0, alphas_cumprod[:-1]))  # tilde_alpha2_{t-1}
        alphas_cumprod_next = torch.tensor(np.append(alphas_cumprod[1:], 0.0)) # tilde_alpha2_{t+1}
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)    # tilde_alpha_t
        sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod) # tilde_beta_t
        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)    # sqrt(1 / tilde_alpha_t)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1) # sqrt(tilde_beta_t / tilde_alpha_t )

        # Gaussian diffusion

        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.from_numpy(
            np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        ).float().to(device)
        self.posterior_mean_coef1 = (
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).float().to(device)
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev)
            * np.sqrt(alphas.numpy())
            / (1.0 - alphas_cumprod)
        ).float().to(device)

        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item() < 1e-5
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < 1.e-5

        # Convert to float32 and register buffers.
        self.register_buffer('alphas', alphas.float().to(device))
        self.register_buffer('log_alpha', log_alpha.float().to(device))
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float().to(device))
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float().to(device))
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float().to(device))
        self.register_buffer('alphas_cumprod', alphas_cumprod.float().to(device))
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float().to(device))
        self.register_buffer('alphas_cumprod_next', alphas_cumprod_next.float().to(device))
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod.float().to(device))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod.float().to(device))
        self.register_buffer('sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod.float().to(device))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', sqrt_recipm1_alphas_cumprod.float().to(device))

        self.register_buffer('Lt_history', torch.zeros(num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(num_timesteps))
    
    # Gaussian part
    def gaussian_q_mean_variance(self, x_start, t):
        mean = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_1_min_cumprod_alpha, t, x_start.shape
        )
        return mean, variance, log_variance
    
    def gaussian_q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )
    
    def gaussian_q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def gaussian_p_mean_variance(
        self, model_output, x, t, clip_denoised=False, denoised_fn=None, model_kwargs=None
    ):
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)

        model_variance = torch.cat([self.posterior_variance[1].unsqueeze(0).to(x.device), (1. - self.alphas)[1:]], dim=0)
        # model_variance = self.posterior_variance.to(x.device)
        model_log_variance = torch.log(model_variance)

        model_variance = extract(model_variance, t, x.shape)
        model_log_variance = extract(model_log_variance, t, x.shape)


        if self.gaussian_parametrization == 'eps':
            pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        elif self.gaussian_parametrization == 'x0':
            pred_xstart = model_output
        else:
            raise NotImplementedError
            
        model_mean, _, _ = self.gaussian_q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        ), f'{model_mean.shape}, {model_log_variance.shape}, {pred_xstart.shape}, {x.shape}'

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }
    
    def _vb_terms_bpd(
        self, model_output, x_start, x_t, t, clip_denoised=False, model_kwargs=None
    ):
        true_mean, _, true_log_variance_clipped = self.gaussian_q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.gaussian_p_mean_variance(
            model_output, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"], "out_mean": out["mean"], "true_mean": true_mean}
    
    def _prior_gaussian(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.gaussian_q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)
    
    def _gaussian_loss(self, model_out, x_start, x_t, t, noise, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        terms = {}
        if self.gaussian_loss_type == 'mse':
            terms["loss"] = mean_flat((noise - model_out) ** 2)
        elif self.gaussian_loss_type == 'kl':
            terms["loss"] = self._vb_terms_bpd(
                model_output=model_out,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]


        return terms['loss']
    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def gaussian_p_sample(
        self,
        model_out,
        x,
        t,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
    ):
        out = self.gaussian_p_mean_variance(
            model_out,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "out": out}

    # dp_sample
    def gaussian_p_sample_dp(
            self,
            model_out,
            x,
            t,
            clip_denoised=False,
            denoised_fn=None,
            model_kwargs=None,
        ):
            out = self.gaussian_p_mean_variance(
                model_out,
                x,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            noise = torch.randn_like(x)
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
            )  # no noise when t == 0

            dp_sensitivity, epsilon = 0.8, 8 # TODO
            
            temp = dp_sensitivity / epsilon  # 需要你设置好这两个值
            lap_noise = sample_laplace_noise(out["mean"].shape, temp, device=out["mean"].device)
            sample = out["mean"] + nonzero_mask * lap_noise
            return {"sample": sample, "pred_xstart": out["pred_xstart"], "out": out}
    
    # Multinomial part

    def multinomial_kl(self, log_prob1, log_prob2):

        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)

        return kl

    def q_pred_one_timestep(self, log_x_t, t):
        log_alpha_t = extract(self.log_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x_t.shape)

        # alpha_t * E[xt] + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t - torch.log(self.num_classes_expanded)
        )

        return log_probs

    def q_pred(self, log_x_start, t):
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_start.shape)
        log_1_min_cumprod_alpha = extract(self.log_1_min_cumprod_alpha, t, log_x_start.shape)

        log_probs = log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - torch.log(self.num_classes_expanded)
        )

        return log_probs

    def predict_start(self, model_out, log_x_t, t):


        assert model_out.size(0) == log_x_t.size(0)
        assert model_out.size(1) == self.num_classes.sum(), f'{model_out.size()}'

        log_pred = torch.empty_like(model_out)
        for ix in self.slices_for_classes:
            log_pred[:, ix] = F.log_softmax(model_out[:, ix], dim=1)
        return log_pred

    def q_posterior(self, log_x_start, log_x_t, t):
        # q(xt-1 | xt, x0) = q(xt | xt-1, x0) * q(xt-1 | x0) / q(xt | x0)
        # where q(xt | xt-1, x0) = q(xt | xt-1).

        # EV_log_qxt_x0 = self.q_pred(log_x_start, t)

        # print('sum exp', EV_log_qxt_x0.exp().sum(1).mean())
        # assert False

        # log_qxt_x0 = (log_x_t.exp() * EV_log_qxt_x0).sum(dim=1)
        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1)

        num_axes = (1,) * (len(log_x_start.size()) - 1)
        t_broadcast = t.to(log_x_start.device).view(-1, *num_axes) * torch.ones_like(log_x_start)
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start, log_EV_qxtmin_x0.to(torch.float32))

        # unnormed_logprobs = log_EV_qxtmin_x0 +
        #                     log q_pred_one_timestep(x_t, t)
        # Note: _NOT_ x_tmin1, which is how the formula is typically used!!!
        # Not very easy to see why this is true. But it is :)
        unnormed_logprobs = log_EV_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t)

        sliced = sliced_logsumexp(unnormed_logprobs, self.offsets)
        log_EV_xtmin_given_xt_given_xstart = unnormed_logprobs - sliced

        return log_EV_xtmin_given_xt_given_xstart

    def p_pred(self, model_out, log_x, t):
        if self.parametrization == 'x0':
            log_x_recon = self.predict_start(model_out, log_x, t=t)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(model_out, log_x, t=t)
        else:
            raise ValueError


        return log_model_pred

    @torch.no_grad()
    def p_sample(self, model_out, log_x, t):
        model_log_prob = self.p_pred(model_out, log_x=log_x, t=t)
        out = self.log_sample_categorical(model_log_prob)
        return out

    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = self.log_alpha.device

        b = shape[0]
        # start with random normal image.
        img = torch.randn(shape, device=device)

        for i in reversed(range(1, self.num_timesteps)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img

    @torch.no_grad()
    def _sample(self, image_size, batch_size = 16):
        return self.p_sample_loop((batch_size, 3, image_size, image_size))

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in reversed(range(0, t)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def log_sample_categorical(self, logits):
        full_sample = []
        for i in range(len(self.num_classes)):
            one_class_logits = logits[:, self.slices_for_classes[i]]
            uniform = torch.rand_like(one_class_logits)
            gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
            sample = (gumbel_noise + one_class_logits).argmax(dim=1)
            full_sample.append(sample.unsqueeze(1))
        full_sample = torch.cat(full_sample, dim=1)
        log_sample = index_to_log_onehot(full_sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t):
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)

        log_sample = self.log_sample_categorical(log_EV_qxt_x0)

        return log_sample

    def nll(self, log_x_start):
        b = log_x_start.size(0)
        device = log_x_start.device
        loss = 0
        for t in range(0, self.num_timesteps):
            t_array = (torch.ones(b, device=device) * t).long()

            kl = self.compute_Lt(
                log_x_start=log_x_start,
                log_x_t=self.q_sample(log_x_start=log_x_start, t=t_array),
                t=t_array)

            loss += kl

        loss += self.kl_prior(log_x_start)

        return loss

    def kl_prior(self, log_x_start):
        b = log_x_start.size(0)
        device = log_x_start.device
        ones = torch.ones(b, device=device).long()

        log_qxT_prob = self.q_pred(log_x_start, t=(self.num_timesteps - 1) * ones)
        log_half_prob = -torch.log(self.num_classes_expanded * torch.ones_like(log_qxT_prob))

        kl_prior = self.multinomial_kl(log_qxT_prob, log_half_prob)

        return sum_except_batch(kl_prior)

    def compute_Lt(self, model_out, log_x_start, log_x_t, t, detach_mean=False):
        log_true_prob = self.q_posterior(
            log_x_start=log_x_start, log_x_t=log_x_t, t=t)
        log_model_prob = self.p_pred(model_out, log_x=log_x_t, t=t)

        if detach_mean:
            log_model_prob = log_model_prob.detach()

        kl = self.multinomial_kl(log_true_prob, log_model_prob)

        # if torch.isinf(kl).nonzero().shape[0] != 0:
        #     idx = torch.isinf(kl).nonzero()[0]
        #     print('KL 0 :', kl[idx])

        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        loss = mask * decoder_nll + (1. - mask) * kl        

        return loss

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = (Lt_sqrt / Lt_sqrt.sum()).to(device)

            t = torch.multinomial(pt_all, num_samples=b, replacement=True).to(device)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def _multinomial_loss(self, model_out, log_x_start, log_x_t, t, pt):

        if self.multinomial_loss_type == 'vb_stochastic':

            kl = self.compute_Lt(
                model_out, log_x_start, log_x_t, t
            )
            kl_prior = self.kl_prior(log_x_start)
            # Upweigh loss term of the kl

            vb_loss = kl / pt + kl_prior


            return vb_loss

        elif self.multinomial_loss_type == 'vb_all':
            # Expensive, dont do it ;).
            # DEPRECATED
            return -self.nll(log_x_start)
        else:
            raise ValueError()

    def log_prob(self, x):
        b, device = x.size(0), x.device
        if self.training:
            return self._multinomial_loss(x)

        else:
            log_x_start = index_to_log_onehot(x, self.num_classes)

            t, pt = self.sample_time(b, device, 'importance')

            kl = self.compute_Lt(
                log_x_start, self.q_sample(log_x_start=log_x_start, t=t), t)

            kl_prior = self.kl_prior(log_x_start)

            # Upweigh loss term of the kl
            loss = kl / (pt + 1e-6) + kl_prior

            return -loss
    
    @torch.no_grad()
    def loss_at_step_t(self, x, step):

        b = x.shape[0]
        device = x.device

        t = (torch.ones((b,)) * step).long().to(device)
        pt = torch.ones_like(t).float() / self.num_timesteps

        x_num = x[:, :self.num_numerical_features]
        x_cat = x[:, self.num_numerical_features:]
        
        x_num_t = x_num
        log_x_cat_t = x_cat
        if x_num.shape[1] > 0:
            noise = torch.randn_like(x_num)
            x_num_t = self.gaussian_q_sample(x_num, t, noise=noise)
        if x_cat.shape[1] > 0:
            log_x_cat = index_to_log_onehot(x_cat.long(), self.num_classes)
            log_x_cat_t = self.q_sample(log_x_start=log_x_cat, t=t)
        
        x_in = torch.cat([x_num_t, log_x_cat_t], dim=1)

        model_out = self._denoise_fn(
            x_in,
            t
        )

        model_out_num = model_out[:, :self.num_numerical_features]
        model_out_cat = model_out[:, self.num_numerical_features:]

        loss_multi = torch.zeros((1,)).float()
        loss_gauss = torch.zeros((1,)).float()
        if x_cat.shape[1] > 0:
            loss_multi = self._multinomial_loss(model_out_cat, log_x_cat, log_x_cat_t, t, pt) / len(self.num_classes)
        
        if x_num.shape[1] > 0:
            loss_gauss = self._gaussian_loss(model_out_num, x_num, x_num_t, t, noise)

        recon_x0_num = self.recon_x0(x_in, model_out, t)[:,:self.num_numerical_features]

        recon_loss = self._gaussian_loss(recon_x0_num, x_num, x_num_t, t, x_num)

        return loss_multi.mean(), loss_gauss.mean(), recon_loss.mean()
    
    @torch.no_grad()
    def recon_x0(self, x, model_out, t):
        # x_num = x[:, :self.num_numerical_features]

        x0 = extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * (x - model_out * extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape))
    
        return x0

    def mixed_loss(self, x, reduce: bool = True):
        b = x.shape[0]
        device = x.device
        t, pt = self.sample_time(b, device, 'uniform')

        x_num = x[:, :self.num_numerical_features]
        x_cat = x[:, self.num_numerical_features:]
        
        x_num_t = x_num
        log_x_cat_t = x_cat
        if x_num.shape[1] > 0:
            noise = torch.randn_like(x_num)
            x_num_t = self.gaussian_q_sample(x_num, t, noise=noise)
        if x_cat.shape[1] > 0:
            log_x_cat = index_to_log_onehot(x_cat.long(), self.num_classes)
            log_x_cat_t = self.q_sample(log_x_start=log_x_cat, t=t)
        
        x_in = torch.cat([x_num_t, log_x_cat_t], dim=1)

        model_out = self._denoise_fn(
            x_in,
            t
        )

        model_out_num = model_out[:, :self.num_numerical_features]
        model_out_cat = model_out[:, self.num_numerical_features:]

        loss_multi = torch.zeros(b, device=device, dtype=x.dtype)
        loss_gauss = torch.zeros(b, device=device, dtype=x.dtype)
        
        if x_cat.shape[1] > 0:
            loss_multi = self._multinomial_loss(model_out_cat, log_x_cat, log_x_cat_t, t, pt) / len(self.num_classes)
        
        if x_num.shape[1] > 0:
            loss_gauss = self._gaussian_loss(model_out_num, x_num, x_num_t, t, noise)
        if reduce:
            return loss_multi.mean(), loss_gauss.mean()
        return loss_multi, loss_gauss
    
    @torch.no_grad()
    def mixed_elbo(self, x0):
        b = x0.size(0)
        device = x0.device

        x_num = x0[:, :self.num_numerical_features]
        x_cat = x0[:, self.num_numerical_features:]
        has_cat = x_cat.shape[1] > 0
        if has_cat:
            log_x_cat = index_to_log_onehot(x_cat.long(), self.num_classes).to(device)

        gaussian_loss = []
        xstart_mse = []
        mse = []
        mu_mse = []
        out_mean = []
        true_mean = []
        multinomial_loss = []
        for t in range(self.num_timesteps):
            t_array = (torch.ones(b, device=device) * t).long()
            noise = torch.randn_like(x_num)

            x_num_t = self.gaussian_q_sample(x_start=x_num, t=t_array, noise=noise)
            if has_cat:
                log_x_cat_t = self.q_sample(log_x_start=log_x_cat, t=t_array)
            else:
                log_x_cat_t = x_cat

            model_out = self._denoise_fn(
                torch.cat([x_num_t, log_x_cat_t], dim=1),
                t_array
            )
            
            model_out_num = model_out[:, :self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features:]

            kl = torch.tensor([0.0])
            if has_cat:
                kl = self.compute_Lt(
                    model_out=model_out_cat,
                    log_x_start=log_x_cat,
                    log_x_t=log_x_cat_t,
                    t=t_array
                )

            out = self._vb_terms_bpd(
                model_out_num,
                x_start=x_num,
                x_t=x_num_t,
                t=t_array,
                clip_denoised=False
            )

            multinomial_loss.append(kl)
            gaussian_loss.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_num) ** 2))
            # mu_mse.append(mean_flat(out["mean_mse"]))
            out_mean.append(mean_flat(out["out_mean"]))
            true_mean.append(mean_flat(out["true_mean"]))

            eps = self._predict_eps_from_xstart(x_num_t, t_array, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        gaussian_loss = torch.stack(gaussian_loss, dim=1)
        multinomial_loss = torch.stack(multinomial_loss, dim=1)
        xstart_mse = torch.stack(xstart_mse, dim=1)
        mse = torch.stack(mse, dim=1)
        # mu_mse = torch.stack(mu_mse, dim=1)
        out_mean = torch.stack(out_mean, dim=1)
        true_mean = torch.stack(true_mean, dim=1)



        prior_gauss = self._prior_gaussian(x_num)

        prior_multin = torch.tensor([0.0])
        if has_cat:
            prior_multin = self.kl_prior(log_x_cat)

        total_gauss = gaussian_loss.sum(dim=1) + prior_gauss
        total_multin = multinomial_loss.sum(dim=1) + prior_multin
        return {
            "total_gaussian": total_gauss,
            "total_multinomial": total_multin,
            "losses_gaussian": gaussian_loss,
            "losses_multinimial": multinomial_loss,
            "xstart_mse": xstart_mse,
            "mse": mse,
            # "mu_mse": mu_mse
            "out_mean": out_mean,
            "true_mean": true_mean
        }

    @torch.no_grad()
    def gaussian_ddim_step(
        self,
        model_out_num,
        x,
        t,
        t_prev,
        clip_denoised=False,
        denoised_fn=None,
        eta=1.0
    ):
        out = self.gaussian_p_mean_variance(
            model_out_num,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=None,
        )

        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = extract(self.alphas_cumprod, t, x.shape)
        
        if t[0] != 0:
            alpha_bar_prev = extract(self.alphas_cumprod, t_prev, x.shape)
        else:
            alpha_bar_prev = extract(self.alphas_cumprod_prev, t_prev, x.shape)
        
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise

        return sample

    
    @torch.no_grad()
    def gaussian_ddim_sample(
        self,
        noise,
        T,
        eta=0.0
    ):
        x = noise
        b = x.shape[0]
        device = x.device
        for t in reversed(range(T)):
            print(f'Sample timestep {t:4d}', end='\r')
            t_array = (torch.ones(b, device=device) * t).long()
            out_num = self._denoise_fn(x, t_array)
            x = self.gaussian_ddim_step(
                out_num,
                x,
                t_array
            )
        print()
        return x


    @torch.no_grad()
    def gaussian_ddim_reverse_step(
        self,
        model_out_num,
        x,
        t,
        clip_denoised=False,
        eta=0.0
    ):
        assert eta == 0.0, "Eta must be zero."
        out = self.gaussian_p_mean_variance(
            model_out_num,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=None,
            model_kwargs=None,
        )

        eps = (
            extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = extract(self.alphas_cumprod_next, t, x.shape)

        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_next)
            + torch.sqrt(1 - alpha_bar_next) * eps
        )

        return mean_pred

    @torch.no_grad()
    def gaussian_ddim_reverse_sample(
        self,
        x,
        T
    ):
        b = x.shape[0]
        device = x.device
        for t in range(T):
            print(f'Reverse timestep {t:4d}', end='\r')
            t_array = (torch.ones(b, device=device) * t).long()
            out_num = self._denoise_fn(x, t_array)
            x = self.gaussian_ddim_reverse_step(
                out_num,
                x,
                t_array,
                eta=0.0
            )
        print()

        return x


    @torch.no_grad()
    def multinomial_ddim_step(
        self,
        model_out_cat,
        log_x_t,
        t,
        t_prev,
        eta=1.0
    ):
        # not ddim, essentially
        log_x0 = self.predict_start(model_out_cat, log_x_t=log_x_t, t=t)

        alpha_bar = extract(self.alphas_cumprod, t, log_x_t.shape)

        if t[0] != 0:
            alpha_bar_prev = extract(self.alphas_cumprod, t_prev, log_x_t.shape)
        else:
            alpha_bar_prev = extract(self.alphas_cumprod_prev, t_prev, log_x_t.shape)
        
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        coef1 = sigma
        coef2 = alpha_bar_prev - sigma * alpha_bar
        coef3 = 1 - coef1 - coef2


        log_ps = torch.stack([
            torch.log(coef1) + log_x_t,
            torch.log(coef2) + log_x0,
            torch.log(coef3) - torch.log(self.num_classes_expanded)
        ], dim=2) 

        log_prob = torch.logsumexp(log_ps, dim=2)

        out = self.log_sample_categorical(log_prob)

        return out

    @torch.no_grad()
    def sample_ddim(self, num_samples, steps = 1000):
        b = num_samples
        device = self.log_alpha.device
        z_norm = torch.randn((b, self.num_numerical_features), device=device)

        has_cat = self.num_classes[0] != 0
        log_z = torch.zeros((b, 0), device=device).float()
        if has_cat:
            uniform_logits = torch.zeros((b, len(self.num_classes_expanded)), device=device)
            log_z = self.log_sample_categorical(uniform_logits)
        
        interval = 1000 // steps
        timesteps = list(np.arange(999, -1, -interval))

        if timesteps[-1] != 0:
            timesteps.append(0)
        
        for i in range(0, len(timesteps)):

            print(f'Sample timestep {i:4d}', end='\r')
            
            t = torch.full((b,), timesteps[i], device=device, dtype=torch.long)
            
           
            if i != len(timesteps) -1 :
                t_prev = torch.full((b,), timesteps[i+1], device=device, dtype=torch.long)
            else:
                t_prev = torch.full((b,), 0, device=device, dtype=torch.long)
                
            model_out = self._denoise_fn(
                torch.cat([z_norm, log_z], dim=1).float(),
                t
            )
            model_out_num = model_out[:, :self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features:]
            z_norm = self.gaussian_ddim_step(model_out_num, z_norm, t, t_prev, clip_denoised=False)
            if has_cat:
                log_z = self.multinomial_ddim_step(model_out_cat, log_z, t, t_prev)

        print()
        z_ohe = torch.exp(log_z).round()
        z_cat = log_z
        if has_cat:
            z_cat = ohe_to_categories(z_ohe, self.num_classes)
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample

    # @torch.no_grad()
    def sample(self, num_samples):
        # === ABLATION CONFIG === # TODO change mode
        # 选项: 'full', 'no_feedback', 'no_structured', 'no_alignment', 'baseline'
        # 每次采样前改这里切换模式
        ABLATION_MODE = 'full'  # <-- 改这里: full / no_feedback / no_structured / no_alignment / baseline
        # =======================

        b = num_samples
        device = self.log_alpha.device
        z_norm = torch.randn((b, self.num_numerical_features), device=device)
        
        has_cat = self.num_classes[0] != 0
        log_z = torch.zeros((b, 0), device=device).float()
        if has_cat:
            uniform_logits = torch.zeros((b, len(self.num_classes_expanded)), device=device)
            print(uniform_logits.shape)
            log_z = self.log_sample_categorical(uniform_logits)
        
        ############### 准备一下数据的处理工具 ##############
        # 1. 参数，和记录工具
        cc = "xx" # 连续值s
        task_name = self.task_name
        mem_list, mem_cat_list, mem_num_list = {}, {}, {}
        # co_ratio_list = {} # TODO 是否记录相关性参数
        
        dataname = "shoppers" # TODO 改成对应的数据集
        
        # 2. 路径和数据
        raw_config = src.load_config(f"./baselines/tabddpm/configs/{dataname}.toml")
        real_data_path = f"data/{dataname}"
        train_data = pd.read_csv(f"./synthetic/{dataname}/real.csv")
        print(f"train_data shape: {train_data.shape}")
        info_path = f'{real_data_path}/info.json'
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        # 3. 数据处理工具
        T = src.Transformations(**raw_config['train']['T'])
        D = make_dataset(
            real_data_path,
            T,
            task_type = raw_config['task_type'],
            change_val = False,
        )
        num_inverse = D.num_transform.inverse_transform
        cat_inverse = D.cat_transform.inverse_transform
        ############### 准备一下数据的处理工具 ##############

        for i in reversed(range(0, self.num_timesteps)):
            print(f'Sample timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            model_out = self._denoise_fn(
                torch.cat([z_norm, log_z], dim=1).float(),
                t
            )

            model_out_num = model_out[:, :self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features:]
            
            ########## gen new fun ##########
            if (i + 1) % 5 == 0: # 每隔5步引导一次 # TODO 是否采用原始方法/dp，或者还是使用新方法
            # if i == -100: # 不执行下面这段
                out = self.gaussian_p_sample(model_out_num, z_norm, t, clip_denoised=False)['out']
                noise = torch.randn_like(z_norm)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(z_norm.shape) - 1)))
                )  # no noise when t == 0
                
                # 获取mem
                z_ohe = torch.exp(log_z).round()
                z_cat = log_z
                if has_cat:
                    z_cat = ohe_to_categories(z_ohe, self.num_classes)
                    
                # 2. 处理成对应的表格sample_data
                alpha_bar_t = extract(self.alphas_cumprod, t, z_norm.shape)
                x0_hat = predict_x0(z_norm, model_out[:, :self.num_numerical_features], alpha_bar_t)
                
                syn_data = torch.cat([x0_hat, z_cat], dim=1).cpu()
                syn_data_np = syn_data[:z_norm.shape[0], :]
                syn_data = syn_data[:train_data.shape[0], :]
                syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse) 
                syn_df = recover_data(syn_num, syn_cat, syn_target, info)

                idx_name_mapping = info['idx_name_mapping']
                idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

                syn_df.rename(columns=idx_name_mapping, inplace=True)

                save_path = f"memorization/mid_{task_name}.csv"
                syn_df.to_csv(save_path, index=False)
                
                # TODO 用于计算列属性关联
                # att_list = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend', 'Revenue']
                # a1, a2 = 'Administrative', 'Revenue'
                # syn_df_att = syn_df.copy()
                # syn_df_att.columns = att_list

                # 3. 开始计算memorization的值
                cat_mem, num_mem, mem_weight = cal_mem_weight(dataname, save_path, train_data)
                
                # cat_mem_shoppers, num_mem_shoppers, mem_weight_shoppers = cal_mem_weight_shoppers(dataname, save_path, train_data, a1, a2)
                
                # 按照步长记录三个mem值
                # if (i + 1) % 5 == 0 or i == 0:
                #     # mem_list[i] = mem_weight
                #     # mem_cat_list[i] = cat_mem
                #     # mem_num_list[i] = num_mem
                #     mem_list[i] = mem_weight_shoppers
                #     mem_cat_list[i] = cat_mem_shoppers
                #     mem_num_list[i] = num_mem_shoppers
                    # 5. 开始计算co_ratio # TODO
                    # _, co_ratio = cal_table_correlation(syn_df_att, a1, a2)
                    # co_ratio_list[i] = co_ratio
                
                mem = mem_weight
                
                # 在这里调整超参数
                aa, bb, cc, threshold = -5, 600000, 1, -1 # -1是为了没有阈值，一定使用mem引导 TODO 修改

                # ablation 开关
                use_stepwise = (ABLATION_MODE != 'no_feedback')
                use_structured = (ABLATION_MODE != 'no_structured')
                use_alignment = (ABLATION_MODE != 'no_alignment')

                if use_stepwise:
                    save_path = f"memorization/mid_{task_name}.csv"
                    syn_df.to_csv(save_path, index=False)
                    cat_mem, num_mem, mem_weight = cal_mem_weight(dataname, save_path, train_data)
                    mem = mem_weight
                else:
                    # v2: 跳过mem计算，用固定值（参数也固定，不用mem动态调整）
                    mem = 0.3

                # 如果大于阈值，就使用mem引导的办法
                if mem > threshold:
                    # 运用分类器概率和梯度
                    x_hat = out['mean'].clone().float().requires_grad_(True).to(device)

                    syn_num, syn_cat, syn_target = split_num_cat_target(syn_data_np, info, num_inverse, cat_inverse)
                    x_hat_np = recover_data(syn_num, syn_cat, syn_target, info)

                    idx_name_mapping = info['idx_name_mapping']
                    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

                    x_hat_np.rename(columns=idx_name_mapping, inplace=True)

                    # 获取最后一列并准备条件
                    last_column = train_data.iloc[:, -1]
                    # last_column = last_column.replace({" <=50K": 1, " >50K": 0})  # TODO 是否是adult?
                    last_column = last_column.values
                    last_column_tensor = torch.tensor(last_column, dtype=torch.float32).reshape(train_data.shape[0], 1)
                    y_condition = last_column_tensor.float().to(device)

                    # y_pred是概率矩阵，x_hat_grad是梯度
                    # gcat_stage = 0.8 # TODO 是否调整引导阶段
                    # x_hat_grad, y_pred, _= model_output(dataname, x_hat_np, y_condition, x_hat.shape[1], stage=gcat_stage)
                    x_hat_grad, y_pred, _= model_output(dataname, x_hat_np, y_condition, x_hat.shape[1])
                    y_pred = torch.tensor(y_pred, dtype=torch.float32, device=device)
                    # y_pred = y_pred.unsqueeze(1)  # 形状从 [12000] -> [12000, 1], 概率矩阵

                    if use_stepwise:
                        para1, para2, para3 = aa * mem, bb * mem, cc * mem
                    else:
                        # v2: 固定参数，取 mem=0.2 时的值（比 typical 小一点）
                        para1, para2, para3 = -1.0, 120000, 0.2

                    if not use_structured:
                        para2 = 0
                    if not use_alignment:
                        para3 = 0

                    z_norm = x_hat + para1 * nonzero_mask * (torch.exp(0.5 * out["log_variance"]) - x_hat_grad * para2 - y_pred * para3) * noise
                    
                # 如果小于阈值，那么就使用原始方法
                else:
                    with torch.no_grad():
                        z_norm = self.gaussian_p_sample(model_out_num, z_norm, t, clip_denoised=False)['sample']
                                    
            ########## gen new fun ##########

            # 采用原始方法
            else:
                with torch.no_grad():
                    # TODO 选择是否采用dp
                    z_norm = self.gaussian_p_sample(model_out_num, z_norm, t, clip_denoised=False)['sample']
                    # 采用dp方式
                    # z_norm = self.gaussian_p_sample_dp(model_out_num, z_norm, t, clip_denoised=False)['sample']
                    
                # mem 记录
                # if (i + 1) % 5 == 0 or i == 0:
                #     # 1. 得到当前的sample数据
                #     z_ohe = torch.exp(log_z).round()
                #     z_cat = log_z
                #     if has_cat:
                #         z_cat = ohe_to_categories(z_ohe, self.num_classes)
                    
                #     syn_data = torch.cat([z_norm, z_cat], dim=1).cpu()
                #     syn_data = syn_data[:train_data.shape[0], :]
                #     syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse) 
                #     syn_df = recover_data(syn_num, syn_cat, syn_target, info)

                #     idx_name_mapping = info['idx_name_mapping']
                #     idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

                #     syn_df.rename(columns = idx_name_mapping, inplace=True)

                #     save_path = f"memorization/mid_{task_name}.csv"
                #     syn_df.to_csv(save_path, index = False)
                #     cat_mem, num_mem, mem_weight = cal_mem_weight(dataname, save_path, train_data)
                #     mem_list[i] = mem_weight
                #     mem_cat_list[i] = cat_mem
                #     mem_num_list[i] = num_mem
            
            if has_cat:
                log_z = self.p_sample(model_out_cat, log_z, t)

        # 5. 保存mem结果
        if mem_list:
            with open(f"eval/result/{task_name}.txt", "w") as f:
                f.write(f"task_name: {task_name}\n")
                for step in mem_list.keys():
                    f.write(f"step: {step}, mem: {mem_list[step]}, cat_mem: {mem_cat_list[step]}, num_mem: {mem_num_list[step]}\n")
                    # f.write(f"step: {step}, mem: {mem_list[step]}, cat_mem: {mem_cat_list[step]}, num_mem: {mem_num_list[step]}, co_ratio: {co_ratio_list[step]}\n") # TODO

        print()
        z_ohe = torch.exp(log_z).round()
        z_cat = log_z
        if has_cat:
            z_cat = ohe_to_categories(z_ohe, self.num_classes)
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample
    
    def sample_all(self, num_samples, batch_size, ddim=False, steps = 1000):
        if ddim:
            print('Sample using DDIM.')
            sample_fn = self.sample_ddim
        else:
            sample_fn = self.sample
        
        b = batch_size

        all_samples = []
        num_generated = 0
        while num_generated < num_samples:
            if not ddim:
                sample  = sample_fn(b)
            else:
                sample = sample_fn(b, steps=steps)
            mask_nan = torch.any(sample.isnan(), dim=1)
            sample = sample[~mask_nan]

            all_samples.append(sample)
      
            if sample.shape[0] != b:
                raise FoundNANsError
            num_generated += sample.shape[0]

        x_gen = torch.cat(all_samples, dim=0)[:num_samples]

        return x_gen