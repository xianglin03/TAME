# ---tools---
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

# ---dp_sample---
def sample_laplace_noise(shape, b, device="cpu"):
    # 使用指数分布构造法更稳定
    exp = torch.distributions.Exponential(1 / b)
    u = torch.rand(shape, device=device)
    sign = torch.where(u < 0.5, -1.0, 1.0)
    noise = sign * exp.sample(sample_shape=shape).to(device)
    return noise

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

# ---gaussian_distribution part---
z_norm = self.gaussian_p_sample_dp(model_out_num, z_norm, t, clip_denoised=False)['sample']