import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (
    cosine_beta_schedule,
    default,
    extract,
    unnormalize_to_zero_to_one,
)

class DiffusionModel(nn.Module):
    def __init__(
        self,
        model,
        timesteps = 1000,
        sampling_timesteps = None,
        ddim_sampling_eta = 1.
    ):
        super(DiffusionModel, self).__init__()
        assert model.channels == model.out_dim
        assert not model.learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.device = torch.cuda.current_device()

        self.betas = cosine_beta_schedule(timesteps).to(self.device)
        self.num_timesteps = self.betas.shape[0]

        alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value = 1.)
    
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod =  torch.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (
            1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(min =1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(
            self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (
            1. - self.alphas_cumprod)

        # sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

    def predict_start_image_from_noise(self, x_t, t, noise):
        # TODO (Q3.1): given a noised image x_t and noise tensor, predict x_0
        x_start = None

        return x_start

    def get_posterior_parameters(self, x_start, x_t, t):
        # Compute the posterior mean and variance for x_{t-1} 
        # using the coefficients, x_t, and x_0
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t):
        # TODO (Q3.1): given a noised image x_t, predict x_0 and the additive noise
        # to predict the additive noise, use the denoising model.
        pred_noise = None
        x_start = None

        return (pred_noise, x_start)

    def mean_variance_at_previous_timestep(self, x, t):
        # TODO (Q3.1): predict the mean and variance for the posterior (x_{t-1})
        # Hint: To do this, you will need a predicted x_0. Which function can do this for you?
        x_start = None
        x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = None, None, None
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def predict_denoised_at_prev_timestep(self, x, t: int):
        # TODO (3.1): given x at timestep t, predict the denoised image at x_{t-1}.
        # also return the predicted starting image.
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img, x_start = None

        return pred_img, x_start

    @torch.no_grad()
    def ddpm_sample(self, shape):
        # TODO (Q3.1): implement the DDPM sampling process.
        # Hint: while returning the final image, ensure it is scaled between [0,1].
        img = None

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, shape):
        # TODO (Q3.2.1): implement the DDIM sampling process.
        img = None

        img = unnormalize_to_zero_to_one(img)
        return img
