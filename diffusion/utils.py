import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def save_samples(samples, fname, nrow=6, title='Samples'):
    grid_img = make_grid(samples, nrow=nrow)
   
    plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.tight_layout()
    plt.savefig(fname)

# helpers functions for diffusion
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# normalization functions
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5
