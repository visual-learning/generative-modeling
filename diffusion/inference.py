import argparse
import os
import torch

from model import DiffusionModel
from unet import Unet
from utils import save_samples


parser = argparse.ArgumentParser(description='Diffusion Model Inference')
parser.add_argument('--ckpt', required=True, type=str, help="Pretrained checkpoint")
parser.add_argument('--num-images', default=8, type=int, help="Number of images per iteration")
parser.add_argument('--sampling-method', default="ddpm", type=str, help="Sampling method (ddpm vs ddim")
args = parser.parse_args()


@torch.no_grad()
def ddim_sample(diffusion_model, num_images=8, image_size=32):
    """
    Use the reverse sampling process from DDIM to go from 
    a tensor of noise to a generated image.
    """
    num_channels = diffusion_model.channels
    return diffusion_model.ddim_sample((num_images, num_channels, image_size, image_size))


@torch.no_grad()
def ddpm_sample(diffusion_model, num_images=8, image_size=32):
    """
    Use the reverse sampling process from DDIM to go from 
    a tensor of noise to a generated image.
    """
    num_channels = diffusion_model.channels
    return diffusion_model.ddpm_sample((num_images, num_channels, image_size, image_size))

def main(
    sampling_method="ddpm",
    ckpt="ckpt_diffusion/epoch_199.pth",
    prefix="data",
    num_images=8,
):
    prefix = f"{prefix}_{sampling_method}/"
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    
    sampling_timesteps = 25 if sampling_method == "ddpm" else None

    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8)
    ).cuda()
    diffusion = DiffusionModel(
        model,
        timesteps=1000,   # number of timesteps
        sampling_timesteps=sampling_timesteps,
    ).cuda()

    # load pre-trained weight
    ckpt = torch.load(ckpt)
    model.load_state_dict(ckpt["model_state_dict"])

    # run inference
    model.eval()
    if sampling_method == "ddpm":
        generated_samples = ddpm_sample(diffusion, num_images)
    elif sampling_method == "ddim":
        generated_samples = ddim_sample(diffusion, num_images)
    save_samples(generated_samples.cpu(), f"{prefix}/output.png", nrow=2)


if __name__ == "__main__":
    main(
        sampling_method=args.sampling_method,
        ckpt=args.ckpt,
        num_images=args.num_images,
    )