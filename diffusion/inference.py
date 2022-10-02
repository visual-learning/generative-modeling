"""
Created for 16-824
by Anirudh Chakravarthy (achakrav@cs.cmu.edu) and Vanshaj Chowdhary (vanshajc@andrew.cmu.edu), 2022.
"""
import argparse
import os
import torch

from model import DiffusionModel
from unet import Unet
from utils import save_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Diffusion Model Inference')
    parser.add_argument('--ckpt', required=True, type=str, help="Pretrained checkpoint")
    parser.add_argument('--num-images', default=8, type=int, help="Number of images per iteration")
    parser.add_argument('--image-size', default=32, type=int, help="Image size to generate")
    parser.add_argument('--sampling-method', choices=['ddpm', 'ddim'])
    parser.add_argument('--ddim-timesteps', type=int, default=25, help="Number of timesteps to sample for DDIM")
    parser.add_argument('--ddim-eta', type=int, default=1, help="Eta for DDIM")
    args = parser.parse_args()

    prefix = f"data_{args.sampling_method}/"
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    
    sampling_timesteps = args.ddim_timesteps if args.sampling_method == "ddpm" else None

    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8)
    ).cuda()
    diffusion = DiffusionModel(
        model,
        timesteps=1000,   # number of timesteps
        sampling_timesteps=sampling_timesteps,
        ddim_sampling_eta=args.ddim_eta,
    ).cuda()

    img_shape = (args.num_images, diffusion.channels, args.image_size, args.image_size)

    # load pre-trained weight
    ckpt = torch.load(args.ckpt)
    model.load_state_dict(ckpt["model_state_dict"])

    # run inference
    model.eval()
    if args.sampling_method == "ddpm":
        generated_samples = diffusion.ddpm_sample(img_shape)
    elif args.sampling_method == "ddim":
        generated_samples = diffusion.ddim_sample(img_shape)
    save_samples(generated_samples.cpu(), f"{prefix}/output.png", nrow=2)
