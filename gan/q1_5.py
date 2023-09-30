import os

import torch
from utils import get_args

from networks import Discriminator, Generator
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    """
    TODO 1.5.1: Implement WGAN-GP loss for discriminator.
    loss = E[D(fake_data)] - E[D(real_data)] + lambda * E[(|| grad wrt interpolated_data (D(interpolated_data))|| - 1)^2]
    """
    ##################################################################
    # TODO 1.5: Implement WGAN-GP loss for discriminator.
    # loss_pt1 = E[D(fake_data)] - E[D(real_data)]
    # loss_pt2 = lambda * E[(|| grad wrt interpolated_data (D(interpolated_data))|| - 1)^2]
    # loss = loss_pt1 + loss_pt2
    ##################################################################
    loss = None
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


def compute_generator_loss(discrim_fake):
    ##################################################################
    # TODO 1.5: Implement WGAN-GP loss for generator.
    # loss = - E[D(fake_data)]
    ##################################################################
    loss = None
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_wgan_gp/"
    os.makedirs(prefix, exist_ok=True)

    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
