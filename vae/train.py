import argparse
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from model import AEModel
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
import time
import os
from utils import *

def ae_loss(model, x):
    """
    TODO 2.2: fill in MSE loss between x and its reconstruction.
    return loss, {recon_loss = loss}
    """

    return loss, OrderedDict(recon_loss=loss)

def vae_loss(model, x, beta = 1):
    """
    TODO 2.5 : Fill in recon_loss and kl_loss.
    NOTE: For the kl loss term for the VAE, implement the loss in closed form, you can find the formula here:
    (https://stats.stackexchange.com/questions/318748/deriving-the-kl-divergence-loss-for-vaes).
    return loss, {recon_loss = loss}
    """
    return total_loss, OrderedDict(recon_loss=recon_loss, kl_loss=kl_loss)


def constant_beta_scheduler(target_val = 1):
    def _helper(epoch):
        return target_val
    return _helper

def linear_beta_scheduler(max_epochs=None, target_val = 1):
    """
    TODO 2.8 : Fill in helper. The value returned should increase linearly
    from 0 at epoch 0 to target_val at epoch max_epochs
    """
    def _helper(epoch):
    return _helper

def run_train_epoch(model, loss_mode, train_loader, optimizer, beta = 1, grad_clip = 1):
    model.train()
    all_metrics = []
    for x, _ in train_loader:
        x = preprocess_data(x)
        if loss_mode == 'ae':
            loss, _metric = ae_loss(model, x)
        elif loss_mode == 'vae':
            loss, _metric = vae_loss(model, x, beta)
        all_metrics.append(_metric)
        optimizer.zero_grad()
        loss.backward()

        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    return avg_dict(all_metrics)


def get_val_metrics(model, loss_mode, val_loader):
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for x, _ in val_loader:
            x = preprocess_data(x)
            if loss_mode == 'ae':
                _, _metric = ae_loss(model, x)
            elif loss_mode == 'vae':
                _, _metric = vae_loss(model, x)
            all_metrics.append(_metric)

    return avg_dict(all_metrics)

def main(log_dir, loss_mode = 'vae', beta_mode = 'constant', num_epochs = 20, batch_size = 256, latent_size = 256,
         target_beta_val = 1, grad_clip=1, lr = 1e-3, eval_interval = 5):

    os.makedirs('data/'+ log_dir, exist_ok = True)
    train_loader, val_loader = get_dataloaders()

    variational = True if loss_mode == 'vae' else False
    model = AEModel(variational, latent_size, input_shape = (3, 32, 32)).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    vis_x = next(iter(val_loader))[0][:36]

    #beta_mode is for part 2.3, you can ignore it for parts 2.1, 2.2
    if beta_mode == 'constant':
        beta_fn = constant_beta_scheduler(target_val = target_beta_val)
    elif beta_mode == 'linear':
        beta_fn = linear_beta_scheduler(max_epochs=num_epochs, target_val = target_beta_val)

    plot_metrics = {}
    val_metrics = get_val_metrics(model, loss_mode, val_loader)

    for epoch in range(num_epochs):
        print('epoch', epoch)
        train_metrics = run_train_epoch(model, loss_mode, train_loader, optimizer, beta_fn(epoch))
        val_metrics = get_val_metrics(model, loss_mode, val_loader)
        for k in val_metrics.keys():
            if k not in plot_metrics:
                plot_metrics[k] = []
            plot_metrics[k].append(val_metrics[k])
        if (epoch+1)%eval_interval == 0:
            print(epoch, train_metrics)
            print(epoch, val_metrics)

            vis_recons(model, vis_x, 'data/'+log_dir+ '/epoch_'+str(epoch))
            if loss_mode == 'vae':
                vis_samples(model, 'data/'+log_dir+ '/epoch_'+str(epoch) )
    for k,v in plot_metrics.items():
        plt.clf()
        save_plot(list(range(len(v))), v, "Epochs", k, "{k} vs. Epochs", 'data/' + log_dir + f'/{k}_vs_iterations')

if __name__ == '__main__':
    # argparser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='')
    parser.add_argument('--loss_mode', type=str, default='ae')
    parser.add_argument('--beta_mode', type=str, default='constant')
    parser.add_argument('--latent_size', type=int, default=1024)
    parser.add_argument('--target_beta_val', type=float, default=1)


    args = parser.parse_args()

    main(args.log_dir, loss_mode = args.loss_mode, beta_mode = args.beta_mode, latent_size = args.latent_size, num_epochs=20, target_beta_val = args.target_beta_val)
