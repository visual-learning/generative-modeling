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

def get_dataloaders(batch_size = 256):

    dataset_cls = datasets.CIFAR10 #datasets.CIFAR100
    train_loader = torch.utils.data.DataLoader(
        dataset_cls(root='data', train=True, transform=transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            #transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        dataset_cls(root='data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
           # transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    return train_loader, val_loader

def preprocess_data(x):
    x = 2*x - 1
    return x.to('cuda')

def avg_dict(all_metrics):
    keys = all_metrics[0].keys()
    avg_metrics = {}
    for key in keys:
        avg_metrics[key] = np.mean([all_metrics[i][key].cpu().detach().numpy() for i in range(len(all_metrics))])
    return avg_metrics

def save_samples(samples, fname, nrow=6, title='Samples'):
    plt.clf()
    samples = (torch.FloatTensor(samples) / 255.).permute(0, 3, 1, 2)
    grid_img = make_grid(samples, nrow=nrow)

    plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.tight_layout()
    plt.savefig(fname)

def vis_samples(model, _file, num_samples = 49):

    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_size).cuda()
        samples = torch.clamp(model.decoder(z), -1, 1)

    samples = samples.cpu().permute(0, 2, 3, 1).numpy() * 0.5 + 0.5
    save_samples(samples*255, _file+'_samples.png')

def vis_recons(model, x, _file):

    with torch.no_grad():

        x = preprocess_data(x)
        enc_out = model.encoder(x)
        if type(enc_out) is tuple:
            z = enc_out[0]
        else:
            z = enc_out
        x_recon = torch.clamp(model.decoder(z), -1, 1)

    reconstructions = torch.stack((x, x_recon), dim=1).view(-1, 3, 32, 32) * 0.5 + 0.5
    reconstructions = reconstructions.permute(0, 2, 3, 1).cpu().numpy() * 255

    save_samples(reconstructions, _file+'_recons.png')

def save_plot(x, y, xlabel, ylabel, title, filename):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename + ".png")
