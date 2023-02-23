## Variational Autoencoders

We will be training AutoEncoders and VAEs on the CIFAR10 dataset.

*2.1*. In model.py, fill in the TODOs where 2.1 is mentioned. This includes the encoder and decoder network architectures, and the forward passes through each.

*2.2*. In train.py, fill in the TODOs where 2.2 is mentioned. This includes the loss function for the autoencoder, which is the MSE loss between the input data and the reconstruction. Important - remember to only average across the batch dimension.

*2.3*. Train the autoencoder for 20 epochs, and try latent sizes 16, 128 and 1024. If your code is correct, the reconstructions should be very clear and sharp.
Commands:
```
python train.py --log_dir ae_latent16 --loss_mode ae --latent_size 16
python train.py --log_dir ae_latent128 --loss_mode ae --latent_size 128
python train.py --log_dir ae_latent1024 --loss_mode ae --latent_size 1024
```

*2.4*. In model.py. fill in the TODOs where 2.4 is mentioned. This only includes the fc layer of the VAEEncoder, and the forward pass through the network.

*2.5*. Fill in the recon_loss and kl_loss that make up the total loss for the VAE, under the TODO where 2.5 is mentioned. Important - remember to only average across the batch dimension.

*2.6*. Train the VAE for 20 epochs. Use the latent size from 2.3 that produces the sharpest reconstructions.
Commands:
```
python train.py --log_dir vae_latent_rep --loss_mode vae --latent_size <>
```

*2.7*. The blurriness of the samples can be reduced by tuning the value of beta. Use the latent size from 2.3 that produces the sharpest reconstructions.
```
python train.py --log_dir vae_latent_beta_.8 --loss_mode vae --latent_size <> --target_beta_val 0.8
python train.py --log_dir vae_latent_beta_1.2 --loss_mode vae --latent_size <> --target_beta_val 1.2
```

*2.8*. Another way to improve the quality of samples is to use an annealing scheme for beta. Fill in TODO for 2.8. The value of beta should increase linearly from 0 at epoch 0 to target_val at epoch max_epochs.

*2.9*. Train the VAE for 20 epochs with beta annealing using the value of beta that results in the best samples, out of 0.8, 1.0, and 1.2. Use the latent size from 2.3 that produces the sharpest reconstructions.
```
python train.py --log_dir vae_latent_beta_annealing --loss_mode vae --latent_size <> --target_beta_val <> --beta_mode linear
```

## Relevant papers:
[1] Tutorial on Variational Autoencoders (Doersch, 2016): https://arxiv.org/pdf/1606.05908.pdf

[2] Understanding disentangling in β-VAE (Burgess et al, 2018): https://arxiv.org/pdf/1804.03599.pdf
