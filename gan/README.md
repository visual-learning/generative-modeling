Please follow the instructions for this part of the assignment in THIS order!

*1.1*. Let's start by setting up our networks for training a Generative Adversarial Network (GAN). As we covered in class, GANs have two networks, a generator and a discriminator. The generator takes in a noise sample z, generally sampled from the standard normal distribution, and maps it to an image. The discriminator takes in images and outputs the probability that the image is real or fake. Fill out `networks.py` wherever `#TODO 1.1` is written.

*1.2*. Now we need to setup the training code for the GAN in `train.py`. Most of the code has been provided but please fill out all of the sections that have `#TODO 1.2`.
Additionally, implement a function to do latent space interpolation (see utils.py).

*1.3*. In general, we train the generator such that it can fool the discriminator, i.e. samples from the generator will have high probability under the discriminator. Analogously, we train the discriminator such that it can tell apart real and fake images. This means our loss term encourages the discriminator to assign high probability to real images while assigning low probability to fake images. To that end, implement the original GAN losses for the generator and discriminator as described in Algorithm 1 of [1]((https://arxiv.org/pdf/1406.2661.pdf)) in `q1_3.py`. Then run `python q1_3.py` and update the report with the plots and FID score obtained.

*1.4*. For this question, please read the LSGAN paper [2](https://arxiv.org/pdf/1611.04076.pdf) and implement equation (2) as the loss for the generator and discriminator with c=1 in `q1_4.py`. Then run `python q1_4.py` and update the report with the plots and FID score obtained.

*1.5*. For this question, please read the WGAN-GP paper [3](https://arxiv.org/pdf/1704.00028.pdf) and implement the generator and discriminator losses from Algorithm 1 in `q1_5.py`. You may also refer to these slides(https://docs.google.com/presentation/d/1kZJ3RBfD-vnwOZQYlpH4qbTcICUcYhK0UXEQlnl3h4g/edit#slide=id.g4ce74c3fd4_0_185) which cover WGAN and WGAN-GP.
Additionally, implement the interpolated batch (which is necessary for the loss) in `train.py`. Then run `python q1_5.py` and update the report with the plots and FID score obtained.

## Relevant papers:
[1] Generative Adversarial Nets (Goodfellow et al, 2014): https://arxiv.org/pdf/1406.2661.pdf

[2] Least Squares Generative Adversarial Networks (Mao etclassification al, 2016): https://arxiv.org/pdf/1611.04076.pdf

[3] Improved Training of Wasserstein GANs (Gulrajani et al, 2017): https://arxiv.org/pdf/1704.00028.pdf
