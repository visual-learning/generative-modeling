Please follow the instructions for this part of the assignment in THIS order!

First, download the pre-trained checkpoint from https://drive.google.com/file/d/1gtn9Jv9jBUol7iJw-94hw4j6KfpG3SZE/view?usp=sharing.

Diffusion models have recently become a very popular generative modeling technique. In this assignment, we will experiment with different sampling methods for diffusion models. Diffusion models apply a series of gaussian noise to an input image, and try to denoise these noisy images by predicting the noise at each timestep. For this assignment, we will use the provided pre-trained diffusion model trained on CIFAR-10 and will implement different sampling techniques for model inference. Please refer to Lilian Weng's blog post here: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ for additional explanation/derivation. In this assignment, we are following the notation from Lilian Weng's blogpost. 

Given an input image $x_0$, the forward process sequentially applies a gaussian noise to the image, producing the following conditional distribution:
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t\mathcal{I})$$

where $\beta$ is defined according to a schedule (cosine schedule in our case), $x_t$ is the image after t applications of noise, $\mathcal{I}$ is the identity matrix.

The above equation depends on $x_{t-1}$ for each $x_t$ which implies that to get the noised output at time $t$, we need the noised output at $t-1$. However, we have fixed the beta schedule beforehand, and therefore we can reparametrize the above equation and generate any arbitrary noised input $x_t$ directly, just given our initial image $x_0$:
$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha_t}} x_0, (1 - \alpha_t)\mathcal{I})$$

$$\alpha_t = 1 - \beta_t, \bar{\alpha_t} = \prod_{i=1}^t \alpha_i$$
Note that we can sample from $q(x_t|x_0)$ to produce $x_t$.

We know how much noise we have added in the forward process. Therefore, we take the output from the forward process $x_t$ and using a denoising network (which takes the noised image and timestamp as inputs), we predict the noise $\epsilon_t$ which was added. Concretely, we have:

$$ \epsilon_t = f(x_t, t)$$

By repeating this several times, we can predict the starting image $x_0$.
$$\hat{x_0} = \frac{1}{\sqrt{\bar{\alpha_t}}} (x_t - \sqrt{1 - \bar{\alpha_t}} \epsilon_t)$$


To run inference using Denoising Diffusion Probabilistic Models (DDPM - [1]), we first sample a random noise vector $x_T$ and apply the denoising process repeatedly with the equations below to generate $x_0$.

$$ z \sim \mathcal{N}(0, \mathcal{I})$$

$$\tilde{\mu_t} = \frac{\sqrt{\alpha_t} (1 - \bar{\alpha_{t-1}})}{1 - \bar{\alpha_t}} x_t + \frac{\sqrt{\bar{\alpha_{t-1}}}\beta_t}{1 - \bar{\alpha_t}} \hat{x_0}$$

$$\sigma_t^2 = \tilde{\beta_t} = \frac{1 - \bar\alpha_{t - 1}}{1 - \bar\alpha_t} \beta_t$$

$$ x_{t-1} \sim  q(x_{t-1} | x_t, \hat{x_0} ) = \mathcal{N}(x_{t-1}; \tilde{\mu_t}, \sigma_{t} \mathcal{I}) $$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (Notation: $q(x_{t-1} | x_t, \hat{x_0} )$ denotes the denoising process, which is denoted by $p_{\theta}$ in class.)


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Since this is Gaussian, we can use the reparameterization trick to implement the sampling:
$$x_{t - 1} = \tilde{\mu}_t + \sigma_t z$$



The algorithm should then look something like:

1. Sample random noise vector $x_T \sim \mathcal{N}(0, \mathcal{I})$.
2. For each t in $[T, 1]$
   1. Sample a noise vector $z$ if t > 0 otherwise, $z$ = 0.
   2. Find $x_{t-1}$ using the equations above.
3. Return the final $\hat{x_0}$

*3.1*. Implement DDPM sampling as described above to generate samples from the pre-trained diffusion model. Fill out all TODOs in the code corresponding to 3.1 in `model.py`. Run inference using the following command:
```
python inference.py --ckpt epoch_199.pth --sampling-method ddpm
```

<hr>
<hr>

The issue with DDPM is that we need to loop over all the timestamps sequentially, which is not very efficient. Denoising Diffusion Implicit Model (DDIM - [2]) samples a small number of timesteps $S$ from the total timestamps. We can do this sampling evenly across the $[1, T]$ range with a step-size of $\frac{T}{S}$ to get a total of S timesteps $[\tau_1, \tau_2, ..., \tau_{S}]$.

$$ z \sim \mathcal{N}(0, \mathcal{I})$$

$$\hat{x_0} = \frac{1}{\sqrt{\bar{\alpha_{\tau_{i}}}}} (x_{\tau_{i}} - \sqrt{1 - \bar{\alpha_t}} \epsilon_t)$$

$$\sigma_{\tau_{i}}^2 = \eta \tilde{\beta_{\tau_{i}}}, \hspace{10px} \tilde{\beta_{\tau_{i}}} = \frac{1 - \bar{\alpha_{\tau_{i - 1}}}}{1 - \bar\alpha_{\tau_{i}}} \beta_{\tau_{i - 1}}$$

<!-- $$q(x_{\tau_{i - 1}} | x_{\tau_t}, x_0) = \mathcal{N}(x_{\tau_{i-1}}; \sqrt{\bar{\alpha_{\tau_{i - 1}}}} x_0 + \sqrt{1 - \bar{\alpha_{\tau_{i - 1}}} - \sigma_{\tau_{i}}^2} \epsilon_{\tau_{i}}; \sigma_{\tau_{i}}^2 \mathcal{I}) $$ -->

$$ \tilde{\mu_{\tau_i}} = \sqrt{\bar{\alpha_{\tau_{i - 1}}}} \hat{x_0} + \sqrt{1 - \bar{\alpha_{\tau_{i - 1}}} - \sigma_{\tau_{i}}^2} \epsilon_{\tau_{i}}$$

$$ x_{\tau_{i-1}} = \tilde{\mu_{\tau_i}} + \sigma_{\tau_i}^2z$$

$$x_{\tau_{i-1}} \sim  q(x_{\tau_{i - 1}} | x_{\tau_t}, \hat{x_0} ) = \mathcal{N}(x_{\tau_{i-1}}; \tilde{\mu_{\tau_i}}, \sigma_{\tau_i} \mathcal{I})$$ 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (Notation: $q(x_{\tau_{i - 1}} | x_{\tau_t}, \hat{x_0} )$ denotes the denoising process, which is denoted by $p_{\theta}$ in class.)


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Since this is Gaussian, we can use the reparameterization trick to implement the sampling:

$$x_{\tau_{i-1}} = \tilde{\mu_{\tau_i}} + \sigma_{\tau_i} z$$



where $\eta$ is a hyperparameter along with $S$. Here $q$ represents the distribution from which we can sample to get $x_{\tau_{i - 1}}$

The algorithm for DDIM should then look something like:
1. Sample random noise vector $x_T \sim \mathcal{N}(0, \mathcal{I})$.
2. For each t in $[\tau_S, \tau_{S - 1}, ..., \tau_1]$
   1. Sample a noise vector $z$ if t > 0 otherwise, $z$ = 0.
   2. Find $x_{\tau_{i - 1}}$ using the equations above.
3. Return $x_0$

*3.2*. Implement the DDIM sampling method as described above to generate samples from the pre-trained diffusion model. Fill out all TODOs in the code corresponding to 3.2 in `model.py`. Run inference using the following command:
```
python inference.py --ckpt epoch_199.pth --sampling-method ddim --ddim-timesteps 100 --ddim-eta 0
```

*3.3*. Finally, we are going to use the sampling methods above to compute the FID of the diffusion model. Fill out all TODOs in the code corresponding to 3.3 in `model.py` and `inference.py`. Compute FID for DDPM and DDIM using the following commands:

```
python inference.py --ckpt epoch_199.pth --sampling-method ddpm --compute_fid
python inference.py --ckpt epoch_199.pth --sampling-method ddim --ddim-timesteps 100 --ddim-eta 0 --compute_fid
```

## Relevant papers:
[1] Denoising diffusion probabilistic models (Jonathan Ho, et al, 2020): https://arxiv.org/abs/2006.11239

[2] Denoising diffusion implicit models (Jiaming Song et al, 2020): https://arxiv.org/abs/2010.02502
