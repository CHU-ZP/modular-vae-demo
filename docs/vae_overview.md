# VAE Overview

A Variational Autoencoder learns a latent-variable model with two distributions:

$$
q_\phi(z \mid x)
$$

is the approximate posterior from data to latent variables, and

$$
p_\theta(x \mid z)
$$

is the decoder likelihood from latent variables back to data.

This demo keeps the probabilistic route fixed:

```text
x -> q_phi(z|x) -> z -> p_theta(x|z)
```

The encoder parameterizes a diagonal Gaussian posterior:

$$
q_\phi(z \mid x) = \mathcal{N}\left(z; \mu_\phi(x), \mathrm{diag}(\sigma_\phi^2(x))\right)
$$

MLP, CNN, and Transformer modules are only replaceable backbones used to parameterize the same distributions.
