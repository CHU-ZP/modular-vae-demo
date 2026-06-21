# ELBO

The training objective is the negative evidence lower bound:

$$
\mathcal{L}(x) =
\mathbb{E}_{q_\phi(z \mid x)}\left[-\log p_\theta(x \mid z)\right]
+ \beta\,\mathrm{KL}\left(q_\phi(z \mid x) \Vert p_\psi(z)\right)
$$

For the standard normal prior, the KL is analytic for a diagonal Gaussian posterior.

$$
p(z) = \mathcal{N}(0, I)
$$

$$
\mathrm{KL}\left(q_\phi(z \mid x) \Vert \mathcal{N}(0,I)\right)
= \frac{1}{2}\sum_j\left(\mu_j^2 + \sigma_j^2 - 1 - \log \sigma_j^2\right)
$$

For arbitrary priors, this demo estimates the KL with one posterior sample:

$$
\mathrm{KL}\left(q_\phi(z \mid x) \Vert p_\psi(z)\right)
\approx
\log q_\phi(z \mid x) - \log p_\psi(z),
\qquad
z \sim q_\phi(z \mid x)
$$

Changing `beta` gives beta-VAE behavior without changing the model class.
