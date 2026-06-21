# Reparameterization

The encoder outputs `mu` and `logvar`, which define:

$$
q_\phi(z \mid x) =
\mathcal{N}\left(z; \mu_\phi(x), \mathrm{diag}(\sigma_\phi^2(x))\right)
$$

Direct sampling would block gradients through the random draw. The reparameterization trick rewrites the sample as:

$$
\epsilon \sim \mathcal{N}(0, I)
$$

$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon
$$

The randomness is isolated in `epsilon`, while `z` remains differentiable with respect to encoder outputs.
