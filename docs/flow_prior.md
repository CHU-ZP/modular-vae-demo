# Flow Prior

The standard VAE prior is:

$$
p(z) = \mathcal{N}(0, I)
$$

The flow prior starts from a standard Gaussian base variable and transforms it:

$$
u \sim \mathcal{N}(0, I)
$$

$$
z = f_\psi(u)
$$

The log probability uses the inverse transformation:

$$
\log p_\psi(z)
=
\log p_0(f_\psi^{-1}(z))
+
\log\left|\det\frac{\partial f_\psi^{-1}}{\partial z}\right|
$$

This demo uses a compact RealNVP-style affine coupling flow. It is intentionally simple, meant to show how the prior can be replaced while the VAE interface stays fixed.
