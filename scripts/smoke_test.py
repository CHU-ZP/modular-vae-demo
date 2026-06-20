"""Fast smoke test for the modular VAE components."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vae.builders import build_loss_config, build_model, build_prior
from vae.losses import vae_loss


def check_config(config: dict) -> None:
    model = build_model(config)
    prior = build_prior(config)
    x = torch.rand(2, 1, 28, 28)
    outputs = model(x)
    losses = vae_loss(outputs, x, prior=prior, **build_loss_config(config))
    losses["loss"].backward()
    assert outputs["x_recon"].shape == x.shape
    assert outputs["z"].shape[0] == x.shape[0]
    assert torch.isfinite(losses["loss"])


def main() -> None:
    configs = [
        {
            "model": {"encoder": "mlp", "decoder": "mlp", "latent_dim": 8, "hidden_dim": 32},
            "prior": {"type": "standard_normal"},
            "loss": {"beta": 1.0, "kl_mode": "analytic", "reconstruction": "bce"},
        },
        {
            "model": {"encoder": "cnn", "decoder": "cnn", "latent_dim": 8, "hidden_dim": 32},
            "prior": {"type": "standard_normal"},
            "loss": {"beta": 1.0, "kl_mode": "analytic", "reconstruction": "bce"},
        },
        {
            "model": {
                "encoder": "transformer",
                "decoder": "transformer",
                "latent_dim": 8,
                "patch_size": 7,
                "d_model": 32,
                "num_layers": 1,
                "num_heads": 4,
                "mlp_ratio": 2.0,
            },
            "prior": {"type": "standard_normal"},
            "loss": {"beta": 1.0, "kl_mode": "analytic", "reconstruction": "bce"},
        },
        {
            "model": {"encoder": "mlp", "decoder": "mlp", "latent_dim": 8, "hidden_dim": 32},
            "prior": {"type": "flow", "num_flow_layers": 2, "flow_hidden_dim": 32},
            "loss": {"beta": 1.0, "kl_mode": "monte_carlo", "reconstruction": "bce"},
        },
    ]
    for config in configs:
        check_config(config)
    print("smoke test passed")


if __name__ == "__main__":
    main()
