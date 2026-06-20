#!/usr/bin/env bash
set -euo pipefail

# Use this after activating .venv, or run it with:
# uv run --no-sync bash scripts/run_all_experiments.sh
#
# The --no-sync part matters when CUDA PyTorch was installed manually from a
# PyTorch wheel index. Plain `uv run` may resync torch/nvidia packages.

python -m vae.train --config configs/mnist_mlp_standard.yaml
python -m vae.train --config configs/mnist_cnn_standard.yaml
python -m vae.train --config configs/mnist_beta_vae.yaml
python -m vae.train --config configs/mnist_transformer_standard.yaml
python -m vae.train --config configs/mnist_flow_prior.yaml
