#!/bin/bash
set -e

echo "=== QuantFlow Cloud Setup ==="

# System deps
apt-get update && apt-get install -y \
  curl build-essential pkg-config libssl-dev \
  git cmake

# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
  sh -s -- -y
source "$HOME/.cargo/env"

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

# Repo
cd /workspace
if [ ! -d "rust-lob-rl-market-maker" ]; then
  git clone https://github.com/dukarevicjosef/rust-lob-rl-market-maker.git
fi
cd rust-lob-rl-market-maker

# Rust build
cargo build --release

# Python deps + native extension
uv sync
uv run maturin develop --release

# Verify
uv run python -c "import quantflow; print('quantflow OK')"
uv run python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

echo "=== Setup complete ==="
echo "Next: export WANDB_API_KEY=your_key"
echo "Then: bash scripts/cloud_train.sh"
