#!/bin/bash
set -e

cd /workspace/rust-lob-rl-market-maker

# Ensure Rust/uv are on PATH (in case shell isn't login)
source "$HOME/.cargo/env" 2>/dev/null || true
source "$HOME/.local/bin/env" 2>/dev/null || true

# W&B key check
if [ -z "$WANDB_API_KEY" ]; then
  echo "ERROR: Set WANDB_API_KEY first"
  echo "  export WANDB_API_KEY=your_key_here"
  exit 1
fi

echo "=== QuantFlow Cloud Training ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no GPU detected')"

uv run python -m quantflow.training.train \
  --timesteps 2000000 \
  --hawkes-params data/btcusdt/calibration/hawkes_params_multi_asia_eu.json \
  --wandb --wandb-project quantflow-mm \
  --run-dir runs/sac_2M_cloud \
  --maker-fee-bps 2.0 \
  --taker-fee-bps 5.0 \
  --dd-penalty-coef 0.001

echo "=== Training complete ==="
echo "Best model: runs/sac_2M_cloud/best_model.zip"
echo "Final model: runs/sac_2M_cloud/final_model.zip"
