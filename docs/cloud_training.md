# Cloud GPU Training (vast.ai)

## Quick Start

1. **Instance starten** auf vast.ai
   - GPU: RTX 4090/5090 oder besser
   - Image: `nvidia/cuda:12.4.0-devel-ubuntu22.04`
   - Disk: 50GB
   - SSH Key muss hinterlegt sein

2. **SSH verbinden:**
   ```
   ssh -p PORT root@HOST
   ```

3. **Setup** (einmalig, ~5-10 min):
   ```bash
   curl -sSL https://raw.githubusercontent.com/dukarevicjosef/rust-lob-rl-market-maker/main/scripts/cloud_setup.sh | bash
   ```

   Oder manuell:
   ```bash
   cd /workspace
   git clone https://github.com/dukarevicjosef/rust-lob-rl-market-maker.git
   cd rust-lob-rl-market-maker
   bash scripts/cloud_setup.sh
   ```

4. **W&B Key setzen:**
   ```bash
   export WANDB_API_KEY=your_key_here
   ```

5. **Training starten:**
   ```bash
   bash scripts/cloud_train.sh
   ```

6. **Monitoring** via W&B Dashboard (nicht SSH)

7. **Nach Training** - Models runterladen:
   ```bash
   scp -P PORT root@HOST:/workspace/rust-lob-rl-market-maker/runs/sac_2M_cloud/*.zip runs/
   ```

8. **Instance stoppen** auf vast.ai!

## Kosten

- RTX 5090: ~$0.40/h x ~8h = ~$3
- 2M Steps bei ~70 fps ~ 8h

## Konfiguration anpassen

Training-Parameter koennen direkt in `scripts/cloud_train.sh` geaendert werden
oder per CLI-Flags:

```bash
uv run python -m quantflow.training.train \
  --timesteps 2000000 \
  --hawkes-params data/btcusdt/calibration/hawkes_params_multi_asia_eu.json \
  --wandb --wandb-project quantflow-mm \
  --run-dir runs/sac_custom \
  --maker-fee-bps 2.0 \
  --taker-fee-bps 5.0 \
  --dd-penalty-coef 0.001 \
  --dd-penalty-threshold 200 \
  --inventory-soft-limit 20
```

## Troubleshooting

- **`cargo build` fails**: Stelle sicher dass `libssl-dev` installiert ist
- **`maturin develop` fails**: Python Version muss 3.12 sein (`uv python install 3.12`)
- **CUDA nicht erkannt**: Image muss CUDA-devel sein, nicht runtime
- **Episode zu kurz (ep_len_mean < 500)**: t_max ist auf 600s gesetzt, sollte reichen. Falls nicht: `--t-max` erhoehen
