# Data Collection Plan: 24h BTCUSDT Coverage

## Goal

Collect ~24 hours of Binance Futures BTCUSDT market data across 4-6
sessions covering different market regimes. Used for multi-session
Hawkes calibration to produce regime-robust simulator parameters.

## Sessions

| # | Timing (UTC) | Regime | Duration | Command |
|---|-------------|--------|----------|---------|
| 1 | Mon 00:00-04:00 | Asia quiet | 4h | See below |
| 2 | Mon 08:00-12:00 | EU open | 4h | See below |
| 3 | Mon 14:00-18:00 | US open, volatile | 4h | See below |
| 4 | Tue 00:00-04:00 | Asia quiet (2) | 4h | See below |
| 5 | Wed 14:00-18:00 | US session (2) | 4h | See below |
| 6 | Thu 20:00-00:00 | Mixed/evening | 4h | See below |

## Commands

```bash
# Session 1: Asia quiet
caffeinate -i uv run python -m quantflow.data.record_session \
  --symbol BTCUSDT --duration 4h \
  --label asia_quiet \
  --output data/btcusdt/sessions/

# Session 2: EU open
caffeinate -i uv run python -m quantflow.data.record_session \
  --symbol BTCUSDT --duration 4h \
  --label eu_open \
  --output data/btcusdt/sessions/

# Session 3: US open
caffeinate -i uv run python -m quantflow.data.record_session \
  --symbol BTCUSDT --duration 4h \
  --label us_open \
  --output data/btcusdt/sessions/

# Session 4: Asia quiet (repeat)
caffeinate -i uv run python -m quantflow.data.record_session \
  --symbol BTCUSDT --duration 4h \
  --label asia_quiet_2 \
  --output data/btcusdt/sessions/

# Session 5: US session (repeat)
caffeinate -i uv run python -m quantflow.data.record_session \
  --symbol BTCUSDT --duration 4h \
  --label us_open_2 \
  --output data/btcusdt/sessions/

# Session 6: Mixed evening
caffeinate -i uv run python -m quantflow.data.record_session \
  --symbol BTCUSDT --duration 4h \
  --label evening_mixed \
  --output data/btcusdt/sessions/
```

## Post-Session Checks

After each session:

1. Verify Parquet is readable:
   ```bash
   uv run python -c "
   import polars as pl
   df = pl.read_parquet('data/btcusdt/sessions/BTCUSDT_*_trades.parquet')
   print(f'Trades: {len(df):,}')
   print(df.head(3))
   "
   ```

2. Check metadata JSON:
   ```bash
   cat data/btcusdt/sessions/BTCUSDT_*_metadata.json | python -m json.tool
   ```

3. Expected event counts for 4h session:
   - Trades: 100K-200K (depends on volatility)
   - Depth updates: 100K-150K (10/sec @ 100ms)
   - BookTicker: 200K-500K (high frequency)

## Calibration

After collecting all sessions, process raw data into events and calibrate:

```bash
# Process each session's trades into classified events
# (requires the existing classify pipeline)

# Multi-session calibration (Option A: concatenated)
uv run python -m quantflow.calibration.calibrate_multi \
  --sessions data/btcusdt/processed/ \
  --output data/btcusdt/calibration/hawkes_params_v2.json

# Multi-session calibration (Option B: median per session)
uv run python -m quantflow.calibration.calibrate_multi \
  --sessions data/btcusdt/processed/ \
  --strategy median \
  --per-session-dir data/btcusdt/calibration/hawkes_params_per_session/ \
  --output data/btcusdt/calibration/hawkes_params_v2.json
```

## Notes

- Mainnet only (testnet has unrealistic liquidity/spreads)
- No API keys needed for public WebSocket streams
- Auto-reconnect with exponential backoff (max 30s) on disconnect
- Buffered writes: 10K events in memory before Parquet flush
- caffeinate -i prevents system sleep during recording
- If a session fails: re-record at a similar time, regime matters more
  than exact timing
