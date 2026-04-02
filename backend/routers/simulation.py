from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.services.websocket import runner

router = APIRouter(tags=["simulation"])


# ── REST control endpoints ───────────────────────────────────────────────────

@router.post("/api/simulation/start")
async def start_simulation(body: dict) -> dict:
    mode = str(body.get("mode", "simulate"))
    if mode == "replay":
        await runner.start(
            mode="replay",
            replay_path=str(body.get("replay_path", "")),
            speed=float(body.get("speed", 1.0)),
        )
    else:
        await runner.start(
            seed=int(body.get("seed", 42)),
            speed=float(body.get("speed", 1.0)),
            strategy=str(body.get("strategy", "as")),
        )
    return {"status": "started"}


@router.post("/api/simulation/stop")
async def stop_simulation() -> dict:
    runner.stop()
    return {"status": "stopped"}


@router.get("/api/simulation/status")
async def get_status() -> dict:
    return {
        "is_running":  runner.is_running,
        "elapsed_s":   round(runner.elapsed, 2),
        "connections": runner.n_connections,
    }


# ── Replay file listing ──────────────────────────────────────────────────────

@router.get("/api/replay/files")
async def list_replay_files() -> list[dict]:
    """
    Scan data/btcusdt/processed/ for *_events.parquet files.
    Returns date, path, size_mb, and event count (read from Parquet footer).
    """
    processed_dir = Path("data/btcusdt/processed")
    if not processed_dir.exists():
        return []

    result: list[dict] = []
    for f in sorted(processed_dir.glob("*_events.parquet")):
        date = f.stem.replace("_events", "")
        size_mb = round(f.stat().st_size / 1_000_000, 2)
        try:
            import pyarrow.parquet as pq
            n_events = pq.read_metadata(f).num_rows
        except Exception:
            n_events = 0
        result.append({
            "date":    date,
            "path":    str(f),
            "size_mb": size_mb,
            "events":  n_events,
        })
    return result


# ── WebSocket stream ────────────────────────────────────────────────────────

@router.websocket("/ws/live")
async def live_ws(ws: WebSocket) -> None:
    await runner.connect(ws)
    try:
        async for msg in ws.iter_json():
            action = msg.get("action", "")
            if action == "start":
                cfg  = msg.get("config", {})
                mode = str(cfg.get("mode", "simulate"))
                if mode == "replay":
                    await runner.start(
                        mode="replay",
                        replay_path=str(cfg.get("replay_path", "")),
                        speed=float(cfg.get("speed", 1.0)),
                    )
                else:
                    await runner.start(
                        seed=int(cfg.get("seed", 42)),
                        speed=float(cfg.get("speed", 1.0)),
                        strategy=str(cfg.get("strategy", "as")),
                    )
            elif action == "stop":
                runner.stop()
            elif action == "set_speed":
                runner.set_speed(float(msg.get("speed", 1.0)))
            elif action == "reset":
                await runner.reset(seed=int(msg.get("seed", 42)))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        runner.disconnect(ws)
