from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.services.websocket import runner

router = APIRouter(tags=["simulation"])


# ── REST control endpoints ───────────────────────────────────────────────────

@router.post("/api/simulation/start")
async def start_simulation(body: dict) -> dict:
    await runner.start(
        seed     = int(body.get("seed",     42)),
        speed    = float(body.get("speed",  1.0)),
        strategy = str(body.get("strategy", "as")),
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


# ── WebSocket stream ────────────────────────────────────────────────────────

@router.websocket("/ws/live")
async def live_ws(ws: WebSocket) -> None:
    await runner.connect(ws)
    try:
        async for msg in ws.iter_json():
            action = msg.get("action", "")
            if action == "start":
                cfg = msg.get("config", {})
                await runner.start(
                    seed     = int(cfg.get("seed",     42)),
                    speed    = float(cfg.get("speed",  1.0)),
                    strategy = str(cfg.get("strategy", "as")),
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
