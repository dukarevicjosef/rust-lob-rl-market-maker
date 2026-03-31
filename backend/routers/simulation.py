from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.models.schemas import SimulationConfig, SimulationStatus
from backend.services.engine import engine
from backend.services.websocket import manager

router = APIRouter(prefix="/api/simulation", tags=["simulation"])


@router.post("/start")
async def start_simulation(config: SimulationConfig) -> dict:
    engine.start(config.model_dump())
    return {"status": "started"}


@router.post("/stop")
async def stop_simulation() -> dict:
    engine.stop()
    return {"status": "stopped"}


@router.get("/status", response_model=SimulationStatus)
async def get_status() -> SimulationStatus:
    return SimulationStatus(
        is_running=engine.is_running,
        step=engine.current_step,
        elapsed_s=engine.elapsed_s,
    )


# ── WebSocket stream ────────────────────────────────────────────────────────

@router.websocket("/ws/live")
async def live_stream(ws: WebSocket) -> None:
    await manager.connect(ws)
    try:
        while True:
            if engine.is_running:
                snapshot = engine.step()
                await ws.send_json({"type": "snapshot", "data": snapshot})
            await asyncio.sleep(0.1)  # 10 Hz
    except WebSocketDisconnect:
        manager.disconnect(ws)
