from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import simulation, evaluation, metrics

app = FastAPI(title="QuantFlow API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(simulation.router)
app.include_router(evaluation.router)
app.include_router(metrics.router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
