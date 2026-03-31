from __future__ import annotations

import time
from typing import Any


class QuantFlowEngine:
    """Singleton bridge to the Rust PyO3 extension."""

    _instance: QuantFlowEngine | None = None

    def __new__(cls) -> QuantFlowEngine:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._simulator: Any = None
        self._env: Any = None
        self._is_running: bool = False
        self._step: int = 0
        self._start_time: float = 0.0
        self._initialized = True

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self, config: dict) -> None:
        try:
            import quantflow_core  # type: ignore[import]
            self._simulator = quantflow_core.PyHawkesSimulator(config)
        except ImportError:
            self._simulator = None  # stub mode — used in tests / before Rust is built

        self._is_running = True
        self._step = 0
        self._start_time = time.monotonic()

    def stop(self) -> None:
        self._is_running = False
        self._simulator = None

    # ── Simulation ─────────────────────────────────────────────────────────────

    def step(self) -> dict:
        if not self._is_running:
            return {}
        self._step += 1
        if self._simulator is not None:
            return self._simulator.step()
        # Stub: return synthetic snapshot so the frontend works without Rust
        import random, math
        mid = 100.0 + math.sin(self._step / 50) * 2.0
        return {
            "bids": [(mid - i * 0.01, random.randint(10, 200)) for i in range(1, 6)],
            "asks": [(mid + i * 0.01, random.randint(10, 200)) for i in range(1, 6)],
            "mid_price": mid,
            "spread": 0.02,
            "timestamp_ns": int(time.time() * 1e9),
        }

    # ── Status ─────────────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def current_step(self) -> int:
        return self._step

    @property
    def elapsed_s(self) -> float:
        if not self._is_running:
            return 0.0
        return time.monotonic() - self._start_time


engine = QuantFlowEngine()
