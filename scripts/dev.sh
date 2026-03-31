#!/bin/bash
set -e
trap 'kill 0' EXIT

echo "Starting QuantFlow dev servers..."
echo "  Frontend: http://localhost:3000"
echo "  Backend:  http://localhost:8000"
echo "  API docs: http://localhost:8000/docs"
echo ""

uv run uvicorn backend.main:app --reload --port 8000 &
cd dashboard && npm run dev &
wait
