#!/bin/bash
set -e
trap 'kill 0' EXIT

echo -e "\033[38;5;208m[QUANTFLOW]\033[0m Starting development servers..."
echo -e "\033[38;5;208m[BACKEND] \033[0m http://localhost:8000/docs"
echo -e "\033[38;5;208m[FRONTEND]\033[0m http://localhost:3000"
echo ""

uv run uvicorn backend.main:app --reload --port 8000 &
cd /Users/josefdukarevic/Developer/rust-lob-rl-market-maker/dashboard && npm run dev &
wait
