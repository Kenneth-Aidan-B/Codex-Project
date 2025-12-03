#!/bin/bash
# Start both backend and frontend servers

# Kill any existing servers
pkill -9 -f uvicorn 2>/dev/null
pkill -9 -f "http.server" 2>/dev/null
sleep 1

echo "Starting Backend API on port 8000..."
cd /workspaces/Codex-Project
uvicorn backend.app:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

sleep 2

echo "Starting Frontend on port 8080..."
cd /workspaces/Codex-Project/frontend
python -m http.server 8080 --bind 0.0.0.0 &
FRONTEND_PID=$!

echo ""
echo "====================================="
echo "Servers started!"
echo "Backend API: http://localhost:8000"
echo "Frontend:    http://localhost:8080"
echo "====================================="
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait
