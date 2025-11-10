#!/usr/bin/env bash
set -euo pipefail

export UVICORN_WORKERS=${UVICORN_WORKERS:-1}
uvicorn backend.app:app --reload --port 8000 --workers ${UVICORN_WORKERS}
