#!/bin/bash

APP_PORT=${PORT:-8000}

# shellcheck disable=SC2164
cd /workspace/

/opt/venv/bin/gunicorn -k uvicorn.workers.UvicornWorker main:app --bind "0.0.0.0:${APP_PORT}"