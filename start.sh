#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

if ! command -v python >/dev/null 2>&1; then
  echo "Python не найден. Установите Python 3.10+ и повторите." >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg не найден. Установите ffmpeg и повторите." >&2
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  python -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install -r "$ROOT_DIR/requirements.txt"

python "$ROOT_DIR/run_gui.py"
