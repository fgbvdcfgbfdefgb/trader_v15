#!/usr/bin/env bash
# Run training on the Clore mining machine (4x T4) WITHOUT stopping the miner.
# Packages live in an isolated ./pylibs dir (system python & jupyter untouched).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d pylibs ]; then
  pip3 install --break-system-packages --progress-bar off \
    --target "$PWD/pylibs" -r requirements.txt
fi

# The miner holds ~110MB VRAM per T4 and 100% SM util; training shares the GPUs
# (tiny models, ~250MB each) and the --min-free-gb guard skips any GPU that is
# too full. All 4 GPUs at 100% util = miner is still running.
export PYTHONPATH="$PWD/pylibs"
exec python3 train.py --epochs "${EPOCHS:-3000}" --device auto --out "${OUT:-runs}"
