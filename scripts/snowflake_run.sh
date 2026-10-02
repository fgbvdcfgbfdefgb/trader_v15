#!/usr/bin/env bash
# Offline training entrypoint (Snowflake 4x A10G container / any machine).
# The repo already contains the dataset - only pip packages are fetched, then
# training runs with zero network access.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# On a 4x A10G box this automatically becomes INTENSE mode:
#   predictor=cuda:0, analyzer=cuda:1, traderA=cuda:2, traderB=cuda:3
#   XL tier models (~9M-param GRU advisors, 147M-param traders)
#   4 random days per epoch, vectorised rollouts, 24-worker precompute pool,
#   EMA precision advisors, batched held-out eval.
python3 train.py --epochs "${EPOCHS:-2000}" --device "${DEVICE:-auto}" \
  --vram-budget-gb "${VRAM:-20}" --bg-steps "${BG:-12}" --out "${OUT:-runs}"
