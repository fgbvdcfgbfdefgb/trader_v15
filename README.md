# trader_v13 — giant agents edition

Offline multi-agent RL trading system. Three cooperating/competing agents
(same v10-family architecture, resized to the v13 **size mandate**):

| agent | architecture | mandate size |
|---|---|---|
| **price predictor** | stacked GRU (3 layers) + LayerNorm + horizon head | **2 B params** |
| **market analyzer** | stacked GRU (3 layers) + embedding + vol/trend heads | **500 M params** |
| **trade maker A + B** | deep MLP (3 hidden layers) + Gaussian PPO policy + value head | **3 B params each** (~6 B both) |

Total at full mandate: **~8.5 B parameters**.

## Sizes are *solved* per machine

At startup the trainer inventories per-GPU free VRAM, free disk and the
optimizer mode, then **solves the hidden dimensions** to fit the mandate —
full sizes when the hardware holds them, otherwise scaled down *proportionally*
(the analyzer:predictor:trader ratio 0.5:2:3 is preserved) to the binding
constraint. It prints exactly what it chose and why:

```
size tier: GIGA (mandate sizes) | predictor GRU h=11520 x3 (1.99B) | analyzer GRU h=5760 x3 (0.498B) | trader MLP (38336, 38336, 38336) (3B each) | ...
```

Memory model used by the solver (bytes/param): weights 4 + grads 4 +
optimizer states 8 (AdamW) or **2 (bitsandbytes AdamW8bit)** + EMA copy 4
(advisors only), plus activation reserves and a disk budget for one rolling
checkpoint (weights + optimizer states).

Rough guide for what you get (4-GPU machines, one agent per GPU):

| machine | optimizer | solved sizes (approx) |
|---|---|---|
| 4× ~40GB+ (A100-40, H100) + 8-bit | 8-bit | **full mandate: 500M / 2B / 3B+3B** |
| 4× 24GB (A10G) | 8-bit | ~250M / 1.2B / 1.7B×2 |
| 4× 16GB (T4) + 24GB disk | 8-bit | ~130M / 510M / 780M×2 |
| CPU | AdamW | tiny XS tier (still runs) |

Overrides: `--pred-hidden/--ana-hidden/--trader-hidden` etc. force any size;
`--vram-budget-gb` caps the per-GPU budget (0 = auto = 85% of free VRAM).

## Big-model training tricks (all automatic)

- **Mixed-precision autocast** for every training forward/backward:
  bf16 on Ampere+ GPUs, fp16 with gradient loss-scaling on Turing (T4).
  Rollouts/eval stay fp32 for exact env dynamics.
- **bitsandbytes 8-bit AdamW/Adam** when the package imports and steps
  successfully (probed at startup); plain torch AdamW otherwise. 8-bit states
  roughly double the parameter count a given GPU can train.
- **EMA without double memory**: the weight-averaged "precision" advisor IS
  the inference model (one extra copy, not two).
- **RAM-safe checkpoints**: state dicts move to CPU one tensor at a time, so
  saving a multi-GB model never spikes system RAM. Resume handles optimizer
  mismatches (e.g. fp32-Adam checkpoint into an 8-bit optimizer) gracefully.

## Task (unchanged)

Each epoch: N random UTC days of 1-minute BTC/ETH/LTC(SOL) bars; predictor +
analyzer train on them (+ replay); traders A and B (competing PPO agents)
trade every minute starting with $20 aiming for $30+, using the EMA advisors'
vol-normalised forecasts and market embedding as observations. Adaptive LRs
(KL-driven for traders, warmup+cosine for advisors). Two traders train in
parallel on separate GPUs; the better one on held-out days wins.

## Quick start

```bash
pip install -r requirements.txt
python train.py                     # sizes auto-solved for your machine
# full mandate needs e.g. 4x40GB GPUs + ~50GB disk:
python train.py --epochs 2000
# optional 2x bigger tiers on CUDA GPUs:
pip install bitsandbytes
```

`bash scripts/snowflake_run.sh` does the same on a Snowflake container.
Everything is offline: dataset lives in `./data`.

## Outputs (`runs/`)

```
epochs/epoch_XXXXX_YYYY-MM-DD.png            <- combined day snapshot (both traders)
epochs/epoch_XXXXX_YYYY-MM-DD_traders.png    <- ONE tall PNG with BOTH agents:
                                                market drawn ONCE (3 price panels,
                                                each with a long/short exposure
                                                ribbon per trader), then A's
                                                equity+weights, then B's below
training_curves.png                          <- loss/corr/equity curves
metrics.csv                                  <- per-epoch metrics
ckpt/                                        <- rolling checkpoint (resume with
                                                --resume runs/ckpt)
eval/                                        <- held-out evaluation (eval.py)
```

## Evaluate / resume

```bash
python eval.py --ckpt runs/ckpt --which both --days 16   # held-out eval + PNGs
python train.py --resume runs/ckpt                        # continue training
```

## Repo layout

```
trader/agents.py    the three agents (GRU predictor/analyzer, MLP trader)
trader/features.py  per-minute features, vol-normalised targets, obs builder
trader/env.py       vectorised 1-minute trading env (fees/slippage/leverage)
trader/ppo.py       clipped-surrogate PPO with GAE (AMP-aware)
trader/plots.py     all graphs (incl. the stacked per-trader PNG)
trader/utils.py     hardware detection + the v13 size solver
train.py            trainer (autocast, 8-bit opts, EMA, resume)
eval.py             held-out evaluation (sizes read from the checkpoint)
data/               1-minute bars for BTC-USD, ETH-USD, SOL-USD
```
