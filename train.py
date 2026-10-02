#!/usr/bin/env python3
"""
trader_v15 - offline multi-agent RL trading training (talking agents edition).

What changed vs v13
-------------------
* THE AGENTS TALK: trader A and trader B roll out the same days in lockstep
  and exchange their lagged effective weights + session return every minute
  (team channel injected into the observation), so they can coordinate,
  imitate or fade each other while trading. A comm-dropout (30% of days)
  keeps each policy robust when the teammate channel is silent (eval).
* TRADER -> ADVISOR FEEDBACK: after each epoch's rollouts, the predictor and
  analyzer train with an activity-weighted loss - they learn hardest on the
  minutes where the traders actually held positions ("forecast where it
  matters"), not uniformly across the day.
* PER-TRADE PROFIT CONDITION: 4th action dim = confidence gate (weight x
  clamp(gate,0,1)); every position round-trip is accounted and fed back into
  the observation (open-trade PnL, last trade PnL, win rate, trade count);
  closed losing trades are punished ~3x harder than winners are rewarded
  (asymmetric reward). Win rate is logged per epoch (wrA/wrB in metrics.csv).
* Everything from v13 stands: solved giant sizes (500M/2B/3B mandate),
  mixed precision, 8-bit optimizers, EMA advisors, stacked per-epoch PNGs.

What changed vs v12
-------------------
* SIZE MANDATE: the v10-family agents (GRU predictor, GRU analyzer, MLP
  traders) are resized toward analyzer 500M / predictor 2B / trader 3B
  params each. Sizes are SOLVED per machine: full mandate when the hardware
  holds it, otherwise scaled down proportionally to per-GPU VRAM, optimizer
  mode and free disk (auto-reported at startup).
* BIG-MODEL TRAINING: mixed-precision autocast (bf16 on Ampere+, fp16 on
  Turing with loss scaling) and bitsandbytes 8-bit AdamW when installable -
  roughly 2x more parameters per GB of VRAM. Optimizer states and EMA copies
  are accounted for when solving sizes; checkpoints are written per-tensor so
  giant models don't spike system RAM.
* Everything else from v12 stands: vectorised multi-day rollouts, parallel
  precompute pool, EMA precision advisors, KL-adaptive trader LRs, per-epoch
  PNGs (market drawn once with per-trader exposure ribbons, then each
  trader's equity + weights stacked), resume support.

What changed vs v10
-------------------
* USE THE WHOLE MACHINE:
  - `--days-per-epoch` (auto: 4 on a 4-GPU + big-RAM box): every epoch draws
    N random days; both traders roll out ALL of them per epoch in ONE
    batched (vectorised) pass and PPO-update on N x 1,376 samples -> lower
    variance gradients, saturated GPUs, faster wall-clock learning.
  - All ~1,340 days are feature/target-precomputed up front with a
    multiprocessing pool across the CPU cores (uses spare RAM).
  - TF32 matmul + cudnn.benchmark on Ampere+ GPUs.
* BIGGER + MORE PRECISE AGENTS:
  - New XL tier for A10G-class (23GB) GPUs: predictor/analyzer GRU 640x3
    (~9M params each), traders MLP 8192-8192-8192 (~147M params each).
  - Predictor/analyzer get more gradient steps per epoch (tier-scaled).
  - EMA (weight-averaged) copies of the predictor/analyzer feed the traders
    and the evaluations - more stable, more precise forecasts.
* PLOTS: on top of all v10 graphs (unchanged), each epoch now saves ONE
  DEDICATED PNG PER TRADER (epoch_XXXXX_<date>_traderA.png / _traderB.png)
  showing that agent's own response to the day: exposure shading, equity,
  and its per-coin portfolio-weight decisions.
* Faster evaluation: held-out days are evaluated in batched vectorised
  rollouts, both traders in parallel.

Each epoch: N random UTC days of 1-minute BTC/ETH/LTC bars; predictor +
analyzer train on them (+ replay); traders A and B (competing PPO agents)
trade every minute of every day starting with $20 aiming for $30+, using
the EMA predictor's forecasts and the EMA analyzer's market embedding.
Fully offline (dataset ships in the repo).
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import threading
import time
from collections import OrderedDict, deque
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trader import data as D  # noqa: E402
from trader import plots  # noqa: E402
from trader.agents import Analyzer, Predictor, Trader  # noqa: E402
from trader.data import PAIRS  # noqa: E402
from trader.env import FB_DIM, TradingEnv  # noqa: E402
from trader.features import (  # noqa: E402
    HORIZONS,
    IDX15,
    IDX60,
    N_FEATURES,
    W,
    build_features,
    build_labels,
    build_targets,
)
from trader.ppo import RolloutBuffer, ppo_update  # noqa: E402
from trader.utils import (choose_plan, detect_resources, format_resources,  # noqa: E402
                          make_opt, probe_8bit, set_seed)

EMB_DIM = 16
CSV_FIELDS = [
    "epoch", "date",
    "pred_loss", "pred_mae15", "pred_mae60", "pred_corr15", "pred_corr60", "lr_pred",
    "ana_loss", "ana_vol_acc", "ana_trend_acc", "lr_ana",
    "pi_loss_a", "v_loss_a", "entropy_a", "kl_a", "lr_trader_a",
    "ep_ret_a", "final_eq_a", "hit_a", "margin_call_a",
    "pi_loss_b", "v_loss_b", "entropy_b", "kl_b", "lr_trader_b",
    "ep_ret_b", "final_eq_b", "hit_b", "margin_call_b",
    "eval_mean_eq_a", "eval_median_eq_a", "eval_hit_a",
    "eval_mean_eq_b", "eval_median_eq_b", "eval_hit_b",
    "wr_a", "ntr_a", "wr_b", "ntr_b",
    "elapsed", "days",
]


class DayProcessed:
    __slots__ = ("date", "day", "feats", "Y_pct", "Y_norm", "M", "vol_scale",
                 "vol_lab", "trend_lab", "lab_mask", "act")

    def __init__(self, date, day):
        self.date = date
        self.day = day
        self.feats = build_features(day)
        self.Y_pct, self.Y_norm, self.M, self.vol_scale = build_targets(day)
        self.vol_lab, self.trend_lab, self.lab_mask = build_labels(day)


def _build_proc(item):
    """Multiprocessing worker: build a DayProcessed from (date, day)."""
    date, day = item
    return DayProcessed(date, day)


def _corr_np(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


class GlobalSched:
    """Warmup + cosine-decay LR schedule over the whole run, stepped per
    optimizer update (only one thread touches a given agent's optimizer at a
    time by construction)."""

    def __init__(self, opt, base_lr, total_steps, warmup=100, floor=0.05):
        self.opt = opt
        self.base = base_lr
        self.total = max(int(total_steps), 1)
        self.warmup = warmup
        self.floor = floor
        self.i = 0

    def step(self):
        if self.i < self.warmup:
            f = (self.i + 1) / self.warmup
        else:
            t = min((self.i - self.warmup) / max(self.total - self.warmup, 1), 1.0)
            f = self.floor + (1 - self.floor) * 0.5 * (1.0 + math.cos(math.pi * t))
        lr = self.base * f
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.i += 1
        return lr


class EmaWeights:
    """Exponential moving average of a model's weights - the 'precision'
    model used for advisor outputs and evaluation.

    v13: the EMA state lives directly in the (separate) inference model, so
    giant advisors cost ONE extra copy instead of two."""

    def __init__(self, live, inf, decay, sync=True):
        self.live, self.inf, self.decay = live, inf, float(decay)
        if sync:
            self.inf.load_state_dict(self.live.state_dict())

    @torch.no_grad()
    def update(self):
        for (_, v), (_, s) in zip(self.live.state_dict().items(),
                                  self.inf.state_dict().items()):
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach().to(s.device), alpha=1.0 - self.decay)
            else:
                s.copy_(v.detach().to(s.device))


def parse_args():
    p = argparse.ArgumentParser(
        description="trader_v13 multi-agent RL training (giant sizes)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", default="./data")
    p.add_argument("--out", default="./runs")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    # hardware / sizes / intensity
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--min-free-gb", type=float, default=3.0)
    p.add_argument("--vram-budget-gb", type=float, default=0.0,
                   help="per-GPU VRAM budget; 0 = auto (85%% of free)")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--pred-hidden", default="auto")
    p.add_argument("--pred-layers", type=int, default=0)
    p.add_argument("--ana-hidden", default="auto")
    p.add_argument("--ana-layers", type=int, default=0)
    p.add_argument("--trader-hidden", default="auto", help="'auto' or comma list")
    p.add_argument("--batch-days", type=int, default=0, help="0 = tier default")
    p.add_argument("--minibatch", type=int, default=0, help="0 = tier default")
    p.add_argument("--pred-steps", type=int, default=0, help="0 = tier default")
    p.add_argument("--ana-steps", type=int, default=0, help="0 = tier default")
    p.add_argument("--days-per-epoch", type=int, default=0,
                   help="random days drawn per epoch (0 = auto: 4 on 4-GPU/big-RAM boxes, else 1)")
    p.add_argument("--no-precompute", action="store_true",
                   help="disable the startup parallel feature precompute pool")
    p.add_argument("--ema-decay", type=float, default=0.999,
                   help="EMA decay for the advisor (predictor/analyzer) weights; 0 disables")
    p.add_argument("--max-days", type=int, default=0)
    # second trader
    p.add_argument("--single-trader", action="store_true")
    p.add_argument("--seed-b", type=int, default=1042)
    p.add_argument("--ent-b", type=float, default=0.008)
    p.add_argument("--lr-b", type=float, default=5e-4)
    # data split
    p.add_argument("--val-days", type=int, default=40)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-limit", type=int, default=16)
    p.add_argument("--eval-batch", type=int, default=8,
                   help="days evaluated per vectorised rollout batch")
    # market / task
    p.add_argument("--capital", type=float, default=20.0)
    p.add_argument("--target", type=float, default=30.0)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--max-lev", type=float, default=5.0)
    p.add_argument("--terminal-bonus", type=float, default=0.5)
    # predictor / analyzer
    p.add_argument("--replay", type=int, default=30)
    p.add_argument("--corr-weight", type=float, default=0.5)
    p.add_argument("--bg-steps", type=int, default=4)
    p.add_argument("--no-background-a", action="store_true")
    # ppo
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--lr-trader", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.003)
    p.add_argument("--lr-agents", type=float, default=1e-3, dest="lr_agents_base")
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--target-kl", type=float, default=0.03)
    # logging
    p.add_argument("--plot-every", type=int, default=1)
    p.add_argument("--ckpt-every", type=int, default=10)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--optimizer", choices=["auto", "adamw", "adamw8bit"], default="auto",
                   help="auto = bitsandbytes 8-bit AdamW when it works, else torch")
    p.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto",
                   help="training autocast dtype (auto: bf16 on Ampere+, fp16 on Turing)")
    p.add_argument("--trade-win-bonus", type=float, default=30.0,
                   help="reward bonus per unit of WINNING closed-trade pnl")
    p.add_argument("--trade-loss-penalty", type=float, default=80.0,
                   help="reward penalty per unit of LOSING closed-trade pnl")
    p.add_argument("--feedback-weight", type=float, default=3.0,
                   help="trader->advisor activity feedback strength")
    p.add_argument("--comm-dropout", type=float, default=0.3,
                   help="fraction of days the teammate channel is silenced")
    p.add_argument("--resume", default="")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "epochs"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "ckpt"), exist_ok=True)
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    # ---------------- data (loaded BEFORE any CUDA init so the precompute
    # pool can fork safely) ----------------
    set_seed(args.seed)
    days = D.load_days(args.data, max_days=args.max_days)
    dates = sorted(days)
    if len(dates) < 20:
        raise RuntimeError("need at least 20 valid days of data")
    n_val = args.val_days
    if n_val >= len(dates) - 10:
        n_val = max(0, (len(dates) - 10) // 2)
    val_dates = dates[len(dates) - n_val:] if n_val > 0 else []
    train_dates = dates[: len(dates) - n_val]

    # ---------------- resources & plan (first CUDA touch) ----------------
    res = detect_resources(args.out)
    n_out = len(PAIRS) * len(HORIZONS)
    obs_dim = W * N_FEATURES + n_out + EMB_DIM + 3 + 1 + FB_DIM
    opt8bit = probe_8bit() if (args.optimizer in ("auto", "adamw8bit")
                               and args.device != "cpu") else (args.optimizer == "adamw8bit")
    plan = choose_plan(args, res, opt8bit=opt8bit, n_in=N_FEATURES, obs_dim=obs_dim)
    tier = plan["tier"]

    # ---- mixed-precision training dtype (bf16 on Ampere+, fp16 on Turing) ----
    amp_dtype = torch.float32
    if plan["n_usable"] > 0:
        want = args.dtype
        if want == "auto":
            major, _ = torch.cuda.get_device_capability(plan["pred"])
            want = "bf16" if major >= 8 else "fp16"
        amp_dtype = {"fp32": torch.float32, "fp16": torch.float16,
                     "bf16": torch.bfloat16}[want]
    if plan["pred"].type != "cuda" and amp_dtype == torch.float16:
        amp_dtype = torch.float32  # fp16 autocast is CUDA-only
    amp_on = amp_dtype != torch.float32
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")  # TF32 on Ampere+

    pred_hidden = tier["pred_hidden"] if args.pred_hidden == "auto" else int(args.pred_hidden)
    pred_layers = tier["pred_layers"] if args.pred_layers <= 0 else args.pred_layers
    ana_hidden = tier["ana_hidden"] if args.ana_hidden == "auto" else int(args.ana_hidden)
    ana_layers = tier["ana_layers"] if args.ana_layers <= 0 else args.ana_layers
    trader_hidden = list(tier["trader_hidden"]) if args.trader_hidden == "auto" \
        else [int(h) for h in str(args.trader_hidden).split(",")]
    batch_days = tier["batch_days"] if args.batch_days <= 0 else args.batch_days
    minibatch = tier["minibatch"] if args.minibatch <= 0 else args.minibatch
    pred_steps = tier["pred_steps"] if args.pred_steps <= 0 else args.pred_steps
    ana_steps = tier["ana_steps"] if args.ana_steps <= 0 else args.ana_steps
    dpe = args.days_per_epoch
    if dpe <= 0:
        dpe = 4 if (plan["n_usable"] >= 4 and (res["ram_gb"] or 0) >= 32) else 1
    dpe = max(1, min(dpe, len(train_dates)))

    # ---------------- parallel precompute of all training days ----------------
    proc_all = {}
    lazy_cache = OrderedDict()
    cache_lock = threading.Lock()
    precompute = (not args.no_precompute) and (res["ram_gb"] or 0) >= 16 \
        and (res["cpu_count"] or 1) >= 8 and len(train_dates) > 200
    if precompute:
        t0 = time.time()
        workers = max(2, min(24, (res["cpu_count"] or 2) // 2))
        items = [(d, days[d]) for d in train_dates]
        with Pool(workers) as pool:
            procs = pool.map(_build_proc, items, chunksize=16)
        proc_all = {p.date: p for p in procs}
        print(f"[precompute] {len(proc_all)} days feature-built on {workers} workers "
              f"in {time.time() - t0:.1f}s", flush=True)

    def get_proc(date):
        if date in proc_all:
            return proc_all[date]
        with cache_lock:
            if date in lazy_cache:
                lazy_cache.move_to_end(date)
                return lazy_cache[date]
        proc = DayProcessed(date, days[date])
        with cache_lock:
            lazy_cache[date] = proc
            while len(lazy_cache) > max(args.replay, 8):
                lazy_cache.popitem(last=False)
        return proc

    print(format_resources(res))
    print(plan["layout"])
    tp = tier.get("params", {})
    print(f"size tier: {tier['name']} | predictor GRU h={pred_hidden} x{pred_layers} "
          f"({tp.get('predictor', 0)/1e9:.3g}B) | analyzer GRU h={ana_hidden} x{ana_layers} "
          f"({tp.get('analyzer', 0)/1e9:.3g}B) | trader MLP {trader_hidden} "
          f"({tp.get('trader', 0)/1e9:.3g}B each) | "
          f"batch_days={batch_days} minibatch={minibatch} pred_steps={pred_steps} "
          f"ana_steps={ana_steps} days/epoch={dpe} ema={args.ema_decay or 'off'}")
    with open(os.path.join(args.out, "resources.txt"), "w") as f:
        f.write(format_resources(res) + "\n\n" + plan["layout"] + "\n"
                + f"tier={tier['name']} ({tier.get('note','')}) "
                f"opt8bit={opt8bit} amp={str(amp_dtype).split('.')[-1]}\n"
                + f"tier={tier['name']} pred h={pred_hidden}x{pred_layers} "
                f"ana h={ana_hidden}x{ana_layers} trader {trader_hidden} "
                f"days/epoch={dpe} precompute={precompute}\n")
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({**vars(args), "plan": {k: str(v) for k, v in plan.items()},
                   "tier": tier, "days_per_epoch": dpe,
                   "batch_days": batch_days, "minibatch": minibatch,
                   "pred_steps": pred_steps, "ana_steps": ana_steps}, f, indent=2)

    # ---------------- agents ----------------
    dev_p, dev_a = plan["pred"], plan["ana"]
    dev_ta, dev_tb = plan["trade_a"], plan["trade_b"]
    n_out = len(PAIRS) * len(HORIZONS)
    obs_dim = W * N_FEATURES + n_out + EMB_DIM + 3 + 1 + FB_DIM

    predictor = Predictor(N_FEATURES, n_out, hidden=pred_hidden, layers=pred_layers).to(dev_p)
    analyzer = Analyzer(N_FEATURES, len(PAIRS), hidden=ana_hidden, layers=ana_layers,
                        emb_dim=EMB_DIM).to(dev_a)
    # EMA ("precision") copies of the advisors that the traders consume;
    # v13 keeps the EMA state inside the inference model (one copy, not two)
    predictor_inf = Predictor(N_FEATURES, n_out, hidden=pred_hidden, layers=pred_layers).to(dev_p)
    analyzer_inf = Analyzer(N_FEATURES, len(PAIRS), hidden=ana_hidden, layers=ana_layers,
                            emb_dim=EMB_DIM).to(dev_a)
    predictor_inf.eval()
    analyzer_inf.eval()
    ema_p = EmaWeights(predictor, predictor_inf, args.ema_decay) if args.ema_decay > 0 else None
    ema_a = EmaWeights(analyzer, analyzer_inf, args.ema_decay) if args.ema_decay > 0 else None
    if ema_p is None:
        predictor_inf = predictor
    if ema_a is None:
        analyzer_inf = analyzer

    torch.manual_seed(args.seed)
    trader_a = Trader(obs_dim, act_dim=4, hidden=trader_hidden).to(dev_ta)
    torch.manual_seed(args.seed_b)
    trader_b = Trader(obs_dim, act_dim=4, hidden=trader_hidden).to(dev_tb)
    set_seed(args.seed)

    opt_p = make_opt(predictor, args.lr_agents_base, opt8bit, adamw=True, weight_decay=1e-5)
    opt_a = make_opt(analyzer, args.lr_agents_base, opt8bit, adamw=True, weight_decay=1e-5)
    opt_ta = make_opt(trader_a, args.lr_trader, opt8bit)
    opt_tb = make_opt(trader_b, args.lr_b, opt8bit)

    # loss scalers for fp16 (bf16/fp32 need none)
    _sc = lambda: torch.amp.GradScaler(dev_p.type, enabled=(amp_on and amp_dtype == torch.float16))
    scaler_p, scaler_a, scaler_ta, scaler_tb = _sc(), _sc(), _sc(), _sc()

    steps_per_epoch = pred_steps + (args.bg_steps if (plan["parallel"] and not args.no_background_a) else 0)
    total_agent_steps = max(1, args.epochs) * steps_per_epoch + 200
    sched_p = GlobalSched(opt_p, args.lr_agents_base, total_agent_steps)
    sched_a = GlobalSched(opt_a, args.lr_agents_base, total_agent_steps)

    for name, m in (("predictor", predictor), ("analyzer", analyzer),
                    ("trader A", trader_a), ("trader B", trader_b)):
        n_par = sum(p.numel() for p in m.parameters())
        print(f"[agents] {name:10s}: {n_par/1e6:.2f}M params ({n_par*4/1e6:.1f} MB fp32)")
    print(f"[agents] trader obs_dim={obs_dim} | actions=4: 3 weights in [-1,1] "
          f"+ confidence gate | "
          f"traders: {'A+B' if not args.single_trader else 'A only'} | "
          f"{dpe} day(s) per epoch")

    # ---------------- resume ----------------
    start_epoch = 1
    mult_a, mult_b = 1.0, 1.0
    if args.resume:
        for name, model, opt in (
            ("predictor", predictor, opt_p),
            ("analyzer", analyzer, opt_a),
            ("trader_a", trader_a, opt_ta),
            ("trader_b", trader_b, opt_tb),
        ):
            pth = os.path.join(args.resume, f"{name}.pt")
            if os.path.exists(pth):
                ck = torch.load(pth, map_location="cpu", weights_only=False)
                model.load_state_dict(ck["model"])
                try:
                    opt.load_state_dict(ck["opt"])
                except Exception as e:  # e.g. fp32-Adam ckpt into 8-bit opt
                    print(f"[resume] {name}: optimizer state skipped ({type(e).__name__})")
                start_epoch = max(start_epoch, int(ck.get("epoch", 0)) + 1)
        for name, sched, ema, inf, live in (
            ("predictor", sched_p, ema_p, predictor_inf, predictor),
            ("analyzer", sched_a, ema_a, analyzer_inf, analyzer),
        ):
            pth = os.path.join(args.resume, f"{name}.pt")
            if os.path.exists(pth):
                ck = torch.load(pth, map_location="cpu", weights_only=False)
                sched.i = int(ck.get("sched_i", 0))
                if ema is not None and "ema" in ck:
                    inf.load_state_dict(ck["ema"])  # EMA lives in inf now
        for name, opt, base, attr in (("trader_a", opt_ta, args.lr_trader, "mult_a"),
                                       ("trader_b", opt_tb, args.lr_b, "mult_b")):
            pth = os.path.join(args.resume, f"{name}.pt")
            if os.path.exists(pth):
                ck = torch.load(pth, map_location="cpu", weights_only=False)
                if attr == "mult_a":
                    mult_a = float(ck.get("lr_mult", 1.0))
                else:
                    mult_b = float(ck.get("lr_mult", 1.0))
        for g in opt_ta.param_groups:
            g["lr"] = args.lr_trader * mult_a
        for g in opt_tb.param_groups:
            g["lr"] = args.lr_b * mult_b
        print(f"[resume] continuing from epoch {start_epoch}")

    # ---------------- replay ----------------
    replay_dates = deque(maxlen=args.replay)

    def sample_batch(todays):
        pool = [get_proc(d) for d in replay_dates] if replay_dates else []
        pool = list(todays) * 2 + pool
        if not pool:
            pool = list(todays)
        if len(pool) < batch_days:
            # pad with repeats so the batch shape is CONSTANT from epoch 1:
            # cudnn then autotunes exactly once (no workspace/fragment churn)
            reps = -(-batch_days // len(pool))
            pool = (pool * reps)[:batch_days]
        return random.sample(pool, batch_days)

    # ---------------- agent training steps ----------------
    def _oom_guard(fn, batch):
        """Retry an advisor step with half the batch if the GPU ran dry
        (shared machines, cudnn workspace spikes)."""
        try:
            return fn(batch)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            half = batch[: max(1, len(batch) // 2)]
            print(f"[oom] {fn.__name__}: retry with {len(half)} day(s)", flush=True)
            return fn(half)

    def predictor_train_batch(batch):
        x = torch.as_tensor(np.stack([d.feats for d in batch]), device=dev_p)
        Yn = torch.as_tensor(np.stack([d.Y_norm for d in batch]), device=dev_p)
        M = torch.as_tensor(np.stack([d.M for d in batch]), device=dev_p)
        # v15 trader->advisor feedback: weight minutes where positions were held
        act = np.ones((x.shape[0], x.shape[1]), dtype=np.float32)
        if args.feedback_weight > 0:
            for bi, d in enumerate(batch):
                a_ = getattr(d, "act", None)
                if a_ is None:
                    continue
                full = np.zeros((x.shape[1],), dtype=np.float32)
                n = min(a_.shape[0], x.shape[1] - (W - 1))
                full[W - 1: W - 1 + n] = np.asarray(a_[:n]).mean(axis=1)
                act[bi] = 1.0 + args.feedback_weight * full
        Aw = torch.as_tensor(act, device=dev_p).unsqueeze(-1)  # (B,T,1)
        with torch.autocast(dev_p.type, dtype=amp_dtype, enabled=amp_on):
            pred = predictor(x)
            mse = ((pred - Yn).pow(2) * M).sum() / M.sum().clamp(min=1.0)
            mse_fb = ((pred - Yn).pow(2) * M * Aw).sum() / (M * Aw).sum().clamp(min=1.0)
            p_ = torch.where(M, pred, torch.zeros_like(pred))
            y_ = torch.where(M, Yn, torch.zeros_like(Yn))
            n = M.sum(dim=(0, 1)).clamp(min=2)
            pm = p_.sum(dim=(0, 1)) / n
            ym = y_.sum(dim=(0, 1)) / n
            cov = (p_ * y_).sum(dim=(0, 1)) / n - pm * ym
            vp = (p_ * p_).sum(dim=(0, 1)) / n - pm.pow(2)
            vy = (y_ * y_).sum(dim=(0, 1)) / n - ym.pow(2)
            corr = cov / (vp.clamp_min(0).sqrt() * vy.clamp_min(0).sqrt() + 1e-6)
            corr = corr.clamp(-1.0, 1.0)
            loss = 0.5 * (mse + mse_fb) + args.corr_weight * (1.0 - corr.mean())
        sched_p.step()
        opt_p.zero_grad(set_to_none=True)
        if amp_on and amp_dtype == torch.float16:
            scaler_p.scale(loss).backward()
            scaler_p.unscale_(opt_p)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            scaler_p.step(opt_p)
            scaler_p.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            opt_p.step()
        return float(loss.item()), float(corr.mean().item())

    def analyzer_train_batch(batch):
        x = torch.as_tensor(np.stack([d.feats for d in batch]), device=dev_a)
        vol = torch.as_tensor(np.stack([d.vol_lab for d in batch]), device=dev_a)
        tr = torch.as_tensor(np.stack([d.trend_lab for d in batch]), device=dev_a)
        mask = torch.as_tensor(np.stack([d.lab_mask for d in batch]), device=dev_a)
        actw = np.ones((mask.shape[0], mask.shape[1]), dtype=np.float32)
        if args.feedback_weight > 0:
            for bi, d in enumerate(batch):
                a_ = getattr(d, "act", None)
                if a_ is None:
                    continue
                full = np.zeros((mask.shape[1],), dtype=np.float32)
                n = min(a_.shape[0], mask.shape[1] - (W - 1))
                full[W - 1: W - 1 + n] = np.asarray(a_[:n]).mean(axis=1)
                actw[bi] = 1.0 + 0.5 * args.feedback_weight * full
        Wt = torch.as_tensor(actw.reshape(-1), device=dev_a)
        with torch.autocast(dev_a.type, dtype=amp_dtype, enabled=amp_on):
            emb, vl, tl = analyzer(x)
            B, T = mask.shape
            v = vl.reshape(B * T, len(PAIRS), 3)
            t_ = tl.reshape(B * T, len(PAIRS), 3)
            yv = vol.reshape(B * T, len(PAIRS))
            yt = tr.reshape(B * T, len(PAIRS))
            m = mask.reshape(B * T)
            wm = Wt[m].unsqueeze(1)          # (n,1): broadcasts over the 3 pairs
            ce_v = F.cross_entropy(v[m], yv[m], reduction="none")
            ce_t = F.cross_entropy(t_[m], yt[m], reduction="none")
            loss = ((ce_v + ce_t) * wm).sum() \
                / wm.expand(ce_v.shape).sum().clamp(min=1.0)
        sched_a.step()
        opt_a.zero_grad(set_to_none=True)
        if amp_on and amp_dtype == torch.float16:
            scaler_a.scale(loss).backward()
            scaler_a.unscale_(opt_a)
            torch.nn.utils.clip_grad_norm_(analyzer.parameters(), 1.0)
            scaler_a.step(opt_a)
            scaler_a.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(analyzer.parameters(), 1.0)
            opt_a.step()
        return float(loss.item())

    @torch.no_grad()
    def predictor_eval(day):
        x = torch.as_tensor(day.feats[None], device=dev_p)
        Yn = torch.as_tensor(day.Y_norm[None], device=dev_p)
        M = torch.as_tensor(day.M[None], device=dev_p)
        pred = predictor(x)
        out = {"loss": float(((pred - Yn).pow(2) * M).sum().item() / max(M.sum().item(), 1.0))}
        pred_np = pred[0].cpu().numpy()
        for tag, k in (("15", IDX15), ("60", IDX60)):
            maes, corrs = [], []
            for i in range(len(PAIRS)):
                c = i * len(HORIZONS) + k
                mc = day.M[:, c]
                if mc.sum() <= 30:
                    continue
                pp = (pred_np[:, c] * day.vol_scale[:, c])[mc]
                yv = day.Y_pct[:, c][mc]
                maes.append(float(np.abs(pp - yv).mean()))
                corrs.append(_corr_np(pp, yv))
            out[f"mae{tag}"] = float(np.mean(maes)) if maes else float("nan")
            out[f"corr{tag}"] = float(np.mean(corrs)) if corrs else float("nan")
        return out

    @torch.no_grad()
    def analyzer_eval(day):
        x = torch.as_tensor(day.feats[None], device=dev_a)
        vol = torch.as_tensor(day.vol_lab[None], device=dev_a)
        tr = torch.as_tensor(day.trend_lab[None], device=dev_a)
        mask = torch.as_tensor(day.lab_mask[None], device=dev_a)
        emb, vl, tl = analyzer(x)
        m = mask[0].cpu().numpy()
        v = vl[0].argmax(-1).cpu().numpy()
        t_ = tl[0].argmax(-1).cpu().numpy()
        yv = tr[0].cpu().numpy()
        yt = vol[0].cpu().numpy()
        va = float((v[m] == yv[m]).mean()) if m.any() else float("nan")
        ta = float((t_[m] == yt[m]).mean()) if m.any() else float("nan")
        B, T = mask.shape
        vv = vl.reshape(B * T, len(PAIRS), 3)
        tt = tl.reshape(B * T, len(PAIRS), 3)
        yv2 = tr.reshape(B * T, len(PAIRS))
        yt2 = vol.reshape(B * T, len(PAIRS))
        mm = mask.reshape(B * T)
        loss = float(F.cross_entropy(vv[mm], yv2[mm]).item()
                     + F.cross_entropy(tt[mm], yt2[mm]).item())
        return {"loss": loss, "vol_acc": va, "trend_acc": ta}

    @torch.no_grad()
    def advisor_outputs_batch(procs):
        """EMA advisors, batched across days (all days are exactly 1440 min)."""
        x = torch.as_tensor(np.stack([p.feats for p in procs]), device=dev_p)
        fn = predictor_inf(x).cpu().numpy().astype(np.float32)          # (B,T,n_out)
        xa = torch.as_tensor(np.stack([p.feats for p in procs]), device=dev_a)
        emb, vl, tl = analyzer_inf(xa)
        emb = emb.cpu().numpy().astype(np.float32)                      # (B,T,E)
        vol_cls = vl.argmax(-1).cpu().numpy().astype(np.int64)          # (B,T,3)
        out = []
        for b, p in enumerate(procs):
            fp = fn[b] * p.vol_scale                                    # % space
            out.append((fn[b], fp, emb[b], vol_cls[b]))
        return out

    # ---------------- trading ----------------
    env_kwargs = dict(capital=args.capital, target=args.target, fee=args.fee,
                      slip=args.slip, max_lev=args.max_lev,
                      terminal_bonus=args.terminal_bonus,
                      trade_win_bonus=args.trade_win_bonus,
                      trade_loss_penalty=args.trade_loss_penalty)
    buf_a = RolloutBuffer()
    buf_b = RolloutBuffer()

    def vec_rollout(trader, buf, procs, advisors, deterministic=False):
        """Roll out N day-environments in lockstep with ONE batched policy
        forward per minute. Trajectories are stored per-env then concatenated
        (PPO's GAE cuts correctly at each episode's done flag)."""
        envs = [TradingEnv(p.day, p.feats, fn, emb, **env_kwargs)
                for p, (fn, _, emb, _) in zip(procs, advisors)]
        obs_list = [e.reset() for e in envs]
        N = len(envs)
        per_env = [dict(obs=[], act=[], logp=[], val=[], rew=[], done=[]) for _ in range(N)]
        ep_rets = [0.0] * N
        dev = next(trader.parameters()).device
        idx_active = list(range(N))
        while idx_active:
            obs = np.stack([obs_list[i] for i in idx_active]).astype(np.float32)
            o_t = torch.as_tensor(obs, device=dev)
            with torch.no_grad():
                mu, std, value = trader(o_t)
                if deterministic:
                    a = mu.clamp(-1.0, 1.0)
                    logp = None
                else:
                    dist = Normal(mu, std)
                    a_raw = dist.sample()
                    logp = dist.log_prob(a_raw).sum(-1)
                    a = a_raw.clamp(-1.0, 1.0)
            a_np = a.cpu().numpy()
            logp_np = logp.cpu().numpy() if logp is not None else None
            v_np = value.cpu().numpy()
            still = []
            for j, i in enumerate(idx_active):
                prev_obs = obs_list[i]
                obs2, r, done, info = envs[i].step(a_np[j])
                obs_list[i] = obs2
                if not deterministic:
                    rec = per_env[i]
                    rec["obs"].append(prev_obs)
                    rec["act"].append(a_np[j])
                    rec["logp"].append(float(logp_np[j]))
                    rec["val"].append(float(v_np[j]))
                    rec["rew"].append(r)
                    rec["done"].append(done)
                    ep_rets[i] += r
                if not done:
                    still.append(i)
            idx_active = still
        if buf is not None and not deterministic:
            for rec in per_env:
                for t in range(len(rec["rew"])):
                    buf.add(rec["obs"][t], rec["act"][t], rec["logp"][t],
                            rec["val"][t], rec["rew"][t], rec["done"][t])
        finals = [float(e.equity) for e in envs]
        return envs, finals, ep_rets

    val_proc_cache = {}

    def joint_vec_rollout(ta, tb, buf_ta_, buf_tb_, procs, advisors,
                          deterministic=False, comm=True, comm_dropout=0.0):
        """v15: BOTH traders roll out the same days in lockstep, exchanging
        their lagged effective weights + session return every minute (team
        channel). Each keeps its own book; each records into its own buffer."""
        mk = lambda: [TradingEnv(p.day, p.feats, fn, emb, **env_kwargs)
                      for p, (fn, _, emb, _) in zip(procs, advisors)]
        envs_a, envs_b = mk(), mk()
        obs_a = [e.reset() for e in envs_a]
        obs_b = [e.reset() for e in envs_b]
        N = len(envs_a)
        rec_a = [dict(obs=[], act=[], logp=[], val=[], rew=[], done=[]) for _ in range(N)]
        rec_b = [dict(obs=[], act=[], logp=[], val=[], rew=[], done=[]) for _ in range(N)]
        rets_a, rets_b = [0.0] * N, [0.0] * N
        dev_a_, dev_b_ = next(ta.parameters()).device, next(tb.parameters()).device
        z3 = np.zeros(3, dtype=np.float32)
        # lagged team state per env pair: (last eff weights, log session return)
        st_a = [(z3, 0.0)] * N
        st_b = [(z3, 0.0)] * N
        comm_day = [comm and (deterministic or random.random() >= comm_dropout)
                    for _ in range(N)]
        act_i, act_j = list(range(N)), list(range(N))
        while act_i or act_j:
            for i in act_i:
                wb, rb = st_b[i] if comm_day[i] else (z3, 0.0)
                envs_a[i].set_other(wb, rb)
            for i in act_j:
                wa, ra = st_a[i] if comm_day[i] else (z3, 0.0)
                envs_b[i].set_other(wa, ra)

            def _fwd(trader, obs_list, idxs, dev):
                o = torch.as_tensor(
                    np.stack([obs_list[i] for i in idxs]).astype(np.float32), device=dev)
                with torch.no_grad():
                    mu, std, value = trader(o)
                    if deterministic:
                        a = mu.clamp(-1.0, 1.0)
                        return a.cpu().numpy(), None, value.cpu().numpy()
                    dist = Normal(mu, std)
                    a_raw = dist.sample()
                    logp = dist.log_prob(a_raw).sum(-1)
                    return a_raw.clamp(-1.0, 1.0).cpu().numpy(), \
                        logp.cpu().numpy(), value.cpu().numpy()

            a_np, lp_a, v_a = _fwd(ta, obs_a, act_i, dev_a_) if act_i else (None, None, None)
            b_np, lp_b, v_b = _fwd(tb, obs_b, act_j, dev_b_) if act_j else (None, None, None)
            still_i, still_j = [], []
            for j, i in enumerate(act_i):
                po = obs_a[i]
                obs_a[i], r, done, _ = envs_a[i].step(a_np[j])
                if not deterministic:
                    ra_ = rec_a[i]
                    ra_["obs"].append(po); ra_["act"].append(a_np[j])
                    ra_["logp"].append(float(lp_a[j])); ra_["val"].append(float(v_a[j]))
                    ra_["rew"].append(r); ra_["done"].append(done)
                rets_a[i] += r
                st_a[i] = (envs_a[i].last_eff.copy(),
                           float(np.clip(np.log(max(envs_a[i].equity, 1e-9)
                                                 / envs_a[i].capital), -4.0, 4.0)))
                if not done:
                    still_i.append(i)
            for j, i in enumerate(act_j):
                po = obs_b[i]
                obs_b[i], r, done, _ = envs_b[i].step(b_np[j])
                if not deterministic:
                    rb_ = rec_b[i]
                    rb_["obs"].append(po); rb_["act"].append(b_np[j])
                    rb_["logp"].append(float(lp_b[j])); rb_["val"].append(float(v_b[j]))
                    rb_["rew"].append(r); rb_["done"].append(done)
                rets_b[i] += r
                st_b[i] = (envs_b[i].last_eff.copy(),
                           float(np.clip(np.log(max(envs_b[i].equity, 1e-9)
                                                 / envs_b[i].capital), -4.0, 4.0)))
                if not done:
                    still_j.append(i)
            act_i, act_j = still_i, still_j
        for buf, recs in ((buf_ta_, rec_a), (buf_tb_, rec_b)):
            if buf is not None and not deterministic:
                for rec in recs:
                    for t_ in range(len(rec["rew"])):
                        buf.add(rec["obs"][t_], rec["act"][t_], rec["logp"][t_],
                                rec["val"][t_], rec["rew"][t_], rec["done"][t_])
        return ((envs_a, [float(e.equity) for e in envs_a], rets_a),
                (envs_b, [float(e.equity) for e in envs_b], rets_b))

    def get_val_proc(date):
        if date not in val_proc_cache:
            val_proc_cache[date] = DayProcessed(date, days[date])
        return val_proc_cache[date]

    def _eval_stats(finals):
        finals = np.asarray(finals, dtype=np.float64)
        return {"mean": float(finals.mean()), "median": float(np.median(finals)),
                "hit": float((finals >= args.target).mean())}

    def evaluate_pair(limit):
        """v15: evaluate BOTH traders jointly with the team channel active -
        the same conditions they trained under (minus comm dropout)."""
        step = max(1, len(val_dates) // max(limit, 1))
        sel = val_dates[::step][:limit]
        fa, fb = [], []
        eb = max(1, args.eval_batch)
        for s in range(0, len(sel), eb):
            batch_dates = sel[s: s + eb]
            procs = [get_val_proc(d) for d in batch_dates]
            advisors = advisor_outputs_batch(procs)
            if args.single_trader:
                _, f, _ = vec_rollout(trader_a, None, procs, advisors,
                                      deterministic=True)
                fa.extend(f)
            else:
                (_, fA, _), (_, fB, _) = joint_vec_rollout(
                    trader_a, trader_b, None, None, procs, advisors,
                    deterministic=True, comm=True)
                fa.extend(fA)
                fb.extend(fB)
        out = {"a": _eval_stats(fa)}
        if not args.single_trader:
            out["b"] = _eval_stats(fb)
        return out

    def adapt_lr(opt, base_lr, mult, kl):
        if not math.isfinite(kl):
            return mult
        if kl > 1.5 * args.target_kl:
            mult *= 0.7
        elif kl < args.target_kl / 1.5:
            mult *= 1.3
        mult = min(max(mult, 0.05), 20.0)
        for g in opt.param_groups:
            g["lr"] = base_lr * mult
        return mult

    def _cpu_sd(sd):
        """Move a state dict to CPU one tensor at a time (RAM-safe for
        multi-GB models on small boxes)."""
        return {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in sd.items()}

    def save_ckpt(directory, epoch, only=None):
        """Full rolling checkpoint, or (only='trader_a'/'trader_b') a small
        best-policy snapshot with just that trader's weights - keeps disk
        usage bounded on small machines."""
        os.makedirs(directory, exist_ok=True)
        if only in ("trader_a", "trader_b"):
            model = trader_a if only == "trader_a" else trader_b
            torch.save({"model": _cpu_sd(model.state_dict()), "epoch": epoch,
                        "lr_mult": mult_a if only == "trader_a" else mult_b},
                       os.path.join(directory, f"{only}.pt"))
            return
        # disk guard: prune best-* snapshots if space runs low
        try:
            if os.statvfs(args.out).f_bavail * os.statvfs(args.out).f_frsize < 10e9:
                for _b in ("best_a", "best_b"):
                    _p = os.path.join(args.out, "ckpt", _b)
                    if os.path.isdir(_p):
                        import shutil
                        shutil.rmtree(_p, ignore_errors=True)
                        print(f"[ckpt] low disk: pruned {_p}", flush=True)
        except Exception:
            pass
        for name, model, opt, extra in (
            ("predictor", predictor, opt_p,
             {"sched_i": sched_p.i,
              "ema": _cpu_sd(predictor_inf.state_dict()) if ema_p else None}),
            ("analyzer", analyzer, opt_a,
             {"sched_i": sched_a.i,
              "ema": _cpu_sd(analyzer_inf.state_dict()) if ema_a else None}),
            ("trader_a", trader_a, opt_ta, {"lr_mult": mult_a}),
            ("trader_b", trader_b, opt_tb, {"lr_mult": mult_b}),
        ):
            torch.save({"model": _cpu_sd(model.state_dict()),
                        "opt": _cpu_sd(opt.state_dict()),
                        "epoch": epoch, **extra},
                       os.path.join(directory, f"{name}.pt"))

    # ---------------- metrics history ----------------
    hist = {k: [] for k in CSV_FIELDS}
    csv_path = os.path.join(args.out, "metrics.csv")
    if args.resume and os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                for k in CSV_FIELDS:
                    v = row.get(k, "")
                    if k == "date":
                        hist[k].append(v)
                    else:
                        try:
                            hist[k].append(float(v))
                        except (TypeError, ValueError):
                            hist[k].append(float("nan"))

    csv_new = not os.path.exists(csv_path)
    csv_f = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
    if csv_new:
        writer.writeheader()

    used_devices = sorted({dev_p, dev_a, dev_ta, dev_tb}, key=str)
    vram_peak = {}

    # ---------------- main loop ----------------
    best_eval = {"a": -1e9, "b": -1e9}
    t_start = time.time()
    epoch = start_epoch - 1
    print(f"[train] start: {args.epochs} epochs x {dpe} day(s) | ${args.capital} -> "
          f"${args.target} | fee={args.fee} slip={args.slip} max_lev={args.max_lev} | "
          f"traders A+B" + ("" if not args.single_trader else " (B disabled)"))

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            date_list = [random.choice(train_dates) for _ in range(dpe)]
            date0 = date_list[0]
            procs = [get_proc(d) for d in date_list]

            # ---- Phase A: predictor + analyzer train on the epoch's days (+replay)
            pred_stats, ana_stats = [], []

            def _pa():
                for _ in range(pred_steps):
                    b = sample_batch(procs)
                    if b:
                        pred_stats.append(_oom_guard(predictor_train_batch, b))

            def _aa():
                for _ in range(ana_steps):
                    b = sample_batch(procs)
                    if b:
                        ana_stats.append(_oom_guard(analyzer_train_batch, b))

            ths = [threading.Thread(target=_pa), threading.Thread(target=_aa)]
            for t in ths:
                t.start()
            for t in ths:
                t.join()

            # refresh the EMA ("precision") advisors the traders will consume
            if ema_p is not None:
                ema_p.update()
            if ema_a is not None:
                ema_a.update()

            pm = predictor_eval(procs[0])
            am = analyzer_eval(procs[0])
            advisors = advisor_outputs_batch(procs)
            forecast_norm0, forecast_pct0, emb0, vol_cls0 = advisors[0]

            # ---- Phase B: BOTH traders roll out ALL the epoch's days
            # (vectorised), while the advisors keep training in background.
            bg = []
            if plan["parallel"] and not args.no_background_a:
                def _bgp():
                    for _ in range(args.bg_steps):
                        b = sample_batch(procs)
                        if b:
                            _oom_guard(predictor_train_batch, b)

                def _bga():
                    for _ in range(args.bg_steps):
                        b = sample_batch(procs)
                        if b:
                            _oom_guard(analyzer_train_batch, b)

                bg = [threading.Thread(target=_bgp), threading.Thread(target=_bga)]
                for t in bg:
                    t.start()

            if args.single_trader:
                e_a, f_a, r_a = vec_rollout(trader_a, buf_a, procs, advisors)
                res_a, res_b = {"envs": e_a, "finals": f_a, "rets": r_a}, {}
            else:
                # v15: both traders roll out in LOCKSTEP with the team channel
                (e_a, f_a, r_a), (e_b, f_b, r_b) = joint_vec_rollout(
                    trader_a, trader_b, buf_a, buf_b, procs, advisors,
                    comm=True, comm_dropout=args.comm_dropout)
                res_a = {"envs": e_a, "finals": f_a, "rets": r_a}
                res_b = {"envs": e_b, "finals": f_b, "rets": r_b}
                # trader -> advisor feedback: remember where positions were held
                for p, ea_i, eb_i in zip(procs, e_a, e_b):
                    p.act = np.maximum(np.abs(ea_i.weights_hist),
                                       np.abs(eb_i.weights_hist)).astype(np.float32)

            def _wr(envs):
                wrs = [e.win_rate for e in envs
                       if e.n_trades > 0 and not math.isnan(e.win_rate)]
                nts = [float(e.n_trades) for e in envs]
                return (float(np.mean(wrs)) if wrs else float("nan"),
                        float(np.mean(nts)) if nts else 0.0)

            wr_a, ntr_a = _wr(res_a["envs"])
            wr_b, ntr_b = _wr(res_b["envs"]) if res_b else (float("nan"), 0.0)
            ppo_a = ppo_update(trader_a, opt_ta, buf_a, dev_ta, gamma=args.gamma,
                               lam=args.lam, ppo_epochs=args.ppo_epochs,
                               minibatch=minibatch, ent_coef=args.ent_coef,
                               amp_dtype=amp_dtype, scaler=scaler_ta) \
                if len(buf_a) >= 16 else {k: float("nan") for k in
                                          ("pi_loss", "v_loss", "entropy", "kl", "clip_frac")}
            mult_a = adapt_lr(opt_ta, args.lr_trader, mult_a, ppo_a.get("kl", float("nan")))
            if not args.single_trader:
                ppo_b = ppo_update(trader_b, opt_tb, buf_b, dev_tb, gamma=args.gamma,
                                   lam=args.lam, ppo_epochs=args.ppo_epochs,
                                   minibatch=minibatch, ent_coef=args.ent_b,
                                   amp_dtype=amp_dtype, scaler=scaler_tb) \
                    if len(buf_b) >= 16 else {k: float("nan") for k in
                                              ("pi_loss", "v_loss", "entropy", "kl", "clip_frac")}
                mult_b = adapt_lr(opt_tb, args.lr_b, mult_b, ppo_b.get("kl", float("nan")))
            else:
                ppo_b = {k: float("nan") for k in ("pi_loss", "v_loss", "entropy", "kl", "clip_frac")}

            for t in bg:
                t.join()
            for d in date_list:
                replay_dates.append(d)

            finals_a = np.asarray(res_a["finals"])
            final_eq_a = float(finals_a.mean())
            hit_a = float((finals_a >= args.target).mean())
            if not args.single_trader:
                finals_b = np.asarray(res_b["finals"])
                final_eq_b = float(finals_b.mean())
                hit_b = float((finals_b >= args.target).mean())
            else:
                finals_b = np.asarray([float("nan")])
                final_eq_b, hit_b = float("nan"), float("nan")
            mc_a = float(any(e.margin_call for e in res_a["envs"]))

            # ---- held-out evaluation (both traders, batched + threaded)
            eval_row = {}
            for k in ("eval_mean_eq_a", "eval_median_eq_a", "eval_hit_a",
                      "eval_mean_eq_b", "eval_median_eq_b", "eval_hit_b"):
                eval_row[k] = float("nan")
            if val_dates and (epoch % args.eval_every == 0 or epoch == args.epochs):
                ev_res = evaluate_pair(args.eval_limit)
                eval_row.update({"eval_mean_eq_a": ev_res["a"]["mean"],
                                 "eval_median_eq_a": ev_res["a"]["median"],
                                 "eval_hit_a": ev_res["a"]["hit"]})
                if ev_res["a"]["mean"] > best_eval["a"]:
                    best_eval["a"] = ev_res["a"]["mean"]
                    save_ckpt(os.path.join(args.out, "ckpt", "best_a"), epoch,
                              only="trader_a")
                if not args.single_trader:
                    eval_row.update({"eval_mean_eq_b": ev_res["b"]["mean"],
                                     "eval_median_eq_b": ev_res["b"]["median"],
                                     "eval_hit_b": ev_res["b"]["hit"]})
                    if ev_res["b"]["mean"] > best_eval["b"]:
                        best_eval["b"] = ev_res["b"]["mean"]
                        save_ckpt(os.path.join(args.out, "ckpt", "best_b"), epoch,
                                  only="trader_b")

            # ---- VRAM accounting
            for d in used_devices:
                if d.type == "cuda":
                    try:
                        vram_peak[str(d)] = max(vram_peak.get(str(d), 0.0),
                                                torch.cuda.max_memory_allocated(d) / 1e9)
                    except RuntimeError:
                        pass

            # ---- logs / plots / checkpoints
            row = {
                "epoch": epoch, "date": date0,
                "pred_loss": pm["loss"], "pred_mae15": pm["mae15"], "pred_mae60": pm["mae60"],
                "pred_corr15": pm["corr15"], "pred_corr60": pm["corr60"],
                "lr_pred": opt_p.param_groups[0]["lr"],
                "ana_loss": am["loss"], "ana_vol_acc": am["vol_acc"],
                "ana_trend_acc": am["trend_acc"], "lr_ana": opt_a.param_groups[0]["lr"],
                "pi_loss_a": ppo_a["pi_loss"], "v_loss_a": ppo_a["v_loss"],
                "entropy_a": ppo_a["entropy"], "kl_a": ppo_a["kl"],
                "lr_trader_a": opt_ta.param_groups[0]["lr"],
                "ep_ret_a": float(np.mean(res_a["rets"])), "final_eq_a": final_eq_a,
                "hit_a": hit_a, "margin_call_a": mc_a,
                "pi_loss_b": ppo_b["pi_loss"], "v_loss_b": ppo_b["v_loss"],
                "entropy_b": ppo_b["entropy"], "kl_b": ppo_b["kl"],
                "lr_trader_b": opt_tb.param_groups[0]["lr"] if not args.single_trader else float("nan"),
                "ep_ret_b": float(np.mean(res_b["rets"])) if not args.single_trader else float("nan"),
                "final_eq_b": final_eq_b, "hit_b": hit_b,
                "margin_call_b": float(any(e.margin_call for e in res_b["envs"]))
                if not args.single_trader else float("nan"),
                "wr_a": wr_a, "ntr_a": ntr_a,
                "wr_b": wr_b if not args.single_trader else float("nan"),
                "ntr_b": ntr_b if not args.single_trader else float("nan"),
                **eval_row, "elapsed": round(time.time() - t0, 2), "days": dpe,
            }
            writer.writerow(row)
            csv_f.flush()
            for k in CSV_FIELDS:
                hist[k].append(row.get(k, float("nan")))

            if args.plot_every and (epoch % args.plot_every == 0 or epoch == args.epochs):
                closes0 = np.stack([procs[0].day[p]["close"] for p in PAIRS], axis=1)
                btc15 = IDX15
                env_a0, env_b0 = res_a["envs"][0], (res_b["envs"][0] if not args.single_trader else None)
                # (unchanged) combined day snapshot: both traders on one chart
                plots.plot_day_snapshot(
                    os.path.join(args.out, "epochs", f"epoch_{epoch:05d}_{date0}.png"),
                    epoch, date0, closes0, env_a0.weights_hist, env_a0.equity_hist,
                    env_b0.equity_hist if env_b0 is not None else None,
                    args.capital, args.target,
                    forecast_pct0[:, btc15], procs[0].Y_pct[:, btc15], pm["corr15"], vol_cls0,
                    hit_a=bool(finals_a[0] >= args.target),
                    hit_b=bool(finals_b[0] >= args.target) if env_b0 is not None else False,
                    margin_call=env_a0.margin_call,
                    extra=f"A PnL {finals_a[0] - args.capital:+.2f}$ ({dpe}d mean {final_eq_a - args.capital:+.2f}$)")
                # v12.1: ONE tall PNG for BOTH traders - A's panels on top,
                # B's directly below (B's earning graph under A's)
                plots.plot_traders_stacked(
                    os.path.join(args.out, "epochs", f"epoch_{epoch:05d}_{date0}_traders.png"),
                    epoch, date0, closes0,
                    {"weights": env_a0.weights_hist, "equity": env_a0.equity_hist,
                     "hit": bool(finals_a[0] >= args.target),
                     "margin_call": env_a0.margin_call},
                    None if env_b0 is None else
                    {"weights": env_b0.weights_hist, "equity": env_b0.equity_hist,
                     "hit": bool(finals_b[0] >= args.target),
                     "margin_call": env_b0.margin_call},
                    args.capital, args.target,
                    extra=f"{dpe}d mean A {final_eq_a - args.capital:+.2f}$"
                          + (f" / B {final_eq_b - args.capital:+.2f}$" if env_b0 is not None else ""))
                plots.plot_curves(os.path.join(args.out, "training_curves.png"),
                                  hist, args.capital, args.target)

            if epoch % args.ckpt_every == 0 or epoch == args.epochs:
                save_ckpt(os.path.join(args.out, "ckpt"), epoch)

            if args.log_every and (epoch % args.log_every == 0 or epoch == args.epochs):
                ev_txt = ""
                if not math.isnan(eval_row["eval_mean_eq_a"]):
                    ev_txt = (f" | EVAL A ${eval_row['eval_mean_eq_a']:.2f}"
                              f"({eval_row['eval_hit_a']*100:.0f}%)"
                              + (f" B ${eval_row['eval_mean_eq_b']:.2f}"
                                 f"({eval_row['eval_hit_b']*100:.0f}%)"
                                 if not math.isnan(eval_row["eval_mean_eq_b"]) else ""))
                vram_txt = " ".join(f"{k}={v:.1f}GB" for k, v in sorted(vram_peak.items()))
                print(f"[{epoch:5d}] {date0}{'+' + str(dpe - 1) + 'd' if dpe > 1 else ''} "
                      f"A=${final_eq_a:6.2f}(max ${finals_a.max():5.2f},"
                      f"{hit_a*100:.0f}%hit)"
                      + (f" B=${final_eq_b:6.2f}(max ${finals_b.max():5.2f},"
                         f"{hit_b*100:.0f}%hit)" if not args.single_trader else "")
                      + f" pCorr15={pm['corr15']:+.3f} aTrd={am['trend_acc']:.2f}"
                      + f" wrA={wr_a:.2f}/{ntr_a:.0f}t"
                      + (f" wrB={wr_b:.2f}/{ntr_b:.0f}t" if not args.single_trader else "")
                      + " "
                      f"lrP={opt_p.param_groups[0]['lr']:.1e} "
                      f"lrA={opt_ta.param_groups[0]['lr']:.1e}"
                      + (f" lrB={opt_tb.param_groups[0]['lr']:.1e}" if not args.single_trader else "")
                      + f"{ev_txt} | {vram_txt} [{time.time() - t0:.1f}s]", flush=True)
    finally:
        csv_f.close()
        save_ckpt(os.path.join(args.out, "ckpt"), epoch)
        print(f"[train] finished {epoch} epochs in {(time.time() - t_start)/60:.1f} min; "
              f"artifacts in {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
