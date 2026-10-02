"""Misc helpers: seeding, hardware detection, device plan, model-size solver.

v13: sizes are SOLVED from the mandate (analyzer 500M, predictor 2B, trader 3B
each) and scaled down automatically to whatever the machine can actually hold
(per-GPU free VRAM, optimizer mode, free disk for checkpoints). Big tiers run
with mixed-precision autocast and (if available) bitsandbytes 8-bit AdamW.
"""

import math
import os
import platform

import numpy as np
import torch


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def total_ram_gb():
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return None


def disk_free_gb(path="."):
    try:
        import shutil

        return shutil.disk_usage(path).free / 1e9
    except Exception:
        return None


def detect_resources(out_path="."):
    """Inventory of the machine: CPUs, RAM, GPUs (with free VRAM), free disk."""
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free_b, total_b = torch.cuda.mem_get_info(i)
            gpus.append(
                {
                    "index": i,
                    "name": props.name,
                    "total_gb": round(total_b / 1e9, 2),
                    "free_gb": round(free_b / 1e9, 2),
                }
            )
    return {
        "hostname": os.uname().nodename if hasattr(os, "uname") else "",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),
        "ram_gb": None if total_ram_gb() is None else round(total_ram_gb(), 2),
        "disk_free_gb": disk_free_gb(out_path),
        "gpus": gpus,
    }


def format_resources(res) -> str:
    lines = [
        f"host   : {res['hostname']}",
        f"python : {res['python']}  torch={res['torch']}  cuda={res['cuda_available']}",
        f"cpu    : {res['cpu_count']} cores   ram={res['ram_gb']} GB   "
        f"disk_free={res.get('disk_free_gb')} GB",
    ]
    if res["gpus"]:
        for g in res["gpus"]:
            lines.append(
                f"gpu {g['index']}  : {g['name']}  total={g['total_gb']}GB free={g['free_gb']}GB"
            )
    else:
        lines.append("gpu    : none -> CPU mode")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# exact parameter-count formulas for the three v10-family agents
# --------------------------------------------------------------------------
def gru_params(H, L, n_in):
    """nn.GRU(n_in, H, num_layers=L) parameter count."""
    return 3 * H * (n_in + H + 2) + (L - 1) * 3 * H * (2 * H + 2)


def predictor_params(H, L, n_in, n_out=12):
    return gru_params(H, L, n_in) + 2 * H + H * n_out + n_out  # + LN + head


def analyzer_params(H, L, n_in, emb_dim=16, n_pairs=3, n_classes=3):
    return (gru_params(H, L, n_in) + 2 * H + H * emb_dim + emb_dim
            + 2 * (emb_dim * n_pairs * n_classes + n_pairs * n_classes))


def trader_params(h, obs_dim, layers=3):
    p = obs_dim * h + h                       # first Linear
    for _ in range(layers - 1):
        p += h * h + h                        # hidden->hidden Linears
    p += 3 * h + 3 + h + 1 + 3                # mu head, value head, log_std
    return p


def _solve_down(target, fn, mult=64):
    """Largest x = k*mult (k>=1) with fn(x) <= target (fn is increasing)."""
    if target < fn(mult):
        return mult
    lo, hi = 1, 2
    while fn(hi * mult) <= target:
        hi *= 2
    # invariant: fn(lo*mult) <= target < fn(hi*mult)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fn(mid * mult) <= target:
            lo = mid
        else:
            hi = mid
    return lo * mult


# the v13 size mandate
MANDATE = {"analyzer": 0.50e9, "predictor": 2.0e9, "trader": 3.0e9}


def solve_tier(min_free_gb, budget_gb, disk_free_gb, opt8bit,
               n_in, obs_dim, mandate=None):
    """Solve agent sizes for this machine.

    Targets are the v13 mandate (analyzer 500M, predictor 2B, trader 3B each);
    if the hardware cannot hold them the sizes are scaled down - preserving
    the mandate's proportions - to the binding constraint:

    - per-GPU VRAM  (weights + grads + optimizer states + EMA copy + activations)
    - free disk     (one rolling checkpoint: weights + optimizer states)

    opt8bit switches the optimizer-state cost from 8 to 2 bytes/param
    (bitsandbytes AdamW8bit), which roughly doubles the model size a given
    GPU can train.
    """
    mandate = mandate or MANDATE
    ana_t, pred_t, trd_t = mandate["analyzer"], mandate["predictor"], mandate["trader"]
    # 0.72 of *currently* free VRAM: co-located workloads (e.g. miners)
    # fluctuate their footprint, so leave a wide third-party margin
    bud = min(float(budget_gb), min_free_gb * 0.72) if min_free_gb > 0 else 0.0

    if bud < 1.0:  # CPU / no usable GPU: fixed tiny tier
        return {"name": "XS (CPU)", "pred_hidden": 64, "pred_layers": 1,
                "ana_hidden": 64, "ana_layers": 1,
                "trader_hidden": (256, 256),
                "batch_days": 2, "minibatch": 128,
                "pred_steps": 6, "ana_steps": 6,
                "note": "cpu tier"}

    opt_b = 2 if opt8bit else 8                       # optimizer bytes/param
    adv_bpp = 4 + 4 + opt_b + 4                       # live+grads+opt+EMA copy
    trd_bpp = 4 + 4 + opt_b                           # traders have no EMA
    ck_bpp = 4 + opt_b                                # checkpoint bytes/param
    act_adv = max(1.2, 0.25 * bud)                    # GRU train activations
    act_trd = max(0.8, 0.12 * bud)                    # PPO minibatch activations

    pred_cap = max(0.0, bud - act_adv) * 1e9 / adv_bpp
    ana_cap = max(0.0, bud - act_adv) * 1e9 / adv_bpp
    trd_cap = max(0.0, bud - act_trd) * 1e9 / trd_bpp

    total_t = pred_t + ana_t + 2 * trd_t              # 8.5B at full mandate
    disk_cap = (float(disk_free_gb) * 0.55) * 1e9 / ck_bpp
    s = min(1.0, disk_cap / total_t)                  # disk keeps proportions
    pred_p = min(pred_t * s, pred_cap)
    ana_p = min(ana_t * s, ana_cap)
    trd_p = min(trd_t * s, trd_cap)

    pred_H = _solve_down(max(pred_p, 1e5), lambda H: predictor_params(H, 3, n_in))
    ana_H = _solve_down(max(ana_p, 1e5), lambda H: analyzer_params(H, 3, n_in))
    trd_h = _solve_down(max(trd_p, 1e5), lambda h: trader_params(h, obs_dim, 3))

    total = (predictor_params(pred_H, 3, n_in) + analyzer_params(ana_H, 3, n_in)
             + 2 * trader_params(trd_h, obs_dim, 3))
    if total >= 4e9:
        bd, mb, ps = 32, 2048, 24
    elif total >= 1e9:
        bd, mb, ps = 24, 1024, 16
    elif total >= 2e8:
        bd, mb, ps = 16, 512, 12
    elif total >= 2e7:
        bd, mb, ps = 8, 512, 10
    else:
        bd, mb, ps = 4, 256, 8
    # activation-aware caps: advisor GRU training (incl. cudnn workspace)
    # measures ~H*8e-5 GB per day on Turing cards; trader PPO ~ h*30 B/sample
    bd = max(1, min(bd, int(act_adv / (pred_H * 8.0e-5))))
    mb = max(64, min(mb, int(act_trd * 1e9 / (trd_h * 30.0))))

    full = (s >= 0.995 and pred_p >= pred_t and ana_p >= ana_t and trd_p >= trd_t)
    name = "GIGA (mandate sizes)" if full else f"AUTO ({total / 1e9:.3g}B total)"
    note = ("full mandate sizes" if full else
            f"scaled to {s * 100:.0f}% of mandate by disk"
            + ("" if s >= 0.995 else "")
            + (f"; VRAM caps pred {pred_cap / 1e9:.2f}B ana {ana_cap / 1e9:.2f}B"
               f" trader {trd_cap / 1e9:.2f}B"
               if not full else ""))
    return {"name": name, "pred_hidden": pred_H, "pred_layers": 3,
            "ana_hidden": ana_H, "ana_layers": 3,
            "trader_hidden": (trd_h, trd_h, trd_h),
            "batch_days": int(bd), "minibatch": int(mb),
            "pred_steps": ps, "ana_steps": ps,
            "note": note,
            "params": {"predictor": predictor_params(pred_H, 3, n_in),
                       "analyzer": analyzer_params(ana_H, 3, n_in),
                       "trader": trader_params(trd_h, obs_dim, 3)}}


def probe_8bit():
    """True if bitsandbytes 8-bit AdamW actually works here (CUDA only)."""
    if not torch.cuda.is_available():
        return False
    try:
        import bitsandbytes as bnb  # noqa

        p = torch.nn.Parameter(torch.zeros(8, device="cuda:0"))
        opt = bnb.optim.AdamW8bit([p], lr=1e-3)
        (p.sum() * 2.0).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        del opt, p
        return True
    except Exception:
        return False


def make_opt(module, lr, opt8bit, adamw=False, weight_decay=0.0):
    """Optimizer factory: bitsandbytes 8-bit when available, else torch."""
    if opt8bit:
        import bitsandbytes as bnb

        cls = bnb.optim.AdamW8bit if adamw else bnb.optim.Adam8bit
        return cls(module.parameters(), lr=lr, weight_decay=weight_decay)
    cls = torch.optim.AdamW if adamw else torch.optim.Adam
    return cls(module.parameters(), lr=lr, weight_decay=weight_decay)


def choose_plan(args, res, opt8bit=False, n_in=23, obs_dim=1504) -> dict:
    """Devices for the four models + the solved size tier.

    4 usable GPUs: predictor=0, analyzer=1, trader A=2, trader B=3
    3 GPUs: traders A+B share GPU 2.  2 GPUs: predictor+analyzer share 0,
    traders share 1.  1 GPU: everything.  0 GPUs (or GPUs too busy, e.g.
    shared with a miner): CPU with the XS tier.

    A GPU is "usable" when it has at least --min-free-gb free VRAM, so we
    never step on other workloads (e.g. a running miner).
    """
    gpus = res["gpus"]
    if args.device == "cpu":
        usable = []
    else:
        usable = [g for g in gpus if g["free_gb"] >= args.min_free_gb]
        if args.device == "cuda" and not usable and gpus:
            usable = gpus  # explicitly forced: ignore the free-memory guard

    def dev(g):
        return torch.device(f"cuda:{g['index']}")

    if len(usable) >= 4:
        plan = {"pred": dev(usable[0]), "ana": dev(usable[1]),
                "trade_a": dev(usable[2]), "trade_b": dev(usable[3]),
                "parallel": True, "n_usable": len(usable),
                "layout": (f"4+ GPUs: predictor={dev(usable[0])}, analyzer={dev(usable[1])}, "
                           f"traderA={dev(usable[2])}, traderB={dev(usable[3])}")}
    elif len(usable) == 3:
        plan = {"pred": dev(usable[0]), "ana": dev(usable[1]),
                "trade_a": dev(usable[2]), "trade_b": dev(usable[2]),
                "parallel": True, "n_usable": len(usable),
                "layout": (f"3 GPUs: predictor={dev(usable[0])}, analyzer={dev(usable[1])}, "
                           f"traders A+B={dev(usable[2])}")}
    elif len(usable) == 2:
        plan = {"pred": dev(usable[0]), "ana": dev(usable[0]),
                "trade_a": dev(usable[1]), "trade_b": dev(usable[1]),
                "parallel": True, "n_usable": len(usable),
                "layout": (f"2 GPUs: predictor+analyzer={dev(usable[0])}, "
                           f"traders A+B={dev(usable[1])}")}
    elif len(usable) == 1:
        plan = {"pred": dev(usable[0]), "ana": dev(usable[0]),
                "trade_a": dev(usable[0]), "trade_b": dev(usable[0]),
                "parallel": False, "n_usable": len(usable),
                "layout": f"1 GPU: all models on {dev(usable[0])}"}
    else:
        plan = {"pred": torch.device("cpu"), "ana": torch.device("cpu"),
                "trade_a": torch.device("cpu"), "trade_b": torch.device("cpu"),
                "parallel": False, "n_usable": 0,
                "layout": "CPU: all models on cpu"}

    min_free = min((g["free_gb"] for g in usable), default=0.0)
    budget = args.vram_budget_gb if args.vram_budget_gb > 0 else 1e9
    plan["tier"] = solve_tier(min_free, budget, res.get("disk_free_gb") or 8.0,
                              opt8bit, n_in, obs_dim)
    plan["opt8bit"] = opt8bit
    return plan
