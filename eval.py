#!/usr/bin/env python3
"""Evaluate saved checkpoints on the most recent (held-out) days.

Usage:
  python eval.py --ckpt runs/ckpt --days 40              # trader A
  python eval.py --ckpt runs/ckpt --which both --days 40 # A-vs-B competition
Evaluation uses the EMA ('precision') advisor weights when present, and runs
the held-out days in batched vectorised rollouts for speed.
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from trader import data as D  # noqa: E402
from trader import plots  # noqa: E402
from trader.agents import Analyzer, Predictor, Trader  # noqa: E402
from trader.data import PAIRS  # noqa: E402
from trader.env import TradingEnv  # noqa: E402
from trader.features import HORIZONS, IDX15, N_FEATURES  # noqa: E402
from train import EMB_DIM, DayProcessed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/ckpt")
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="runs/eval")
    ap.add_argument("--which", choices=["a", "b", "both"], default="a")
    ap.add_argument("--days", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8, help="days per vectorised rollout batch")
    ap.add_argument("--capital", type=float, default=20.0)
    ap.add_argument("--target", type=float, default=30.0)
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--slip", type=float, default=0.0002)
    ap.add_argument("--max-lev", type=float, default=5.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    days = D.load_days(args.data)
    dates = sorted(days)[-args.days:]
    print(f"[eval] {len(dates)} days on {device}")

    n_out = len(PAIRS) * len(HORIZONS)
    ck_p = torch.load(os.path.join(args.ckpt, "predictor.pt"),
                      map_location=device, weights_only=False)
    pred_hid = ck_p["model"]["gru.weight_hh_l0"].shape[0] // 3
    pred_layers = sum(1 for k in ck_p["model"] if k.startswith("gru.weight_ih_l"))
    predictor = Predictor(N_FEATURES, n_out, hidden=pred_hid, layers=pred_layers).to(device)
    # prefer the EMA ("precision") weights for evaluation
    predictor.load_state_dict(ck_p.get("ema") or ck_p["model"])
    predictor.eval()

    ck_a = torch.load(os.path.join(args.ckpt, "analyzer.pt"),
                      map_location=device, weights_only=False)
    ana_hid = ck_a["model"]["gru.weight_hh_l0"].shape[0] // 3
    ana_layers = sum(1 for k in ck_a["model"] if k.startswith("gru.weight_ih_l"))
    analyzer = Analyzer(N_FEATURES, len(PAIRS), hidden=ana_hid, layers=ana_layers,
                        emb_dim=EMB_DIM).to(device)
    analyzer.load_state_dict(ck_a.get("ema") or ck_a["model"])
    analyzer.eval()

    traders = {}
    for tag in (["a"] if args.which == "a" else ["b"] if args.which == "b" else ["a", "b"]):
        ck_t = torch.load(os.path.join(args.ckpt, f"trader_{tag}.pt"),
                          map_location=device, weights_only=False)
        obs_dim = ck_t["model"]["trunk.0.weight"].shape[1]
        widths = [v.shape[0] for k, v in ck_t["model"].items()
                  if k.startswith("trunk.") and k.endswith(".weight")]
        act_dim = ck_t["model"]["mu.weight"].shape[0]   # v15: 3 weights + gate
        traders[tag] = Trader(obs_dim, act_dim=act_dim, hidden=widths).to(device)
        traders[tag].load_state_dict(ck_t["model"])
        traders[tag].eval()

    env_kwargs = dict(capital=args.capital, target=args.target, fee=args.fee,
                      slip=args.slip, max_lev=args.max_lev)

    def vec_eval(trader, procs, advisors):
        envs = [TradingEnv(p.day, p.feats, fn, emb, **env_kwargs)
                for p, (fn, _, emb, _) in zip(procs, advisors)]
        obs_list = [e.reset() for e in envs]
        idx_active = list(range(len(envs)))
        while idx_active:
            obs = np.stack([obs_list[i] for i in idx_active]).astype(np.float32)
            with torch.no_grad():
                mu, _, _ = trader(torch.as_tensor(obs, device=device))
                a = mu.clamp(-1, 1).cpu().numpy()
            still = []
            for j, i in enumerate(idx_active):
                obs_list[i], r, done, info = envs[i].step(a[j])
                if not done:
                    still.append(i)
            idx_active = still
        return envs, [float(e.equity) for e in envs]

    all_rows = {tag: [] for tag in traders}
    eb = max(1, args.batch)
    for s in range(0, len(dates), eb):
        batch_dates = dates[s: s + eb]
        procs = [DayProcessed(d, days[d]) for d in batch_dates]
        with torch.no_grad():
            x = torch.as_tensor(np.stack([p.feats for p in procs]), device=device)
            fn_all = predictor(x).cpu().numpy().astype(np.float32)
            fp_all = fn_all * np.stack([p.vol_scale for p in procs])
            emb_all, vl_all, _ = analyzer(x)
            emb_all = emb_all.cpu().numpy().astype(np.float32)
            vol_all = vl_all.argmax(-1).cpu().numpy()
        advisors = [(fn_all[b], fp_all[b], emb_all[b], vol_all[b]) for b in range(len(procs))]
        env_by_tag = {}
        for tag, trader in traders.items():
            envs, finals = vec_eval(trader, procs, advisors)
            env_by_tag[tag] = envs
            for d, e, f in zip(batch_dates, envs, finals):
                wr = e.win_rate if e.n_trades > 0 else float("nan")
                all_rows[tag].append({"date": d, "final_eq": f,
                                      "hit": float(f >= args.target),
                                      "margin_call": float(e.margin_call),
                                      "win_rate": wr, "n_trades": e.n_trades})
                print(f"  [{d}] trader {tag.upper()}: ${f:7.2f} "
                      f"{'HIT' if f >= args.target else '   '} "
                      f"wr={wr:.2f}({e.n_trades}t)")
        # per-day PNG for the first trader in the batch (as before, trader A)
        snap_tag = "a" if "a" in traders else list(traders)[0]
        for b, d in enumerate(batch_dates):
            closes = np.stack([procs[b].day[p]["close"] for p in PAIRS], axis=1)
            plots.plot_day_snapshot(
                os.path.join(args.out, f"eval_{d}.png"), 0, d, closes,
                env_by_tag[snap_tag][b].weights_hist, env_by_tag[snap_tag][b].equity_hist,
                None, args.capital, args.target, fp_all[b][:, IDX15],
                procs[b].Y_pct[:, IDX15], 0.0, vol_all[b],
                hit_a=bool(all_rows[snap_tag][-len(batch_dates) + b]["hit"]),
                hit_b=False,
                margin_call=bool(all_rows[snap_tag][-len(batch_dates) + b]["margin_call"]),
                extra=f"held-out evaluation (trader {snap_tag.upper()})")
            specs = {}
            for tag in traders:
                e = env_by_tag[tag][b]
                specs[tag] = {"weights": e.weights_hist, "equity": e.equity_hist,
                              "hit": bool(all_rows[tag][-len(batch_dates) + b]["hit"]),
                              "margin_call": bool(e.margin_call)}
            first_tag = "a" if "a" in specs else list(specs)[0]
            second = specs.get("b")
            plots.plot_traders_stacked(
                os.path.join(args.out, f"eval_{d}_traders.png"), 0, d, closes,
                specs[first_tag], second, args.capital, args.target,
                extra="held-out evaluation",
                tag_a=first_tag.upper(),
                tag_b=("b" if second is not None else first_tag).upper())

    for tag, rows in all_rows.items():
        with open(os.path.join(args.out, f"eval_results_{tag}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["date", "final_eq", "hit", "margin_call", "win_rate", "n_trades"])
            w.writeheader()
            w.writerows(rows)

    fig, ax = plt.subplots(figsize=(15, 6), dpi=110)
    width = 0.8 / len(all_rows)
    for j, (tag, rows) in enumerate(sorted(all_rows.items())):
        finals = np.array([r["final_eq"] for r in rows])
        hits = np.array([r["hit"] for r in rows])
        x = np.arange(len(rows)) + (j - len(all_rows) / 2 + 0.5) * width
        ax.bar(x, finals, width=width,
               color=("#8e44ad" if tag == "a" else "#e67e22"),
               label=f"trader {tag.upper()}: mean ${finals.mean():.2f}, "
                     f"hit {hits.mean()*100:.1f}%")
    ax.axhline(args.capital, color="gray", ls="--", lw=1, label=f"start ${args.capital:.0f}")
    ax.axhline(args.target, color="#2ecc71", ls=":", lw=1.4, label=f"target ${args.target:.0f}")
    ax.set_xticks(np.arange(len(dates)), [d[5:] for d in dates], rotation=90, fontsize=7)
    ax.set_ylabel("final equity ($)")
    ax.set_title(f"Held-out evaluation: {len(dates)} days | "
                 + " vs ".join(f"{t.upper()} mean ${np.mean([r['final_eq'] for r in rows]):.2f}"
                               for t, rows in sorted(all_rows.items())))
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "eval_summary.png"))
    plt.close(fig)
    for tag, rows in sorted(all_rows.items()):
        finals = np.array([r["final_eq"] for r in rows])
        hits = np.array([r["hit"] for r in rows])
        print(f"[eval] trader {tag.upper()}: mean ${finals.mean():.2f} "
              f"median ${np.median(finals):.2f} hit {hits.mean()*100:.1f}% "
              f"max ${finals.max():.2f}")
    print(f"[eval] artifacts -> {args.out}")


if __name__ == "__main__":
    main()
