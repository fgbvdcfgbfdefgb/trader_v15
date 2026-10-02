"""PNG plotting (v10): per-epoch day snapshots with BOTH traders' response to
that day's market, and cumulative training curves. Agg backend, no display."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from .data import PAIRS  # noqa: E402
from .features import W  # noqa: E402

COLORS = {"BTCUSDT": "#f7931a", "ETHUSDT": "#627eea", "LTCUSDT": "#41a5ee"}


def plot_day_snapshot(
    path,
    epoch,
    date,
    closes,
    weights,
    equity_a,
    equity_b,
    capital,
    target,
    forecast15,
    realized15,
    corr15,
    vol_classes,
    hit_a=False,
    hit_b=False,
    margin_call=False,
    extra="",
):
    """One PNG per epoch: prices + trader-A exposure per coin, BOTH equity
    curves, the price-predictor's forecast (in %) vs reality, and the
    market-analyzer's regime read of the day."""
    T = closes.shape[0]
    n = len(equity_a)
    fig = plt.figure(figsize=(16, 13), dpi=100)
    gs = fig.add_gridspec(6, 1, height_ratios=[1, 1, 1, 1.3, 1.1, 1.0], hspace=0.6)
    status = f"A {'HIT' if hit_a else 'miss'} / B {'HIT' if hit_b else 'miss'}"
    if margin_call:
        status += " | margin call"
    fig.suptitle(
        f"epoch {epoch} | {date} UTC | A ${equity_a[-1]:.2f} / B "
        + (f"${equity_b[-1]:.2f}" if equity_b is not None and len(equity_b) else "-")
        + f" / target ${target:.2f} | {status}"
        + (f" | {extra}" if extra else ""),
        fontsize=13,
    )

    for i, pair in enumerate(PAIRS):
        ax = fig.add_subplot(gs[i])
        y = closes[:, i] / max(closes[0, i], 1e-9)
        ax.plot(np.arange(T), y, color=COLORS[pair], lw=0.9)
        lo, hi = float(np.min(y)), float(np.max(y))
        if hi - lo < 1e-12:
            hi = lo + 1e-3
        pad = (hi - lo) * 0.08
        xw = np.arange(W - 1, W - 1 + n)
        ax.fill_between(xw, lo - pad, hi + pad, where=weights[:, i] > 0.05, color="#2ecc71", alpha=0.25, linewidth=0)
        ax.fill_between(xw, lo - pad, hi + pad, where=weights[:, i] < -0.05, color="#e74c3c", alpha=0.25, linewidth=0)
        ax.set_xlim(0, T)
        ax.set_title(f"{pair} close (normalised) - green=trader-A long, red=trader-A short", fontsize=9)
        ax.set_ylabel("norm price")
        ax.grid(alpha=0.15)

    ax = fig.add_subplot(gs[3])
    xw = np.arange(W - 1, W - 1 + n)
    ax.plot(xw, equity_a, color="#8e44ad", lw=1.2, label="trader A")
    if equity_b is not None and len(equity_b):
        ax.plot(xw[: len(equity_b)], equity_b, color="#e67e22", lw=1.0, ls="--", label="trader B")
    ax.axhline(capital, color="gray", ls="--", lw=0.8, label=f"start ${capital:.0f}")
    ax.axhline(target, color="#2ecc71", ls="--", lw=0.9, label=f"target ${target:.0f}")
    ax.set_title("trade-maker equity through the day (both traders)", fontsize=10)
    ax.set_ylabel("$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)

    ax = fig.add_subplot(gs[4])
    ax.plot(np.arange(T), realized15, color="#555555", lw=0.8, label="realized next-15m return (BTC, %)")
    ax.plot(np.arange(T), forecast15, color="#e67e22", lw=0.8, alpha=0.9, label="predictor forecast (%)")
    ax.set_title(f"price-predictor agent - 15-minute horizon (BTC, vol-normalized model) - corr {corr15:+.3f}", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)

    ax = fig.add_subplot(gs[5])
    cmap = ListedColormap(["#2ecc71", "#f1c40f", "#e74c3c"])
    ax.imshow(
        np.asarray(vol_classes).T,
        aspect="auto",
        cmap=cmap,
        vmin=-0.5,
        vmax=2.5,
        extent=[0, T, 0, len(PAIRS)],
        interpolation="nearest",
    )
    ax.set_yticks([i + 0.5 for i in range(len(PAIRS))], labels=list(PAIRS))
    ax.set_title("market-analyzer agent - predicted volatility regime (green=calm, amber=normal, red=stormy)", fontsize=9)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _ema(x, a=0.08):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    e = 0.0
    started = False
    for i, v in enumerate(x):
        if not np.isfinite(v):
            out[i] = np.nan
            continue
        e = v if not started else a * v + (1 - a) * e
        started = True
        out[i] = e
    return out


def _panel(ax, hist, keys, title, labels=None, hlines=()):
    plotted = False
    for j, k in enumerate(keys):
        x = np.asarray(hist.get(k, []), dtype=float)
        if x.size == 0:
            continue
        ax.plot(x, lw=0.5, alpha=0.30, color=f"C{j}")
        ax.plot(_ema(x), lw=1.5, color=f"C{j}", label=(labels or keys)[j])
        plotted = True
    for y, c in hlines:
        ax.axhline(y, color=c, ls="--", lw=0.7)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.2)
    if plotted:
        ax.legend(fontsize=7)


def plot_curves(path, hist, capital, target):
    """Cumulative training curves PNG (rewritten every epoch)."""
    fig, axes = plt.subplots(4, 3, figsize=(16, 13), dpi=100)
    fig.suptitle("trader_v10 training curves", fontsize=13)
    _panel(axes[0, 0], hist, ["pred_loss"], "price-predictor loss (MSE + corr, norm-space)")
    _panel(axes[0, 1], hist, ["pred_mae15", "pred_mae60"], "predictor MAE (% , vs raw returns)", ["15m", "60m"])
    _panel(axes[0, 2], hist, ["pred_corr15", "pred_corr60"], "predictor correlation with realized (flat-line detector)", ["15m", "60m"], hlines=[(0.0, "#555")])
    _panel(axes[1, 0], hist, ["ana_vol_acc", "ana_trend_acc"], "market-analyzer accuracy", ["vol regime", "trend"], hlines=[(1 / 3, "#555")])
    _panel(axes[1, 1], hist, ["pi_loss_a", "v_loss_a"], "trader A PPO losses", ["policy", "value"])
    _panel(axes[1, 2], hist, ["pi_loss_b", "v_loss_b"], "trader B PPO losses", ["policy", "value"])
    _panel(axes[2, 0], hist, ["entropy_a", "entropy_b"], "policy entropy", ["A", "B"])
    _panel(axes[2, 1], hist, ["ep_ret_a", "ep_ret_b"], "epoch return", ["A", "B"])
    _panel(axes[2, 2], hist, ["final_eq_a", "final_eq_b"], f"final equity per epoch (target ${target:.0f})", ["A", "B"], hlines=[(capital, "gray"), (target, "#2ecc71")])
    _panel(axes[3, 0], hist, ["lr_trader_a", "lr_trader_b"], "trader learning rates (KL-adaptive)", ["A", "B"])
    _panel(axes[3, 1], hist, ["eval_mean_eq_a", "eval_mean_eq_b"], "held-out eval mean equity", ["A", "B"], hlines=[(capital, "gray")])
    _panel(axes[3, 2], hist, ["eval_hit_a", "eval_hit_b"], "held-out eval target hit-rate", ["A", "B"], hlines=[(0.0, "#555")])
    axes[3, 2].set_ylim(-0.05, 1.05)
    for ax in axes[3]:
        ax.set_xlabel("epoch")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path)
    plt.close(fig)


def plot_trader_day(
    path,
    tag,
    epoch,
    date,
    closes,
    weights,
    equity,
    capital,
    target,
    hit=False,
    margin_call=False,
    extra="",
):
    """v12: one dedicated PNG per trade-making agent per epoch - that agent's
    own response to the day's market: prices with ITS exposure shading, ITS
    equity curve, and ITS target-portfolio-weight decisions through the day."""
    from .features import W as _W

    T = closes.shape[0]
    n = len(equity)
    fig = plt.figure(figsize=(16, 12), dpi=100)
    gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 1.3, 1.2], hspace=0.6)
    status = "TARGET HIT" if hit else "target missed"
    if margin_call:
        status += " | MARGIN CALL"
    fig.suptitle(
        f"trader {tag} | epoch {epoch} | {date} UTC | final equity ${equity[-1]:.2f} "
        f"/ target ${target:.2f} | {status}" + (f" | {extra}" if extra else ""),
        fontsize=13,
    )

    for i, pair in enumerate(PAIRS):
        ax = fig.add_subplot(gs[i])
        y = closes[:, i] / max(closes[0, i], 1e-9)
        ax.plot(np.arange(T), y, color=COLORS[pair], lw=0.9)
        lo, hi = float(np.min(y)), float(np.max(y))
        if hi - lo < 1e-12:
            hi = lo + 1e-3
        pad = (hi - lo) * 0.08
        xw = np.arange(_W - 1, _W - 1 + n)
        ax.fill_between(xw, lo - pad, hi + pad, where=weights[:, i] > 0.05, color="#2ecc71", alpha=0.25, linewidth=0)
        ax.fill_between(xw, lo - pad, hi + pad, where=weights[:, i] < -0.05, color="#e74c3c", alpha=0.25, linewidth=0)
        ax.set_xlim(0, T)
        ax.set_title(f"{pair} close (normalised) - green=long, red=short", fontsize=9)
        ax.set_ylabel("norm price")
        ax.grid(alpha=0.15)

    ax = fig.add_subplot(gs[3])
    ax.plot(np.arange(_W - 1, _W - 1 + n), equity, color="#8e44ad", lw=1.2)
    ax.fill_between(np.arange(_W - 1, _W - 1 + n), capital, equity,
                    where=equity >= capital, color="#2ecc71", alpha=0.15, linewidth=0)
    ax.fill_between(np.arange(_W - 1, _W - 1 + n), capital, equity,
                    where=equity < capital, color="#e74c3c", alpha=0.15, linewidth=0)
    ax.axhline(capital, color="gray", ls="--", lw=0.8, label=f"start ${capital:.0f}")
    ax.axhline(target, color="#2ecc71", ls="--", lw=0.9, label=f"target ${target:.0f}")
    stats = (f"final ${equity[-1]:.2f}   max ${equity.max():.2f}   min ${equity.min():.2f}   "
             f"return {100 * (equity[-1] / capital - 1):+.1f}%")
    ax.set_title(f"trader {tag} equity through the day | {stats}", fontsize=10)
    ax.set_ylabel("$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)

    ax = fig.add_subplot(gs[4])
    xw = np.arange(_W - 1, _W - 1 + n)
    for i, pair in enumerate(PAIRS):
        ax.plot(xw, weights[:, i], lw=0.8, color=COLORS[pair], label=pair)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylim(-1.15, 1.15)
    n_switch = int((np.abs(np.diff(weights, axis=0)) > 0.15).sum())
    ax.set_title(f"trader {tag} target portfolio weights per coin "
                 f"(position sizing; ~{n_switch} coin position changes)", fontsize=10)
    ax.set_ylabel("weight")
    ax.set_xlabel("minute of day (UTC)")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.15)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)



def plot_traders_stacked(path, epoch, date, closes, spec_a, spec_b, capital,
                         target, extra="", tag_a="A", tag_b="B"):
    """v12.2: ONE tall PNG per epoch for BOTH trade makers. The market data
    (3 normalised price panels) is drawn ONCE; each price panel carries a thin
    exposure ribbon per trader (top ribbon = first trader, bottom = second).
    Below the market: each trader's equity curve with stats, then per-coin
    portfolio weights. No market data is duplicated."""
    from .features import W as _W

    T = closes.shape[0]
    specs = [(tag_a, spec_a)]
    if spec_b is not None:
        specs.append((tag_b, spec_b))
    n_sections = len(specs)
    rows = 3 + 2 * n_sections
    fig = plt.figure(figsize=(16, 2.3 * rows + 1.6), dpi=100)
    gs = fig.add_gridspec(rows, 1, hspace=0.8)

    eq_a = spec_a["equity"]
    head = f"trader {tag_a} ${eq_a[-1]:.2f}"
    if spec_b is not None:
        head += f"  vs  trader {tag_b} ${spec_b['equity'][-1]:.2f}"
    fig.suptitle(
        f"{head} | epoch {epoch} | {date} UTC | start ${capital:.0f} "
        f"target ${target:.2f}" + (f" | {extra}" if extra else ""),
        fontsize=14,
    )

    # ---- market section: 3 price panels, drawn once, with exposure ribbons ----
    n = len(eq_a)
    xw = np.arange(_W - 1, _W - 1 + n)
    for i, pair in enumerate(PAIRS):
        ax = fig.add_subplot(gs[i])
        y = closes[:, i] / max(closes[0, i], 1e-9)
        ax.plot(np.arange(T), y, color=COLORS[pair], lw=0.9)
        lo, hi = float(np.min(y)), float(np.max(y))
        if hi - lo < 1e-12:
            hi = lo + 1e-3
        pad = (hi - lo) * 0.08
        strip = (hi - lo) * 0.05
        gap = (hi - lo) * 0.015
        y_bot = lo - pad - (2 * strip + gap)
        ax.set_xlim(0, T)
        ax.set_ylim(y_bot - strip * 0.4, hi + pad)
        for r, (tag, spec) in enumerate(specs):
            w = spec["weights"][:, i]
            yb = y_bot + strip + gap if r == 0 else y_bot
            ax.fill_between(xw, yb, yb + strip, where=w > 0.05, color="#2ecc71", alpha=0.6, linewidth=0)
            ax.fill_between(xw, yb, yb + strip, where=w < -0.05, color="#e74c3c", alpha=0.6, linewidth=0)
        badge = "  [TARGET HIT]" if (i == 0 and (spec_a.get("hit") or (spec_b or {}).get("hit"))) else ""
        rib = f"ribbons = exposure (green long / red short): upper {tag_a}"
        if spec_b is not None:
            rib += f", lower {tag_b}"
        ax.set_title(f"{pair} close (normalised) | {rib}{badge}", fontsize=9)
        ax.set_ylabel("norm price")
        ax.grid(alpha=0.15)

    # ---- per-trader section: equity + weights (no market duplication) ----
    for s, (tag, spec) in enumerate(specs):
        weights, equity = spec["weights"], spec["equity"]
        hit, mc = spec.get("hit", False), spec.get("margin_call", False)
        color = "#8e44ad" if s == 0 else "#e67e22"
        nx = np.arange(_W - 1, _W - 1 + len(equity))

        ax = fig.add_subplot(gs[3 + 2 * s])
        ax.plot(nx, equity, color=color, lw=1.2)
        ax.fill_between(nx, capital, equity, where=equity >= capital, color="#2ecc71", alpha=0.15, linewidth=0)
        ax.fill_between(nx, capital, equity, where=equity < capital, color="#e74c3c", alpha=0.15, linewidth=0)
        ax.axhline(capital, color="gray", ls="--", lw=0.8, label=f"start ${capital:.0f}")
        ax.axhline(target, color="#2ecc71", ls="--", lw=0.9, label=f"target ${target:.0f}")
        stats = (f"final ${equity[-1]:.2f}   max ${equity.max():.2f}   min ${equity.min():.2f}   "
                 f"return {100 * (equity[-1] / capital - 1):+.1f}%")
        ax.set_title(f"TRADER {tag} equity | {stats}" + ("  | MARGIN CALL" if mc else ""), fontsize=10)
        ax.set_ylabel("$")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.15)

        ax = fig.add_subplot(gs[4 + 2 * s])
        for i, pair in enumerate(PAIRS):
            ax.plot(nx, weights[:, i], lw=0.8, color=COLORS[pair], label=pair)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_ylim(-1.15, 1.15)
        n_switch = int((np.abs(np.diff(weights, axis=0)) > 0.15).sum())
        ax.set_title(f"TRADER {tag} target portfolio weights per coin (~{n_switch} position changes)", fontsize=10)
        ax.set_ylabel("weight")
        ax.set_xlabel("minute of day (UTC)")
        ax.legend(fontsize=8, ncol=3)
        ax.grid(alpha=0.15)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
