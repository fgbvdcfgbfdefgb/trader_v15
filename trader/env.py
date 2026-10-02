"""Trading environment: one episode = one full UTC day of 1-minute bars.

The agent starts every episode with `capital` ($20) and must reach `target`
($30+). Each minute it outputs target portfolio weights in [-1, 1] per coin
(negative = short), plus a confidence gate. Exposure per coin is
weight * gate * max_lev * equity. Trades execute at the current minute's
close with a taker fee + slippage. A margin call liquidates the book if
equity falls below the maintenance margin.

v15 additions
-------------
* 4-dim action [w_eth, w_btc, w_sol, gate]: effective weight is
  w * clamp(gate, 0, 1). The gate lets the agent veto or size down any
  position - the mechanism behind "only take trades you expect to win".
* Per-trade accounting + feedback: every position round-trip
  (open -> flat/flip, threshold 0.05) is measured. The observation carries
  the open trades' unrealised PnL per coin, the last closed trade's PnL,
  the win rate so far and the trade count - the agent sees its own trade
  results while it trades.
* Asymmetric per-trade reward: a closed losing round-trip costs
  `trade_loss_penalty` x |pnl| , a winning one earns `trade_win_bonus` x pnl
  (losses are punished ~3x harder than wins are rewarded) - the training
  pressure behind "always profit on every trade".
* Team channel: the rollout can inject the teammate trader's lagged
  effective weights + session return (4 dims) into the observation, so the
  two trade makers communicate while they trade.
"""

import numpy as np

from .data import PAIRS
from .features import N_FEATURES, W

FLAT_TAU = 0.05          # |effective weight| below this = flat
FB_DIM = 10              # 6 self-feedback + 4 teammate-channel obs dims


class TradingEnv:
    def __init__(
        self,
        day,
        feats,
        forecast,
        emb,
        capital=20.0,
        target=30.0,
        fee=5e-4,
        slip=2e-4,
        max_lev=5.0,
        maint_margin=0.01,
        terminal_bonus=0.5,
        reward_scale=100.0,
        trade_win_bonus=30.0,
        trade_loss_penalty=80.0,
    ):
        self.close = np.stack([day[p]["close"] for p in PAIRS], axis=1).astype(np.float64)
        self.feats = np.asarray(feats, dtype=np.float32)
        self.forecast = np.asarray(forecast, dtype=np.float32)
        self.emb = np.asarray(emb, dtype=np.float32)
        self.capital = float(capital)
        self.target = float(target)
        self.fee = float(fee)
        self.slip = float(slip)
        self.max_lev = float(max_lev)
        self.maint_margin = float(maint_margin)
        self.terminal_bonus = float(terminal_bonus)
        self.reward_scale = float(reward_scale)
        self.trade_win_bonus = float(trade_win_bonus)
        self.trade_loss_penalty = float(trade_loss_penalty)
        self.T = len(self.close)
        self.n_steps = self.T - 1 - (W - 1)
        if forecast.shape[0] != self.T or emb.shape[0] != self.T:
            raise ValueError("forecast/emb must cover every minute of the day")
        # obs: window feats + forecasts + embedding + weights + log-eq
        #      + 6 self trade-feedback + 4 teammate channel
        self.obs_dim = W * N_FEATURES + forecast.shape[1] + emb.shape[1] + 3 + 1 + FB_DIM

    # ------------------------------------------------------------------ #
    def reset(self):
        self.t = W - 1
        self.step_i = 0
        self.cash = self.capital
        self.units = np.zeros(3, dtype=np.float64)
        self.equity = self.capital
        self.margin_call = False
        self.weights_hist = np.zeros((self.n_steps, 3), dtype=np.float32)
        self.equity_hist = np.zeros(self.n_steps, dtype=np.float32)
        # ---- v15 trade tracking ----
        self.pos_dir = np.zeros(3, dtype=np.float64)   # 0 flat, +/-1 direction
        self.open_gross = np.zeros(3)                  # price PnL accrued per open trade
        self.trade_cost = np.zeros(3)                  # fees+slip paid per open trade
        self.open_eq = np.full(3, self.capital)        # equity when trade opened
        self.last_pnl = 0.0                            # last closed trade (rel. to equity)
        self.n_trades = 0
        self.n_wins = 0
        self.last_eff = np.zeros(3)                    # last effective weights
        self.other_w = np.zeros(3)                     # teammate lagged weights
        self.other_ret = 0.0                           # teammate lagged log session return
        return self._obs()

    def set_other(self, w, ret):
        """Inject the teammate's lagged weights + session return (team channel)."""
        self.other_w = np.asarray(w, dtype=np.float32).reshape(3)
        self.other_ret = float(np.clip(ret, -4.0, 4.0))

    @property
    def win_rate(self):
        return self.n_wins / self.n_trades if self.n_trades > 0 else float("nan")

    # ------------------------------------------------------------------ #
    def _obs(self):
        t = self.t
        win = self.feats[t - W + 1 : t + 1].reshape(-1)
        w_now = (self.units * self.close[t]) / max(self.equity, 1e-6) / self.max_lev
        w_now = np.clip(w_now, -1.5, 1.5)
        eq = np.clip(np.log(max(self.equity, 1e-9) / self.capital), -4.0, 4.0)
        # self trade feedback (6)
        unreal = np.zeros(3, dtype=np.float32)
        for i in range(3):
            if self.pos_dir[i] != 0:
                rel = (self.open_gross[i] - self.trade_cost[i]) / max(self.open_eq[i], 1e-9)
                unreal[i] = np.tanh(10.0 * rel)
        last = np.tanh(10.0 * self.last_pnl)
        wr = self.win_rate if self.n_trades > 0 else 0.5
        ntr = min(self.n_trades, 20) / 20.0
        # teammate channel (4)
        obs = np.concatenate([
            win, self.forecast[t], self.emb[t], w_now, [eq],
            unreal, [last], [wr], [ntr],
            self.other_w, [self.other_ret],
        ])
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # ------------------------------------------------------------------ #
    def _close_trade(self, i, exit_fee):
        """Settle coin i's open round-trip; returns the shaped reward delta."""
        rel = (self.open_gross[i] - self.trade_cost[i] - exit_fee) / max(self.open_eq[i], 1e-9)
        self.last_pnl = rel
        self.n_trades += 1
        if rel > 0:
            self.n_wins += 1
        self.pos_dir[i] = 0.0
        self.open_gross[i] = self.trade_cost[i] = 0.0
        self.open_eq[i] = self.equity
        if rel > 0:
            return self.trade_win_bonus * rel
        return -self.trade_loss_penalty * abs(rel)

    def step(self, action):
        a_raw = np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[:3], -1.0, 1.0)
        gate = float(np.asarray(action, dtype=np.float64).reshape(-1)[3]) \
            if np.asarray(action).size > 3 else 1.0
        gate = float(np.clip(gate, 0.0, 1.0))          # gate<0 = flat (no trade)
        a = a_raw * gate                               # effective weights
        t = self.t
        prices = self.close[t]
        eq_prev = self.cash + float(self.units @ prices)

        # rebalance to the requested weights at this minute's close.
        # Self-financing: cash pays for the net notional traded (cash may go
        # negative = margin borrowing), so a trade only costs fees+slippage.
        tgt_notional = a * self.max_lev * eq_prev
        tgt_units = tgt_notional / np.maximum(prices, 1e-9)
        delta = tgt_units - self.units
        trade_value = float(delta @ prices)  # + = net buying
        turnover = float(np.abs(delta) @ prices)
        fees = turnover * (self.fee + self.slip)
        self.cash -= trade_value + fees
        self.units = tgt_units
        fee_i = np.abs(delta) * prices * (self.fee + self.slip)

        # ---- v15: per-coin trade open/close detection ----
        shaped = 0.0
        for i in range(3):
            d_new = 0.0 if abs(a[i]) < FLAT_TAU else float(np.sign(a[i]))
            if self.pos_dir[i] == 0.0 and d_new != 0.0:
                # open a round-trip
                self.pos_dir[i] = d_new
                self.open_gross[i] = 0.0
                self.trade_cost[i] = fee_i[i]
                self.open_eq[i] = max(eq_prev, 1e-9)
            elif self.pos_dir[i] != 0.0 and (d_new == 0.0 or d_new != self.pos_dir[i]):
                # close (and maybe flip into the opposite direction)
                shaped += self._close_trade(i, fee_i[i])
                if d_new != 0.0:
                    self.pos_dir[i] = d_new
                    self.open_gross[i] = 0.0
                    self.trade_cost[i] = fee_i[i]
                    self.open_eq[i] = max(eq_prev, 1e-9)
            elif self.pos_dir[i] != 0.0:
                self.trade_cost[i] += fee_i[i]   # resizing cost accrues to the trade

        # one minute passes
        self.t += 1
        p_next = self.close[self.t]
        self.open_gross += self.units * (p_next - prices)   # accrue price PnL
        equity = self.cash + float(self.units @ p_next)

        gross = float(np.abs(self.units) @ p_next)
        done = False
        if gross > 0 and equity < self.maint_margin * gross:
            # margin call: liquidate everything (settle open trades too)
            liquid_fee = gross * (self.fee + self.slip)
            equity = max(equity - liquid_fee, 1e-9)
            for i in range(3):
                if self.pos_dir[i] != 0.0:
                    shaped += self._close_trade(i, 0.0)
            self.cash, self.units = equity, np.zeros(3)
            self.margin_call = True
            done = True
        if equity <= 1e-6:
            for i in range(3):
                if self.pos_dir[i] != 0.0:
                    shaped += self._close_trade(i, 0.0)
            equity = 1e-9
            self.cash, self.units = equity, np.zeros(3)
            done = True

        self.equity = equity
        r = float(
            np.clip(
                self.reward_scale * np.log(max(equity, 1e-9) / max(eq_prev, 1e-9)),
                -50.0,
                50.0,
            )
        )
        r += float(np.clip(shaped, -50.0, 50.0))       # asymmetric trade shaping
        if self.t >= self.T - 1:
            done = True
            for i in range(3):                          # end of day: settle
                if self.pos_dir[i] != 0.0:
                    r += float(np.clip(self._close_trade(i, 0.0), -50.0, 50.0))
        if done and equity >= self.target:
            r += self.terminal_bonus * self.reward_scale

        self.last_eff = a
        if self.step_i < self.n_steps:
            self.weights_hist[self.step_i] = a
            self.equity_hist[self.step_i] = equity
        self.step_i += 1
        return self._obs(), r, done, {
            "equity": equity,
            "margin_call": self.margin_call,
            "steps": self.step_i,
            "win_rate": self.win_rate,
            "n_trades": self.n_trades,
        }
