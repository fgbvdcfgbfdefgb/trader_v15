"""Compact PPO (clipped surrogate + GAE) for the trade-maker agent."""

import numpy as np
import torch
from torch.distributions import Normal


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.obs, self.act, self.logp, self.val, self.rew, self.done = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

    def add(self, obs, act, logp, val, rew, done):
        self.obs.append(obs)
        self.act.append(act)
        self.logp.append(logp)
        self.val.append(val)
        self.rew.append(rew)
        self.done.append(done)

    def __len__(self):
        return len(self.rew)

    def compute_gae(self, gamma, lam):
        T = len(self.rew)
        adv = np.zeros(T, dtype=np.float32)
        last = 0.0
        for t in range(T - 1, -1, -1):
            next_v = self.val[t + 1] if t + 1 < T else 0.0
            nonterm = 0.0 if self.done[t] else 1.0
            delta = self.rew[t] + gamma * next_v * nonterm - self.val[t]
            last = delta + gamma * lam * nonterm * last
            adv[t] = last
        ret = adv + np.asarray(self.val, dtype=np.float32)
        return adv, ret


def ppo_update(
    trader,
    optimizer,
    buf,
    device,
    gamma=0.999,
    lam=0.95,
    clip=0.2,
    ppo_epochs=4,
    minibatch=256,
    ent_coef=3e-3,
    vf_coef=0.5,
    max_grad=0.5,
    target_kl=0.05,
    amp_dtype=None,
    scaler=None,
):
    adv, ret = buf.compute_gae(gamma, lam)
    obs = torch.as_tensor(np.asarray(buf.obs, dtype=np.float32), device=device)
    act = torch.as_tensor(np.asarray(buf.act, dtype=np.float32), device=device)
    old_logp = torch.as_tensor(np.asarray(buf.logp, dtype=np.float32), device=device)
    adv_t = torch.as_tensor(adv, device=device)
    ret_t = torch.as_tensor(ret, device=device)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    T = obs.shape[0]
    mb = int(min(minibatch, T))
    stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "kl": 0.0, "clip_frac": 0.0}
    n_updates = 0
    stop = False
    for _ in range(ppo_epochs):
        if stop:
            break
        idx = torch.randperm(T, device=device)
        for s in range(0, T, mb):
            j = idx[s : s + mb]
            with torch.autocast(device.type, dtype=amp_dtype or torch.float32,
                                enabled=amp_dtype is not None
                                and amp_dtype != torch.float32):
                mu, std, value = trader(obs[j])
                dist = Normal(mu, std)
                logp = dist.log_prob(act[j]).sum(-1)
                ratio = (logp - old_logp[j]).exp()
                s_adv = adv_t[j]
                l1 = -s_adv * ratio
                l2 = s_adv * ratio.clamp(1.0 - clip, 1.0 + clip)
                pi_loss = -torch.min(l1, l2).mean()
                v_loss = 0.5 * (value - ret_t[j]).pow(2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pi_loss + vf_coef * v_loss - ent_coef * ent
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trader.parameters(), max_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trader.parameters(), max_grad)
                optimizer.step()
            with torch.no_grad():
                stats["pi_loss"] += pi_loss.item()
                stats["v_loss"] += v_loss.item()
                stats["entropy"] += ent.item()
                stats["kl"] += (old_logp[j] - logp).mean().abs().item()
                stats["clip_frac"] += ((ratio - 1.0).abs() > clip).float().mean().item()
            n_updates += 1
            if stats["kl"] / n_updates > 50 * target_kl:  # policy diverging: bail out
                stop = True
                break
    for k in stats:
        stats[k] /= max(n_updates, 1)
    return stats
