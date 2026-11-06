"""The four steps, each as a small pure function."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .network import Network


# ---------------------------------------------------------------- 1. generation
def trip_generation(pop: np.ndarray, jobs: np.ndarray, prod_rate: float = 0.12,
                    attr_rate: float = 0.20) -> tuple[np.ndarray, np.ndarray]:
    """AM-peak-hour person trips. Attractions are scaled to match productions."""
    P = prod_rate * pop
    A = attr_rate * jobs
    return P, A * P.sum() / A.sum()


# -------------------------------------------------------------- 2. distribution
def gravity_model(P, A, cost, beta, tol=1e-8, max_iter=500) -> np.ndarray:
    """Doubly constrained gravity model, balanced with Furness / IPF."""
    f = np.exp(-beta * cost)
    T = np.outer(P, A) * f
    for _ in range(max_iter):
        T *= (P / np.maximum(T.sum(1), 1e-12))[:, None]
        T *= (A / np.maximum(T.sum(0), 1e-12))[None, :]
        if np.abs(T.sum(1) - P).max() < tol * P.max():
            break
    return T


def calibrate_beta(P, A, cost, target, measure=None, lo=1e-4, hi=2.0) -> float:
    """Find the deterrence parameter that reproduces an observed mean of
    ``measure`` (default: the cost itself), e.g. the average trip length from
    a household travel survey."""
    measure = cost if measure is None else measure

    def err(beta):
        T = gravity_model(P, A, cost, beta)
        return (T * measure).sum() / T.sum() - target
    return brentq(err, lo, hi, xtol=1e-6)


# ---------------------------------------------------------------- 3. mode split
def mode_choice(utilities: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Multinomial logit. Returns per-mode probabilities and the logsum
    (expected maximum utility), used as the composite cost for distribution."""
    V = np.stack(list(utilities.values()))
    vmax = V.max(0)
    expV = np.exp(V - vmax)
    denom = expV.sum(0)
    probs = {m: expV[k] / denom for k, m in enumerate(utilities)}
    return probs, vmax + np.log(denom)


# ------------------------------------------------------------------ 4. assignment
def all_or_nothing(net: Network, demand: np.ndarray, link_cost: np.ndarray) -> np.ndarray:
    """Load every OD flow onto its current shortest path."""
    _, pred = net.shortest_paths(link_cost)
    idx = net.link_index()
    flow = np.zeros(net.n_links)
    n = net.n_nodes
    for o in range(n):
        for d in range(n):
            q = demand[o, d]
            if o == d or q <= 0:
                continue
            v = d
            while v != o:
                u = pred[o, v]
                if u < 0:
                    raise ValueError(f"no path {o}->{d}")
                flow[idx[(u, v)]] += q
                v = u
    return flow


def frank_wolfe(net: Network, demand: np.ndarray, max_iter: int = 500,
                gap_tol: float = 1e-4) -> tuple[np.ndarray, list[float]]:
    """User-equilibrium traffic assignment (Wardrop's first principle).

    Relative gap = (TSTT - SPTT) / TSTT, where SPTT is total travel time if
    everyone used the current shortest paths. 0 at exact equilibrium.
    """
    x = all_or_nothing(net, demand, net.fft_min)
    gaps = []
    for _ in range(max_iter):
        t = net.link_time(x)
        y = all_or_nothing(net, demand, t)
        tstt = float(t @ x)
        gap = (tstt - float(t @ y)) / tstt
        gaps.append(gap)
        if gap < gap_tol:
            break
        d = y - x
        # exact line search on the Beckmann objective: its derivative in lam is
        # sum_a d_a * t_a(x + lam d), monotone increasing -> bisection
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if d @ net.link_time(x + mid * d) > 0:
                hi = mid
            else:
                lo = mid
        x = x + 0.5 * (lo + hi) * d
    return x, gaps


def two_link_equilibrium(t0, cap, demand) -> float:
    """Analytic check: flow on link 0 that equalises BPR times on two parallel links."""
    from .network import ALPHA, BETA

    def t(i, v):
        return t0[i] * (1 + ALPHA * (v / cap[i]) ** BETA)
    if t(0, demand) <= t(1, 0):
        return demand
    if t(1, demand) <= t(0, 0):
        return 0.0
    return brentq(lambda v: t(0, v) - t(1, demand - v), 0, demand)
