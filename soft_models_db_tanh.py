"""
h-model, soft device, with the four jump channels of Chaintron et al. (3.2)-(3.5)
and detailed balance built in.

State (alpha, X, h); h in T_d = [-d/2, d/2); s = X + h* = nearest actin site (h* = lift of h).
Reference (Gibbs) weights, per unit dX dh (alpha=0) and per unit dX (alpha=1):
    pi_0(x) = exp(-w0(x)/kT),   pi_1(y) = exp(-w1(y)/kT)
Four channels. Notation: K(.,.) = rate FUNCTION of the paper; arguments are given in the
coordinates in which each detailed-balance relation is written.

 attachment, pre-jump state (x,h) -> (x+h*, pi(0))           [alpha = 0 -> 1]
   A   : rate K01(x,h)                       (Poisson N_{0->1}(dt,du))
   A^r : rate K10rev(x+h*, h)                (Poisson N^rev_{1->0}(dt,du)),  K10rev indexed by POST-jump x
 detachment, pre-jump state (y,pi(0)) -> (y-h*, h), mark h    [alpha = 1 -> 0]
   D   : rate density K10(y,h)               (Poisson N_{1->0}(dt,du,dh))
   D^r : rate density K01rev(y-h*, h)        (Poisson N^rev_{0->1}(dt,du,dh)), K01rev indexed by POST-jump x

Detailed balance (3.4)-(3.5):
   K01(x,h)  = exp[-(w1(x+h*) - w0(x))/kT]          K01rev(x,h)
   K10(y,h)  = exp[-(w0(y-h*) - mu_T - w1(y))/kT]   K10rev(y,h)
Both members of a pair vanish at h = +-d/2 (3.3).  Choice here (Metropolis):
   K01 = G_a(h) min(1, e^{-dE_a/kT}),  K01rev = G_a(h) min(1, e^{+dE_a/kT}),  dE_a = w1(x+h)-w0(x)
   K10 = G_d(h) min(1, e^{-dE_d/kT}),  K10rev = G_d(h) min(1, e^{+dE_d/kT}),  dE_d = w0(y-h)-mu_T-w1(y)
so every rate is bounded by G_a or G_d (no exponential blow-up), and the ratio is exactly (3.4)/(3.5).
mu_T = 0  => both pairs balanced w.r.t. the same Gibbs measure (equilibrium); mu_T != 0 => driven cycle.

Soft closure:  gamma*ydot = sigma + F_mot,  F_mot = -(1/N) sum_i alpha_i w1'(X_i).
"""
import numpy as np
from dataclasses import dataclass

@dataclass
class Params:
    d: float = 1.0
    k: float = 1.0; k0: float = 1.0; xs: float = 0.0   # w1 = k (x-xs)^2/2, w0 = k0 x^2/2
    eta: float = 1.0; kT: float = 0.05
    mu_T: float = 0.0
    gamma: float = 1.0; sigma: float = 0.0
    # envelopes (prefactors Gamma of the two reversible pairs): tanh windows in h, origins (l, r) + slope lam
    k_on: float = 5.0;  a_l: float = -0.40; a_r: float = -0.20; lam_a: float = 40.0   # attachment pair (3.4)
    k_off: float = 1.0; d_l: float = -0.30; d_r: float = 0.30;  lam_d: float = 40.0   # detachment pair (3.5)
    # initial law of a head: X ~ N(X0, sX^2) (sX=None -> thermal sqrt(kT/k0)); h law; probability to start attached
    X0: float = 0.0; sX: float = None
    h_init: str = "uniform"       # 'uniform' on the torus | 'window' (law ~ Gamma_d window) | 'zero'
    p_att0: float = 0.0
    dt: float = 0.01; T: float = 60.0
    rule: str = 'tanh'       # 'tanh' (Glauber) or 'metropolis'

wrap = lambda h, d: (h + d / 2) % d - d / 2
w0 = lambda x, p: 0.5 * p.k0 * x**2
w1 = lambda x, p: 0.5 * p.k * (x - p.xs) ** 2
dw1 = lambda x, p: p.k * (x - p.xs)

def window(h, l, r, lam):
    """tanh window: 1/2 [tanh(lam (h-l)) - tanh(lam (h-r))]  -- two steps, origins l < r, slope lam; values in [0,1)."""
    return 0.5 * (np.tanh(lam * (h - l)) - np.tanh(lam * (h - r)))

def cut(h, p):
    """exact zero at h = +-d/2 (condition (3.3)), ~1 elsewhere; Lipschitz."""
    return 1.0 - np.abs(2 * h / p.d) ** 16

def G_a(h, p): return p.k_on * window(h, p.a_l, p.a_r, p.lam_a) * cut(h, p)
def G_d(h, p): return p.k_off * window(h, p.d_l, p.d_r, p.lam_d) * cut(h, p)

def fwd(dE, p):
    """forward factor f(dE); the reverse jump uses f(-dE). Both give f(dE)/f(-dE) = exp(-dE/kT)."""
    if p.rule == "tanh":
        return 0.5 * (1.0 - np.tanh(dE / (2 * p.kT)))     # = 1/(1+e^{dE/kT}),  f(dE)+f(-dE) = 1
    return np.minimum(1.0, np.exp(-dE / p.kT))              # Metropolis

def rates_attach(x, h, p):
    """total rate of 0->1 jumps from pre-state (x,h): A + A^r."""
    dEa = w1(x + h, p) - w0(x, p)
    K01 = G_a(h, p) * fwd(dEa, p)
    y = x + h                                           # post-jump position (A^r is indexed by it)
    dEd = w0(y - h, p) - p.mu_T - w1(y, p)
    K10rev = G_d(h, p) * fwd(-dEd, p)
    return K01 + K10rev

def rho_detach(y, h, p):
    """rate density in the mark h of 1->0 jumps from y: D + D^r."""
    dEd = w0(y - h, p) - p.mu_T - w1(y, p)
    K10 = G_d(h, p) * fwd(dEd, p)
    x = y - h                                           # post-jump position (D^r is indexed by it)
    dEa = w1(x + h, p) - w0(x, p)
    K01rev = G_a(h, p) * fwd(-dEa, p)
    return K10 + K01rev

def sample_window(shape, l, r, lam, p, rng):
    """h with density ~ window(h; l, r, lam) * cut(h), by rejection."""
    out = np.empty(shape); todo = np.ones(shape, bool)
    while todo.any():
        c = rng.uniform(-p.d / 2, p.d / 2, shape)
        ok = todo & (rng.random(shape) < window(c, l, r, lam) * cut(c, p))
        out[ok] = c[ok]; todo &= ~ok
    return out

def init_state(R, N, p, rng):
    """random initial condition: X ~ N(X0, sX^2), h from p.h_init, alpha ~ Bernoulli(p_att0) (then h = 0)."""
    sX = np.sqrt(p.kT / p.k0) if p.sX is None else p.sX
    X = rng.normal(p.X0, sX, (R, N))
    if p.h_init == "uniform": h = rng.uniform(-p.d / 2, p.d / 2, (R, N))
    elif p.h_init == "window": h = sample_window((R, N), p.d_l, p.d_r, p.lam_d, p, rng)
    else: h = np.zeros((R, N))
    alpha = rng.random((R, N)) < p.p_att0
    return alpha, X, np.where(alpha, 0.0, h)

def simulate(model, N, R, p, seed=0, v_fixed=None, init=None, rec=10):
    rng = np.random.default_rng(seed)
    nt = int(round(p.T / p.dt))
    alpha, X, h = init_state(R, N, p, rng)
    phi = wrap(X + h, p.d)                                  # lattice phase (conserved if ydot = 0 and alpha = 0)
    y = np.zeros((R, 1)); M = p.k_on + p.k_off          # bound on rho_detach
    nr = nt // rec
    o = dict(t=np.arange(nr) * p.dt * rec, y=np.zeros((nr, R)), v=np.zeros((nr, R)),
             F=np.zeros((nr, R)), frac=np.zeros((nr, R)), phi=phi.copy())
    s2 = np.sqrt(2 * p.kT * p.dt / p.eta)
    for n in range(nt):
        f = -alpha.astype(float) * dw1(X, p)
        Fm = f.mean(axis=1, keepdims=True)
        if model == 2: Fm = np.full((R, 1), f.mean())
        v = (p.sigma + Fm) / p.gamma if v_fixed is None else np.full((R, 1), v_fixed)
        if n % rec == 0 and n // rec < nr:
            i = n // rec
            o["y"][i] = y[:, 0]; o["v"][i] = v[:, 0]; o["F"][i] = Fm[:, 0]; o["frac"][i] = alpha.mean(1)
        free = ~alpha
        dX = np.where(free, -(p.k0 / p.eta) * X * p.dt + s2 * rng.standard_normal(X.shape), v * p.dt)
        X = X + dX; h = wrap(h + v * p.dt - dX, p.d)
        # alpha=0: A and A^r, Poisson N(dt,du) thinned: jump iff u <= rate
        att = free & (rng.random(X.shape) < 1 - np.exp(-rates_attach(X, h, p) * p.dt))
        # alpha=1: Poisson on [0,M] x T_d with intensity dt du dh, accepted if u <= rho(y,h)
        cand = alpha & (rng.random(X.shape) < 1 - np.exp(-M * p.d * p.dt))
        hc = rng.uniform(-p.d / 2, p.d / 2, X.shape); u = rng.uniform(0, M, X.shape)
        det = cand & (u <= rho_detach(X, hc, p))
        Xn = np.where(att, X + h, np.where(det, X - hc, X))
        h = np.where(att, 0.0, np.where(det, hc, h)); X = Xn
        alpha = (alpha | att) & ~det
        y = y + v * p.dt
    return o

def predicted_attached(phi, p):
    """hard device, mu_T=0: fixed lattice phase phi => P(att|phi) = S1/(S1+Z0)."""
    n = np.arange(-6, 7)[:, None]
    S1 = np.exp(-w1(phi.ravel()[None, :] + n * p.d, p) / p.kT).sum(0)
    Z0 = np.sqrt(2 * np.pi * p.kT / p.k0)
    return (S1 / (S1 + Z0))


# ======================================================================================
#  Driver: three closures of the soft-device h-model
#    Model 1 : own y per realisation, N1 heads, NR1 independent realisations
#    Model 2 : mean field, N2 heads, NR2 realisations SHARING one y (force = mean over realisations and heads)
#    Model 3 : own y per realisation, N3 heads, NR3 realisations
#  Usage:  python soft_models_db_tanh.py --N1 2 --N2 2 --NR1 500 --sigma 0 --mu_T 0.1 [--show]
# ======================================================================================
import argparse
import matplotlib.pyplot as plt

CONFIG = dict(N1=1, NR1=200, N2=1, NR2=200, N3=500, NR3=100,
              sigma=0.0, mu_T=0.1, T=80.0, seed=0, rule='tanh',
              h_init='uniform', p_att0=0.0, X0=0.0)

def main():
    ap = argparse.ArgumentParser()
    for k, v in CONFIG.items():
        ap.add_argument("--" + k, type=type(v), default=v)
    ap.add_argument("--out", default="trajectories_tanh.png")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    p = Params(sigma=a.sigma, mu_T=a.mu_T, T=a.T, rule=a.rule, h_init=a.h_init, p_att0=a.p_att0, X0=a.X0)
    runs = {f"Model 1: N={a.N1}, N_R={a.NR1}": simulate(1, a.N1, a.NR1, p, a.seed + 1),
            f"Model 2: N={a.N2}, N_R={a.NR2}, mean field": simulate(2, a.N2, a.NR2, p, a.seed + 2),
            f"Model 3: N={a.N3}, N_R={a.NR3}": simulate(3, a.N3, a.NR3, p, a.seed + 3)}
    rows = [("y", "y(t)"), ("F", "F_mot per head"), ("frac", "fraction attached")]
    fig, ax = plt.subplots(3, 4, figsize=(16, 10), sharex=True)
    for j, (name, o) in enumerate(runs.items()):
        for i, (key, lab) in enumerate(rows):
            ax[i, j].plot(o["t"], o[key][:, :4], lw=0.6, alpha=0.7)
            ax[i, j].plot(o["t"], o[key].mean(axis=1), "k", lw=1.8)
            if j == 0: ax[i, j].set_ylabel(lab)
            if i == 0: ax[i, j].set_title(name, fontsize=9)
            ax[i, 3].plot(o["t"], o[key].mean(axis=1), lw=1.5, label=name.split(":")[0] if i == 0 else None)
    ax[0, 3].set_title("means over realisations", fontsize=9); ax[0, 3].legend(frameon=False)
    for j in range(4): ax[2, j].set_xlabel("t")
    fig.suptitle(f"soft device, sigma={a.sigma}, mu_T={a.mu_T}  (thin: 4 realisations, thick: mean)")
    fig.tight_layout(); fig.savefig(a.out, dpi=130)
    sl = next(iter(runs.values()))["t"] > p.T / 2
    for name, o in runs.items():
        print(f"{name:40s} <ydot>={o['v'][sl].mean():+.4f}  <F>={o['F'][sl].mean():+.4f}  attached={o['frac'][sl].mean():.3f}")
    if a.show: plt.show()

if __name__ == "__main__":
    main()
