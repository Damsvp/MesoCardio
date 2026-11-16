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
    k_on: float = 5.0; h_a: float = -0.3; ell: float = 0.15   # attachment window G_a
    k_off: float = 1.0                                         # detachment envelope G_d
    dt: float = 0.01; T: float = 60.0

wrap = lambda h, d: (h + d / 2) % d - d / 2
w0 = lambda x, p: 0.5 * p.k0 * x**2
w1 = lambda x, p: 0.5 * p.k * (x - p.xs) ** 2
dw1 = lambda x, p: p.k * (x - p.xs)

def G_a(h, p): return p.k_on * np.exp(-(h - p.h_a) ** 2 / (2 * p.ell**2)) * np.cos(np.pi * h / p.d) ** 2
def G_d(h, p): return p.k_off * np.cos(np.pi * h / p.d) ** 2

def rates_attach(x, h, p):
    """total rate of 0->1 jumps from pre-state (x,h): A + A^r."""
    dEa = w1(x + h, p) - w0(x, p)
    K01 = G_a(h, p) * np.minimum(1.0, np.exp(-dEa / p.kT))
    y = x + h                                           # post-jump position (A^r is indexed by it)
    dEd = w0(y - h, p) - p.mu_T - w1(y, p)
    K10rev = G_d(h, p) * np.minimum(1.0, np.exp(dEd / p.kT))
    return K01 + K10rev

def rho_detach(y, h, p):
    """rate density in the mark h of 1->0 jumps from y: D + D^r."""
    dEd = w0(y - h, p) - p.mu_T - w1(y, p)
    K10 = G_d(h, p) * np.minimum(1.0, np.exp(-dEd / p.kT))
    x = y - h                                           # post-jump position (D^r is indexed by it)
    dEa = w1(x + h, p) - w0(x, p)
    K01rev = G_a(h, p) * np.minimum(1.0, np.exp(dEa / p.kT))
    return K10 + K01rev

def simulate(model, N, R, p, seed=0, v_fixed=None, init=None, rec=10):
    rng = np.random.default_rng(seed)
    nt = int(round(p.T / p.dt))
    alpha = np.zeros((R, N), bool)
    X = rng.normal(0, np.sqrt(p.kT / p.k0), (R, N))
    phi = rng.uniform(-p.d / 2, p.d / 2, (R, N)) if init is None else init
    h = wrap(phi - X, p.d)
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
#  Usage:  python soft_models_all.py --N1 2 --N2 2 --NR1 500 --sigma 0 --mu_T 0.1 [--show]
# ======================================================================================
import argparse
import matplotlib.pyplot as plt

CONFIG = dict(N1=1, NR1=200, N2=1, NR2=1000, N3=1000, NR3=100,
              sigma=10.0, mu_T=0.1, T=80.0, seed=0)

def main():
    ap = argparse.ArgumentParser()
    for k, v in CONFIG.items():
        ap.add_argument("--" + k, type=type(v), default=v)
    ap.add_argument("--out", default="trajectories_db.png")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    p = Params(sigma=a.sigma, mu_T=a.mu_T, T=a.T)
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
