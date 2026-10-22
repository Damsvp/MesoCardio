"""
Two-rate version of hmodel_soft.py (same interface: HModel, init_state, run).

Instead of the four detailed-balance channels, a head has exactly two jump kernels, densities in
the mark h (units ms^-1 nm^-1) on the window h in [-d/2, d/2]:

    k_+(x,h) = lambda_+(x) * q_G(h)     attachment 0->1, Gaussian mark   (q_G: truncated Gaussian)
    k_-(x,h) = lambda_-(x) * q_U(h)     detachment 1->0, uniform mark    (q_U = 1/d)

q_G, q_U are probability densities (nm^-1), so  int k_pm dh = lambda_pm(x)  (total rates, ms^-1).
x is the tip position X (attachment) or the attached position X (detachment).
Detailed balance is NOT imposed: k_+ and k_- are independent inputs.

Dynamics per head, state (alpha, X); h is a mark, not a state:
    alpha=0: dX = -w0'(X)/eta dt + sqrt(2 kBT/eta) dB        alpha=1: dX = xdot_c dt
    0->1 at rate lambda_+(X):  X -> X + h*,  h* ~ q_G
    1->0 at rate lambda_-(X):  X -> X - h',  h' ~ q_U
Soft device (as in hmodel_soft.py), f_i = alpha_i w1'(X_i), traction > 0:
    nu dx_c/dt = -F_ext - sum_i f_i          (mode='soft')      dx_c/dt = v   (mode='clamp')
Units: nm, ms, pN, zJ.  Needs hmodel_soft.py beside this file (potentials w0, w1, parameters P).

Run `python hmodel_soft_2rates.py` for a single soft-ramp demo with plots.
"""
from math import erf, sqrt, pi
import numpy as np
import hmodel_soft as base

P = base.P

# =============================================================================================
# CHOOSE THE RATES HERE
# =============================================================================================
RATES = "custom"            # "const" | "x" | "custom"
SIGMA_G = 4.0          # nm, width of the Gaussian attachment mark q_G

# --- RATES = "const": constant total rates (ms^-1)
LAMBDA_PLUS_CONST = 0.20
LAMBDA_MINUS_CONST = 0.10

# --- RATES = "x": smooth bounded x-dependent rates (ms^-1), values matched to the four-channel
#     h-model: lambda_+ = plateau * tanh window (plateau = average of K_{0->1} over uniform h),
#     lambda_- = tanh step fitted to int K_{1->0}(x,h) dh (max error ~0.015 ms^-1).
#       lambda_+(x) = LP_X * (1+tanh(KAP_X (x-XL_X)))/2 * (1-tanh(KAP_X (x-XR_X)))/2
#       lambda_-(x) = LM_X * (1-tanh(KAPM_X (x-XM_X)))/2
LP_X = P.kp_max * 2 * P.h12_p / P.d       # ~0.0805 ms^-1
XL_X, XR_X, KAP_X = P.xl_p, P.xr_p, P.lam_x_p   # 1.5 nm, 17 nm, 8 nm^-1
LM_X, XM_X, KAPM_X = 0.311, 0.5, 0.65     # ms^-1, nm, nm^-1


# --- RATES = "custom": Gaussian x-dependent rates (ms^-1)

LAMBDA_0 = 0.1  # Maximum rate (ms^-1)
X_0 = 0.0       # Center of the rate profile (nm)
ELL_X = 5.0     # Width of the rate profile (nm)

def lam_plus_custom(x):
    return LAMBDA_0 * np.exp(-0.5 * ((x - X_0) / ELL_X)**2)

def lam_minus_custom(x):
    return LAMBDA_0 * np.exp(-0.5 * ((x - X_0) / ELL_X)**2)
# =============================================================================================


class HModel:
    """Potentials, total rates lambda_pm(x), marks laws q_G, q_U and their samplers."""

    def __init__(self, rates=None, sigma_g=None):
        self.rates = RATES if rates is None else rates
        if self.rates not in ("const", "x", "custom"):
            raise ValueError("rates must be 'const', 'x' or 'custom'")
        self.sigma_g = SIGMA_G if sigma_g is None else sigma_g
        if self.sigma_g <= 0:
            raise ValueError("sigma_g must be positive")
        self.W1 = base.Potential(P.w1)
        self.W0 = base.Potential(P.w0)
        s, d = self.sigma_g, P.d
        self._Z = s * sqrt(2 * pi) * erf(d / (2 * s * sqrt(2)))     # normalisation of q_G

    # --- total rates (ms^-1)
    def lam_plus(self, x):
        x = np.asarray(x, float)
        if self.rates == "const":
            return LAMBDA_PLUS_CONST * np.ones_like(x)
        if self.rates == "custom":
            return np.asarray(lam_plus_custom(x), float)
        return (LP_X * 0.5 * (1 + np.tanh(KAP_X * (x - XL_X)))
                * 0.5 * (1 - np.tanh(KAP_X * (x - XR_X))))

    def lam_minus(self, x):
        x = np.asarray(x, float)
        if self.rates == "const":
            return LAMBDA_MINUS_CONST * np.ones_like(x)
        if self.rates == "custom":
            return np.asarray(lam_minus_custom(x), float)
        return LM_X * 0.5 * (1 - np.tanh(KAPM_X * (x - XM_X)))

    # --- mark densities (nm^-1) and kernels (ms^-1 nm^-1)
    def q_G(self, h):
        h = np.asarray(h, float)
        return np.where(np.abs(h) <= P.d / 2,
                        np.exp(-h ** 2 / (2 * self.sigma_g ** 2)) / self._Z, 0.0)

    def q_U(self, h):
        h = np.asarray(h, float)
        return np.where(np.abs(h) <= P.d / 2, 1.0 / P.d, 0.0)

    def k_plus(self, x, h):
        return self.lam_plus(x) * self.q_G(h)

    def k_minus(self, x, h):
        return self.lam_minus(x) * self.q_U(h)

    # --- samplers for the marks
    def sample_att(self, n, rng):
        """h* ~ q_G on [-d/2, d/2] (exact, rejection)."""
        out = np.empty(n)
        todo = np.arange(n)
        while todo.size:
            prop = rng.normal(0.0, self.sigma_g, todo.size)
            ok = np.abs(prop) <= P.d / 2
            out[todo[ok]] = prop[ok]
            todo = todo[~ok]
        return out

    def sample_det(self, n, rng):
        """h' ~ q_U on [-d/2, d/2)."""
        return rng.uniform(-P.d / 2, P.d / 2, n)


def init_state(N, model, rng):
    """All heads detached, tip ~ Boltzmann in w0 (Gaussian around the left-well minimum)."""
    X = -2.0 + rng.normal(size=N) * np.sqrt(P.kBT / P.w0["kl"] * 0.5)
    return dict(alpha=np.zeros(N, dtype=bool), X=X, xc=0.0)


def run(model, N, T, dt, mode, rng, state=None, v=0.0, nu=None, Fext_fn=None, record_every=50,
        burn=0.0, x_clip=100.0):
    """
    mode='clamp': dx_c/dt = v.   mode='soft': nu dx_c/dt = -F_ext(t) - sum f_i, semi-implicit in
    the attached stiffness. Same return structure as hmodel_soft.run: dict with t, xc, vel,
    tau (= sum f / N), na (attached fraction), tau_mean, na_mean, vel_mean (after burn-in),
    counts = [attachments, detachments] after burn-in, state.
    """
    if mode not in ("clamp", "soft"):
        raise ValueError("mode must be 'clamp' or 'soft'")
    if mode == "soft" and nu is None:
        raise ValueError("soft mode needs nu")
    s = init_state(N, model, rng) if state is None else state
    al, X, xc = s["alpha"], s["X"], s["xc"]
    nsteps = int(round(T / dt))
    sig = np.sqrt(2.0 * P.kBT * dt / P.eta)
    rec = []
    counts = np.zeros(2)
    acc_tau = acc_na = acc_v = 0.0
    nacc = 0
    for n in range(nsteps):
        t = n * dt
        att = al
        f = np.where(att, model.W1.dw(X), 0.0)
        S = f.sum()
        if mode == "clamp":
            dxc = v * dt
        else:
            Kst = np.where(att, np.maximum(model.W1.d2w(X), 0.0), 0.0).sum()
            F = Fext_fn(t, xc) if Fext_fn is not None else 0.0
            dxc = -(F + S) * dt / (nu + Kst * dt)
        # --- events decided from the state at time t: probability 1 - exp(-lambda(X) dt)
        u_a, u_d = rng.random(N), rng.random(N)
        do_att = (~att) & (u_a < -np.expm1(-model.lam_plus(X) * dt))
        do_det = att & (u_d < -np.expm1(-model.lam_minus(X) * dt))
        # --- continuous motion
        det = ~att
        Xn = X + dxc                                      # attached heads follow x_c
        nd = int(det.sum())
        if nd:
            Xn[det] = X[det] - dt * model.W0.dw(X[det]) / P.eta + sig * rng.normal(size=nd)
        # --- jumps
        if do_att.any():
            idx = np.flatnonzero(do_att)
            Xn[idx] = X[idx] + model.sample_att(idx.size, rng) + dxc
            al[idx] = True
            if burn <= t:
                counts[0] += idx.size
        if do_det.any():
            idx = np.flatnonzero(do_det)
            Xn[idx] = X[idx] + dxc - model.sample_det(idx.size, rng)
            al[idx] = False
            if burn <= t:
                counts[1] += idx.size
        X = np.clip(Xn, -x_clip, x_clip)
        xc += dxc
        if t >= burn:
            acc_tau += S / N
            acc_na += att.mean()
            acc_v += dxc / dt
            nacc += 1
        if n % record_every == 0:
            rec.append((t, xc, dxc / dt, S / N, att.mean()))
    s.update(alpha=al, X=X, xc=xc)
    rec = np.array(rec)
    return dict(t=rec[:, 0], xc=rec[:, 1], vel=rec[:, 2], tau=rec[:, 3], na=rec[:, 4],
                tau_mean=acc_tau / max(nacc, 1), na_mean=acc_na / max(nacc, 1),
                vel_mean=acc_v / max(nacc, 1), counts=counts, state=s)


# ----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=400)
    ap.add_argument("--rates", choices=["const", "x", "custom"], default=None,
                    help="override RATES set at the top of this file")
    ap.add_argument("--nu-per-head", type=float, default=0.05)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--hold", type=float, default=10.0)
    ap.add_argument("--ramp", type=float, default=30.0)
    ap.add_argument("--tau0", type=float, default=0.88)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()

    model = HModel(rates=a.rates)
    p_lo, p_hi = 0.25 * a.tau0, 1.30 * a.tau0

    def pbar(t):
        if t < a.hold:
            return p_lo
        s = t - a.hold
        if s < a.ramp:
            return p_lo + (p_hi - p_lo) * s / a.ramp
        s -= a.ramp
        return p_hi - (p_hi - p_lo) * s / a.ramp if s < a.ramp else p_lo

    T = a.hold + 2 * a.ramp
    out = run(model, N=a.N, T=T, dt=a.dt, mode="soft", nu=a.N * a.nu_per_head,
              Fext_fn=lambda t, xc: -a.N * pbar(t), rng=np.random.default_rng(a.seed))
    print(f"rates={model.rates}, N={a.N}: attachments={int(out['counts'][0])}, "
          f"detachments={int(out['counts'][1])}")
    print(f"mean tau={out['tau_mean']:.4f} pN, mean vel={out['vel_mean']:.4f} nm/ms, "
          f"mean attached={out['na_mean']:.4f}")

    xs = np.linspace(-40, 40, 801)
    hh = np.linspace(-P.d / 2, P.d / 2, 4001)
    print("check int k dh at x=5:", float(np.sum(model.k_plus(5.0, hh)) * (hh[1] - hh[0])),
          float(np.sum(model.k_minus(5.0, hh)) * (hh[1] - hh[0])), "ms^-1")
    fig, ax = plt.subplots()
    ax.plot(xs, model.lam_plus(xs), label=r"$\lambda_+(x)$")
    ax.plot(xs, model.lam_minus(xs), label=r"$\lambda_-(x)$")
    ax.set(xlabel="x (nm)", ylabel=r"total rate (ms$^{-1}$)"); ax.legend(); fig.tight_layout()
    fig, ax = plt.subplots()
    ax.plot(out["t"], out["tau"], label=r"$\tau_c$")
    ax.plot(out["t"], [pbar(t) for t in out["t"]], "k--", label=r"load $\bar p(t)$")
    ax.set(xlabel="time (ms)", ylabel="force per head (pN)"); ax.legend(); fig.tight_layout()
    fig, ax = plt.subplots()
    ax.plot(out["t"], out["vel"])
    ax.set(xlabel="time (ms)", ylabel=r"$\dot x_c$ (nm/ms)"); fig.tight_layout()
    fig, ax = plt.subplots()
    ax.plot(out["t"], out["na"])
    ax.set(xlabel="time (ms)", ylabel="attached fraction"); fig.tight_layout()
    if not a.no_show:
        plt.show()
