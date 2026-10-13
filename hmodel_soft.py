"""
h-model of actin-myosin interaction (Chaintron, Caruel, Kimmig, Math. in Action 12, 2023)
with a soft-device closure for the sliding variable x_c:

    nu * dx_c/dt = -F_ext - sum_i alpha_i * w1'(X_i)         (soft device, total force imposed)
    dx_c/dt      = v                                          (velocity clamp, as in the paper)

Conventions (paper): x_c-dot < 0 is contraction, f_i = alpha_i w1'(X_i) > 0 is traction.
Shortening speed u = -x_c-dot.  Load opposing contraction: P = -F_ext.
Units: nm, ms, pN, zJ (1 pN nm = 1 zJ).

State per head: alpha in {0,1}, X (tip position), h in [-d/2,d/2) (signed distance tip -> nearest site).
Four jump channels (Algorithm 1, App. A of the paper), each with a clock c and threshold e ~ Exp(1):
    A: 0->1  K01(x,h)               B: 0->1  K10rev(x,h)   (reverse of detachment)
    C: 1->0  k10(x)   (integral over h of K10)   D: 1->0  k01rev(x) (integral over h of K01rev)
Detailed balance (3.4)-(3.5), with the pairing x_det = x_att - h:
    K01rev(x_att,h) = K01(x_att-h,h)    exp[( w1(x_att) - w0(x_att-h)        )/kBT]
    K10(x_att,h)    = K10rev(x_att-h,h) exp[( w1(x_att) - w0(x_att-h) + muT  )/kBT]
"""
import numpy as np

# ----------------------------------------------------------------------------------------------
# Parameters (Table D.1).  NOTE: transcribed from the PDF text; check against Fig. 5.1.
# ----------------------------------------------------------------------------------------------
class P:
    kB = 1.380649e-2            # zJ/K  (1.380649e-23 J/K)
    T = 298.0
    kBT = kB * T                # ~4.11 zJ
    eta = 0.097                 # pN ms/nm   (micro drag of the detached tip)
    d = 40.0                    # nm
    muT = 100.0                 # zJ
    # attached potential w1
    w1 = dict(kl=0.60, kr=1.0, xtl=-12.5, xtr=-1.5, xa=2.0, xl=1.3, xr=2.5, wchk=5.7, E=50.0)
    # detached potential w0
    w0 = dict(kl=0.12, kr=1.0, xtl=-2.0, xtr=2.0, xa=-2.0, xl=-3.5, xr=0.5, wchk=5.7, E=70.0)
    # K0->1 (direct attachment)
    kp_max, lam_h_p, h12_p, lam_x_p, xl_p, xr_p = 0.413, 1.6, 3.9, 8.0, 1.5, 17.0
    # K^rev_{1->0}
    k_rev_max, lam_h_rm, h12_rm, lam_x_rm, x_rm = 0.206, 5.0, 2.0, 10.0, -28.0
    k_rev_p, lam_h_rp, h12_rp, lam_x_rp, x_rp = 3.1, 5.0, 2.0, 20.0, 0.5
    k_rev_b, lam_h_rb, h12_rb = 0.07, 5.0, 0.2
    EXP_CLIP = 30.0
    H_NORM = 40.0   # length factor in detailed balance for the h-measure (see notes); 1.0 = literal (3.4)-(3.5)


def make_potential(kl, kr, xtl, xtr, xa, xl, xr, wchk, E):
    """Regularised piecewise quadratic potential of Table D.1 (value and derivative as numpy funcs)."""
    wt = -0.5 * kl * (xa - xtl) ** 2 + 0.5 * kr * (xa - xtr) ** 2
    L = xr - xl

    def basis(x):
        p1 = (x - xr) ** 2 / L ** 2 * (2 * (x - xl) / L + 1)
        p2 = (x - xr) ** 2 / L ** 2 * (x - xl)
        p3 = (x - xl) ** 2 / L ** 2 * (3 - 2 * (x - xl) / L)
        p4 = (x - xl) ** 2 / L ** 2 * (x - xr)
        return p1, p2, p3, p4

    p1a, _, p3a, _ = basis(np.float64(xa))

    def w(x):
        x = np.asarray(x, dtype=float)
        left = 0.5 * kl * (x - xtl) ** 2 + wt
        right = 0.5 * kr * (x - xtr) ** 2
        p1, p2, p3, p4 = basis(x)
        mid = ((0.5 * kl * (xl - xtl) ** 2 + wt) * p1 + kl * (xl - xtl) * p2
               + 0.5 * kr * (xr - xtr) ** 2 * p3 + kr * (xr - xtr) * p4
               + wchk * p1 * p3 / (p1a * p3a))
        return E + np.where(x < xl, left, np.where(x > xr, right, mid))
    return w


class Potential:
    """Tabulated w, w', w'' on a fine grid (fast np.interp)."""
    def __init__(self, params, lo=-150.0, hi=150.0, dx=0.01):
        self.x = np.arange(lo, hi + dx, dx)
        w = make_potential(**params)(self.x)
        self.w_tab = w
        self.dw_tab = np.gradient(w, dx)
        self.d2w_tab = np.gradient(self.dw_tab, dx)

    def w(self, x):   return np.interp(x, self.x, self.w_tab)
    def dw(self, x):  return np.interp(x, self.x, self.dw_tab)
    def d2w(self, x): return np.interp(x, self.x, self.d2w_tab)


def box_h(h, lam, hh):
    return np.where(h < 0, 0.5 * (1 + np.tanh(lam * (h + hh))), 0.5 * (1 - np.tanh(lam * (h - hh))))


class HModel:
    """Rates, potentials and tabulated global detachment rates / landing laws."""
    def __init__(self, muT=None):
        self.muT = P.muT if muT is None else muT
        self.W1 = Potential(P.w1)
        self.W0 = Potential(P.w0)
        d = P.d
        dh = 0.25
        self.hgrid = np.arange(-d / 2, d / 2, dh) + dh / 2          # midpoints
        self.dh = dh
        self.xgrid = np.arange(-100.0, 100.0 + 1e-9, 0.2)           # attached positions
        self._build_tables()

    # --- primitive rates (detached state, arguments x = tip, h = signed distance to nearest site) ---
    def K01(self, x, h):
        bx = np.where(x < 0.5 * (P.xl_p + P.xr_p),
                      0.5 * (1 + np.tanh(P.lam_x_p * (x - P.xl_p))),
                      0.5 * (1 - np.tanh(P.lam_x_p * (x - P.xr_p))))
        return P.kp_max * box_h(h, P.lam_h_p, P.h12_p) * bx

    def K10rev(self, x, h):
        e = np.clip((self.W0.w(x) - self.W1.w(x + h) - self.muT) / P.kBT, -P.EXP_CLIP, P.EXP_CLIP)
        t1 = P.k_rev_max * 0.5 * (1 - np.tanh(P.lam_x_rm * (x - P.x_rm))) * box_h(h, P.lam_h_rm, P.h12_rm)
        t2 = P.k_rev_p * 0.5 * (1 - np.tanh(P.lam_x_rp * (x - P.x_rp))) * box_h(h, P.lam_h_rp, P.h12_rp)
        t3 = P.k_rev_b * box_h(h, P.lam_h_rb, P.h12_rb)
        return t1 + (t2 + t3) * np.exp(e)

    # --- derived rates (attached state, arguments x_att, h) ---
    def K01rev(self, xa, h):
        xd = xa - h
        e = np.clip((self.W1.w(xa) - self.W0.w(xd)) / P.kBT, -P.EXP_CLIP, P.EXP_CLIP)
        return self.K01(xd, h) * np.exp(e) / P.H_NORM

    def K10(self, xa, h):
        xd = xa - h
        e = np.clip((self.W1.w(xa) - self.W0.w(xd) + self.muT) / P.kBT, -P.EXP_CLIP, P.EXP_CLIP)
        return self.K10rev(xd, h) * np.exp(e) / P.H_NORM

    def _build_tables(self):
        XA, H = np.meshgrid(self.xgrid, self.hgrid, indexing="ij")
        dens_det = self.K10(XA, H)
        dens_rev = self.K01rev(XA, H)
        self.k_det = dens_det.sum(1) * self.dh          # global rate C(x)
        self.k_rev = dens_rev.sum(1) * self.dh          # global rate D(x)
        self.cdf_det = np.cumsum(dens_det, 1)
        self.cdf_rev = np.cumsum(dens_rev, 1)
        for c in (self.cdf_det, self.cdf_rev):
            tot = c[:, -1:].copy()
            tot[tot == 0] = 1.0
            c /= tot

    def _ix(self, x):
        return np.clip(np.rint((x - self.xgrid[0]) / 0.2).astype(int), 0, len(self.xgrid) - 1)

    def rate_det(self, x):  return self.k_det[self._ix(x)]
    def rate_rev(self, x):  return self.k_rev[self._ix(x)]

    def sample_h(self, x, channel, rng):
        """Landing h' for heads detaching from attached position x. channel 'C' or 'D'."""
        cdf = self.cdf_det if channel == "C" else self.cdf_rev
        rows = cdf[self._ix(x)]
        u = rng.random(len(x))[:, None]
        k = np.minimum((rows < u).sum(1), len(self.hgrid) - 1)
        return self.hgrid[k]


# ----------------------------------------------------------------------------------------------
def wrap(h, d=P.d):
    return (h + d / 2) % d - d / 2


def init_state(N, model, rng):
    """All heads detached, tip ~ Boltzmann in w0 (Gaussian around the left-well minimum), h uniform."""
    X = -2.0 + rng.normal(size=N) * np.sqrt(P.kBT / P.w0["kl"] * 0.5)
    h = rng.uniform(-P.d / 2, P.d / 2, size=N)
    return dict(alpha=np.zeros(N, dtype=np.int8), X=X, h=h,
                c=np.zeros((N, 4)), e=rng.exponential(size=(N, 4)), xc=0.0)


def run(model, N, T, dt, mode, rng, state=None, v=0.0, nu=None, Fext_fn=None, record_every=50,
        burn=0.0, x_clip=100.0):
    """
    mode='clamp': dx_c/dt = v (velocity clamp).
    mode='soft' : nu dx_c/dt = -F_ext(t) - sum f_i, semi-implicit in the attached stiffness.
    Returns dict with time series (t, xc, vel, tau (=sum f / N), na (=attached fraction)) and event counts.
    """
    s = init_state(N, model, rng) if state is None else state
    al, X, h, c, e, xc = s["alpha"], s["X"], s["h"], s["c"], s["e"], s["xc"]
    nsteps = int(round(T / dt))
    rec = []
    counts = np.zeros(4)                       # events A,B,C,D after burn-in
    sig = np.sqrt(2.0 * P.kBT * dt / P.eta)
    acc_tau = acc_na = acc_v = 0.0
    nacc = 0
    for n in range(nsteps):
        t = n * dt
        att = al == 1
        f = np.where(att, model.W1.dw(X), 0.0)
        S = f.sum()
        if mode == "clamp":
            dxc = v * dt
        else:
            K = np.where(att, np.maximum(model.W1.d2w(X), 0.0), 0.0).sum()
            F = Fext_fn(t) if Fext_fn is not None else 0.0
            dxc = -(F + S) * dt / (nu + K * dt)
        # --- rates for the active channels
        det = ~att
        r = np.zeros((N, 4))
        if det.any():
            r[det, 0] = model.K01(X[det], h[det])
            r[det, 1] = model.K10rev(X[det], h[det])
        if att.any():
            r[att, 2] = model.rate_det(X[att])
            r[att, 3] = model.rate_rev(X[att])
        c += r * dt
        res = np.where(r > 0, c - e, -np.inf)
        win = res.argmax(1)
        jump = res[np.arange(N), win] > 0
        # --- continuous motion
        Xn = X.copy()
        hn = h.copy()
        if det.any():
            dX = -dt * model.W0.dw(X[det]) / P.eta + sig * rng.normal(size=det.sum())
            Xn[det] = X[det] + dX
            hn[det] = wrap(h[det] + dxc - dX)
        Xn[att] = X[att] + dxc
        # --- jumps
        if jump.any():
            ij = np.where(jump)[0]
            for ch in (0, 1, 2, 3):
                sel = ij[win[ij] == ch]
                if len(sel) == 0:
                    continue
                if burn <= t:
                    counts[ch] += len(sel)
                if ch in (0, 1):                       # attach to the site at time n+1
                    Xn[sel] = X[sel] + h[sel] + dxc
                    hn[sel] = 0.0
                    al[sel] = 1
                else:                                  # detach, land at h'
                    hp = model.sample_h(X[sel], "C" if ch == 2 else "D", rng)
                    Xn[sel] = X[sel] + dxc - hp
                    hn[sel] = hp
                    al[sel] = 0
            c[ij] = 0.0
            e[ij] = rng.exponential(size=(len(ij), 4))
        X, h = np.clip(Xn, -x_clip, x_clip), hn
        xc += dxc
        if t >= burn:
            acc_tau += S / N
            acc_na += att.mean()
            acc_v += dxc / dt
            nacc += 1
        if n % record_every == 0:
            rec.append((t, xc, dxc / dt, S / N, att.mean()))
    s.update(alpha=al, X=X, h=h, c=c, e=e, xc=xc)
    rec = np.array(rec)
    out = dict(t=rec[:, 0], xc=rec[:, 1], vel=rec[:, 2], tau=rec[:, 3], na=rec[:, 4],
               tau_mean=acc_tau / max(nacc, 1), na_mean=acc_na / max(nacc, 1),
               vel_mean=acc_v / max(nacc, 1), counts=counts, state=s)
    return out
