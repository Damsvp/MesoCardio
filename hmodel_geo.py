# %% h = distance tip -> nearest actin site, recomputed at EVERY step from the geometry (no dynamics for h).
#    P sites on the filament, spacing d; in the myosin frame they sit at a_j + y(t), y = V t  (paper dynamics for X).
#    Large sample of independent heads (random site offset per head); imposed sigma and V => tau_c = sigma - nu_ph*V.
import numpy as np, matplotlib.pyplot as plt
import hmodel_soft as base
P_ = base.P; d = P_.d

SIGMA, NU_PH, V = 0.5, 0.05, 0.5                     # sigma per head, nu per head, speed ydot [nm/ms]
TAU_C_MACRO = SIGMA - NU_PH*V
H_THR   = 0.0                                        # step: a detached head can bind only if h > H_THR [nm]  (0 <= H_THR < d/2)
K_ATT   = 0.10                                       # attachment rate when h > H_THR [ms^-1]  (np.inf = bind as soon as h > H_THR)
LAM_DET = 0.10                                       # detachment rate [ms^-1]; landing mark h' ~ U(-d/2, 0): X -> X - h'
R, T, T_BURN, dt, seed, rec_every = 4000, 150.0, 60.0, 0.002, 4, 5
NSITES = int(np.ceil(V*T/d)) + 5                     # P: enough sites to cover the filament sliding for the whole run
xbins = np.linspace(-60, 100, 641); dx = xbins[1] - xbins[0]; xm = 0.5*(xbins[1:] + xbins[:-1])

W0, W1 = base.Potential(P_.w0), base.Potential(P_.w1)
rng = np.random.default_rng(seed); noise = np.sqrt(2*P_.kBT*dt/P_.eta)
a_start = -(NSITES - 1)*d + rng.uniform(0, d, R)     # site j of head i lies at a_start[i] + j d on the filament
X = -2.0 + rng.normal(size=R)*np.sqrt(P_.kBT/P_.w0["kl"]*0.5)
al = np.zeros(R, bool); y = 0.0
p_att = 1.0 if np.isinf(K_ATT) else -np.expm1(-K_ATT*dt); p_det = -np.expm1(-LAM_DET*dt)

def geo_h(X, y):
    """nearest site to the tip, h = s - X, valid only if it lies in the window [X-d/2, X+d/2]"""
    j = np.clip(np.rint((X - y - a_start)/d), 0, NSITES - 1)
    h = a_start + j*d + y - X
    return h, np.abs(h) <= d/2

acc1 = np.zeros(len(xm)); acc0 = np.zeros(len(xm)); ns = 0; tau_s = 0.0; traj = []; nosite = 0; hb_sum = 0.0; hb_n = 0
for n in range(int(round(T/dt))):
    h, ok = geo_h(X, y)                              # 1. look at the nearest site
    nosite += (~ok).sum()
    if n % rec_every == 0:
        traj.append((n*dt, X[0], h[0], al[0]))
        if n*dt >= T_BURN:
            acc1 += np.histogram(X[al], xbins)[0]/(R*dx); acc0 += np.histogram(X[~al], xbins)[0]/(R*dx)
            tau_s += np.where(al, W1.dw(X), 0.0).mean(); ns += 1
    up = (~al) & ok & (h > H_THR) & (rng.random(R) < p_att)             # 2. step rule: h > H_THR -> can bind, X -> X + h
    dn = al & (rng.random(R) < p_det)                                    #    detach: X -> X - h', h' ~ U(-d/2,0)
    if n*dt >= T_BURN: hb_sum += h[up].sum(); hb_n += up.sum()
    X[up] += h[up]; X[dn] -= rng.uniform(-d/2, 0.0, dn.sum())
    al = (al | up) & ~dn
    de = ~al                                                             # 3. X moves with the dynamics of its state
    X[al] += V*dt
    X[de] += -dt*W0.dw(X[de])/P_.eta + noise*rng.normal(size=de.sum())
    y += V*dt
p1, p0 = acc1/ns, acc0/ns; tau_p = np.sum(W1.dw(xm)*p1)*dx
print(f"P={NSITES} sites; head-steps without a site in the window: {nosite/(R*int(round(T/dt))):.2e}")
print(f"step threshold H_THR = {H_THR} nm: mean h at binding = {hb_sum/max(hb_n,1):.2f} nm over {hb_n} bindings")
print(f"n_a = {p1.sum()*dx:.3f} ; int p1 = {p1.sum()*dx:.3f}, int p0 = {p0.sum()*dx:.3f}")
print(f"tau_c from p = {tau_p:.4f} pN (direct {tau_s/ns:.4f}); macro sigma - nu_ph*V = {TAU_C_MACRO:.4f} pN; residual {TAU_C_MACRO - tau_p:+.4f}")
tr = np.array(traj); t_, Xt, ht, at = tr.T; z = t_ <= 80
fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
ax[0].plot(t_[z], Xt[z]); ax[0].set(ylabel="X (nm)")
ax[1].plot(t_[z], ht[z]); ax[1].axhline(0, color="k", lw=0.5); ax[1].set(ylabel="h (nm)", ylim=(-d/2, d/2))
ax[2].step(t_[z], at[z], where="post"); ax[2].set(ylabel=r"$\alpha$", xlabel="t (ms)", yticks=[0, 1]); fig.tight_layout()
fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(xm, p1, label=r"$p(1,x)$"); ax.plot(xm, p0, label=r"$p(0,x)$")
ax.set(xlabel="x (nm)", ylabel=r"$p(\alpha,x)$ (nm$^{-1}$)", xlim=(-40, 80)); ax.legend(); fig.tight_layout(); plt.show()
