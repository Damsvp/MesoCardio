# %% h-model (3.2) with the macro force balance (soft device, sigma imposed), N heads (N = 1: the case of the remarks)
#    macro equation (per head, nu_ph = nu/N):   nu_ph * ydot = sigma - (1/N) sum_i alpha_i w1'(X_i)
#    detached : dX = -w0'(X)/eta dt + sqrt(2 kBT/eta) dB ;  attached : dX = ydot dt ;  h = distance to the nearest site (sites slide with y)
#    0 -> 1 : rate K_ATT * 1{h > H_THR}, X -> X + h        (K_ATT = np.inf : the threshold alone decides, bind as soon as h > H_THR)
#    1 -> 0 : rate LAM_DET, mark h' ~ U(-d/2, 0), X -> X - h'
#    LAM_DET = 0 : no detachment (the system must stop) ;  K_ATT = 0 : no attachment (ydot = sigma/nu_ph, linear in sigma)
import numpy as np, matplotlib.pyplot as plt
import hmodel_soft as base
P = base.P; d = P.d

N = 1
SIGMA, NU_PH = 0.5, 0.05                             # sigma per head [pN], nu per head [pN ms/nm]
H_THR, K_ATT, LAM_DET = 5.0, 0.10, 0.10              # [nm], [ms^-1], [ms^-1]
T, dt, seed, T_RELAX = 300.0, 0.002, 3, 5.0          # T_RELAX: time after a detachment before h is read as "free" detached drift

wrap = lambda u: (u + d/2) % d - d/2
W0, W1 = base.Potential(P.w0), base.Potential(P.w1)
rng = np.random.default_rng(seed); noise = np.sqrt(2*P.kBT*dt/P.eta)
p_att = 1.0 if np.isinf(K_ATT) else -np.expm1(-K_ATT*dt); p_det = -np.expm1(-LAM_DET*dt)
nsteps = int(round(T/dt))
al = np.zeros(N, bool); X = -2.0 + rng.normal(size=N)*np.sqrt(P.kBT/P.w0["kl"]*0.5)
s0 = X + rng.uniform(-d/2, d/2, N); y = 0.0
rec = np.empty((nsteps, 7)); events = []
for n in range(nsteps):
    tau = np.where(al, W1.dw(X), 0.0).mean()                                  # tau_c = (1/N) sum alpha_i w1'(X_i)
    Kst = np.where(al, np.maximum(W1.d2w(X), 0.0), 0.0).mean()
    ydot = (SIGMA - tau)/(NU_PH + Kst*dt)                                     # macro equation (semi-implicit)
    h = wrap(s0 + y - X)                                                      # nearest site in the window centred on each tip
    rec[n] = (n*dt, ydot, tau, al.mean(), X[0], h[0], al[0])
    up = (~al) & (h > H_THR) & (rng.random(N) < p_att)
    dn = al & (rng.random(N) < p_det)
    if up[0]: events.append((n*dt, "ATTACH", h[0]))
    hp = rng.uniform(-d/2, 0.0, dn.sum())
    if dn[0]: events.append((n*dt, "DETACH", hp[np.cumsum(dn)[0] - 1]))
    X[up] += h[up]; X[dn] -= hp
    al = (al | up) & ~dn
    dxc = ydot*dt; de = ~al
    X[al] += dxc
    X[de] += -dt*W0.dw(X[de])/P.eta + noise*rng.normal(size=de.sum())
    y += dxc
t_, yd_, tau_, na_, X_, h_, a_ = rec.T
for t, k, hh in events[:12]: print(f"t = {t:6.1f} ms: {k}  (h = {hh:5.1f} nm)")
print(f"N = {N}, sigma = {SIGMA} pN, nu_ph = {NU_PH}; <n_a> = {na_.mean():.3f}; free sliding speed sigma/nu_ph = {SIGMA/NU_PH:.3f} nm/ms")

# ---- checks suggested in the remarks (quadratic w1: w1'' = k)
xg = np.linspace(-50, 100, 301); kk = W1.d2w(xg); quad = bool(np.allclose(kk, kk[0], rtol=1e-6)); k = float(kk[0])
print(f"w1'' = {k:.4g} pN/nm {'(constant: quadratic w1)' if quad else '(NOT constant: the exponential checks below do not apply)'}")
dh = wrap(np.diff(h_)); det_free = np.zeros(nsteps - 1, bool)                  # detached steps at least T_RELAX after the last detachment
last_det = -1e9; i = 0
for j in range(nsteps - 1):
    if a_[j] == 1: last_det = t_[j]
    det_free[j] = (a_[j] == 0) and (a_[j + 1] == 0) and (t_[j] - last_det > T_RELAX)
if N == 1 and det_free.any():
    print(f"[detached] mean slope of h = {dh[det_free].sum()/(det_free.sum()*dt):.3f} nm/ms vs sigma/nu_ph = {SIGMA/NU_PH:.3f} nm/ms ({det_free.sum()} steps)")
if N == 1 and quad:
    rates = []; i = 0
    while i < nsteps:
        if a_[i] == 1:
            j = i
            while j < nsteps and a_[j] == 1: j += 1
            e = np.abs(X_[i:j] - SIGMA/k); m = e > 1e-4*e[0] if e.size else e
            if m.sum() >= 8: rates.append(-np.polyfit(t_[i:j][m], np.log(e[m]), 1)[0])
            i = j
        else: i += 1
    if rates: print(f"[attached] X(t) - sigma/k decays exponentially, fitted rate = {np.mean(rates):.3f} ms^-1 vs k/nu_ph = {k/NU_PH:.3f} ms^-1 ({len(rates)} segments)")
print(f"end of run: ydot = {yd_[-1]:.4g} nm/ms, tau_c = {tau_[-1]:.4f} pN (sigma = {SIGMA}), X of head 0 = {X_[-1]:.3f} nm (sigma/k = {SIGMA/k:.3f} nm)")
print(f"[between jumps] nu_ph*yddot = -(k n_a) ydot  =>  ydot decays at rate k*n_a/nu_ph (= {k/NU_PH:.2f} ms^-1 with one head attached); it stops when tau_c = sigma")
fig, ax = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
ax[0].plot(t_, X_); ax[0].axhline(SIGMA/k, color="g", lw=0.6, ls="--"); ax[0].set(ylabel="X (nm), head 0")
ax[1].plot(t_, h_); ax[1].axhline(H_THR, color="r", lw=0.8); ax[1].axhline(0, color="k", lw=0.4); ax[1].set(ylabel="h (nm)", ylim=(-d/2, d/2))
ax[2].step(t_, a_, where="post"); ax[2].set(ylabel=r"$\alpha$ (head 0)", yticks=[0, 1])
ax[3].plot(t_, yd_); ax[3].axhline(SIGMA/NU_PH, color="k", lw=0.5, ls=":"); ax[3].set(ylabel=r"$\dot y$ (nm/ms)", xlabel="t (ms)"); fig.tight_layout(); plt.show()
