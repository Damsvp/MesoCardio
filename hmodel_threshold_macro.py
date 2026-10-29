# %% SIMPLEST model: ONE head, sigma imposed, macro force balance, attachment decided by a threshold on h only.
#    quadratic w1 (force of an attached head f = -K1 X), harmonic w0, no external module.
#    macro equation:      NU * ydot = SIGMA - alpha * K1 * X
#    detached (alpha = 0): dX = -K0 X/ETA dt + sqrt(2 KBT/ETA) dB ;  ydot = SIGMA/NU ;  h = s - X rises at ~ SIGMA/NU
#    h > H_THR            -> the head attaches at once: X -> X + h   (no attachment probability)
#    attached (alpha = 1): dX = ydot dt = (SIGMA - K1 X)/NU dt   =>   X(t) - SIGMA/K1 = (X_att - SIGMA/K1) exp(-K1 (t - t_att)/NU)
#    LAM_DET = 0: no detachment => the system stops (ydot -> 0, tau_c = SIGMA, X = SIGMA/K1).  LAM_DET > 0: constant-rate detachment
#    jump (mark h' ~ U(-D/2,0), X -> X - h'): the only element that is not a threshold on h.
import numpy as np, matplotlib.pyplot as plt

D, KBT, ETA, K0, K1, NU = 40.0, 4.1, 0.2, 0.2, 0.5, 0.05          # d [nm], kBT [zJ], eta, k0, k1 [pN/nm], nu [pN ms/nm]
SIGMA, H_THR, LAM_DET = 0.5, 5.0, 0.0
T, dt, seed = 60.0, 0.002, 3
wrap = lambda u: (u + D/2) % D - D/2

rng = np.random.default_rng(seed); noise = np.sqrt(2*KBT*dt/ETA); p_det = -np.expm1(-LAM_DET*dt)
X = rng.normal()*np.sqrt(KBT/K0); s0 = X + rng.uniform(-D/2, D/2); y = 0.0; al = 0
nsteps = int(round(T/dt)); rec = np.empty((nsteps, 5)); events = []
for n in range(nsteps):
    t = n*dt
    ydot = (SIGMA - al*K1*X)/(NU + al*K1*dt)                      # macro equation (semi-implicit)
    h = wrap(s0 + y - X)                                           # nearest site in the window centred on the tip
    rec[n] = (t, X, h, al, ydot)
    if not al and h > H_THR:                                       # threshold: attach, X -> X + h
        events.append((t, "ATTACH", h, X)); X += h; al = 1
    elif al and rng.random() < p_det:                              # optional detachment jump
        hp = rng.uniform(-D/2, 0.0); events.append((t, "DETACH", hp, X)); X -= hp; al = 0
    if al: X += ydot*dt
    else:  X += -dt*K0*X/ETA + noise*rng.normal()
    y += ydot*dt
t_, X_, h_, a_, yd_ = rec.T
for t, k, hh, xx in events: print(f"t = {t:5.2f} ms: {k}  (h = {hh:5.2f} nm, X before = {xx:6.2f} nm)")
print(f"detached: mean slope of h before the first attachment = {np.mean(wrap(np.diff(h_[:int(events[0][0]/dt)]))/dt):.2f} nm/ms vs sigma/nu = {SIGMA/NU:.2f}")
i0 = int(round(events[0][0]/dt)) + 1; seg = np.arange(i0, nsteps) if LAM_DET == 0 else np.arange(i0, i0 + int(1.0/dt))
e = np.abs(X_[seg] - SIGMA/K1); m = e > 1e-4*e[0]
print(f"attached: X - sigma/k1 = (X_att - sigma/k1) exp(-r t): fitted r = {-np.polyfit(t_[seg][m], np.log(e[m]), 1)[0]:.3f} ms^-1 vs k1/nu = {K1/NU:.3f}")
print(f"end: ydot = {yd_[-1]:.2e} nm/ms, tau_c = {a_[-1]*K1*X_[-1]:.4f} pN (sigma = {SIGMA}), X = {X_[-1]:.3f} nm (sigma/k1 = {SIGMA/K1:.3f})")
fig, ax = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
ax[0].plot(t_, X_); ax[0].axhline(SIGMA/K1, color="g", lw=0.6, ls="--", label="sigma/k1"); ax[0].set(ylabel="X (nm)"); ax[0].legend()
ax[1].plot(t_, h_); ax[1].axhline(H_THR, color="r", lw=0.8, label="threshold"); ax[1].axhline(0, color="k", lw=0.4); ax[1].set(ylabel="h (nm)", ylim=(-D/2, D/2)); ax[1].legend()
ax[2].step(t_, a_, where="post"); ax[2].set(ylabel=r"$\alpha$", yticks=[0, 1])
ax[3].plot(t_, yd_); ax[3].set(ylabel=r"$\dot y$ (nm/ms)", xlabel="t (ms)"); fig.tight_layout(); plt.show()
