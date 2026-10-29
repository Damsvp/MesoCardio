# %% ONE head, the h-model dynamics (3.2), the simplest version: step attachment rate  K01 = K_ATT * 1{h > H_THR}
#    detached : dX = -w0'(X)/eta dt + sqrt(2 kBT/eta) dB ;  the sites slide at xdot_c = V, so h = s - X rises at ~V
#    attached : dX = xdot_c dt = V dt ,  h = 0 (X is carried with its site)
#    0 -> 1   : if h > H_THR, with rate K_ATT :  X -> X + h  (tip lands on the site)
#    1 -> 0   : rate LAM_DET, landing mark h' ~ U(-d/2, 0) :  X -> X - h'
import numpy as np, matplotlib.pyplot as plt
import hmodel_soft as base
P = base.P; d = P.d

V, H_THR = 0.5, 5.0                                  # xdot_c [nm/ms], threshold on h [nm]
K_ATT, LAM_DET = 0.10, 0.10                          # attachment rate above the threshold, detachment rate [ms^-1]
T, dt, seed = 300.0, 0.002, 3

wrap = lambda u: (u + d/2) % d - d/2
W0 = base.Potential(P.w0); rng = np.random.default_rng(seed); noise = np.sqrt(2*P.kBT*dt/P.eta)
p_att, p_det = -np.expm1(-K_ATT*dt), -np.expm1(-LAM_DET*dt)
X = -2.0; s0 = X + rng.uniform(-d/2, d/2); y = 0.0; al = 0
tr, events = [], []
for n in range(int(round(T/dt))):
    t = n*dt
    h = wrap(s0 + y - X)                             # nearest site in the window centred on the tip
    tr.append((t, X, h, al))
    if not al and h > H_THR and rng.random() < p_att:        # step: only a head with h above the threshold can bind
        events.append((t, "ATTACH", h)); X += h; al = 1
    elif al and rng.random() < p_det:
        hp = rng.uniform(-d/2, 0.0); events.append((t, "DETACH", hp)); X -= hp; al = 0
    if al: X += V*dt                                          # dX = xdot_c dt
    else:  X += -dt*W0.dw(X)/P.eta + noise*rng.normal()       # dX = -w0'/eta dt + noise
    y += V*dt
for t, k, h in events: print(f"t = {t:6.1f} ms: {k}  (h = {h:5.1f} nm)")
t_, X_, h_, a_ = np.array(tr).T
print(f"n_a = {a_.mean():.3f}; {sum(e[1]=='ATTACH' for e in events)} attachments, {sum(e[1]=='DETACH' for e in events)} detachments in {T:.0f} ms")
fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
ax[0].plot(t_, X_); ax[0].set(ylabel="X (nm)")
ax[1].plot(t_, h_); ax[1].axhline(H_THR, color="r", lw=0.8, label=f"threshold {H_THR} nm"); ax[1].axhline(0, color="k", lw=0.4)
ax[1].set(ylabel="h (nm)", ylim=(-d/2, d/2)); ax[1].legend()
ax[2].step(t_, a_, where="post"); ax[2].set(ylabel=r"$\alpha$ (1 = attached)", xlabel="t (ms)", yticks=[0, 1]); fig.tight_layout(); plt.show()