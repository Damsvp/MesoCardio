# %% Pure on/off: alpha = 1 if h > H_THR else 0, h = distance tip -> nearest site (window of size d), recomputed at every step.
#    No attachment/detachment rates, no jumps. Sites slide at V (paper frame); many independent heads (random site offset).
#    CARRY = False : X always follows the detached dynamics (diffusion in w0); alpha only labels the time spent with h > H_THR
#    CARRY = True  : an "on" head is carried with the filament, dX = V dt (paper attached dynamics)  -> never switches off
import numpy as np, matplotlib.pyplot as plt
import hmodel_soft as base
P_ = base.P; d = P_.d

SIGMA, NU_PH, V = 0.5, 0.05, 0.5
H_THR, CARRY = 0.0, True
R, T, T_BURN, dt, seed, rec_every = 3000, 150.0, 40.0, 0.002, 4, 5
NSITES = int(np.ceil(V*T/d)) + 5
xbins = np.linspace(-60, 100 + (V*T if CARRY else 0), 801); dx = xbins[1] - xbins[0]; xm = 0.5*(xbins[1:] + xbins[:-1])

W0, W1 = base.Potential(P_.w0), base.Potential(P_.w1)
rng = np.random.default_rng(seed); noise = np.sqrt(2*P_.kBT*dt/P_.eta)
a_start = -(NSITES - 1)*d + rng.uniform(0, d, R)
X = -2.0 + rng.normal(size=R)*np.sqrt(P_.kBT/P_.w0["kl"]*0.5); y = 0.0
acc1 = np.zeros(len(xm)); acc0 = np.zeros(len(xm)); ns = 0; tau_s = 0.0; traj = []
for n in range(int(round(T/dt))):
    j = np.clip(np.rint((X - y - a_start)/d), 0, NSITES - 1)
    h = a_start + j*d + y - X                        # new window, new h
    al = (np.abs(h) <= d/2) & (h > H_THR)            # on/off: a function of h only
    if n % rec_every == 0:
        traj.append((n*dt, X[0], h[0], al[0]))
        if n*dt >= T_BURN:
            acc1 += np.histogram(X[al], xbins)[0]/(R*dx); acc0 += np.histogram(X[~al], xbins)[0]/(R*dx)
            tau_s += np.where(al, W1.dw(X), 0.0).mean(); ns += 1
    de = ~al if CARRY else np.ones(R, bool)
    if CARRY: X[al] += V*dt
    X[de] += -dt*W0.dw(X[de])/P_.eta + noise*rng.normal(size=de.sum())
    y += V*dt
p1, p0 = acc1/ns, acc0/ns
print(f"n_a = {p1.sum()*dx:.3f} (duty cycle (d/2-H_THR)/d = {(d/2 - H_THR)/d:.3f}); tau_c from p = {np.sum(W1.dw(xm)*p1)*dx:.4f} pN")
tr = np.array(traj); t_, Xt, ht, at = tr.T; z = t_ <= 160
fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
ax[0].plot(t_[z], Xt[z]); ax[0].set(ylabel="X (nm)")
ax[1].plot(t_[z], ht[z]); ax[1].axhline(H_THR, color="r", lw=0.7); ax[1].set(ylabel="h (nm)", ylim=(-d/2, d/2))
ax[2].step(t_[z], at[z], where="post"); ax[2].set(ylabel=r"$\alpha$", xlabel="t (ms)", yticks=[0, 1]); fig.tight_layout()
fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(xm, p1, label=r"$p(1,x)$"); ax.plot(xm, p0, label=r"$p(0,x)$")
ax.set(xlabel="x (nm)", ylabel=r"$p(\alpha,x)$ (nm$^{-1}$)"); ax.legend(); fig.tight_layout(); plt.show()
